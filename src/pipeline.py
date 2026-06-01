import cv2
import numpy as np
import sys
import os

from src.config import *
from src.geometry import detectar_tablero, warp_tablero
from src.grid import roi_casilla, is_light_square
from src.occupancy import calcular_ocupacion_diff_por_color
from src.features import cargar_dataset_prototypes, ak_global, matcher_global
from src.classification import init_state_by_matching, classify_piece_by_akaze_proto
from src.motion import (farneback_mag, mag_por_casilla, detect_castle_by_motion, 
                        movimiento_con_captura, key_mov, validar_movimiento_con_farneback)
from src.chess_logic import commit_move_logic
from src.video_processing import inicializar_contexto_desde_video

def flip_turn(color):
    return 'B' if color == 'W' else 'W'

def run_pipeline():
    # Load dataset
    db_proto = cargar_dataset_prototypes("dataset")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo. Revisa VIDEO_PATH: {VIDEO_PATH}")

    # INICIALIZACIÓN
    H, tpl_light, tpl_dark, idx0 = inicializar_contexto_desde_video(
        cap, detectar_tablero, S=S, margin=MARGIN, max_frames=800, step=5
    )
    print("Inicialización OK en frame:", idx0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)

    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("No se pudo leer el frame inicial.")

    tab_prev = cv2.warpPerspective(frame0, H, (S, S))
    occ_prev, _, _, _ = calcular_ocupacion_diff_por_color(
        tab_prev, tpl_light, tpl_dark, S=S, margin=MARGIN, debug=True, factor=FACTOR
    )

    idx_prev = idx0
    idx = idx0

    cand = None
    cand_count = 0
    movs = []

    # Estado lógico (verdad absoluta)
    state_prev = init_state_by_matching(tab_prev, db_proto, ak_global, matcher_global, occ_prev, 
                                        tpl_light=tpl_light, tpl_dark=tpl_dark, S=S, margin=MARGIN)

    # Variable de turno (W empieza)
    side_to_move = "W"
    pending_castle = None
    castle_cooldown_until_frame = -1

    # VIDEO WRITER
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if not fps_input or fps_input == 0:
        fps_input = 30.0

    # Reducimos los fps de salida para que no vaya tan rapido
    fps_output = FPS_OUTPUT

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("salida_anotada.mp4", fourcc, fps_output, (S, S))
    if not out.isOpened():
        raise RuntimeError("No se pudo crear salida_anotada.mp4")

    overlay_text = ""
    overlay_until = -1

    # Buffer para recuperación de Enroques (Resync)
    history_mag = [] 
    HIST_LEN = 10 

    # LOOP PRINCIPAL
    try:
        while True:
            frame = None
            for _ in range(STEP):
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1

            if not ret or frame is None:
                break

            tab_cur = cv2.warpPerspective(frame, H, (S, S))
            occ_cur, _, _, _ = calcular_ocupacion_diff_por_color(
                tab_cur, tpl_light, tpl_dark, S=S, margin=MARGIN, debug=True, factor=FACTOR
            )

            try:
                # 1. Calcular Flujo (Farneback) para TODO análisis de movimiento
                mag_flow = farneback_mag(tab_prev, tab_cur)
                mcell = mag_por_casilla(mag_flow, S=S, margin=MARGIN)
                
                # Store history for resync
                history_mag.append({'idx': idx, 'mcell': mcell})
                if len(history_mag) > HIST_LEN:
                    history_mag.pop(0)

                # CHECK CASTLE (MOTION PATTERN) DIRECTLY
                # Antes que el movimiento normal, ver si es un enroque del turno actual
                if idx >= castle_cooldown_until_frame:
                    c_res_motion = detect_castle_by_motion(mcell, side_to_move)
                    
                    if c_res_motion:
                        c_type, kf, kt, rf, rt = c_res_motion
                        
                        state_prev, log_text, san_res = commit_move_logic(
                            state_prev, None, None, None, False, False, side_to_move, enroque_type=c_type
                        )
                        castle_cooldown_until_frame = idx + CASTLE_COOLDOWN
                        
                        print(f"Frame {idx}: {log_text} [CASTLE DETECTED by MOTION pattern]")
                        movs.append({"start": idx_prev, "end": idx, "log": log_text, "san": san_res, "color": side_to_move})
                        overlay_text = log_text
                        
                        overlay_until = idx + int(3.0 * fps_output * STEP)
                        
                        prev_side = side_to_move
                        side_to_move = flip_turn(side_to_move)
                        print(f"TURN_UPDATE: prev={prev_side} (Castle) -> new={side_to_move}")
                        
                        occ_prev = occ_cur
                        tab_prev = tab_cur
                        idx_prev = idx
                        cand = None
                        cand_count = 0
                        continue


                # MOVIMIENTO NORMAL / CAPTURA
                o_rc, d_rc, is_capture = movimiento_con_captura(
                    tab_prev, tab_cur, occ_prev, occ_cur, S=S, margin=MARGIN
                )

                k = key_mov(o_rc, d_rc)
                if k is None:
                    occ_prev = occ_cur
                    tab_prev = tab_cur
                    idx_prev = idx
                    continue

                if cand == k:
                    cand_count += 1
                else:
                    cand = k
                    cand_count = 1

                if cand_count < CONFIRM:
                    continue

                o = (cand[0], cand[1])
                d = (cand[2], cand[3])

                # VALIDAR FARNEBACK 
                # Reimplementamos validación con mcell pre-cal
                flat_idx = np.argsort(mcell.ravel())[::-1]
                top_rc = [tuple(map(int, np.unravel_index(i, (8, 8)))) for i in flat_idx[:8]]
                top_set = set(top_rc)
                
                ok_fb = (o in top_set) and (d in top_set)
                if not ok_fb:
                    cand = None; cand_count = 0
                    occ_prev = occ_cur; tab_prev = tab_cur; idx_prev = idx
                    continue

                # NOTACIÓN CON COLOR (SAN)
                pieza_memoria = state_prev[o] 
                
                # Matching Checks
                roi_o_prev = roi_casilla(tab_prev, o[0], o[1], S=S, margin=None, pad_factor=0.12)
                
                is_light = is_light_square(o[0], o[1])
                tpl = tpl_light if is_light else tpl_dark
                pred_o = classify_piece_by_akaze_proto(roi_o_prev, db_proto, ak_global, matcher_global, ratio=0.75,
                                                    empty_template=tpl, is_light_square=is_light,
                                                    init_mode=False, expected_type=pieza_memoria.upper())
                best_lbl, best_sc, sec_lbl, sec_sc, ratio_val, conf_val, nkp = pred_o
                
                MIN_BEST = { "P": 8, "R": 10, "N": 10, "B": 10, "Q": 12, "K": 12, "EMPTY": 0 }
                MIN_RATIO = { "P": 1.10, "R": 1.20, "N": 1.20, "B": 1.20, "Q": 1.25, "K": 1.25, "EMPTY": 0.0 }
                HARD_BEST = { "P": 14, "R": 16, "N": 16, "B": 16, "Q": 20, "K": 20, "EMPTY": 999 }
                
                visual_label = "EMPTY"
                accepted = False
                
                if best_lbl != "EMPTY":
                    mb = MIN_BEST.get(best_lbl, 12)
                    mr = MIN_RATIO.get(best_lbl, 1.25)
                    hb = HARD_BEST.get(best_lbl, 20)
                    
                    if best_sc >= hb: accepted = True
                    elif best_sc >= mb and ratio_val >= mr: accepted = True
                        
                    if accepted: visual_label = best_lbl

                # Ghost Check
                if visual_label == "EMPTY":
                    # Guardarrail occ
                    if occ_prev[o] and len(movs) < EARLY_HEAL_MOVES:
                        visual_label = "P" # Force accept P
                    else:
                        print(f"Frame {idx}: REJECT Ghost from_empty (sc={best_sc})")
                        cand = None; cand_count = 0
                        occ_prev = occ_cur; tab_prev = tab_cur; idx_prev = idx
                        continue

                # Validación de turno y Resync INTELIGENTE (Castle Recovery)
                piece_from = state_prev[o]
                actual_side = side_to_move
                skip_standard_toggle = False
                
                if piece_from != ".":
                    color_real = "W" if piece_from.isupper() else "B"
                    
                    if color_real != side_to_move:
                        dest_empty = (state_prev[d] == ".")
                        move_confident = (is_capture or dest_empty)
                        
                        if move_confident:
                            # DETECTADA JUGADA FUERA DE TURNO -> ¿PERDIMOS UN ENROQUE?
                            missed_side = side_to_move # El turno que nos saltamos
                            print(f"Frame {idx}: [TURN_RESYNC TRIGGER] {color_real} moved, expected {missed_side}. Checking history for missed Castle...")
                            
                            found_castle = False
                            for h_item in reversed(history_mag):
                                # Check si en algun frame reciente hubo enroque del missing side
                                c_res = detect_castle_by_motion(h_item['mcell'], missed_side, th_ratio=0.4) # th mas bajo
                                if c_res:
                                    # FOUND MISSED CASTLE!
                                    c_type, _, _, _, _ = c_res
                                    c_idx = h_item['idx']
                                    
                                    # Apply OLD castle first
                                    state_prev, log_c, san_res_c = commit_move_logic(
                                        state_prev, None, None, None, False, False, missed_side, enroque_type=c_type
                                    )
                                    print(f"   >>> RECOVERED MISSED CASTLE: {log_c} (found at frame {c_idx})")
                                    movs.append({"start": c_idx, "end": c_idx+STEP, "log": log_c + " [RECOVERED]", "san": san_res_c, "color": missed_side})
                                    
                                    side_to_move = flip_turn(missed_side)
                                    actual_side = side_to_move 
                                    
                                    found_castle = True
                                    break
                            
                            if not found_castle:
                                # No castle found, just standard resync
                                print(f"   >>> No missed castle found. Forcing Resync.")
                                actual_side = color_real
                                skip_standard_toggle = True # Manual toggle
                        else:
                            cand = None; cand_count = 0
                            occ_prev = occ_cur; tab_prev = tab_cur; idx_prev = idx
                            continue

                # Commit Normal
                state_new, log_text, san_res = commit_move_logic(
                    state_prev, o, d, visual_label, occ_prev[o], is_capture, actual_side
                )

                state_prev = state_new
                if skip_standard_toggle: log_text += " [RESYNC]"
                        
                print(f"Frame {idx_prev}->{idx}: {log_text} [COMMITTED]")
                movs.append({"start": idx_prev, "end": idx, "log": log_text, "san": san_res, "color": actual_side})
                overlay_text = log_text
                # Duración overlay ~3 seg en salida
                overlay_until = idx + int(3.0 * fps_output * STEP)
                
                # Turn Update
                if not skip_standard_toggle:
                    side_to_move = flip_turn(actual_side)
                else:
                    side_to_move = flip_turn(actual_side)
                
                print(f"TURN_UPDATE: -> new={side_to_move}")

                occ_prev = occ_cur
                tab_prev = tab_cur
                idx_prev = idx
                cand = None
                cand_count = 0

            finally:
                # ----- overlay -----
                frame_vis = tab_cur.copy()
                if idx <= overlay_until and overlay_text:
                    cv2.rectangle(frame_vis, (10, 10), (780, 70), (0, 0, 0), -1)
                    cv2.putText(frame_vis, overlay_text, (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 255, 255), 2, cv2.LINE_AA)
                out.write(frame_vis)
            tab_prev = tab_cur
            idx_prev = idx
            cand = None
            cand_count = 0

    except KeyboardInterrupt:
        print("Interrumpido por el usuario...")

    # CIERRE
    cap.release()
    out.release()

    with open("jugadas_detectadas.txt", "w", encoding="utf-8") as f:
        move_counter = 1
        i = 0
        while i < len(movs):
            m = movs[i]
            san = m.get('san', '?')
            color = m.get('color', 'W')
            
            if color == 'W':
                line = f"{move_counter}. {san}"
                # Check next is Black
                if i + 1 < len(movs):
                    next_m = movs[i+1]
                    if next_m.get('color') == 'B':
                        line += f" {next_m['san']}"
                        i += 1
                f.write(line + "\n")
                move_counter += 1
            else:
                # Black unexpected start or out of sync
                f.write(f"{move_counter}... {san}\n")
                move_counter += 1
            i += 1

    print("Guardado en jugadas_detectadas.txt")
    print("Guardado en salida_anotada.mp4")

if __name__ == "__main__":
    run_pipeline()
