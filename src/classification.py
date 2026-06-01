import cv2
import numpy as np
from src.features import extract_features_robust
from src.grid import roi_casilla, is_light_square

# --- C) Clasificador por matching ROBUSTO ---

def pawn_fallback(roi_bgr, empty_template, is_light_square):
    """
    Intenta detectar un peón (especialmente negro) usando background subtraction.
    Devuelve 'P' si detecta algo compatible con un peón, o None.
    """
    if empty_template is None:
        return None
        
    g_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g_roi, empty_template)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
        
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    h_roi, w_roi = g_roi.shape
    roi_area = h_roi * w_roi
    
    # Filtros de forma para Peón
    if area / roi_area < 0.05 or area / roi_area > 0.70:
        return None
        
    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = w / h
    extent = area / (w*h) 
    
    if 0.3 < aspect_ratio < 1.2 and extent > 0.4:
        return "P"
        
    return None

def classify_piece_by_akaze_proto(roi_bgr, db_proto, detector, matcher, ratio=0.75, 
                                  empty_template=None, is_light_square=None,
                                  init_mode=False, row=None, col=None, expected_type=None):
    """
    Versión ROBUSTA con fallback AKAZE -> ORB.
    Retorna: best_label, best_score, second_label, second_score, ratio, conf, nkp
    """
    # 1. Extracción Robusta
    method_used, kp, des, _ = extract_features_robust(roi_bgr)
    nkp = len(kp)
    
    # --- Debug Init Custom (User Request) ---
    debug_target = (init_mode and row == 1 and col in [3, 6, 7])
    
    best_label, best_score = "EMPTY", 0
    second_label, second_score = "None", 0
    use_fallback = False

    if method_used == "NONE" or des is None:
        if debug_target:
            print(f"[DEBUG Init r={row} c={col}] method={method_used} len(kp)={nkp} -> NO FEATURES")
        pass 
    else:
        label_scores = {}
        
        for label, protos in db_proto.items():
            max_good = 0
            for proto in protos:
                proto_des = None
                if method_used == "AKAZE":
                    proto_des = proto.get("des_akaze")
                elif method_used == "ORB":
                    proto_des = proto.get("des_orb")
                
                if proto_des is None:
                    continue
                    
                try:
                    matches = matcher.knnMatch(des, proto_des, k=2)
                except cv2.error:
                    continue
                    
                good_count = 0
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < ratio * n.distance:
                            good_count += 1
                
                if good_count > max_good:
                    max_good = good_count
            
            label_scores[label] = max_good
            
        sorted_scores = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = sorted_scores[0]
        second_label, second_score = sorted_scores[1] if len(sorted_scores) > 1 else ("None", 0)
        
        if debug_target:
            gap = best_score - second_score
            print(f"[DEBUG Init r={row} c={col}] Method={method_used} KP={nkp} "
                  f"Best={best_label}({best_score}) 2nd={second_label}({second_score}) Gap={gap}")

        if best_score < 8:
            use_fallback = True
            
    if nkp < 5:
        use_fallback = True

    # --- FALLBACK LOGIC ---
    allow_fallback = False
    if init_mode:
        if row in [1, 6]:
            allow_fallback = True
    else:
        if expected_type == "P":
            allow_fallback = True

    if use_fallback and allow_fallback and empty_template is not None:
        fb_res = pawn_fallback(roi_bgr, empty_template, is_light_square)
        if fb_res == "P":
            return "P", 999, "Fallback", 0, 9.9, 1.0, nkp
            
    ratio_val = best_score / (second_score + 1e-6)
    conf_val = best_score / (nkp + 1e-6)
    
    return best_label, best_score, second_label, second_score, ratio_val, conf_val, nkp

def classify_retry_aggressive(roi_bgr, db_proto, matcher):
    """
    Intento de recuperación agresiva con AdaptiveThreshold + ORB.
    Retorna: best_label, best_score, second_label, second_score, nkp
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Adaptive Threshold (Gaussian)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Opcional: dilate 1 iter
    thresh = cv2.dilate(thresh, None, iterations=1)

    # ORB directo
    orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8)
    kp, des = orb.detectAndCompute(thresh, None)
    
    if des is None or len(kp) < 5:
        return "EMPTY", 0, "None", 0, len(kp) if kp else 0
        
    # Match vs DB (ORB parts)
    label_scores = {}
    ratio = 0.75
    
    for label, protos in db_proto.items():
        max_good = 0
        for proto in protos:
            p_des = proto.get("des_orb")
            if p_des is None: continue
            
            try:
                matches = matcher.knnMatch(des, p_des, k=2)
            except: continue
            
            good = 0
            for m_n in matches:
                if len(m_n) == 2 and m_n[0].distance < ratio * m_n[1].distance:
                    good += 1
            if good > max_good: max_good = good
            
        label_scores[label] = max_good
        
    # Ranking
    sorted_scores = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    best, score = sorted_scores[0]
    sec, sec_score = sorted_scores[1] if len(sorted_scores) > 1 else ("None", 0)
    
    return best, score, sec, sec_score, len(kp)

def init_state_by_matching(board_canon_bgr, db_proto, detector, matcher, occ_grid, 
                           tpl_light=None, tpl_dark=None, S=800, margin=8):
    state = np.full((8,8), ".", dtype=object)
    print("--- Init State by Robust Matching (with Retry) ---")
    
    MIN_BEST = { "P": 8, "R": 10, "N": 10, "B": 10, "Q": 12, "K": 12, "EMPTY": 0 }
    MIN_RATIO = { "P": 1.10, "R": 1.20, "N": 1.20, "B": 1.20, "Q": 1.25, "K": 1.25, "EMPTY": 0.0 }
    HARD_BEST = { "P": 14, "R": 16, "N": 16, "B": 16, "Q": 20, "K": 20, "EMPTY": 999 }

    # Primer Pass
    for r in range(8):
        for c in range(8):
            roi = roi_casilla(board_canon_bgr, r, c, S=S, margin=None, pad_factor=0.12)
            is_light = is_light_square(r,c)
            tpl = tpl_light if is_light else tpl_dark
            
            pred = classify_piece_by_akaze_proto(roi, db_proto, detector, matcher, ratio=0.75,
                                                 empty_template=tpl, is_light_square=is_light,
                                                 init_mode=True, row=r, col=c)
            best_lbl, best_sc, sec_lbl, sec_sc, ratio_val, conf_val, nkp = pred
            
            final_lbl = "."
            if best_lbl != "EMPTY":
                mb = MIN_BEST.get(best_lbl, 12)
                mr = MIN_RATIO.get(best_lbl, 1.25)
                hb = HARD_BEST.get(best_lbl, 20)
                
                accepted = False
                if best_sc >= hb: accepted = True
                elif best_sc >= mb and ratio_val >= mr: accepted = True
                
                if accepted:
                    final_lbl = best_lbl
                    if r <= 1: final_lbl = final_lbl.lower()
                    elif r >= 6: final_lbl = final_lbl.upper()

            state[r,c] = final_lbl

    # Segundo Pass: Retry on Empty
    print("--- Init Retry Pass for Empty Squares ---")
    for r in range(8):
        for c in range(8):
            if state[r,c] != ".":
                continue
                
            candidates = []
            
            roi_a = roi_casilla(board_canon_bgr, r, c, S=S, margin=0) 
            roi_b = roi_casilla(board_canon_bgr, r, c, S=S, margin=2)
            
            # --- DEBUG VARIANTES ---
            for i, roi_var in enumerate([roi_a, roi_b]):
                res = classify_retry_aggressive(roi_var, db_proto, matcher)
                # res es (best, score, sec, sec_score, nkp)
                candidates.append(res)
                
                if r == 1 and c == 3:
                     lb, sc, slb, ssc, kn = res
                     gp = sc - ssc
                     nm = "Pad0" if i == 0 else "PadSmall"
                     print(f"[DEBUG Retry {nm} r=1 c=3] Method=AggressiveORB KP={kn} "
                           f"Best={lb}({sc}) 2nd={slb}({ssc}) Gap={gp}")

            candidates.sort(key=lambda x: x[1], reverse=True)
            best_res = candidates[0]
            
            lbl, score, sec_lbl, sec_score, nkp_retry = best_res
            
            if lbl != "EMPTY" and lbl != "None":
                gap = score - sec_score
                accepted_retry = False
                rule_matched = ""
                
                # --- LOGICA DE ACEPTACION (RETRY) ---
                # (3) Anti-Ghost de KP y Ambigüedad
                is_ambiguous = (sec_lbl != "EMPTY" and sec_lbl != "None" and gap == 0)
                is_low_kp = (nkp_retry < 8)
                
                if not is_low_kp and not is_ambiguous:
                    # (1) Strong Evidence
                    if score >= 8 and gap >= 2:
                        accepted_retry = True
                        rule_matched = "1 (Strong)"
                        
                    # (2) Moderate Evidence
                    elif (sec_lbl == "EMPTY" or sec_lbl == "None") and score >= 6 and gap >= 1 and nkp_retry >= 12:
                        accepted_retry = True
                        rule_matched = "2 (Moderate)"
                
                # --- DEBUG DECISION ---
                if r == 1 and c == 3:
                    if accepted_retry:
                        print(f"[DEBUG Retry Decision r=1 c=3] ACCEPTED (rule={rule_matched}, score={score}, gap={gap}, kp={nkp_retry})")
                    else:
                        print(f"[DEBUG Retry Decision r=1 c=3] REJECTED (score={score}, gap={gap}, sec={sec_lbl}, kp={nkp_retry})")

                if accepted_retry:
                     final_lbl = lbl
                     if r <= 1: final_lbl = final_lbl.lower()
                     elif r >= 6: final_lbl = final_lbl.upper()
                     
                     state[r,c] = final_lbl
                     print(f"INIT_RETRY filled ({r},{c}) label={final_lbl} score={score} gap={gap} rule={rule_matched}")

    # Print state final
    for r in range(8):
        row_s = "".join([f"{str(x)} " for x in state[r]])
        print(f"Row {r}: {row_s}")

    return state
