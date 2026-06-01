import numpy as np
from src.grid import rc_a_alg

def move_to_san(o_rc, d_rc, piece_type, is_capture, side_to_move, enroque=None):
    """
    Genera notación SAN mínima.
    piece_type: P, N, B, R, Q, K (Standard). Si es '.' se asume P.
    """
    if enroque:
        return enroque[1] # "O-O" o "O-O-O"
        
    # Corrección de tipo: si viene vacío o desconocido, asumimos Peón para evitar ".xh5"
    pt = piece_type
    if pt == "." or pt is None:
        pt = "P"
        
    # Usar mayúsculas para determinar la letra SAN
    pt_upper = pt.upper()
    
    # P -> ""
    if pt_upper == "P":
        san_letter = ""
    else:
        san_letter = pt_upper
    
    dest = rc_a_alg(*d_rc)
    
    if pt_upper == "P":
        if is_capture:
            # file_from + x + dest
            file_from = "abcdefgh"[o_rc[1]]
            return f"{file_from}x{dest}"
        else:
            return dest
    else:
        # Pieza mayor
        capture_char = "x" if is_capture else ""
        return f"{san_letter}{capture_char}{dest}"

def commit_move_logic(board, from_sq, to_sq, matched_label, occ_from, is_capture, side_to_move, enroque_type=None):
    """
    Centraliza la lógica de aplicar el movimiento y generar el log.
    Maneja self-healing de peones y uso de pieza visual si la memoria está vacía.
    """
    st = board.copy()
    log_notes = []
    san_move = ""
    
    # --- CASTLING ---
    if enroque_type:
        # Fila según turno
        row = 7 if side_to_move == "W" else 0
        
        # Simbolos Correctos
        k_char = "K" if side_to_move == "W" else "k"
        r_char = "R" if side_to_move == "W" else "r"
        
        # Mover Rey y Torre explícitamente
        if enroque_type == "O-O":
            # Rey e->g (4->6)
            st[row, 6] = st[row, 4] if st[row, 4] != '.' else k_char
            st[row, 4] = "."
            # Torre h->f (7->5)
            st[row, 5] = st[row, 7] if st[row, 7] != '.' else r_char
            st[row, 7] = "."
        elif enroque_type == "O-O-O":
            # Rey e->c (4->2)
            st[row, 2] = st[row, 4] if st[row, 4] != '.' else k_char
            st[row, 4] = "."
            # Torre a->d (0->3)
            st[row, 3] = st[row, 0] if st[row, 0] != '.' else r_char
            st[row, 0] = "."
            
        san_move = enroque_type
        # Log formateado robusto
        color_log = "White" if side_to_move == "W" else "Black"
        log_text = f"{side_to_move}: {san_move} ({color_log} Castle)"
        return st, log_text, san_move

    # --- NORMAL MOVE ---
    piece_mem = st[from_sq]
    
    # --- HARD ANTI-GHOST RULE (STRICT) ---
    # Si board[from] == '.' y NO es Peón (ni Castle), RECHAZAR.
    if piece_mem == "." and matched_label != "P":
        o_alg = rc_a_alg(*from_sq)
        d_alg = rc_a_alg(*to_sq)
        reason = f"REJECT from_empty_non_pawn (label={matched_label}, from={o_alg}, to={d_alg})"
        return board, reason, None

    # --- REGLA EXTRA: REY FANTASMA ---
    # Comparar insensible a Mayúsculas
    if matched_label == 'K' and (piece_mem == "." or piece_mem.upper() != 'K') and (not occ_from):
        o_alg = rc_a_alg(*from_sq)
        d_alg = rc_a_alg(*to_sq)
        reason = f"REJECT ghost_king (label=K, mem={piece_mem}, from={o_alg})"
        return board, reason, None

    piece_effective = piece_mem
    
    # 1. Sincronización Memoria vs Visual
    if piece_mem == ".":
        # Memoria vacía: intentamos usar visual
        if matched_label and matched_label != "EMPTY":
            
            # Asignar color (inferencia)
            # Si aparece una pieza de la nada en medio del juego, asumimos que es del color que mueve (side_to_move), 
            if side_to_move == "B":
                piece_effective = matched_label.lower()
            else:
                piece_effective = matched_label.upper() # Visual ya es upper, pero por seguridad
            
            # Caso especial: Peón + Ocupación = Self-Heal (Persistir)
            # Peón siempre es 'P' o 'p'
            if matched_label == "P" and occ_from:
                # Usar el color efectivo calculado
                st[from_sq] = piece_effective
                log_notes.append(f"[SELF-HEAL: filled from-square as {piece_effective}]")
            else:
                # Otros: Usar para este movimiento pero avisar (No persistir en from, se mueve directo)
                log_notes.append(f"[WARN board_from_empty; printed_using_visual={piece_effective}]")
        else:
            # Fallback total (Peón del color que toca)
            piece_effective = "p" if side_to_move == "B" else "P"
            log_notes.append(f"[WARN board_from_empty; defaulting to {piece_effective}]")
            
    elif matched_label and matched_label != "EMPTY" and piece_mem.upper() != matched_label.upper():
        # Conflicto real (Memoria dice X, Visual dice Y)
        # Comparación insensible a mayúsculas
        log_notes.append(f"[CONFLICT: Visual={matched_label} vs Memory={piece_mem}]")

    # 2. Aplicar movimiento
    st[to_sq] = piece_effective
    st[from_sq] = "."
    
    # 3. Generar SAN (move_to_san ya maneja .upper())
    san_move = move_to_san(from_sq, to_sq, piece_effective, is_capture, side_to_move)
    
    # 4. Construir Log
    o_alg = rc_a_alg(*from_sq)
    d_alg = rc_a_alg(*to_sq)
    notes_str = "".join(log_notes)
    log_text = f"{side_to_move}: {san_move} ({o_alg}->{d_alg}){notes_str}"
    
    # 5. Debug Assert (Opcional)
    if st[to_sq] == '.':
        log_text += " [BOARD_DESYNC: Dest empty after move]"
        
    return st, log_text, san_move
