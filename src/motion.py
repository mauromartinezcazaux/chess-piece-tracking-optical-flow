import cv2
import numpy as np
from src.grid import casilla_a_roi

def cambio_por_casilla(tab1, tab2, S=800, margin=8):
    g1 = cv2.cvtColor(tab1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(tab2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)

    scores = np.zeros((8,8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            x0,y0,x1,y1 = casilla_a_roi(r,c,S)
            x0 += margin; y0 += margin; x1 -= margin; y1 -= margin
            scores[r,c] = float(np.mean(diff[y0:y1, x0:x1]))
    return scores

def detectar_enroque(occ1, occ2):
    vaciado = np.argwhere((occ1 == True) & (occ2 == False))
    llenado = np.argwhere((occ1 == False) & (occ2 == True))
    if len(vaciado) == 2 and len(llenado) == 2:
        return [tuple(x) for x in vaciado], [tuple(x) for x in llenado]
    return None

def detectar_enroque_agnostico(vac, ll):
    """
    Detecta enroque sin saber el turno.
    Retorna (color, tipo) ej: ("W", "O-O") o None.
    """
    vac = set((int(r), int(c)) for r,c in vac)
    ll  = set((int(r), int(c)) for r,c in ll)

    # Check White (Row 7)
    if (7, 4) in vac: # King e1
        if (7, 6) in ll: return ("W", "O-O")
        if (7, 2) in ll: return ("W", "O-O-O")
    
    # Check Black (Row 0)
    if (0, 4) in vac: # King e8
        if (0, 6) in ll: return ("B", "O-O")
        if (0, 2) in ll: return ("B", "O-O-O")
        
    return None

def destino_por_max_cambio(tab1, tab2, occ1, o_rc, S=800, margin=8):
    scores = cambio_por_casilla(tab1, tab2, S=S, margin=margin)
    idxs = np.dstack(np.unravel_index(np.argsort(scores.ravel())[::-1], (8,8)))[0]
    for r,c in idxs:
        r,c = int(r), int(c)
        if (r,c) == o_rc:
            continue
        if occ1[r,c]:
            return (r,c)
    return None

def movimiento_con_captura(tab1, tab2, occ1, occ2, S=800, margin=8, max_cambios=6):
    vaciado = np.argwhere((occ1 == True) & (occ2 == False))
    llenado = np.argwhere((occ1 == False) & (occ2 == True))

    if len(vaciado) + len(llenado) > max_cambios:
        return None, None, False

    if len(vaciado) == 1 and len(llenado) == 1:
        return tuple(vaciado[0]), tuple(llenado[0]), False

    if len(vaciado) == 1 and len(llenado) == 0:
        o = tuple(vaciado[0])
        d = destino_por_max_cambio(tab1, tab2, occ1, o, S=S, margin=margin)
        if d is not None:
            return o, d, True

    return None, None, False

def key_mov(o_rc, d_rc):
    if o_rc is None or d_rc is None:
        return None
    return (int(o_rc[0]), int(o_rc[1]), int(d_rc[0]), int(d_rc[1]))

def farneback_mag(tab1_bgr, tab2_bgr):
    """
    Calcula magnitud de flujo óptico Farneback en tablero canónico.
    Devuelve matriz mag (SxS) float32.
    """
    g1 = cv2.cvtColor(tab1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(tab2_bgr, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    return mag


def mag_por_casilla(mag, S=800, margin=8):
    """
    Agrega magnitud de flujo por casilla 8x8 (media).
    """
    scores = np.zeros((8,8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            x0, y0, x1, y1 = casilla_a_roi(r, c, S)
            x0 += margin; y0 += margin
            x1 -= margin; y1 -= margin
            scores[r,c] = float(np.mean(mag[y0:y1, x0:x1]))
    return scores


def validar_movimiento_con_farneback(o_rc, d_rc, tab_prev, tab_cur, S=800, margin=8, topk=8, min_mag=0.20):
    """
    Valida que origen y destino están entre las top-k casillas con mayor magnitud
    y que su magnitud supera un umbral mínimo (min_mag).
    """
    if o_rc is None or d_rc is None:
        return False, None

    mag = farneback_mag(tab_prev, tab_cur)
    mcell = mag_por_casilla(mag, S=S, margin=margin)

    flat_idx = np.argsort(mcell.ravel())[::-1]
    top_rc = [tuple(map(int, np.unravel_index(i, (8, 8)))) for i in flat_idx[:topk]]
    top_set = set(top_rc)

    o_rc = (int(o_rc[0]), int(o_rc[1]))
    d_rc = (int(d_rc[0]), int(d_rc[1]))

    ok = (o_rc in top_set) and (d_rc in top_set)
    ok = ok and (mcell[o_rc] >= min_mag) and (mcell[d_rc] >= min_mag)

    return ok, mcell

def detect_castle_by_motion(change_matrix, side_to_move, th_ratio=0.5):
    """
    Detecta patrón de enroque basado en las casillas con mayor movimiento.
    change_matrix: 8x8 float (mag de flujo/diff)
    Returns: (castle_type, k_from, k_to, r_from, r_to) or None
    """
    # 1. Definir patrones (r, c)
    if side_to_move == "W":
        row = 7
        # Short: e1->g1 (K), h1->f1 (R) => Active: e1,h1,f1,g1
        pat_short = {(row,4), (row,7), (row,5), (row,6)}
        # Long: e1->c1 (K), a1->d1 (R) => Active: e1,a1,c1,d1
        pat_long  = {(row,4), (row,0), (row,2), (row,3)}
    else:
        row = 0
        pat_short = {(row,4), (row,7), (row,5), (row,6)}
        pat_long  = {(row,4), (row,0), (row,2), (row,3)}
        
    # 2. Obtener casillas activas (Top 4 score absoluto)
    flat_idx = np.argsort(change_matrix.ravel())[::-1]
    # threshold dinámico: al menos X% del máximo
    max_val = change_matrix.max()
    threshold = max_val * th_ratio
    
    # Tomamos candidatos que superen umbral, mínimo 4 para castle
    extracted = []
    for idx_flat in flat_idx:
        val = change_matrix.ravel()[idx_flat]
        if val < threshold and len(extracted) >= 4:
            break
        r_i, c_i = np.unravel_index(idx_flat, (8,8))
        extracted.append((r_i, c_i))
        
    active_set = set(extracted[:4]) # Miramos las 4 más fuertes
    
    # 3. Match
    # Relajamos: si 3 de 4 coinciden y el score es alto, a veces falta 1
    # Pero para robustez exigimos 4/4 o intersección muy fuerte
    if len(active_set.intersection(pat_short)) == 4:
        return ("O-O", (row,4), (row,6), (row,7), (row,5))
        
    if len(active_set.intersection(pat_long)) == 4:
        return ("O-O-O", (row,4), (row,2), (row,0), (row,3))
        
    return None

def get_diff_score(roi1, roi2):
    g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    return float(np.mean(diff)) / 255.0
