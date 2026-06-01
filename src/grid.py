import numpy as np

def casilla_a_roi(r, c, S=800):
    cell = S // 8
    x0, y0 = c*cell, r*cell
    x1, y1 = (c+1)*cell, (r+1)*cell
    return x0, y0, x1, y1

def rc_a_alg(r, c):
    # r=0 arriba (8), r=7 abajo (1)
    file_ = "abcdefgh"[c]
    rank_ = str(8 - r)
    return f"{file_}{rank_}"

def roi_casilla(tablero, r, c, S=800, margin=None, pad_factor=0.12):
    """
    Extrae ROI de la casilla (r,c).
    Si margin es None, se calcula como pad_factor * cell_size.
    """
    cell = S // 8
    x0, y0 = c*cell, r*cell
    x1, y1 = (c+1)*cell, (r+1)*cell
    
    if margin is None:
        margin = int(cell * pad_factor)
        
    x0 += margin; y0 += margin
    x1 -= margin; y1 -= margin
    
    # Clip por seguridad
    h, w = tablero.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w, x1); y1 = min(h, y1)
    
    return tablero[y0:y1, x0:x1]

def is_light_square(r,c):
    return (r+c) % 2 == 0
