import cv2
import numpy as np

def ordenar_esquinas(pts):
    # pts: (4,2)
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def detectar_tablero(frame):
    """Detecta el contorno del tablero como un cuadrilátero (4 esquinas) en el frame.
    Devuelve esquinas ordenadas (TL,TR,BR,BL) o None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Bordes
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    h, w = gray.shape[:2]
    min_area = 0.10 * (h*w)  # evita pequeños

    for cnt in cnts[:20]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4,2)
            return ordenar_esquinas(pts)

    return None

def warp_tablero(frame, esquinas, S=800, debug=True):
    """Warpea el tablero a un cuadrado SxS y devuelve (tablero, H).
    Aplica recorte automático de bordes negros si existen y actualiza H.
    """
    dst = np.array([[0,0],[S-1,0],[S-1,S-1],[0,S-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(esquinas.astype(np.float32), dst)
    board_warp = cv2.warpPerspective(frame, H, (S, S))
    
    # 1) Gray
    gray = cv2.cvtColor(board_warp, cv2.COLOR_BGR2GRAY)
    
    # 2) threshold 5th element
    t = max(10, int(np.percentile(gray, 5)))
    
    # 3) Mask > t
    mask = gray > t
    
    # 4) Count check
    if np.sum(mask) < 0.2 * S * S:
        return board_warp, H
        
    # 5) Bbox
    ys, xs = np.where(mask)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    # 6) Margin 1%
    margin = int(0.01 * S)
    x0 = max(0, x_min - margin)
    x1 = min(S, x_max + margin) # Fix S
    y0 = max(0, y_min - margin)
    y1 = min(S, y_max + margin)
    
    # 7) Validation (70% size)
    w_crop = x1 - x0
    h_crop = y1 - y0
    if w_crop < S * 0.7 or h_crop < S * 0.7:
        return board_warp, H
        
    # 8) Crop
    board_crop = board_warp[y0:y1, x0:x1]
    
    # 9) Resize back
    board_clean = cv2.resize(board_crop, (S, S), interpolation=cv2.INTER_LINEAR)
    
    # 10) Update H
    sx = S / float(w_crop)
    sy = S / float(h_crop)
    
    M_crop = np.array([
        [sx, 0, -x0 * sx],
        [0, sy, -y0 * sy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    H_new = M_crop @ H
    
    # -- DEBUG --
    if debug and not getattr(warp_tablero, '_debug_done', False):
        print(f"[warp_tablero] Auto-crop applied: x={x0}:{x1} y={y0}:{y1} (size {w_crop}x{h_crop})")
        warp_tablero._debug_done = True
        
    return board_clean, H_new
