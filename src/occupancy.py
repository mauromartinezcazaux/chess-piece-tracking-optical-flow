import cv2
import numpy as np
from src.grid import roi_casilla, is_light_square

def build_empty_templates(tablero, S=800, margin=8, sample_rows=(2,3,4,5)):
    light, dark = [], []
    for r in sample_rows:
        for c in range(8):
            patch = roi_casilla(tablero, r, c, S=S, margin=margin)
            g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            (light if is_light_square(r,c) else dark).append(g)
    if len(light) < 4 or len(dark) < 4:
        raise RuntimeError("No hay suficientes casillas vacías para plantillas de fondo.")
    tpl_light = np.mean(np.stack(light, axis=0), axis=0).astype(np.uint8)
    tpl_dark  = np.mean(np.stack(dark,  axis=0), axis=0).astype(np.uint8)
    return tpl_light, tpl_dark

def grad_mag(gray_u8):
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def occupancy_score_diff(patch_bgr, tpl_gray):
    # robusto a highlights: diferencia estructural (gradiente)
    g = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    # Calculamos diff absoluta con el template
    diff = cv2.absdiff(g, tpl_gray)
    
    # Suavizamos la diferencia
    diff = cv2.GaussianBlur(diff, (3,3), 0)
    
    # Calculamos magnitud del gradiente sobre la diferencia
    # Esto resalta los bordes de los objetos que son diferentes al fondo
    mg = grad_mag(diff)
    
    return float(np.mean(mg))

def threshold_kmeans_1d(x_flat, factor=0.10):
    x = np.array(x_flat, dtype=np.float32).reshape(-1,1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    _, _, centers = cv2.kmeans(x, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    c0, c1 = sorted(centers.flatten())  # bajo=vacío, alto=ocupado
    return float(c0 + (c1 - c0) * factor)

def calcular_ocupacion_diff_por_color(tablero, tpl_light, tpl_dark, S=800, margin=8, factor=0.10, debug=False):
    scores = np.zeros((8,8), dtype=np.float32)
    light_scores, dark_scores = [], []

    for r in range(8):
        for c in range(8):
            patch = roi_casilla(tablero, r, c, S=S, margin=margin)
            tpl = tpl_light if is_light_square(r,c) else tpl_dark
            s = occupancy_score_diff(patch, tpl)
            scores[r,c] = s
            (light_scores if is_light_square(r,c) else dark_scores).append(s)

    thr_light = threshold_kmeans_1d(light_scores, factor=factor)
    thr_dark  = threshold_kmeans_1d(dark_scores,  factor=factor)

    occ = np.zeros((8,8), dtype=bool)
    for r in range(8):
        for c in range(8):
            thr = thr_light if is_light_square(r,c) else thr_dark
            occ[r,c] = scores[r,c] > thr

    if debug:
        return occ, scores, thr_light, thr_dark
    return occ, scores
