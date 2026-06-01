import cv2
import numpy as np
from src.geometry import warp_tablero
from src.occupancy import build_empty_templates

def inicializar_contexto_desde_video(
    cap,
    detectar_tablero_fn,
    S=800,
    margin=12,
    max_frames=800,
    step=5
):
    """
    Busca el primer frame del vídeo donde se detecta correctamente el tablero.
    Fija:
      - homografía H
      - plantillas de casillas vacías (claras y oscuras)
    Devuelve: H, tpl_light, tpl_dark, idx_frame
    """

    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # probamos solo cada 'step' frames para ir rápido
        if i % step != 0:
            continue

        esquinas = detectar_tablero_fn(frame)
        if esquinas is None:
            continue

        # Warp del tablero
        tablero, H = warp_tablero(frame, esquinas, S=S)

        # Construir plantillas de casillas vacías
        tpl_light, tpl_dark = build_empty_templates(
            tablero, S=S, margin=margin
        )

        idx_frame = start_pos + i
        return H, tpl_light, tpl_dark, idx_frame

    raise RuntimeError(
        f"No se detectó el tablero en los primeros {max_frames} frames."
    )
