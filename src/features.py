import os
from glob import glob
import cv2
import numpy as np

# --- A) utilitaria: extract_features_robust ---
def extract_features_robust(roi_bgr):
    """
    Extracción de características robusta para piezas con poca textura.
    Pipeline:
    1. Gray
    2. Resize x2 si es pequeño (<120px) medante INTER_CUBIC
    3. CLAHE (clip=2.0, tile=(8,8))
    4. AKAZE (si >= 15 KPs) -> Retorna 'AKAZE'
    5. Fallback: ORB (800 feats) -> Retorna 'ORB'
    Devuelve: (method_used, kp, des, gray_processed)
    """
    # 1. Convertir a gray
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Si roi es pequeña (<120 px de lado), reescalar x2
    h, w = gray.shape
    if h < 120 or w < 120:
        gray = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        
    # 3. Aplicar CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. Intentar AKAZE
    ak = cv2.AKAZE_create()
    kp, des = ak.detectAndCompute(enhanced, None)
    
    if des is not None and len(kp) >= 15:
        return "AKAZE", kp, des, enhanced
        
    # 5. Intentar ORB
    orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8)
    kp_orb, des_orb = orb.detectAndCompute(enhanced, None)
    
    if des_orb is not None and len(kp_orb) > 0:
        return "ORB", kp_orb, des_orb, enhanced
        
    return "NONE", [], None, enhanced


# --- B) Carga del dataset (DUAL: AKAZE + ORB) ---
def cargar_dataset_prototypes(dataset_dir, max_protos=50):
    # Detectores para pre-cálculo
    ak = cv2.AKAZE_create()
    orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8)
    
    db_proto = {}
    
    print("Cargando dataset de PROTOTIPOS (Dual AKAZE + ORB)...")
    
    if not os.path.exists(dataset_dir):
        print(f"Warning: {dataset_dir} no existe.")
        return {}
        
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for folder in subdirs:
        label_internal = folder
        valid_labels = ["P", "N", "B", "R", "Q", "K", "EMPTY"]
        if label_internal not in valid_labels:
            continue
            
        if label_internal not in db_proto:
            db_proto[label_internal] = []
            
        protos = db_proto[label_internal]
        path_pattern = os.path.join(dataset_dir, folder, "*.*")
        files = glob(path_pattern)
        
        count_added = 0
        for fpath in files:
            if len(protos) >= max_protos:
                break
                
            img = cv2.imread(fpath)
            if img is None: continue
            
            # --- Preprocesado consistente con extract_features_robust ---
            # 1, 2, 3
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            if h < 120 or w < 120:
                gray = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Calcular AMBOS descriptores
            kp_a, des_a = ak.detectAndCompute(enhanced, None)
            kp_o, des_o = orb.detectAndCompute(enhanced, None)
            
            # Guardamos si al menos uno es válido
            valid_a = (des_a is not None and len(kp_a) >= 5)
            valid_o = (des_o is not None and len(kp_o) >= 5)
            
            if valid_a or valid_o:
                protos.append({
                    "name": os.path.basename(fpath),
                    "des_akaze": des_a, # Puede ser None
                    "des_orb": des_o,   # Puede ser None
                    "kp_akaze_n": len(kp_a) if kp_a else 0,
                    "kp_orb_n": len(kp_o) if kp_o else 0
                })
                count_added += 1
                
        print(f"  [{folder}]: {count_added} prototipos cargados.")
            
    return db_proto

# Se inicializan al importar el modulo
ak_global = cv2.AKAZE_create() 
matcher_global = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
