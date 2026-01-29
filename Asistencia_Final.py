import os
import cv2
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURACIÓN ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "dataset"
BASE_ALUMNOS = "Base_Alumnos.xlsx"
EXCEL_ASISTENCIA = "Registro_Asistencia.xlsx"
UMBRAL = 0.80  # Ajusta según qué tan estricto quieras el reconocimiento
FRAMES_CONFIRMACION = 3

# --- 2. INICIALIZAR MODELOS IA ---
print(f"Iniciando sistema en {DEVICE}...")
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# --- 3. VINCULACIÓN EXCEL - DATASET ---
print("Cargando base de datos de alumnos...")
df_base = pd.read_excel(BASE_ALUMNOS)

# Creamos el nombre de la carpeta como aparece en tu dataset para mapear al ID
# Formato esperado: ApellidoPaterno_ApellidoMaterno_Nombres
df_base['Folder_Name'] = (df_base['Apellido Paterno'] + "_" + 
                          df_base['Apellido Materno'].fillna('') + "_" + 
                          df_base['Nombres']).str.replace('__', '_').str.strip('_')

# Diccionario para obtener ID rápido: { 'Nombre_Carpeta': 'ID' }
mapa_ids = dict(zip(df_base['Folder_Name'], df_base['ID / Matrícula']))

# --- 4. CARGAR ROSTROS (EMBEDDINGS) ---
db_rostros = {}
print("Procesando imágenes del dataset...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path): continue
    
    embeddings = []
    for img_name in os.listdir(person_path):
        img = cv2.imread(os.path.join(person_path, img_name))
        if img is None: continue
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)
        
        if face is not None:
            with torch.no_grad():
                emb = model(face.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
                embeddings.append(emb)
    
    if embeddings:
        db_rostros[person] = embeddings

# --- 5. FUNCIONES DE APOYO ---
def registrar_asistencia(nombre_carpeta):
    matricula = mapa_ids.get(nombre_carpeta, "DESCONOCIDO")
    ahora = datetime.now()
    
    nuevo = pd.DataFrame({
        "Fecha": [ahora.strftime("%Y-%m-%d")],
        "ID / Matrícula": [matricula],
        "Nombre Completo": [""], # Excel llenará esto con la fórmula BUSCARV
        "Estatus": ["Asistió"]
    })
    
    if not os.path.isfile(EXCEL_ASISTENCIA):
        nuevo.to_excel(EXCEL_ASISTENCIA, index=False)
    else:
        existente = pd.read_excel(EXCEL_ASISTENCIA)
        # Evitar duplicar asistencia el mismo día
        if not ((existente['ID / Matrícula'] == matricula) & 
                (existente['Fecha'] == ahora.strftime("%Y-%m-%d"))).any():
            pd.concat([existente, nuevo], ignore_index=True).to_excel(EXCEL_ASISTENCIA, index=False)
            print(f"✅ Registrado: {nombre_carpeta} ({matricula})")

def reconocer(emb_actual):
    mejor_nombre, mejor_sim = "Desconocido", 0
    for nombre, embs_db in db_rostros.items():
        for emb_db in embs_db:
            sim = cosine_similarity(emb_actual.reshape(1,-1), emb_db.reshape(1,-1))[0][0]
            if sim > mejor_sim:
                mejor_sim, mejor_nombre = sim, nombre
    return mejor_nombre, mejor_sim

# --- 6. BUCLE PRINCIPAL (CÁMARA) ---
cap = cv2.VideoCapture(1) # Cambia a 0 si usas cámara integrada
asistidos_hoy = set()
confirmaciones = {}

print("Cámara lista. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    
    if face is not None:
        with torch.no_grad():
            emb_actual = model(face.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
        
        nombre, score = reconocer(emb_actual)
        
        if score >= UMBRAL and nombre not in asistidos_hoy:
            confirmaciones[nombre] = confirmaciones.get(nombre, 0) + 1
            if confirmaciones[nombre] >= FRAMES_CONFIRMACION:
                registrar_asistencia(nombre)
                asistidos_hoy.add(nombre)
        
        color = (0, 255, 0) if score >= UMBRAL else (0, 0, 255)
        cv2.putText(frame, f"{nombre} ({score:.2f})", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Sistema de Asistencia Pro", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
