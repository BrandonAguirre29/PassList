import os
import cv2
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# -------- CONFIGURACIÓN --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "dataset"
UMBRAL = 0.85
FRAMES_CONFIRMACION = 3
EXCEL_FILE = "Control_Asistencia.xlsx"
# --------------------------------

print("🔄 Iniciando sistema...")
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# 1. PROCESAR DATASET Y GENERAR LISTA DE ALUMNOS
db = {}
nombres_alumnos = []

if not os.path.exists(DATASET_DIR):
    print(f"❌ Carpeta '{DATASET_DIR}' no encontrada.")
    exit()

print("📂 Generando base de datos y lista maestra...")
for person in sorted(os.listdir(DATASET_DIR)):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path): continue
    
    nombres_alumnos.append(person) # Guardamos el nombre para el Excel
    embeddings = []
    for img_name in os.listdir(person_path):
        img = cv2.imread(os.path.join(person_path, img_name))
        if img is None: continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)
        if face is not None:
            face = face.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = model(face).cpu().numpy()[0]
                embeddings.append(emb)
    if embeddings:
        db[person] = embeddings

# 2. CREAR O CARGAR EL EXCEL CON TODOS LOS NOMBRES
if not os.path.isfile(EXCEL_FILE):
    # Si el archivo no existe, lo creamos con todos como "FALTA"
    df_asistencia = pd.DataFrame({
        "Nombre": nombres_alumnos,
        "Estatus": ["FALTA"] * len(nombres_alumnos),
        "Fecha": ["-"] * len(nombres_alumnos),
        "Hora": ["-"] * len(nombres_alumnos)
    })
    df_asistencia.to_excel(EXCEL_FILE, index=False)
    print(f"📄 Lista maestra creada con {len(nombres_alumnos)} alumnos.")
else:
    # Si ya existe, lo cargamos
    df_asistencia = pd.read_excel(EXCEL_FILE)
    print("📄 Lista maestra cargada desde el archivo existente.")

# 3. FUNCIÓN PARA ACTUALIZAR EL ESTATUS
def marcar_presente(nombre):
    global df_asistencia
    ahora = datetime.now()
    
    # Buscamos la fila del alumno y actualizamos
    if nombre in df_asistencia["Nombre"].values:
        # Solo actualizamos si aún tiene "FALTA" para no sobreescribir la hora de entrada
        idx = df_asistencia.index[df_asistencia["Nombre"] == nombre][0]
        if df_asistencia.at[idx, "Estatus"] == "FALTA":
            df_asistencia.at[idx, "Estatus"] = "PRESENTE"
            df_asistencia.at[idx, "Fecha"] = ahora.strftime("%Y-%m-%d")
            df_asistencia.at[idx, "Hora"] = ahora.strftime("%H:%M:%S")
            
            # Guardamos los cambios en el Excel
            try:
                df_asistencia.to_excel(EXCEL_FILE, index=False)
                print(f"✅ {nombre} marcado como PRESENTE.")
            except PermissionError:
                print(f"⚠️ ¡Cierra el archivo Excel para poder guardar la asistencia de {nombre}!")

def reconocer(embedding_actual):
    mejor_nombre, mejor_sim = "Desconocido", 0
    for nombre, lista_embs in db.items():
        for emb_db in lista_embs:
            sim = cosine_similarity(embedding_actual.reshape(1, -1), emb_db.reshape(1, -1))[0][0]
            if sim > mejor_sim:
                mejor_sim = sim
                mejor_nombre = nombre
    return mejor_nombre, mejor_sim

# 4. LOOP DE CÁMARA
print("📷 Cámara activada...")
cap = cv2.VideoCapture(1)
asistencia_sesion = set()
contador_frames = {}

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)

    if face is not None:
        face = face.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb_actual = model(face).cpu().numpy()[0]

        nombre, score = reconocer(emb_actual)

        if score >= UMBRAL and nombre not in asistencia_sesion:
            contador_frames[nombre] = contador_frames.get(nombre, 0) + 1
            if contador_frames[nombre] >= FRAMES_CONFIRMACION:
                asistencia_sesion.add(nombre)
                marcar_presente(nombre)
        
        color = (0, 255, 0) if score >= UMBRAL else (0, 0, 255)
        cv2.putText(frame, f"{nombre} ({score:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Control de Asistencia Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
