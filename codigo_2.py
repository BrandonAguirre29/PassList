import os
import cv2
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# -------- CONFIGURACIÓN --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "dataset"
UMBRAL = 0.85
FRAMES_CONFIRMACION = 3
EXCEL_FILE = "asistencia_clase.xlsx"
# --------------------------------

# INICIALIZAR MODELOS
print("🔄 Cargando modelos de IA...")
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)

model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# CREAR BASE DE DATOS EN MEMORIA (Desde la carpeta dataset)
db = {}
if not os.path.exists(DATASET_DIR):
    print(f" Error: No existe la carpeta '{DATASET_DIR}'")
    exit()

print(" Procesando imágenes del dataset...")
for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path): continue

    embeddings = []
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
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
        #print(f"✔ {person} registrado ({len(embeddings)} fotos)")

if not db:
    print(" No se encontraron rostros en el dataset. Revisa tus carpetas.")
    exit()

# 3. FUNCIONES DE APOYO
def registrar_en_excel(nombre):
    ahora = datetime.now()
    nuevo_registro = {
        "Nombre": [nombre], 
        "Fecha": [ahora.strftime("%Y-%m-%d")], 
        "Hora": [ahora.strftime("%H:%M:%S")]
    }
    df_nuevo = pd.DataFrame(nuevo_registro)
    if not os.path.isfile(EXCEL_FILE):
        df_nuevo.to_excel(EXCEL_FILE, index=False)
    else:
        df_existente = pd.read_excel(EXCEL_FILE)
        pd.concat([df_existente, df_nuevo], ignore_index=True).to_excel(EXCEL_FILE, index=False)
    #print(f"Excel actualizado: {nombre}")

def reconocer(embedding_actual):
    mejor_nombre, mejor_sim = "Desconocido", 0
    for nombre, lista_embs in db.items():
        for emb_db in lista_embs:
            sim = cosine_similarity(embedding_actual.reshape(1, -1), emb_db.reshape(1, -1))[0][0]
            if sim > mejor_sim:
                mejor_sim = sim
                mejor_nombre = nombre
    return mejor_nombre, mejor_sim

# 4. INICIO DE CÁMARA Y ASISTENCIA
print("Abriendo cámara para pase de lista...")
cap = cv2.VideoCapture(1) # Cambia a 1 si tienes cámara externa
asistencia_hoy = set()
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

        if score >= UMBRAL and nombre not in asistencia_hoy:
            contador_frames[nombre] = contador_frames.get(nombre, 0) + 1
            if contador_frames[nombre] >= FRAMES_CONFIRMACION:
                asistencia_hoy.add(nombre)
                registrar_en_excel(nombre)
        
        # Dibujar info en pantalla
        color = (0, 255, 0) if score >= UMBRAL else (0, 0, 255)
        txt = f"{nombre} ({score:.2f})"
        cv2.putText(frame, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Sistema de Asistencia Integrado", frame)
    if cv2.waitKey(1) & 0xFF == 27: break # ESC para salir

cap.release()
cv2.destroyAllWindows()