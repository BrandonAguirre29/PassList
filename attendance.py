import cv2
import torch
import time
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# -------- CONFIGURACIÓN --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UMBRAL = 0.85                # seguridad alta
FRAMES_CONFIRMACION = 3     # frames seguidos
TIEMPO_CAMARA = 10          # segundos
CAMARA_ID = 1
# --------------------------------

mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(classify=False).eval().to(DEVICE)
model.load_state_dict(
    torch.load("20180402-114759-vggface2.pt", map_location=DEVICE),
    strict=False
)

db = torch.load("embeddings_db.pth")
print(" Alumnos registrados:", list(db.keys()))

asistencia = set()
contador_frames = {}
inicio = time.time()

def reconocer(embedding):
    mejor_nombre = None
    mejor_sim = 0

    for nombre, embs in db.items():
        for emb_db in embs:
            sim = cosine_similarity(
                embedding.reshape(1, -1),
                emb_db.reshape(1, -1)
            )[0][0]

            if sim > mejor_sim:
                mejor_sim = sim
                mejor_nombre = nombre

    return mejor_nombre, mejor_sim

cap = cv2.VideoCapture(CAMARA_ID)
print(" Pase de lista en curso...")

while time.time() - inicio < TIEMPO_CAMARA:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)

    if face is not None:
        face = face.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model(face).cpu().numpy()[0]

        nombre, score = reconocer(embedding)

        if score >= UMBRAL and nombre not in asistencia:
            contador_frames[nombre] = contador_frames.get(nombre, 0) + 1

            if contador_frames[nombre] >= FRAMES_CONFIRMACION:
                asistencia.add(nombre)
                print(f"✅ ASISTENCIA: {nombre}")
        else:
            contador_frames.clear()

        texto = f"{nombre if score>=UMBRAL else 'DESCONOCIDO'} ({score:.2f})"
        color = (0,255,0) if score>=UMBRAL else (0,0,255)

        cv2.putText(frame, texto, (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Pase de Lista", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\n ASISTENCIA FINAL:")
for a in asistencia:
    print("✔", a)
