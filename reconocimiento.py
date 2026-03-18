import os
import sys
import cv2
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.configuracion import DEVICE_PREFERENCE, UMBRAL, FRAMES_CONFIRMACION, CAMERA_INDEX
from services.embeddings import cargar_embeddings
from database.operaciones import obtener_horario_activo, registrar_asistencia

DEVICE = "cuda" if (DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available()) else "cpu"

print(f"[RecognitionService] Cargando modelos en {DEVICE}...")
_mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
print("[RecognitionService]  Modelos listos.")

_db = cargar_embeddings()


def reconocer(embedding_actual: np.ndarray) -> tuple[str, float]:
    mejor_nombre, mejor_sim = "Desconocido", 0.0
    for nombre, lista_embs in _db.items():
        for emb_db in lista_embs:
            sim = cosine_similarity(
                embedding_actual.reshape(1, -1),
                emb_db.reshape(1, -1)
            )[0][0]
            if sim > mejor_sim:
                mejor_sim = sim
                mejor_nombre = nombre
    return mejor_nombre, float(mejor_sim)


def iniciar_camara():
    horario = obtener_horario_activo()
    horario_id  = horario["horario_id"]
    hora_inicio = horario["hora_inicio"]

    if horario_id is None:
        print("\n[RecognitionService]  No hay clase activa en este momento.")
        respuesta = input("¿Deseas continuar de todas formas? (s/n): ").strip().lower()
        if respuesta != "s":
            return

    print(f"\n[RecognitionService]  Abriendo cámara (índice {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[RecognitionService]  No se pudo abrir la cámara {CAMERA_INDEX}.")
        return

    print("[RecognitionService]  Cámara activa. Presiona ESC para salir.\n")

    asistencia_sesion = set()
    contador_frames   = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = _mtcnn(rgb)

        if face is not None:
            face = face.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb_actual = _model(face).cpu().numpy()[0]

            nombre, score = reconocer(emb_actual)

            if score >= UMBRAL and nombre not in asistencia_sesion:
                contador_frames[nombre] = contador_frames.get(nombre, 0) + 1
                if contador_frames[nombre] >= FRAMES_CONFIRMACION:
                    asistencia_sesion.add(nombre)
                    if horario_id:
                        registrar_asistencia(nombre, horario_id, hora_inicio)

            color    = (0, 255, 0) if score >= UMBRAL else (0, 0, 255)
            etiqueta = f"{nombre}  ({score:.2f})"
            cv2.putText(frame, etiqueta, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("PassLi — Control de Asistencia", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[RecognitionService] Cámara cerrada.")
    print(f"[RecognitionService] Registrados esta sesión: {len(asistencia_sesion)}")
    for nombre in sorted(asistencia_sesion):
        print(f"  · {nombre}")


if __name__ == "__main__":
    iniciar_camara()