import os
import sys
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Agregar raíz del proyecto al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.configuracion import DATASET_DIR, EMBEDDINGS_FILE, DEVICE_PREFERENCE


# ── Inicializar device ──────────────────────────────────────────────────────
DEVICE = "cuda" if (DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available()) else "cpu"


def _cargar_modelos():
    """Carga MTCNN e InceptionResnetV1. Solo se llama internamente."""
    print(f" Cargando modelos en {DEVICE}...")
    mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    return mtcnn, model


def generar_embeddings() -> dict:

    if not os.path.exists(DATASET_DIR):
        print(f" Carpeta '{DATASET_DIR}' no encontrada.")
        sys.exit(1)

    mtcnn, model = _cargar_modelos()
    db = {}

    print(" Procesando imágenes del dataset...")
    for persona in sorted(os.listdir(DATASET_DIR)):
        persona_path = os.path.join(DATASET_DIR, persona)
        if not os.path.isdir(persona_path):
            continue

        embeddings = []
        for img_name in os.listdir(persona_path):
            img_path = os.path.join(persona_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb)
            if face is not None:
                face = face.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = model(face).cpu().numpy()[0]
                    embeddings.append(emb)

        if embeddings:
            db[persona] = embeddings
            print(f"   {persona}  ({len(embeddings)} foto(s) procesada(s))")
        else:
            print(f"   {persona}  — no se detectaron rostros, revisa las fotos")

    if not db:
        print("No se encontraron rostros en el dataset.")
        sys.exit(1)

    # Guardar en disco para no recalcular la próxima vez
    np.save(EMBEDDINGS_FILE, db)
    print(f"\n Embeddings guardados en '{EMBEDDINGS_FILE}'")
    print(f"Total de personas registradas: {len(db)}\n")
    return db


def cargar_embeddings() -> dict:
   
    if os.path.isfile(EMBEDDINGS_FILE):
        print(f"Cargando embeddings desde '{EMBEDDINGS_FILE}'...")
        db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        print(f"{len(db)} persona(s) cargadas desde caché.\n")
        return db
    else:
        print("No se encontró caché. Generando desde dataset...")
        return generar_embeddings()


# ── Ejecutar directamente para regenerar embeddings ─────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  GENERADOR DE EMBEDDINGS")
    print("  Ejecuta esto cuando agregues nuevas fotos al dataset")
    print("=" * 50 + "\n")
    generar_embeddings()
    print("Listo.")
