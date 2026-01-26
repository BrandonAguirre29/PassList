import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "dataset"

mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(classify=False).eval().to(DEVICE)
model.load_state_dict(
    torch.load("20180402-114759-vggface2.pt", map_location=DEVICE),
    strict=False
)

db = {}

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
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
        db[person] = embeddings
        print(f"✔ {person}: {len(embeddings)} rostros")

torch.save(db, "embeddings_db.pth")
print(" Base de datos creada correctamente")
