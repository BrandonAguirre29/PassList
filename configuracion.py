import os

DEVICE_PREFERENCE = "cuda"   # "cuda" si tienes GPU, si no usa "cpu" automáticamente
DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "embeddings.npy"   # archivo donde se guardan los embeddings calculados
UMBRAL = 0.85               # confianza mínima para reconocer un rostro
FRAMES_CONFIRMACION = 3               # frames consecutivos para confirmar asistencia
CAMERA_INDEX = 1                 # 0 = cámara integrada, 1 = cámara externa
MINUTOS_ANTES_CLASE = 5
MINUTOS_PRESENTE    = 5
MINUTOS_RETARDO     = 10