import os
import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCFACE_MODEL = os.path.join(BASE_DIR, "models", "arcface_r100.onnx")

session = ort.InferenceSession(ARCFACE_MODEL, providers=["CPUExecutionProvider"])

def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, 0)

def get_embedding(img_bgr):
    x = preprocess(img_bgr)
    emb = session.run(None, {"input": x})[0][0]
    emb = emb / np.linalg.norm(emb)
    return emb
