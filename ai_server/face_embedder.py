# ai_api/face_embedder.py

import numpy as np
from deepface import DeepFace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_MODEL_NAME = "ArcFace"  # ArcFace 사용이다.


def get_embedding(bgr_face: np.ndarray):
    """
    BGR 얼굴 이미지(numpy)를 입력받아 ArcFace 임베딩(정규화된 벡터)을 반환하는 함수이다.
    """
    if bgr_face is None or bgr_face.size == 0:
        return None

    # DeepFace는 RGB를 입력으로 기대하므로 변환한다.
    rgb_face = bgr_face[:, :, ::-1]

    # 🔵 DeepFace.represent 호출
    #   - model 인자는 사용하지 않고
    #   - model_name 으로만 ArcFace를 지정한다.
    #   - detector_backend="skip" 으로 이미 얼굴만 잘린 상태라고 알려준다.
    reps = DeepFace.represent(
        rgb_face,
        model_name=_MODEL_NAME,
        enforce_detection=False,
        detector_backend="skip",
    )

    if not reps:
        return None

    emb = np.array(reps[0]["embedding"], dtype=np.float32)

    # ArcFace는 보통 L2 정규화된 벡터를 사용한다.
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb
