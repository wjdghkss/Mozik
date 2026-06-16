# mozik/api_service.py

import cv2
import numpy as np

from detect import load_yolov5, detect_faces_yolov5
from face_embedder import get_embedding
from face_registry import FaceRegistry

# ----------------------------------------------------
# 모델/레지스트리 로드
# ----------------------------------------------------
# 얼굴 탐지 모델 로드
face_detector = load_yolov5()

# 얼굴 정보 레지스트리 로드
face_registry = FaceRegistry()


# ----------------------------------------------------
# API 서비스 함수
# ----------------------------------------------------
def register_face(img_bytes: bytes, name: str):
    """
    이미지(bytes)와 이름을 받아 얼굴을 등록하는 서비스이다.
    """
    try:
        # 이미지 바이트를 numpy array로 디코딩
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "유효하지 않은 이미지입니다."}

        # 얼굴 탐지
        faces = detect_faces_yolov5(face_detector, img)

        if len(faces) == 0:
            return {"status": "error", "message": "얼굴을 찾을 수 없습니다."}

        if len(faces) > 1:
            return {"status": "error", "message": "이미지에 한 명의 얼굴만 있어야 합니다."}

        # 얼굴 임베딩 추출
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        embedding = get_embedding(face_img)

        if embedding is None:
            return {"status": "error", "message": "얼굴 임베딩을 추출할 수 없습니다."}

        # 레지스트리에 등록
        face_registry.register(embedding, name, "exclude")
        face_registry.save()

        return {"status": "ok", "message": f"{name}님의 얼굴을 등록했습니다."}

    except Exception as e:
        return {"status": "error", "message": f"얼굴 등록 중 오류 발생: {e}"}


def detect_and_match(img_bytes: bytes):
    """
    이미지(bytes)를 받아 얼굴을 탐지하고, 등록된 얼굴과 매칭하는 서비스이다.
    """
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "유효하지 않은 이미지입니다."}

        faces = detect_faces_yolov5(face_detector, img)
        
        results = []
        for x, y, w, h in faces:
            face_img = img[y:y+h, x:x+w]
            embedding = get_embedding(face_img)
            
            if embedding is None:
                match_result = {"name": "Unknown", "policy": "include"}
            else:
                match = face_registry.match(embedding)
                if match:
                    match_result = {"name": match.name, "policy": match.policy}
                else:
                    match_result = {"name": "Unknown", "policy": "include"}
            
            results.append({
                "box": [x, y, w, h],
                "match": match_result
            })

        return {"status": "ok", "faces": results}

    except Exception as e:
        return {"status": "error", "message": f"얼굴 매칭 중 오류 발생: {e}"}


def reload_registry():
    """
    얼굴 레지스트리를 다시 로드하는 서비스이다.
    """
    try:
        face_registry.load()
        return {"status": "ok", "message": "얼굴 레지스트리를 새로고침했습니다."}
    except Exception as e:
        return {"status": "error", "message": f"레지스트리 새로고침 중 오류 발생: {e}"}
