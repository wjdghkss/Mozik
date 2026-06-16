from typing import List, Tuple, Optional
import os
import pathlib

import cv2
import numpy as np
import torch

# ----------------------------------------------------
# Windows에서 리눅스용 PosixPath가 들어있는 pt를 언피클할 때 에러 방지용 패치이다.
# (YOLOv5 가중치가 리눅스에서 학습된 경우 자주 발생함)
# ----------------------------------------------------
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# 한 번 로드한 모델을 재사용하기 위한 전역 캐시이다.
_YOLOV5_MODEL: Optional[torch.nn.Module] = None


def load_yolov5(model_path: str = "best.pt",
                device: Optional[str] = None):
    """
    YOLOv5 커스텀 모델(best.pt)을 로드하는 함수이다.
    torch.hub + ultralytics/yolov5 레포를 사용함이다.
    """
    global _YOLOV5_MODEL

    # 이미 로드했다면 캐시된 모델을 그대로 반환한다.
    if _YOLOV5_MODEL is not None:
        return _YOLOV5_MODEL

    # 디바이스 선택 (GPU 있으면 cuda, 아니면 cpu)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 상대 경로로 들어오면 현재 파일 기준으로 절대경로로 보정한다.
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO 가중치 파일을 찾을 수 없음: {model_path}")

    # 🔹 YOLOv5 hub 모델 로드
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=model_path,
        trust_repo=True,
    )

    model.to(device)
    model.eval()

    _YOLOV5_MODEL = model
    return model


def detect_faces_yolov5(
    model,
    frame_bgr: np.ndarray,
    conf_thres: float = 0.6,
    thumb_width: int = 640,
    **kwargs,
) -> List[Tuple[int, int, int, int]]:
    """
    YOLOv5 모델과 BGR 프레임을 받아
    (x, y, w, h) 형태의 얼굴 박스 리스트를 반환하는 함수이다.

    conf_thres : 신뢰도 임계값이다.
    thumb_width : YOLO 입력으로 리사이즈할 가로 길이이다. (기본 640)
                  예전 코드에서 thumb_width=640 또는 thumb_w=640 으로
                  부르는 경우를 모두 지원한다.
    """
    if model is None:
        return []

    # kwargs 에서 넘어온 thumb_width / thumb_w 도 지원한다.
    if "thumb_width" in kwargs and kwargs["thumb_width"] is not None:
        thumb_width = int(kwargs["thumb_width"])
    if "thumb_w" in kwargs and kwargs["thumb_w"] is not None:
        thumb_width = int(kwargs["thumb_w"])

    if thumb_width is None:
        thumb_width = 640

    size = int(thumb_width)

    # YOLOv5는 BGR 이미지를 그대로 넣어도 동작함이다.
    # thumb_width 값을 size 로 사용한다.
    results = model(frame_bgr, size=size)

    # results.xyxy[0]: [N, 6] 텐서 => x1, y1, x2, y2, conf, cls
    if not hasattr(results, "xyxy") or len(results.xyxy) == 0:
        return []

    det = results.xyxy[0].cpu().numpy()
    boxes: List[Tuple[int, int, int, int]] = []

    for x1, y1, x2, y2, conf, cls in det:
        if conf < conf_thres:
            continue

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        if w > 0 and h > 0:
            boxes.append((x1, y1, w, h))

    return boxes
