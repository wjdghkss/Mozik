import os
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from core import gaussian_blur_region, expand_box
from detect import load_yolov5, detect_faces_yolov5
from face_registry import FaceRegistry
from face_embedder import get_embedding

# =============================
# 기본 설정
# =============================
YOLOV5_WEIGHTS = os.path.join(os.path.dirname(__file__), "best.pt")
DETECT_THUMB_W = 640
FACE_PAD = 0.05   # 얼굴 부분만 타이트하게 잡기 위한 패딩

EMOJI_DIR = os.path.join(os.path.dirname(__file__), "emoji")
DEFAULT_EMOJI_FILES = ["emoji1.png", "emoji2.png", "emoji3.png", "emoji4.png", "emoji5.png"]


# =============================
# Dataclasses
# =============================
@dataclass
class UBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class DBox(UBox):
    enabled: bool = True
    identity: Optional[str] = None
    emb: Optional[np.ndarray] = None


# =============================
# 모델 로드
# =============================
try:
    DETECTOR = load_yolov5(YOLOV5_WEIGHTS)
except Exception as e:
    print(f"[ERROR] YOLOv5 로드 실패: {e}")
    DETECTOR = None


# =============================
# 얼굴 탐지
# =============================
def _detect_on_frame(detector, frame) -> List[DBox]:
    h, w = frame.shape[:2]

    if w > DETECT_THUMB_W:
        scale = DETECT_THUMB_W / float(w)
        small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    else:
        scale = 1.0
        small = frame

    try:
        rects_small = detect_faces_yolov5(detector, small)
    except Exception:
        rects_small = []

    faces: List[DBox] = []
    for (x, y, w0, h0) in rects_small:
        if scale != 1.0:
            x, y, w0, h0 = int(x / scale), int(y / scale), int(w0 / scale), int(h0 / scale)

        ex, ey, ew, eh = expand_box(x, y, w0, h0, FACE_PAD, FACE_PAD, w, h)
        if ew >= 12 and eh >= 12:
            faces.append(DBox(ex, ey, ew, eh, True))

    return faces


# =============================
# 얼굴 매칭 (ArcFace)
# =============================
def _annotate_authorized_faces(registry: FaceRegistry, faces: List[DBox], frame_bgr):
    if not faces:
        return

    h, w = frame_bgr.shape[:2]

    for b in faces:
        if b.identity is not None:
            continue

        x1, y1 = max(0, b.x), max(0, b.y)
        x2, y2 = min(w, b.x + b.w), min(h, b.y + b.h)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        try:
            if b.emb is None:
                b.emb = get_embedding(crop)

            name, dist = registry.match(b.emb)
            b.identity = name

        except Exception:
            b.identity = None


# =============================
# 정책: exclude / force / normal
# =============================
def _should_blur_auto_face(registry: FaceRegistry, b: DBox, auth_exclude_enable: bool) -> bool:
    if not b.enabled:
        return False

    if auth_exclude_enable and b.identity:
        mode = registry.get_policy(b.identity)

        if mode == "exclude":
            return False  # 항상 제외 → 모자이크/블러 하지 않음

        if mode == "force":
            return True   # 항상 보호 → 무조건 블러/이모지/모자이크 처리

    return True


# =============================
# 모자이크 효과
# =============================
def _apply_mosaic(frame, x, y, w, h, mosaic_size=15):
    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    small_w = max(1, (x2 - x1) // mosaic_size)
    small_h = max(1, (y2 - y1) // mosaic_size)

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    frame[y1:y2, x1:x2] = mosaic


# =============================
# 이모지 오버레이
# =============================
def _load_emoji_image(emoji_path: Optional[str], emoji_index: int = 0) -> Optional[np.ndarray]:
    if emoji_path:
        path = emoji_path
    else:
        idx = emoji_index if 0 <= emoji_index < len(DEFAULT_EMOJI_FILES) else 0
        path = os.path.join(EMOJI_DIR, DEFAULT_EMOJI_FILES[idx])

    if not os.path.exists(path):
        print(f"[emoji] 파일 없음: {path}")
        return None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def _overlay_emoji(frame, x, y, w, h, emoji_img):
    if emoji_img is None:
        return

    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    target_w = x2 - x1
    target_h = y2 - y1

    emoji_resized = cv2.resize(emoji_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if emoji_resized.shape[2] == 4:  # 알파 존재
        b, g, r, a = cv2.split(emoji_resized)
        emoji_rgb = cv2.merge((b, g, r))
        alpha = (a / 255.0)[..., None]

        roi = frame[y1:y2, x1:x2].astype(float)
        blended = alpha * emoji_rgb + (1 - alpha) * roi
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = emoji_resized


# =============================
# 메인 처리 함수
# =============================
def process_image(
    input_path: str,
    output_path: str,
    blur_strength: int,
    auth_exclude_enable: bool = False,
    style: str = "blur",
    emoji_path: Optional[str] = None,
    emoji_index: int = 0,
):
    if DETECTOR is None:
        raise RuntimeError("YOLOv5 face detector could not be loaded.")

    frame = cv2.imread(input_path)
    if frame is None:
        raise IOError(f"이미지 읽기 실패: {input_path}")

    registry = FaceRegistry()

    detected_faces = _detect_on_frame(DETECTOR, frame)

    if auth_exclude_enable:
        _annotate_authorized_faces(registry, detected_faces, frame)

    emoji_img = None
    if style == "emoji":
        emoji_img = _load_emoji_image(emoji_path, emoji_index)
        if emoji_img is None:
            style = "blur"  # fallback

    for box in detected_faces:
        if not _should_blur_auto_face(registry, box, auth_exclude_enable):
            continue

        if style == "blur":
            gaussian_blur_region(frame, box.x, box.y, box.w, box.h, blur_strength)

        elif style == "mosaic":
            _apply_mosaic(frame, box.x, box.y, box.w, box.h,
                          mosaic_size=max(8, min(30, blur_strength)))

        elif style == "emoji":
            _overlay_emoji(frame, box.x, box.y, box.w, box.h, emoji_img)

        else:
            gaussian_blur_region(frame, box.x, box.y, box.w, box.h, blur_strength)

    ext = os.path.splitext(output_path)[1].lower()
    encode_ext = {
        ".png": ".png", ".jpg": ".jpg", ".jpeg": ".jpg",
        ".webp": ".webp", ".bmp": ".bmp"
    }.get(ext, ".png")

    try:
        ok, enc = cv2.imencode(encode_ext, frame)
        if not ok:
            raise RuntimeError("Encoding failed")
        with open(output_path, "wb") as f:
            f.write(enc)
    except Exception:
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_img.save(output_path)
        except Exception as e:
            raise IOError(f"이미지 저장 실패: {e}")
