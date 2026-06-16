import os
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import json
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


# ArcFace 매칭 임계값이다.
# 다른 사람까지 같은 사람으로 잡히는 상황이므로 꽤 높게 잡는다.
FACE_SIM_THRESHOLD = 0.60  # 필요하면 0.65, 0.70으로 더 올려도 된다.
FACE_SIM_MARGIN = 0.08

EMOJI_DIR = os.path.join(os.path.dirname(__file__), "emoji")
DEFAULT_EMOJI_FILES = ["emoji1.png", "emoji2.png", "emoji3.png", "emoji4.png", "emoji5.png"]



EMOJI_CHAR_TO_INDEX = {
    "🙂": 0,
    "😢": 1,
    "😎": 2,
    "🤔": 3,
    "🥰": 4,
}
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
    sim: Optional[float] = None


# =============================
# YOLO 모델 로드
# =============================
try:
    DETECTOR = load_yolov5(YOLOV5_WEIGHTS)
    print("[YOLO] 모델 로드 성공")
except Exception as e:
    print(f"[ERROR] YOLOv5 로드 실패: {e}")
    DETECTOR = None


# =============================
# 얼굴 탐지: YOLO → 실패 시 Haar
# =============================
def _detect_on_frame(detector, frame) -> List[DBox]:
    """
    1단계: YOLOv5로 얼굴 탐지
    2단계: 결과가 없으면 HaarCascade로 한 번 더 탐지
    """
    h, w = frame.shape[:2]
    faces: List[DBox] = []

    # ---------- 1단계: YOLO ----------
    if detector is not None:
        if w > DETECT_THUMB_W:
            scale = DETECT_THUMB_W / float(w)
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        else:
            scale = 1.0
            small = frame

        try:
            rects_small = detect_faces_yolov5(detector, small)
        except Exception as e:
            print(f"[WARN] YOLO detect 실패: {e}")
            rects_small = []

        pad = float(FACE_PAD)

        for (x, y, w0, h0) in rects_small:
            if scale != 1.0:
                x, y, w0, h0 = int(x / scale), int(y / scale), int(w0 / scale), int(h0 / scale)

            ex, ey, ew, eh = expand_box(x, y, w0, h0, pad, pad, w, h)
            if ew >= 12 and eh >= 12:
                faces.append(DBox(ex, ey, ew, eh, True))

    # ---------- 2단계: YOLO 결과가 없으면 Haar ----------
    if not faces:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            rects = haar.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            pad = float(FACE_PAD)
            for (x, y, w0, h0) in rects:
                ex, ey, ew, eh = expand_box(x, y, w0, h0, pad, pad, w, h)
                if ew >= 12 and eh >= 12:
                    faces.append(DBox(ex, ey, ew, eh, True))
        except Exception as e:
            print(f"[WARN] Haar detect 실패: {e}")

    print(f"[DETECT] faces={len(faces)}")
    return faces


# =============================
# 얼굴 매칭 (ArcFace + FaceRegistry)
# =============================
from typing import Dict, List

def _annotate_authorized_faces(registry: FaceRegistry, faces: List[DBox], frame_bgr):
    """
    1) ArcFace 임베딩으로 identity와 sim(유사도)을 붙이고,
    2) 같은 identity로 인식된 얼굴이 여러 개라면
       '가장 큰 얼굴(박스 면적이 큰 것)' 하나만 등록 얼굴로 남기는 함수이다.
    """
    if not faces:
        return

    h, w = frame_bgr.shape[:2]

    # ---------------------------
    # 1) 각 얼굴에 identity, sim 채우기
    # ---------------------------
    for b in faces:
        # 이미 identity가 있다면 그대로 둔다.
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

            # FaceRegistry.match는 (identity, sim) 을 반환한다고 가정
            name, sim = registry.match(b.emb)

            # 유사도가 threshold보다 낮으면 다른 사람으로 간주
            if sim is None or sim < FACE_SIM_THRESHOLD:
                b.identity = None
                b.sim = None
            else:
                b.identity = name
                b.sim = sim

        except Exception:
            b.identity = None
            b.sim = None

    # ---------------------------
    # 2) 같은 identity가 여러 개면
    #    "가장 큰 얼굴" 하나만 남기고 나머지는 일반 얼굴로 내리기
    # ---------------------------
    id_to_best: Dict[str, DBox] = {}

    for b in faces:
        if not b.identity:
            continue

        cur_best = id_to_best.get(b.identity)

        # 면적(= w*h)이 더 큰 얼굴을 대표로 선택
        if cur_best is None or (b.w * b.h) > (cur_best.w * cur_best.h):
            id_to_best[b.identity] = b

    # 대표가 아닌 동일 identity 얼굴들은 전부 일반 얼굴로 처리
    for b in faces:
        if not b.identity:
            continue

        if id_to_best.get(b.identity) is not b:
            b.identity = None
            b.sim = None


    # 2) 같은 identity로 인식된 얼굴이 여러 개면
    #    sim이 가장 높은 한 개만 유지하고 나머지는 등록 해제
    # 2) 같은 identity가 여러 개면
    #    "가장 큰 얼굴(박스 면적이 큰 것)" 하나만 등록 얼굴로 남긴다.
    id_to_best: Dict[str, DBox] = {}
    for b in faces:
        if not b.identity:
            continue

        cur_best = id_to_best.get(b.identity)
        # 면적(= w*h)이 더 큰 얼굴을 선택
        if cur_best is None or (b.w * b.h) > (cur_best.w * cur_best.h):
            id_to_best[b.identity] = b

    for b in faces:
        if not b.identity:
            continue
        # 이 identity의 대표(가장 큰 얼굴)가 아닌 나머지는 전부 일반 얼굴로 내린다.
        if id_to_best.get(b.identity) is not b:
            b.identity = None
            b.sim = None




# =============================
# 정책: exclude / force / neutral
# =============================
def _should_blur_auto_face(registry: FaceRegistry, b: DBox, auth_exclude_enable: bool) -> bool:
    """
    FaceRegistry 정책에 따라 이 얼굴을 기본적으로 모자이크 대상에 포함할지 여부를 결정하는 함수이다.
    - exclude: 기본적으로 모자이크 하지 않음
    - force  : 기본적으로 항상 모자이크
    - neutral: 일반 얼굴과 동일하게 처리
    """
    # UI 등에서 enabled를 꺼놓은 경우라면 무조건 모자이크 안 함이다.
    if not b.enabled:
        return False

    if auth_exclude_enable and b.identity:
        mode = registry.get_policy(b.identity)

        if mode == "exclude":
            # 등록된 얼굴 중 "exclude" → 항상 제외 (모자이크 X)
            return False

        if mode == "force":
            # 등록된 얼굴 중 "force" → 항상 모자이크
            return True

    # 기본값: 모자이크 대상이다.
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
    if target_w <= 0 or target_h <= 0:
        return

    emoji_resized = cv2.resize(emoji_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if emoji_resized.shape[2] == 4:  # 알파 채널 존재
        b, g, r, a = cv2.split(emoji_resized)
        emoji_rgb = cv2.merge((b, g, r))
        alpha = (a / 255.0)[..., None]

        roi = frame[y1:y2, x1:x2].astype(float)
        blended = alpha * emoji_rgb + (1 - alpha) * roi
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = emoji_resized


# =============================
# 디버그용: 얼굴 bounding box 그리기
# =============================
def draw_frames(
    frame,
    faces: List[DBox],
    show_index: bool = True,
    show_identity: bool = True,
    color=(0, 255, 0),
    thickness: int = 2,
):
    """
    현재 프레임(이미지) 위에 얼굴 bounding box와 텍스트(인덱스/identity)를 그려주는 함수이다.
    """
    if frame is None or not faces:
        return

    h, w = frame.shape[:2]

    for idx, box in enumerate(faces):
        x1 = max(0, box.x)
        y1 = max(0, box.y)
        x2 = min(w - 1, box.x + box.w)
        y2 = min(h - 1, box.y + box.h)

        # 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 라벨 텍스트
        label_parts = []
        if show_index:
            label_parts.append(f"#{idx}")
        if show_identity and box.identity:
            label_parts.append(str(box.identity))

        if label_parts:
            label = " ".join(label_parts)
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

def process_image(
    input_path: str,
    output_path: str,
    blur_strength: int,
    auth_exclude_enable: bool = False,
    style: str = "blur",             # "blur" / "mosaic" / "emoji"
    emoji_path: Optional[str] = None,
    emoji_index: int = 0,
    excluded_ids: Optional[List[int]] = None,
    detect_only: bool = False,
    blur_frames: Optional[List[dict]] = None,
    sticker_mode: bool = False,
    sticker_positions: Optional[List[dict]] = None,
    sticker_face_ids: Optional[List[int]] = None,
    selected_emoji: Optional[str] = None,
    force_ids: Optional[List[int]] = None,
    delete_frames: Optional[List[dict]] = None,
    draw_box: bool = False,   # 디버그용 박스
    **kwargs,
) -> Tuple[str, List[Dict[str, int]]]:

    # -----------------------------
    # 0. 인자 기본값 / 정규화
    # -----------------------------
    if not draw_box and "draw_frames" in kwargs:
        draw_box = bool(kwargs.get("draw_frames"))

    if blur_frames is None:
        blur_frames = []
    if sticker_positions is None:
        sticker_positions = []
    if force_ids is None:
        force_ids = []
    if sticker_face_ids is None:
        sticker_face_ids = []
    if delete_frames is None:
        delete_frames = []

    # excluded_ids 문자열/JSON 방어
    if isinstance(excluded_ids, str):
        try:
            if excluded_ids.strip().startswith("["):
                excluded_ids = json.loads(excluded_ids)
            else:
                excluded_ids = [
                    int(x.strip())
                    for x in excluded_ids.split(",")
                    if x.strip()
                ]
        except (json.JSONDecodeError, ValueError):
            excluded_ids = []
    elif excluded_ids is None:
        excluded_ids = []

    # style 정리
    style = (style or "blur").strip().lower()
    if style not in {"blur", "mosaic", "emoji"}:
        style = "blur"

    # blur_strength 0~100
    try:
        blur_strength = int(blur_strength)
    except Exception:
        blur_strength = 50
    blur_strength = max(0, min(100, blur_strength))

    # -----------------------------
    # 1. 원본 이미지 로드
    # -----------------------------
    frame = cv2.imread(input_path)
    if frame is None:
        raise IOError(f"이미지 읽기 실패: {input_path}")

    h, w = frame.shape[:2]

    # -----------------------------
    # 2. 강도 → 커널/블록 사이즈
    # -----------------------------
    strength_ratio = (blur_strength / 100.0) * 0.10 + 0.02

    base_len = min(w, h)
    mosaic_block = int(base_len * strength_ratio)
    mosaic_block = max(8, min(60, mosaic_block))

    blur_kernel_size = int(w * strength_ratio)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    blur_kernel_size = max(5, min(299, blur_kernel_size))

    # ArcFace 매칭 기준을 강화한 레지스트리를 사용한다.
    registry = FaceRegistry(sim_threshold=FACE_SIM_THRESHOLD)

    # -----------------------------
    # 3. 얼굴 자동 탐지
    # -----------------------------
    detected_faces = _detect_on_frame(DETECTOR, frame)

    face_boxes_for_api = [
        {"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in detected_faces
    ]

    if draw_box:
        draw_frames(frame, detected_faces)

    # -----------------------------
    # 4. ArcFace identity 부여
    # -----------------------------
    if detected_faces:
        _annotate_authorized_faces(registry, detected_faces, frame)

    # -----------------------------
    # 5. 이모지 이미지 준비
    # -----------------------------
    emoji_img = None

    has_emoji_frame = any(
        isinstance(bf, dict)
        and str(bf.get("effect_type", "")).lower() == "emoji"
        for bf in (blur_frames or [])
    )

    if style == "emoji" or sticker_mode or has_emoji_frame:
        idx = emoji_index
        if selected_emoji:
            idx = EMOJI_CHAR_TO_INDEX.get(selected_emoji, emoji_index)

        emoji_img = _load_emoji_image(emoji_path, idx)
        if emoji_img is None and style == "emoji":
            print("[emoji] 이모지 이미지를 찾을 수 없어 blur로 대체함")
            style = "blur"

    # -----------------------------
    # 6. blur_frames 좌표/크기 정규화
    #    - 두 종류 모두 허용
    #      1) rect 모드: x,y,w,h (% / 좌상단 + 폭/높이)
    #      2) center 모드: x,y,size (% / 중심 + 크기)
    # -----------------------------
    norm_blur_frames: List[dict] = []
    for bf in blur_frames:
        if not isinstance(bf, dict):
            continue

        effect_type = (
            bf.get("effect_type")
            or bf.get("effectType")
            or "blur"
        )
        effect_type = str(effect_type).lower()

        # --- rect 모드 ---
        has_rect = (
            bf.get("w") is not None and
            bf.get("h") is not None
        )
        if has_rect:
            try:
                xp = float(bf.get("x"))
                yp = float(bf.get("y"))
                wp = float(bf.get("w"))
                hp = float(bf.get("h"))
            except (TypeError, ValueError):
                continue

            xp = max(0.0, min(100.0, xp))
            yp = max(0.0, min(100.0, yp))
            wp = max(1.0, min(100.0, wp))
            hp = max(1.0, min(100.0, hp))

            nd = {
                "mode": "rect",
                "x": xp,
                "y": yp,
                "w": wp,
                "h": hp,
                "effect_type": effect_type,
            }
            if "selected_emoji" in bf:
                nd["selected_emoji"] = bf.get("selected_emoji")
            norm_blur_frames.append(nd)
            continue

        # --- center 모드 (x,y,size) ---
        x_val = (
            bf.get("x")
            if bf.get("x") is not None
            else bf.get("xPercent", bf.get("xp"))
        )
        y_val = (
            bf.get("y")
            if bf.get("y") is not None
            else bf.get("yPercent", bf.get("yp"))
        )

        try:
            xp = float(x_val)
            yp = float(y_val)
        except (TypeError, ValueError):
            continue

        size_val = bf.get("size")
        if size_val is None:
            size_val = bf.get("boxSize", bf.get("width", 20))

        try:
            size_percent = float(size_val)
        except (TypeError, ValueError):
            size_percent = 20.0

        size_percent = max(1.0, min(100.0, size_percent))

        nd = {
            "mode": "center",
            "x": xp,
            "y": yp,
            "size": size_percent,
            "effect_type": effect_type,
        }
        if "selected_emoji" in bf:
            nd["selected_emoji"] = bf.get("selected_emoji")
        norm_blur_frames.append(nd)

    blur_frames = norm_blur_frames
    print("[studio_image] blur_frames 정규화 결과 개수:", len(blur_frames))
    if blur_frames:
        print("[studio_image] blur_frames[0]:", blur_frames[0])

    # -----------------------------
    # 7-a. 얼굴별 자동 효과 적용
    # -----------------------------
    for idx, box in enumerate(detected_faces):
        if idx in excluded_ids:
            continue

        if idx not in force_ids:
            if not _should_blur_auto_face(registry, box, auth_exclude_enable):
                continue

        if detect_only:
            cv2.rectangle(
                frame,
                (box.x, box.y),
                (box.x + box.w, box.y + box.h),
                (0, 255, 255),
                2,
            )
            continue

        current_style = style
        if sticker_mode and emoji_img is not None:
            if (not sticker_face_ids) or (idx in sticker_face_ids):
                current_style = "emoji"
        if style == "emoji" and emoji_img is not None:
            current_style = "emoji"

        x, y, bw, bh = box.x, box.y, box.w, box.h
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))

        if current_style == "blur":
            gaussian_blur_region(frame, x, y, bw, bh, blur_kernel_size)
        elif current_style == "mosaic":
            _apply_mosaic(frame, x, y, bw, bh, mosaic_size=mosaic_block)
        elif current_style == "emoji" and emoji_img is not None:
            _overlay_emoji(frame, x, y, bw, bh, emoji_img)
        else:
            gaussian_blur_region(frame, x, y, bw, bh, blur_kernel_size)

    # -----------------------------
    # 7-b. 수동 프레임 기반 추가 블러 / 이모지
    #      (얼굴 루프 바깥에서 한 번만 수행)
    # -----------------------------
    if not detect_only and blur_frames:
        print("[studio_image] 수동 프레임 적용 시작:", len(blur_frames))
        for bf in blur_frames:
            mode = bf.get("mode", "center")
            frame_effect = (bf.get("effect_type") or "blur").lower()

            # rect 모드
            if mode == "rect" and all(k in bf for k in ("x", "y", "w", "h")):
                try:
                    xp = float(bf.get("x", 0.0)) / 100.0
                    yp = float(bf.get("y", 0.0)) / 100.0
                    wp = float(bf.get("w", 0.0)) / 100.0
                    hp = float(bf.get("h", 0.0)) / 100.0
                except (TypeError, ValueError):
                    continue

                x1 = int(xp * w)
                y1 = int(yp * h)
                bw = int(wp * w)
                bh = int(hp * h)

            else:
                # center 모드
                try:
                    xp = float(bf.get("x", 0)) / 100.0
                    yp = float(bf.get("y", 0)) / 100.0
                except (TypeError, ValueError):
                    continue

                try:
                    size_percent = float(bf.get("size", 20.0))
                except (TypeError, ValueError):
                    size_percent = 20.0

                size_percent = max(1.0, min(100.0, size_percent))

                box_size = int(w * (size_percent / 100.0))
                box_size = max(20, min(600, box_size))
                half = box_size // 2

                cx = int(xp * w)
                cy = int(yp * h)

                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(w, cx + half)
                y2 = min(h, cy + half)

                if x2 <= x1 or y2 <= y1:
                    continue

                bw = x2 - x1
                bh = y2 - y1

            # 안전 클램프
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            bw = max(1, min(bw, w - x1))
            bh = max(1, min(bh, h - y1))

            if frame_effect == "emoji" and emoji_img is not None:
                _overlay_emoji(frame, x1, y1, bw, bh, emoji_img)
            else:
                gaussian_blur_region(frame, x1, y1, bw, bh, blur_kernel_size)

    # -----------------------------
    # 7-c. 자유 스티커 좌표
    # -----------------------------
    if not detect_only and sticker_mode and emoji_img is not None and sticker_positions:
        for sp in sticker_positions:
            try:
                xp = float(sp.get("x", 0)) / 100.0
                yp = float(sp.get("y", 0)) / 100.0
            except (TypeError, ValueError):
                continue

            raw_size = sp.get("size", 120)
            try:
                box_size = int(raw_size)
            except (TypeError, ValueError):
                box_size = 120
            box_size = max(20, min(600, box_size))
            half = box_size // 2

            cx = int(xp * w)
            cy = int(yp * h)

            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            if x2 <= x1 or y2 <= y1:
                continue

            bw = x2 - x1
            bh = y2 - y1
            _overlay_emoji(frame, x1, y1, bw, bh, emoji_img)

    # -----------------------------
    # 8. 결과 저장
    # -----------------------------
    ext = os.path.splitext(output_path)[1].lower()
    encode_ext = {
        ".png": ".png",
        ".jpg": ".jpg",
        ".jpeg": ".jpg",
        ".webp": ".webp",
        ".bmp": ".bmp",
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

    return output_path, face_boxes_for_api
