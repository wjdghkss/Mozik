import os
import cv2
import json
import numpy as np

from core import gaussian_blur_region, expand_box
from detect import load_yolov5, detect_faces_yolov5
from face_registry import FaceRegistry
from face_embedder import get_embedding
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

# ----------------------------------
# 전역 설정
# ----------------------------------
YOLOV5_WEIGHTS = os.path.join(os.path.dirname(__file__), "best.pt")
DETECT_THUMB_W = 640
FACE_PAD = 0.05

FACE_SIM_THRESHOLD = 0.60
FACE_SIM_MARGIN = 0.08

VIDEO_FACE_SIM_THRESHOLD = 0.72   # 0.65~0.70 정도 추천
VIDEO_FACE_SIM_MARGIN    = 0.05 

EMOJI_DIR = os.path.join(os.path.dirname(__file__), "emoji")
DEFAULT_EMOJI_FILES = ["emoji1.png", "emoji2.png", "emoji3.png", "emoji4.png", "emoji5.png"]

_DETECTOR = None

EMOJI_CHAR_TO_INDEX = {
    "🙂": 0,
    "😢": 1,
    "😎": 2,
    "🤔": 3,
    "🥰": 4,
}

# ----------------------------------
# 데이터 클래스
# ----------------------------------
@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    identity: Optional[str] = None
    emb: Optional[np.ndarray] = None
    enabled: bool = True
    sim: Optional[float] = None


@dataclass
class DetectedFace:
    id: int
    frame: int
    box: List[int]  # [x, y, w, h]
    mosaic: bool = True

    # 🔵 등록 얼굴 관련 메타데이터 (영상 프리뷰에서 사용)
    registered: bool = False       # 단순히 "등록된 얼굴인가"
    is_registered: bool = False    # 프론트에서 같이 체크하는 플래그
    matched: bool = False          # 매칭 성공 여부
    excluded: bool = False         # 정책상 기본 제외 얼굴이면 True


# ----------------------------------
# 내부 유틸
# ----------------------------------
def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = load_yolov5(YOLOV5_WEIGHTS)
    return _DETECTOR


def _detect_on_frame(detector, frame) -> List[FaceBox]:
    h, w = frame.shape[:2]
    faces: List[FaceBox] = []

    if detector is None:
        return faces

    if w > DETECT_THUMB_W:
        scale = DETECT_THUMB_W / float(w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        small = frame

    try:
        rects = detect_faces_yolov5(detector, small)
    except Exception:
        rects = []

    pad = float(FACE_PAD)

    for (x, y, w0, h0) in rects:
        if scale != 1.0:
            x, y, w0, h0 = int(x / scale), int(y / scale), int(w0 / scale), int(h0 / scale)

        ex, ey, ew, eh = expand_box(x, y, w0, h0, pad, pad, w, h)
        if ew >= 12 and eh >= 12:
            faces.append(FaceBox(ex, ey, ew, eh))

    return faces


# ----------------------------------
# 🔹 등록 얼굴 정보 채우는 함수 (identity 세팅)
# ----------------------------------
def _annotate_authorized_faces(registry: FaceRegistry, faces: List[FaceBox], frame_bgr):
    """
    1) ArcFace 임베딩으로 identity와 sim(유사도)을 붙이고,
    2) 같은 identity로 인식된 얼굴이 여러 개라면
       '가장 큰 얼굴(박스 면적이 큰 것)' 하나만 등록 얼굴로 남기는 함수이다.
    """
    if not faces:
        return

    h, w = frame_bgr.shape[:2]

    # 1) 각 얼굴에 identity, sim 채우기
    for b in faces:
        if b.identity is not None:
            continue

        x1, y1 = max(0, b.x), max(0, b.y)
        x2, y2 = min(w, b.x + b.w), min(h, b.y + b.h)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

        try:
            if b.emb is None:
                b.emb = get_embedding(crop)

            name, sim = registry.match(
                b.emb,
                min_threshold=VIDEO_FACE_SIM_THRESHOLD,
                margin=VIDEO_FACE_SIM_MARGIN,
            )

            # 디버그용으로 유사도 한 번 찍어보면 튜닝에 도움 됨
            print(f"[VIDEO MATCH] name={name}, sim={sim}")

            if sim is None or sim < VIDEO_FACE_SIM_THRESHOLD:
                b.identity = None
                b.sim = None
            else:
                b.identity = name
                b.sim = sim


        except Exception:
            b.identity = None
            b.sim = None

    # 2) 같은 identity가 여러 개면 "가장 큰 얼굴" 하나만 남기기
    id_to_best: Dict[str, FaceBox] = {}
    for b in faces:
        if not b.identity:
            continue

        cur_best = id_to_best.get(b.identity)
        if cur_best is None or (b.w * b.h) > (cur_best.w * cur_best.h):
            id_to_best[b.identity] = b

    for b in faces:
        if not b.identity:
            continue
        if id_to_best.get(b.identity) is not b:
            b.identity = None
            b.sim = None


# ----------------------------------
# 🔹 어떤 얼굴을 실제로 블러/모자이크/이모지 처리할지 결정
# ----------------------------------
def _should_blur_auto_face(registry: FaceRegistry, box: FaceBox, auth_exclude_enable: bool) -> bool:
    """
    - auth_exclude_enable == False 이면: 무조건 처리(True)
    - auth_exclude_enable == True 이면:
        - box.identity 가 None 이면: 등록 안 된 얼굴 → 처리(True)
        - box.identity 가 있으면: FaceRegistry 정책에 따라 처리 여부 결정
          * "exclude" → 처리하지 않음(False)
          * "force"   → 항상 처리(True)
          * "neutral" → 기본 처리(True)
    """
    if not box.enabled:
        return False

    if not auth_exclude_enable:
        return True

    if box.identity:
        mode = registry.get_policy(box.identity)
        if mode == "exclude":
            return False
        if mode == "force":
            return True

    return True


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


def _overlay_emoji_pixel(frame, px, py, emoji_img,
                         target_w: int | None = None,
                         target_h: int | None = None):
    if emoji_img is None:
        return

    eh, ew = emoji_img.shape[:2]
    h, w = frame.shape[:2]

    if target_w is None or target_h is None:
        tw = ew
        th = eh
    else:
        tw = target_w
        th = target_h

    max_w = int(w * 0.4)
    max_h = int(h * 0.4)
    tw = max(10, min(tw, max_w))
    th = max(10, min(th, max_h))

    x1 = int(px - tw // 2)
    y1 = int(py - th // 2)
    x2 = x1 + tw
    y2 = y1 + th

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    emoji_resized = cv2.resize(emoji_img, (roi.shape[1], roi.shape[0]))

    if emoji_resized.shape[2] == 4:
        emoji_rgb = emoji_resized[:, :, :3]
        alpha = emoji_resized[:, :, 3:] / 255.0
        frame[y1:y2, x1:x2] = (alpha * emoji_rgb + (1 - alpha) * roi).astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = emoji_resized[:, :, :3]


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

    if emoji_resized.shape[2] == 4:
        b, g, r, a = cv2.split(emoji_resized)
        emoji_rgb = cv2.merge((b, g, r))
        alpha = (a / 255.0)[..., None]

        roi = frame[y1:y2, x1:x2].astype(float)
        blended = alpha * emoji_rgb + (1 - alpha) * roi
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = emoji_resized


# ----------------------------------
# 🔹 디버그용: 얼굴 bounding box 그려주는 함수
# ----------------------------------
def draw_frames(
    frame,
    faces: List[FaceBox],
    show_index=False,
    show_identity=False,
    color=(0, 255, 0),
    thickness: int = 2,
):
    if frame is None or not faces:
        return

    h, w = frame.shape[:2]

    for idx, box in enumerate(faces):
        x1 = max(0, box.x)
        y1 = max(0, box.y)
        x2 = min(w - 1, box.x + box.w)
        y2 = min(h - 1, box.y + box.h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

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


# ----------------------------------
# ★ 메인
# ----------------------------------
def process_video(
    input_path: str,
    output_path: str,
    blur_strength: int,
    auth_exclude_enable: bool = False,
    style: str = "blur",
    emoji_path: Optional[str] = None,
    emoji_index: int = 0,
    excluded_ids: Optional[List[int]] = None,
    draw_box=False,
    blur_frames: Optional[List[dict]] = None,
    sticker_mode: bool = False,
    sticker_positions: Optional[List[dict]] = None,
    sticker_face_ids: Optional[List[int]] = None,
    force_ids: Optional[List[int]] = None,
    delete_frames: Optional[List[dict]] = None,

    # --- New params for two-phase processing ---
    detect_only: bool = False,
    json_output_path: Optional[str] = None,
    face_data_path: Optional[str] = None,
    selected_emoji: Optional[str] = None,

    **kwargs,
) -> Tuple[Optional[str], Optional[str]]:
    """
    - detect_only=True  : 얼굴만 탐지해서 JSON으로 저장하고 (None, json_path) 반환
    - detect_only=False :
        - face_data_path 가 있으면 해당 JSON 기반으로 모자이크 적용
        - 없으면 프레임마다 YOLO로 탐지해서 바로 모자이크 적용
    """

    # --- Parameter normalization ---
    if blur_frames is None:
        blur_frames = []
    if sticker_positions is None:
        sticker_positions = []
    if sticker_face_ids is None:
        sticker_face_ids = []
    if force_ids is None:
        force_ids = []
    if delete_frames is None:
        delete_frames = []

    if isinstance(excluded_ids, str):
        try:
            if excluded_ids.strip().startswith("["):
                excluded_ids = json.loads(excluded_ids)
            else:
                excluded_ids = [int(x.strip()) for x in excluded_ids.split(",") if x.strip()]
        except (json.JSONDecodeError, ValueError):
            excluded_ids = []
    elif excluded_ids is None:
        excluded_ids = []

    style = (style or "blur").strip().lower()
    if style not in {"blur", "mosaic", "emoji"}:
        style = "blur"

    try:
        blur_strength = int(blur_strength)
    except Exception:
        blur_strength = 50
    blur_strength = max(0, min(100, blur_strength))

    # --- Video setup ---
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector = _get_detector()
    registry = FaceRegistry(sim_threshold=VIDEO_FACE_SIM_THRESHOLD)

    # 블러/모자이크 강도 계산
    strength_ratio = (blur_strength / 100.0) * 0.10 + 0.02
    base_len = min(width, height)

    mosaic_block = int(base_len * strength_ratio)
    mosaic_block = max(8, min(60, mosaic_block))

    blur_kernel_size = int(width * strength_ratio)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    blur_kernel_size = max(5, min(299, blur_kernel_size))

    # -----------------------------
    # Mode 1: Detect-Only
    # -----------------------------
    if detect_only:
        if not json_output_path:
            raise ValueError("json_output_path is required for detect_only mode")

        detected_faces_data: List[DetectedFace] = []
        face_counter = 0
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                faces = _detect_on_frame(detector, frame)

                if faces:
                    # detect-only 모드에서는 항상 identity를 붙여서 JSON으로 넘겨주는 게 좋음
                    _annotate_authorized_faces(registry, faces, frame)

                for box in faces:
                    is_reg = box.identity is not None

                    policy = None
                    try:
                        if is_reg:
                            policy = registry.get_policy(box.identity)
                    except Exception:
                        policy = None

                    detected_faces_data.append(
                        DetectedFace(
                            id=face_counter,
                            frame=frame_idx,
                            box=[box.x, box.y, box.w, box.h],
                            mosaic=True,
                            registered=is_reg,
                            is_registered=is_reg,
                            matched=is_reg,
                            excluded=(policy == "exclude"),
                        )
                    )
                    face_counter += 1

                frame_idx += 1
        finally:
            cap.release()

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(face) for face in detected_faces_data],
                f,
                indent=2,
                ensure_ascii=False,
            )

        return None, json_output_path

    # -----------------------------
    # Mode 2: 실제 영상 처리
    # -----------------------------
    writer: Optional[cv2.VideoWriter] = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # face_data 기반 처리 여부
    faces_to_process_map: Dict[int, List[Dict]] = {}
    use_face_data = bool(face_data_path and os.path.exists(face_data_path))
    if use_face_data:
        with open(face_data_path, "r", encoding="utf-8") as f:
            all_faces_data = json.load(f)
            for face_data in all_faces_data:
                if face_data.get("mosaic", True):
                    frame_num = face_data.get("frame")
                    if frame_num not in faces_to_process_map:
                        faces_to_process_map[frame_num] = []
                    faces_to_process_map[frame_num].append(face_data)

    # 이모지 이미지 준비
    emoji_img = None
    if style == "emoji" or sticker_mode:
        idx = emoji_index
        if selected_emoji:
            idx = EMOJI_CHAR_TO_INDEX.get(selected_emoji, emoji_index)

        emoji_img = _load_emoji_image(emoji_path, idx)
        if emoji_img is None:
            sticker_mode = False
            if style == "emoji":
                style = "blur"

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if use_face_data:
                # A) JSON 기반 처리
                if frame_idx in faces_to_process_map:
                    for face_data in faces_to_process_map[frame_idx]:
                        x, y, w, h = face_data["box"]

                        if style == "blur":
                            gaussian_blur_region(frame, x, y, w, h, blur_kernel_size)
                        elif style == "mosaic":
                            _apply_mosaic(frame, x, y, w, h, mosaic_size=mosaic_block)
                        elif style == "emoji" and emoji_img is not None:
                            _overlay_emoji(frame, x, y, w, h, emoji_img)
                        else:
                            gaussian_blur_region(frame, x, y, w, h, blur_kernel_size)

            else:
                # B) 프레임마다 YOLO 탐지 + (필요 시) ArcFace 매칭
                faces = _detect_on_frame(detector, frame)

                if faces and auth_exclude_enable:
                    # 등록 얼굴 제외 기능이 켜져 있을 때만 ArcFace 매칭 수행 → 속도 절약
                    _annotate_authorized_faces(registry, faces, frame)

                for idx, box in enumerate(faces):
                    if idx in excluded_ids:
                        continue

                    if idx not in force_ids:
                        if not _should_blur_auto_face(registry, box, auth_exclude_enable):
                            continue

                    if draw_box:
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
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    bw = max(1, min(bw, width - x))
                    bh = max(1, min(bh, height - y))

                    if current_style == "blur":
                        gaussian_blur_region(frame, x, y, bw, bh, blur_kernel_size)
                    elif current_style == "mosaic":
                        _apply_mosaic(frame, x, y, bw, bh, mosaic_size=mosaic_block)
                    elif current_style == "emoji" and emoji_img is not None:
                        _overlay_emoji(frame, x, y, bw, bh, emoji_img)
                    else:
                        gaussian_blur_region(frame, x, y, bw, bh, blur_kernel_size)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()

    return output_path, None


def _is_inside_delete_list(bf, delete_frames, W, H):
    """
    (This function is unchanged)
    """
    try:
        bx = float(bf.get("x", 0))
        by = float(bf.get("y", 0))
    except:
        return False

    for df in delete_frames:
        try:
            dx = float(df.get("x", 0))
            dy = float(df.get("y", 0))
        except:
            continue

        if abs(bx - dx) < 3 and abs(by - dy) < 3:
            return True

    return False
