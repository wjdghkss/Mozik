# api_service.py
import os
import io
import uuid
import shutil
import json
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

from face_embedder import get_embedding
from face_registry import FaceRegistry
from detect import load_yolov5, detect_faces_yolov5
from studio_image import process_image
from studio_video import process_video

YOLO_MODEL = None
REGISTRY: Optional[FaceRegistry] = None

# 디렉토리 설정
UPLOADS_DIR = "uploads"
RESULTS_DIR = "results"
TEMP_DIR = "temp"  # 임시 파일을 위한 디렉토리
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def _get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = load_yolov5("best.pt")
    return YOLO_MODEL


def _get_registry() -> FaceRegistry:
    global REGISTRY
    if REGISTRY is None:
        REGISTRY = FaceRegistry(db_dir="face_db", sim_threshold=0.40)
    return REGISTRY


def bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img


def bgr_to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("이미지 인코딩 실패")
    return buf.tobytes()


def register_face(img_bytes: bytes, name: str) -> Dict[str, Any]:
    # (기존 함수 내용과 동일)
    pass


def detect_and_match(img_bytes: bytes) -> Dict[str, Any]:
    # (기존 함수 내용과 동일)
    pass


def reload_registry() -> Dict[str, Any]:
    global REGISTRY
    REGISTRY = FaceRegistry(db_path="face_db", sim_threshold=0.40)
    return {"success": True}


# ----------------------------------------------
# 옵션 키 정규화 유틸 (camelCase → snake_case)
# ----------------------------------------------
def _normalize_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """
    프론트에서 넘어온 options(dict)를 studio_image / studio_video에서
    기대하는 snake_case 형식으로 맞춰주는 함수이다.
    예:
      - blurStrength      -> blur_strength
      - blurFrames        -> blur_frames
      - selectedEmoji     -> selected_emoji
      - stickerMode       -> sticker_mode
      - authExcludeEnable -> auth_exclude_enable
    """
    if not isinstance(options, dict):
        return {}

    opt = dict(options)  # 원본을 건드리지 않기 위해 복사본 사용

    def move(old_key: str, new_key: str):
        if old_key in opt and new_key not in opt:
            opt[new_key] = opt.pop(old_key)

    # 자주 쓰는 옵션들 매핑
    move("blurStrength", "blur_strength")
    move("blur_frames", "blur_frames")  # 혹시 이미 snake_case로 올 때 대비 (no-op)
    move("blurFrames", "blur_frames")

    move("style", "style")
    move("detectOnly", "detect_only")
    move("drawBox", "draw_box")

    move("stickerMode", "sticker_mode")
    move("stickerPositions", "sticker_positions")
    move("stickerFaceIds", "sticker_face_ids")

    move("forceIds", "force_ids")
    move("deleteFrames", "delete_frames")

    move("selectedEmoji", "selected_emoji")

    move("authExcludeEnable", "auth_exclude_enable")
    move("excludedIds", "excluded_ids")

    # 문자열로 넘어온 JSON을 파싱(필요할 수 있는 필드들)
    for key in ("blur_frames", "sticker_positions", "delete_frames"):
        v = opt.get(key)
        if isinstance(v, str):
            v_str = v.strip()
            # JS 쪽에서 JSON.stringify 해서 보내는 경우
            if v_str.startswith("[") or v_str.startswith("{"):
                try:
                    opt[key] = json.loads(v_str)
                except Exception:
                    # 파싱 실패하면 그냥 원본 문자열 유지
                    pass

    return opt


# ----------------------------------------------
# ★ 1단계: 얼굴 탐지 및 데이터 추출 서비스
# ----------------------------------------------
def detect_faces_in_media(file_bytes: bytes, original_filename: str) -> Dict[str, Any]:
    """
    미디어 파일에서 얼굴을 탐지하여 좌표 데이터 JSON을 생성한다.
    """
    _, ext = os.path.splitext(original_filename)
    unique_id = uuid.uuid4().hex

    # 원본 파일 저장
    input_filename = f"{unique_id}{ext}"
    input_path = os.path.join(UPLOADS_DIR, input_filename)
    with open(input_path, "wb") as f:
        f.write(file_bytes)

    # 얼굴 데이터 JSON 파일 경로 설정
    json_filename = f"faces_{unique_id}.json"
    json_output_path = os.path.join(RESULTS_DIR, json_filename)

    try:
        # `studio_video.process_video`를 detect_only 모드로 호출
        _, result_json_path = process_video(
            input_path=input_path,
            output_path="",  # detect_only 모드에선 사용되지 않음
            blur_strength=0,  # detect_only 모드에선 사용되지 않음
            detect_only=True,
            json_output_path=json_output_path,
        )

        if not result_json_path or not os.path.exists(result_json_path):
            raise RuntimeError("JSON 파일 생성에 실패했습니다.")

        return {
            "success": True,
            "original_media_path": input_path,
            "face_data_path": result_json_path,
        }
    except Exception as e:
        return {"success": False, "reason": str(e)}


# ----------------------------------------------
# ★ 2단계: 수정된 데이터 기반 모자이크 적용 서비스
# ----------------------------------------------
def apply_mosaics_to_media(
    original_media_path: str,
    face_data_content: Dict[str, Any],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """
    원본 미디어 경로와 수정된 얼굴 데이터를 받아 최종 처리된 영상을 생성한다.
    """
    if not os.path.exists(original_media_path):
        return {"success": False, "reason": "Original media file not found."}

    _, ext = os.path.splitext(original_media_path)
    unique_id = uuid.uuid4().hex

    # 최종 결과 영상 경로
    output_filename = f"processed_{unique_id}{ext}"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    # 임시 JSON 파일 생성
    temp_json_path = os.path.join(TEMP_DIR, f"temp_{unique_id}.json")
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(face_data_content, f)

    # 옵션 정규화 (여기서 selected_emoji, blur_frames 등 정리)
    norm_options = _normalize_options(options or {})

    try:
        # `studio_video.process_video`를 처리 모드로 호출
        processed_path, _ = process_video(
            input_path=original_media_path,
            output_path=output_path,
            face_data_path=temp_json_path,
            **norm_options,
        )

        if not processed_path or not os.path.exists(processed_path):
            raise RuntimeError("미디어 처리 실패")

        return {
            "success": True,
            "processed_media_path": processed_path,
        }
    except Exception as e:
        return {"success": False, "reason": str(e)}
    finally:
        # 임시 JSON 파일 삭제
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)


# ----------------------------------------------
# 기존 통합 함수 (레거시)
# ----------------------------------------------
def process_media_for_interactive_blur(
    file_bytes: bytes,
    original_filename: str,
    options: dict
) -> Dict[str, Any]:
    """
    인터랙티브 미리보기용 공통 처리 함수이다.
    - 파일을 uploads/results에 저장하고
    - blur_frames, blur_strength 등만 정규화해서 process_image / process_video 로 넘긴다.
    """
    # ----- 원본 파일명 / 확장자 처리 -----
    if original_filename is None:
        original_filename = ""

    _, ext = os.path.splitext(original_filename)
    ext = (ext or "").lower()

    unique_id = uuid.uuid4().hex
    input_filename = f"{unique_id}{ext}"
    input_path = os.path.join(UPLOADS_DIR, input_filename)

    with open(input_path, "wb") as f:
        f.write(file_bytes)

    output_filename = f"result_{unique_id}{ext}"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    boxes_data: List[Dict[str, Any]] = []

    # ---- options 정리 시작 ----
    raw = options or {}

    # 🔵 여기서 mediaType/fileType 값 확인
    media_type = (raw.get("mediaType") or raw.get("fileType") or "").lower()

    # 🔵 ext + media_type 둘 다 보고 영상 여부 결정
    is_video = (
        media_type == "video"
        or ext in [".mp4", ".mov", ".avi", ".mkv"]
    )

    # --------- 작은 헬퍼들 ---------
    def to_bool(v, default=False):
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        s = str(v).strip().lower()
        if s in ("1", "true", "on", "yes"):
            return True
        if s in ("0", "false", "off", "no"):
            return False
        return default

    def to_int(v, default=0):
        try:
            return int(v)
        except Exception:
            return default


    # --------- blur_frames 정규화 ---------
    raw_blur_frames = raw.get("blurFrames", raw.get("blur_frames")) or []
    if isinstance(raw_blur_frames, str):
        try:
            blur_frames = json.loads(raw_blur_frames)
        except Exception:
            blur_frames = []
    elif isinstance(raw_blur_frames, list):
        blur_frames = raw_blur_frames
    else:
        blur_frames = []

    # 디버그용 로그
    print("[process_media] blur_frames len:", len(blur_frames))
    # print("[process_media] blur_frames sample:", blur_frames[:1])

    # excluded_ids는 항상 문자열로
    excluded_ids = raw.get("excludedIds", raw.get("excluded_ids", "")) or ""

    # --------- 실제로 넘길 옵션 ---------
    normalized: Dict[str, Any] = {
        "blur_strength": to_int(
            raw.get("blurStrength", raw.get("blur_strength", 50)), 50
        ),
        "style": raw.get("style", "blur"),
        "auth_exclude_enable": to_bool(
            raw.get("authExcludeEnable", raw.get("auth_exclude_enable", False)),
            False,
        ),
        "detect_only": to_bool(
            raw.get("detectOnly", raw.get("detect_only", False)), False
        ),
        "excluded_ids": excluded_ids,

        # ★ 수동 블러 프레임
        "blur_frames": blur_frames,

        # 이모지/스티커는 일단 무시 (selected_emoji만 형식 맞춰서 보냄)
        "selected_emoji": raw.get(
            "selectedEmoji", raw.get("selected_emoji", "😎")
        ),

        # 등록 얼굴/프리뷰 관련
        "identity": raw.get("identity", "anonymous"),
        "is_preview": to_bool(
            raw.get("isPreview", raw.get("is_preview", True)), True
        ),
        "draw_frames": False,  # 결과물에 프레임 선은 안 그림
    }

    print("[process_media] normalized keys:", list(normalized.keys()))

    try:
        if is_video:
            processed_path, boxes_data = process_video(
                input_path=input_path,
                output_path=output_path,
                **normalized,
            )
        else:
            processed_path, boxes_data = process_image(
                input_path=input_path,
                output_path=output_path,
                **normalized,
            )
                # ---------------------------------------------
        # ★ 등록 얼굴 후처리:
        #    한 identity(또는 is_registered=True 그룹)에서
        #    가장 큰 얼굴 1개만 등록 얼굴로 남기고,
        #    나머지는 is_registered/registered를 False 로 내린다.
        # ---------------------------------------------
        try:
            if isinstance(boxes_data, list):
                # 1) identity 기준으로 그룹 나누기
                groups: Dict[str, List[Dict[str, Any]]] = {}
                for box in boxes_data:
                    if not isinstance(box, dict):
                        continue
                    ident = box.get("identity")
                    if not ident:
                        continue
                    key = str(ident)
                    groups.setdefault(key, []).append(box)

                # 2) identity가 하나도 없고, 대신 is_registered만 true인 경우도 묶어준다.
                if not groups:
                    reg_list = [
                        b for b in boxes_data
                        if isinstance(b, dict) and b.get("is_registered")
                    ]
                    if len(reg_list) > 1:
                        groups["__noid__"] = reg_list

                # 3) 각 그룹마다 가장 큰 박스만 등록 얼굴로 유지
                for ident, items in groups.items():
                    if len(items) <= 1:
                        continue

                    # 면적이 가장 큰 박스 찾기
                    best = None
                    best_area = -1
                    for b in items:
                        try:
                            w = int(b.get("w", 0) or 0)
                            h = int(b.get("h", 0) or 0)
                        except Exception:
                            w, h = 0, 0
                        area = w * h
                        if area > best_area:
                            best_area = area
                            best = b

                    # 대표가 아닌 나머지 박스는 등록 상태 해제
                    for b in items:
                        if b is best:
                            # 대표 한 개는 그대로 둔다.
                            continue
                        b["is_registered"] = False
                        b["registered"] = False
                        # 필요하면 identity도 지울 수 있다.
                        # b["identity"] = None
        except Exception as e:
            print("[process_media] post-process registered faces failed:", e)


        return {
            "success": True,
            "original_path": input_path,
            "processed_path": processed_path,
            "blur_areas": boxes_data,
            "face_data": boxes_data,
        }
    except Exception as e:
        return {"success": False, "reason": str(e)}
    finally:
        # 필요하면 여기서 임시 파일 정리
        pass
