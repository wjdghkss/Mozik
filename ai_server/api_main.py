#############################################
# ai_api/api_main.py  (AI 서버 - FastAPI + ArcFace 버전)
#############################################

import io
import os
import uuid
import traceback
import json
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from werkzeug.utils import secure_filename

# 내부 모듈들
from detect import load_yolov5, detect_faces_yolov5
from face_registry import FaceRegistry
from face_embedder import get_embedding
from api_service import (
    process_media_for_interactive_blur,
    detect_faces_in_media,
    apply_mosaics_to_media,
)
from pydantic import BaseModel

#############################################
# 경로 / 전역 설정
#############################################

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

FACE_DIR = os.path.join(BASE_DIR, "detected_face")
os.makedirs(FACE_DIR, exist_ok=True)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# YOLOv5 공통 설정이다.
YOLOV5_WEIGHTS = os.path.join(BASE_DIR, "best.pt")
detector = load_yolov5(YOLOV5_WEIGHTS)
DETECT_THUMB_W = 640

# 🔵 ArcFace FaceRegistry 전역 설정이다.
FACE_DB_DIR = os.path.join(BASE_DIR, "face_db")
os.makedirs(FACE_DB_DIR, exist_ok=True)

FACE_SIM_THRESHOLD = 0.35
FACE_SIM_MARGIN = 0.08

# 첫 시작 시에는 None으로 두고, get_registry()에서 실제로 생성한다.
_REGISTRY: Optional[FaceRegistry] = None

#############################################
# 비디오 프리뷰용 전역 설정이다.
#############################################

VIDEO_PREVIEW_MAX_FRAMES = 60   # 처음 60프레임까지만 후보로 본다.
VIDEO_PREVIEW_STEP = 5          # 5프레임마다 한 번씩만 검사한다.

# YOLOv5 검출기 캐시용 전역 변수이다.
_video_detector = None



def find_preview_frame_and_faces(video_path: str):
    """
    프리뷰용으로 쓸 비디오 프레임 인덱스와 해당 프레임의 얼굴 박스 목록을 찾는 함수이다.
    - 전체 영상을 모자이크하지 않고, 앞부분 몇 프레임만 빠르게 스캔한다.
    - 리턴 값:
        frame_index: int
        faces: [
            {"id": 0, "box": [x1, y1, x2, y2]},
            ...
        ]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없음: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("영상 메타데이터가 올바르지 않음")

    detector = get_video_detector()

    best_faces = []
    best_index = 0

    max_idx = min(total_frames, VIDEO_PREVIEW_MAX_FRAMES)

    for idx in range(0, max_idx, VIDEO_PREVIEW_STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]

        # 썸네일 크기로 줄여서 빠르게 검출한다.
        scale = 1.0
        if w > DETECT_THUMB_W:
            scale = DETECT_THUMB_W / float(w)
            thumb_w = DETECT_THUMB_W
            thumb_h = int(h * scale)
            frame_small = cv2.resize(frame, (thumb_w, thumb_h))
        else:
            frame_small = frame

        # YOLOv5 얼굴 검출 (기존 detect_faces_yolov5 형식에 맞춘다)
        det_boxes = detect_faces_yolov5(detector, frame_small)

        faces_this = []
        for j, b in enumerate(det_boxes):
            # b가 [x1, y1, x2, y2] 또는 [x1, y1, x2, y2, score] 형식이라고 가정한다.
            x1, y1, x2, y2 = map(float, b[:4])

            # 리사이즈 되었다면 원본 좌표로 되돌린다.
            if scale != 1.0:
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            # 얼굴이 아닌 긴 띠/간판 등을 걸러내기 위한 안전 필터이다.
            aspect = bw / bh
            area_ratio = (bw * bh) / float(w * h)

            # 너무 작거나 너무 큰 박스는 버린다.
            if area_ratio < 0.003 or area_ratio > 0.35:
                continue
            # 가로/세로 비율이 너무 극단적인 박스도 버린다.
            if aspect < 0.3 or aspect > 3.5:
                continue

            faces_this.append({
                "id": len(faces_this),           # 0,1,2,... 번호 부여
                "box": [x1, y1, x2, y2],         # 프론트에서는 이 좌표를 그대로 쓴다.
            })

        # 가장 얼굴이 많이 검출된 프레임을 선택한다.
        if len(faces_this) > len(best_faces):
            best_faces = faces_this
            best_index = idx

        # 충분히 많이 찾았으면 더 이상 안 도는 것도 가능하다.
        if len(best_faces) >= 5:
            break

    cap.release()
    return best_index, best_faces

def get_video_detector():
    """
    비디오 프리뷰용 YOLO 검출기를 1번만 로드해서 캐시하는 함수이다.
    """
    global _video_detector
    if _video_detector is None:
        print("[VIDEO_PREVIEW] YOLOv5 로딩 중...")
        # studio_image / studio_video 에서 쓰던 것과 동일한 weight 경로를 사용한다.
        # ex) YOLOV5_WEIGHTS = os.path.join(BASE_DIR, "best.pt")
        _video_detector = load_yolov5(YOLOV5_WEIGHTS)
        print("[VIDEO_PREVIEW] YOLOv5 로딩 완료")
    return _video_detector


#############################################
# FastAPI 앱 생성
#############################################

app = FastAPI(
    title="Mozik AI Server (ArcFace)",
    description="이미지/영상 모자이크 + ArcFace 기반 얼굴 등록을 제공하는 FastAPI 서버이다.",
    version="1.0.0",
)

# KT-Cloud Flask 서버에서 호출하므로 CORS를 널널하게 열어둔다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 정적 파일 서빙 설정 ---
# /uploads/filename, /results/filename URL로 접근 가능하다.
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")


#############################################
# Pydantic 모델
#############################################
class ApplyMosaicsRequest(BaseModel):
    original_media_path: str
    face_data: List[Dict[str, Any]]
    options: Dict[str, Any] = {}


#############################################
# 공통 유틸 함수
#############################################

def safe_log(prefix: str, e: Exception) -> None:
    print(f"[{prefix}] {e}")
    print(traceback.format_exc())


def safe_read_image(upload_file: UploadFile) -> np.ndarray:
    """
    FastAPI UploadFile -> OpenCV BGR 이미지로 변환하는 함수이다.
    """
    from PIL import Image  # lazy import
    try:
        upload_file.file.seek(0)
        img = Image.open(upload_file.file).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"이미지 디코딩 실패: {e}")


def apply_mosaic(img: np.ndarray, x: int, y: int, w: int, h: int, factor: int = 20) -> np.ndarray:
    """
    간단 픽셀 모자이크 함수이다. (테스트용 /api/mosaic/face 에서도 사용한다.)
    """
    h_img, w_img = img.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img

    small_w = max(1, (x2 - x1) // factor)
    small_h = max(1, (y2 - y1) // factor)

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    img[y1:y2, x1:x2] = mosaic
    return img


#############################################
# ArcFace / FaceRegistry 관련 유틸
#############################################

def get_registry() -> FaceRegistry:
    """
    ArcFace 임베딩을 관리하는 FaceRegistry 싱글톤이다.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = FaceRegistry(
            db_dir=FACE_DB_DIR,
            sim_threshold=FACE_SIM_THRESHOLD,
        )
    return _REGISTRY



def _parse_id_list(s: str) -> List[int]:
    """
    "0,1,2" 또는 "[0, 1, 2]" 같은 문자열을 정수 리스트로 바꾸는 유틸 함수이다.
    """
    s = (s or "").strip()
    if not s:
        return []
    try:
        if s.startswith("["):
            arr = json.loads(s)
            return [int(x) for x in arr]
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return []


def _extract_main_face_emb(bgr_img: np.ndarray) -> Optional[np.ndarray]:
    """
    이미지에서 가장 큰 얼굴 한 개만 골라 ArcFace 임베딩을 추출하는 함수이다.
    detect_faces_yolov5()가
    - 객체(b.x, b.y, b.w, b.h)를 반환하든
    - 튜플/리스트 (x, y, w, h) 또는 (x1, y1, x2, y2, ...)를 반환하든
    둘 다 처리할 수 있게 만든다.
    """
    if bgr_img is None:
        return None

    h, w = bgr_img.shape[:2]
    boxes = detect_faces_yolov5(detector, bgr_img, conf_thres=0.6)
    if not boxes:
        return None

    norm_boxes = []  # (x, y, bw, bh) 형태로 정규화된 박스 리스트이다.

    for b in boxes:
        # 1) dataclass / 객체 형태 (b.x, b.y, b.w, b.h)
        if hasattr(b, "w") and hasattr(b, "h"):
            x = float(b.x)
            y = float(b.y)
            bw = float(b.w)
            bh = float(b.h)

        # 2) 튜플/리스트 형태
        elif isinstance(b, (list, tuple)) and len(b) >= 4:
            x1 = float(b[0])
            y1 = float(b[1])
            x2 = float(b[2])
            y2 = float(b[3])

            # 보통 YOLO 출력은 (x1, y1, x2, y2)이므로 이렇게 처리한다.
            bw = x2 - x1
            bh = y2 - y1
            x = x1
            y = y1

        else:
            # 알 수 없는 형식은 건너뛴다.
            continue

        if bw <= 0 or bh <= 0:
            continue

        norm_boxes.append((x, y, bw, bh))

    if not norm_boxes:
        return None

    # 가장 큰 박스를 선택한다.
    x, y, bw, bh = max(norm_boxes, key=lambda t: t[2] * t[3])

    # 이미지 경계 내로 보정한다.
    x = int(max(0, min(x, w - 1)))
    y = int(max(0, min(y, h - 1)))
    bw = int(max(1, min(bw, w - x)))
    bh = int(max(1, min(bh, h - y)))

    face = bgr_img[y:y + bh, x:x + bw]
    if face.size == 0:
        return None

    return get_embedding(face)


#############################################
# 1. 헬스 체크
#############################################

@app.get("/api/health")
def health():
    """
    서버 상태를 확인하는 엔드포인트이다.
    """
    try:
        face_files = [
            fn for fn in os.listdir(FACE_DIR)
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        return {"status": "ok", "faces": len(face_files)}
    except Exception as e:
        safe_log("HEALTH", e)
        raise HTTPException(status_code=500, detail="health check failed")


#############################################
# 1-1. 등록 얼굴 디버그용 엔드포인트
#############################################

@app.get("/api/face/debug")
def api_face_debug():
    """
    face_db 디렉터리에 어떤 파일들이 저장되어 있는지 확인하는 디버그용 엔드포인트이다.
    실제 서비스에서는 막아두는 것이 좋다.
    """
    db_dir = os.path.join(BASE_DIR, "face_db")
    files = []
    if os.path.isdir(db_dir):
        try:
            files = sorted(os.listdir(db_dir))
        except Exception as e:
            safe_log("FACE_DEBUG", e)
    return {"db_dir": db_dir, "files": files}


#############################################
# 2. YOLO 얼굴 검출 (디버그용)
#############################################

@app.post("/api/detect/face")
async def api_detect_face(file: UploadFile = File(...)):
    """
    업로드된 이미지에서 YOLOv5로 얼굴 박스를 검출한다.
    (디버깅/테스트 용도)
    """
    try:
        img = safe_read_image(file)
        boxes = detect_faces_yolov5(detector, img, conf_thres=0.6)
        return {
            "faces": [
                {"x": int(b.x), "y": int(b.y), "w": int(b.w), "h": int(b.h)}
                for b in boxes
            ]
        }
    except Exception as e:
        safe_log("DETECT", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "detect failed", "details": str(e)},
        )


#############################################
# 3. ArcFace 기반 얼굴 등록 API
#############################################

@app.post("/api/face/register")
async def api_register_face(
    identity: str = Form(..., description="사용자 고유 ID(예: user_id, email 등)이다."),
    policy: str = Form("exclude", description="exclude | force | neutral 중 하나이다."),
    images: List[UploadFile] = File(..., description="여러 장의 얼굴 이미지이다."),
):
    """
    마이페이지에서 사용자가 여러 장의 얼굴 사진을 업로드하면
    ArcFace 임베딩을 추출하여 FaceRegistry에 등록하는 엔드포인트이다.
    """
    if not identity:
        raise HTTPException(status_code=400, detail="identity가 비어 있다.")

    registry = get_registry()
    added_count = 0
    failed_count = 0

    for uf in images:
        try:
            img = safe_read_image(uf)
            if img is None:
                failed_count += 1
                continue

            emb = _extract_main_face_emb(img)
            if emb is None:
                failed_count += 1
                continue

            registry.add(identity, emb)
            added_count += 1
        except Exception as e:
            safe_log("FACE_REGISTER_IMG", e)
            failed_count += 1

    registry.set_policy(identity, policy or "exclude")

    print(f"[FACE_REGISTER] identity={identity}, added={added_count}, failed={failed_count}, policy={policy}")

    return {
        "success": True,
        "identity": identity,
        "added": added_count,
        "failed": failed_count,
        "policy": registry.get_policy(identity),
    }


#############################################
# 4. 신규 2단계 처리 API
#############################################

@app.post("/api/detect-faces")
async def api_detect_faces_in_media(request: Request, file: UploadFile = File(...)):
    """
    1단계: 미디어(영상)를 업로드하여 얼굴을 탐지하고, 좌표 데이터가 담긴 JSON 파일을 생성한다.
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        result = detect_faces_in_media(
            file_bytes=file_bytes,
            original_filename=file.filename,
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("reason", "Face detection failed"))

        face_data: List[Dict[str, Any]] = []
        face_data_path = result.get("face_data_path")
        if face_data_path and os.path.exists(face_data_path):
            try:
                with open(face_data_path, "r", encoding="utf-8") as f:
                    face_data = json.load(f)
            except Exception as e:
                safe_log("DETECT_FACES_JSON_READ", e)

        # 읽어온 리스트를 응답에 그대로 포함시킨다.
        result["face_data"] = face_data

        base_url = str(request.base_url)
        result["original_media_url"] = f"{base_url}{result['original_media_path']}".replace("\\", "/")
        result["face_data_url"] = f"{base_url}{result['face_data_path']}".replace("\\", "/")

        return JSONResponse(content=result)

    except Exception as e:
        safe_log("DETECT_FACES", e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.post("/api/apply-mosaics")
async def api_apply_mosaics_to_media(request: Request, body: ApplyMosaicsRequest):
    """
    2단계: 원본 미디어 경로와 수정된 얼굴 데이터를 받아 최종 모자이크 처리된 영상을 생성한다.
    """
    try:
        result = apply_mosaics_to_media(
            original_media_path=body.original_media_path,
            face_data_content=body.face_data,
            options=body.options,
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("reason", "Failed to apply mosaics"))

        base_url = str(request.base_url)
        result["processed_media_url"] = f"{base_url}{result['processed_media_path']}".replace("\\", "/")

        return JSONResponse(content=result)

    except Exception as e:
        safe_log("APPLY_MOSAICS", e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


#############################################
# 4-legacy. 기존 통합 미디어 처리 API
#############################################

@app.post("/api/process/media")
async def process_media(
    request: Request,
    file: UploadFile = File(...),
    options: str = Form("{}", description="JSON string of processing options"),
    blur_frames: Optional[str] = Form(None),
):
    """
    이미지/비디오 파일을 받아 블러/모자이크 등 효과를 적용하고,
    결과물 경로와 함께 탐지된 얼굴의 좌표 데이터를 JSON으로 반환한다.
    """
    try:
        try:
            proc_options = json.loads(options)
        except json.JSONDecodeError:
            proc_options = {}

        try:
            proc_options = json.loads(options)
        except json.JSONDecodeError:
            proc_options = {}

        # 🔵 디버그용 로그 추가 (선택)
        try:
            print("[PROCESS_MEDIA] options keys:", list(proc_options.keys()))
            if "blur_frames" in proc_options:
                print("[PROCESS_MEDIA] blur_frames length from options:",
                      len(proc_options.get("blur_frames") or []))
        except Exception:
            pass


        # 🔵 수동 프레임(blur_frames) 문자열이 들어오면 options 안에 합치기
        if blur_frames:
            try:
                parsed_blur_frames = json.loads(blur_frames)
                proc_options["blur_frames"] = parsed_blur_frames
                print("[PROCESS_MEDIA] blur_frames merged into options:",
                      len(parsed_blur_frames), "개")
            except Exception as e:
                safe_log("PROCESS_MEDIA_BLUR_FRAMES_PARSE", e)


        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # api_service.process_media_for_interactive_blur 호출
        result = process_media_for_interactive_blur(
            file_bytes=file_bytes,
            original_filename=file.filename,
            options=proc_options,
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("reason", "Unknown processing error"))

        # -----------------------------
        # ★ face_data 설정 로직 수정
        # -----------------------------
        # 1) api_service 쪽에서 이미 face_data를 넣어줬으면 그걸 우선 사용
        face_data: List[Dict[str, Any]] = result.get("face_data")

        # 2) face_data가 없으면, 예전 방식처럼 face_data_path가 있다면 JSON 파일을 읽어서 채움
        if face_data is None:
            face_data = []
            face_data_path = result.get("face_data_path")
            if face_data_path and os.path.exists(face_data_path):
                try:
                    with open(face_data_path, "r", encoding="utf-8") as f:
                        face_data = json.load(f)
                except Exception as e:
                    safe_log("PROCESS_MEDIA_JSON_READ", e)

        # 3) 최종 face_data를 result에 다시 넣는다.
        result["face_data"] = face_data

        # -----------------------------
        # URL 구성
        # -----------------------------
        base_url = str(request.base_url)
        if "original_path" in result:
            result["original_url"] = f"{base_url}{result['original_path']}"
        if "processed_path" in result:
            result["processed_url"] = f"{base_url}{result['processed_path']}"

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        safe_log("PROCESS_MEDIA", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "Media processing failed", "details": str(e)},
        )


#############################################
# 5. 전역 예외 처리
#############################################

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    safe_log("GLOBAL", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Error", "details": str(exc)},
    )


#############################################
# 로컬 개발용 실행
#############################################

if __name__ == "__main__":
    import uvicorn

    print("[AI_API] FastAPI 서버를 시작한다.")
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=False)
