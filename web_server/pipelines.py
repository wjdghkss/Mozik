import os, sys, csv, cv2
from typing import List, Dict
from .core import Box, ensure_dir, expand_box, pixelate_region, gaussian_blur_region, assign_ids
from .yolo_detector import load_yolo_model, detect_faces_yolo
from .face_embedder import FaceEmbedder
from .face_registry import FaceRegistry

# ---------- CSV I/O ----------
def write_csv(csv_path: str, boxes: List[Box]):
    ensure_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame","t_ms","id","x","y","w","h","source"])
        for b in boxes:
            w.writerow([b.frame, b.t_ms, b.id, b.x, b.y, b.w, b.h, b.source])

def read_csv(csv_path: str) -> List[Box]:
    out: List[Box] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(Box(
                frame=int(row["frame"]),
                t_ms=int(row["t_ms"]),
                id=int(row["id"]),
                x=int(row["x"]),
                y=int(row["y"]),
                w=int(row["w"]),
                h=int(row["h"]),
                source=row.get("source","manual")
            ))
    return out

# ---------- scan ----------
def cmd_scan(args):
    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 동영상 없음: {args.input}"); sys.exit(1)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없음: {args.input}"); sys.exit(1)

    try:
        load_yolo_model() # Ensure YOLO model is loaded
    except Exception as e:
        print(f"[ERROR] 얼굴 모델 로드 실패: {e}"); sys.exit(1)

    boxes_all: List[Box] = []
    prev_boxes: List[Box] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Not needed for YOLO

        rects = detect_faces_yolo(frame) # Use YOLO detector
        curr_boxes = assign_ids(prev_boxes, rects, frame_idx, t_ms)
        for b in curr_boxes: b.source = "auto"
        boxes_all.extend(curr_boxes)
        prev_boxes = curr_boxes
        frame_idx += 1

        if args.preview and frame_idx % args.preview_stride == 0:
            preview = frame.copy()
            for (x,y,w,h) in rects:
                x2,y2,w2,h2 = expand_box(x,y,w,h, args.pad_x, args.pad_y, preview.shape[1], preview.shape[0])
                cv2.rectangle(preview, (x2,y2), (x2+w2, y2+h2), (0,255,0), 2)
            cv2.imshow("scan preview (q to stop preview)", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                args.preview = False
                cv2.destroyAllWindows()

    cap.release(); cv2.destroyAllWindows()
    write_csv(args.output_csv, boxes_all)
    print(f"[OK] CSV 저장: {args.output_csv}")

# ---------- render ----------
def cmd_render(args):
    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 동영상 없음: {args.input}"); sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없음: {args.input}"); sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir(args.output)
    out = cv2.VideoWriter(args.output, fourcc, fps, (W, H))
    if not out.isOpened():
        print(f"[ERROR] 출력 비디오 생성 실패: {args.output}"); sys.exit(1)

    boxes_by_frame: Dict[int, List[Box]] = {}
    # face_cascade = None # Removed Haar Cascade variable

    # 얼굴 인식기 초기화
    try:
        face_embedder = FaceEmbedder()
        face_registry = FaceRegistry(
            db_path=args.db_path,
            sim_threshold=args.sim_threshold,
            unknown_policy=args.unknown_policy
        )
        if len(face_registry.people) > 0:
            print(f"[OK] {len(face_registry.people)}명 얼굴 로드.")
    except Exception as e:
        print(f"[ERROR] 얼굴 DB 로드 실패: {e}"); sys.exit(1)

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"[ERROR] CSV 없음: {args.csv}"); sys.exit(1)
        for b in read_csv(args.csv):
            boxes_by_frame.setdefault(b.frame, []).append(b)
        print(f"[OK] CSV에서 {sum(len(v) for v in boxes_by_frame.values())}개 박스 로드.")
    else:
        print("[INFO] CSV 없이 자동 탐지로 렌더.")
        try:
            load_yolo_model() # Ensure YOLO model is loaded
        except Exception as e:
            print(f"[ERROR] 얼굴 모델 로드 실패: {e}"); sys.exit(1)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            if boxes_by_frame:
                curr = boxes_by_frame.get(frame_idx, [])
                rects = [(b.x, b.y, b.w, b.h) for b in curr]
            else:
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Not needed for YOLO
                rects = detect_faces_yolo(frame) # Use YOLO detector

            # 얼굴 인식 및 정책 기반 처리
            for (x,y,w,h) in rects:
                x2,y2,w2,h2 = expand_box(x,y,w,h, args.pad_x, args.pad_y, frame.shape[1], frame.shape[0])
                
                # 얼굴 영역 자르기
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0: continue

                # 얼굴 임베딩 및 정책 확인
                embedding = face_embedder.get_embedding(face_img)
                name, _ = face_registry.match(embedding)

                policy = face_registry.unknown_policy
                if name:
                    policy = face_registry.get_policy(name)

                # 정책에 따라 모자이크 처리
                apply_mosaic = (policy == 'force')

                if apply_mosaic:
                    if args.mode == "pixelate":
                        pixelate_region(frame, x2,y2,w2,h2, strength=args.strength)
                    else:
                        gaussian_blur_region(frame, x2,y2,w2,h2, strength=args.strength)
                
                if args.draw_box:
                    # 정책에 따라 박스 색상 및 텍스트 변경
                    color = (0, 255, 0) # green (neutral)
                    if policy == 'force': color = (0,0,255) # red
                    elif policy == 'exclude': color = (255,0,0) # blue
                    
                    cv2.rectangle(frame, (x2,y2), (x2+w2, y2+h2), color, 2)
                    display_name = name if name else "unknown"
                    cv2.putText(frame, display_name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


            out.write(frame); frame_idx += 1

            if args.preview and frame_idx % args.preview_stride == 0:
                cv2.imshow("render preview (q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    args.preview = False
                    cv2.destroyAllWindows()
    finally:
        out.release(); cap.release(); cv2.destroyAllWindows()
    print(f"[OK] 저장: {args.output}")

# ---------- preview (키보드/슬라이더) ----------
def cmd_preview(args):
    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 동영상 없음: {args.input}"); sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없음: {args.input}"); sys.exit(1)

    try:
        load_yolo_model() # Ensure YOLO model is loaded
        face_embedder = FaceEmbedder()
        face_registry = FaceRegistry(
            db_path=args.db_path,
            sim_threshold=args.sim_threshold,
            unknown_policy=args.unknown_policy
        )
        if len(face_registry.people) > 0:
            print(f"[OK] {len(face_registry.people)}명 얼굴 로드.")
    except Exception as e:
        print(f"[ERROR] 얼굴 모델/DB 로드 실패: {e}"); sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    win = "Preview (q:종료, m:모드, b:박스, r:녹화)"
    cv2.namedWindow(win)
    cv2.createTrackbar("Strength", win, 18, 60, lambda v: None)
    cv2.createTrackbar("PadX%", win, int(args.pad_x*100), 50, lambda v: None)
    cv2.createTrackbar("PadY%", win, int(args.pad_y*100), 50, lambda v: None)
    cv2.createTrackbar("MinFace", win, args.min_face, 200, lambda v: None)
    cv2.createTrackbar("Neighbors", win, args.min_neighbors, 15, lambda v: None)

    mode_pixelate = True
    draw_box = False
    recording = False
    writer = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            strength = max(1, cv2.getTrackbarPos("Strength", win))
            pad_x = cv2.getTrackbarPos("PadX%", win) / 100.0
            pad_y = cv2.getTrackbarPos("PadY%", win) / 100.0
            min_face = max(1, cv2.getTrackbarPos("MinFace", win))
            neighbors = max(0, cv2.getTrackbarPos("Neighbors", win))

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Not needed for YOLO
            rects = detect_faces_yolo(frame) # Use YOLO detector

            vis = frame.copy()
            for (x,y,w,h) in rects:
                x2,y2,w2,h2 = expand_box(x,y,w,h, pad_x, pad_y, vis.shape[1], vis.shape[0])
                
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0: continue

                embedding = face_embedder.get_embedding(face_img)
                name, _ = face_registry.match(embedding)
                
                policy = face_registry.unknown_policy
                if name:
                    policy = face_registry.get_policy(name)

                apply_mosaic = (policy == 'force')

                if apply_mosaic:
                    if mode_pixelate:
                        pixelate_region(vis, x2,y2,w2,h2, strength=strength)
                    else:
                        gaussian_blur_region(vis, x2,y2,w2,h2, strength=strength)
                
                if draw_box:
                    color = (0, 255, 0) # green
                    if policy == 'force': color = (0,0,255) # red
                    elif policy == 'exclude': color = (255,0,0) # blue
                    
                    cv2.rectangle(vis, (x2,y2), (x2+w2, y2+h2), color, 2)
                    display_name = name if name else "unknown"
                    cv2.putText(vis, display_name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if recording:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_path = args.output if args.output else "preview_record.mp4"
                    ensure_dir(out_path)
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
                if writer is not None:
                    writer.write(vis)

            overlay = vis.copy()
            txt = f"Mode:{'PIXEL' if mode_pixelate else 'GAUSS'} Box:{'ON' if draw_box else 'OFF'} Rec:{'ON' if recording else 'OFF'}"
            cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,220,10), 2, cv2.LINE_AA)
            cv2.imshow(win, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'): mode_pixelate = not mode_pixelate
            elif key == ord('b'): draw_box = not draw_box
            elif key == ord('r'):
                recording = not recording
                if not recording and writer is not None:
                    writer.release(); writer = None; print("[OK] 녹화 저장")
    finally:
        if writer is not None: writer.release()
        cap.release(); cv2.destroyAllWindows()

# ---------- GUI (버튼: 재생/일시정지/저장/종료) ----------
def cmd_gui(args):
    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 동영상 없음: {args.input}"); sys.exit(1)
    try:
        import PySimpleGUI as sg
    except ImportError:
        print("[ERROR] PySimpleGUI 설치 필요: pip install PySimpleGUI"); sys.exit(1)

    mode_pixelate = True
    is_playing = False
    is_saving = False

    control_col = [
        [sg.Text("모자이크 모드")],
        [sg.Button("모드 전환 (픽셀↔블러)", key="-MODE-")],
        [sg.Checkbox("박스 표시", default=False, key="-BOX-")],
        [sg.Text("강도"), sg.Slider(range=(1,60), default_value=18, orientation="h", size=(28,15), key="-STRENGTH-")],
        [sg.Text("PadX%"), sg.Slider(range=(0,50), default_value=int(args.pad_x*100), orientation="h", size=(28,15), key="-PADX-")],
        [sg.Text("PadY%"), sg.Slider(range=(0,50), default_value=int(args.pad_y*100), orientation="h", size=(28,15), key="-PADY-")],
        [sg.Text("MinFace"), sg.Slider(range=(1,200), default_value=args.min_face, orientation="h", size=(28,15), key="-MINFACE-")],
        [sg.Text("Neighbors"), sg.Slider(range=(0,15), default_value=args.min_neighbors, orientation="h", size=(28,15), key="-NEIGH-")],
        [sg.HorizontalSeparator()],
        [sg.Button("재생/일시정지", key="-PLAY-"), sg.Button("저장 시작/중지", key="-SAVE-")],
        [sg.Button("처음으로", key="-REWIND-"), sg.Button("종료", key="-EXIT-")],
        [sg.Text("상태: 정지", key="-STATUS-", size=(34,1))]
    ]
    layout = [[sg.Image(key="-IMAGE-"), sg.Column(control_col, vertical_alignment='top')]]
    window = sg.Window("Face Mosaic GUI", layout, finalize=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sg.popup_error(f"영상을 열 수 없음: {args.input}"); window.close(); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        load_yolo_model() # Ensure YOLO model is loaded
        face_embedder = FaceEmbedder()
        face_registry = FaceRegistry(
            db_path=args.db_path,
            sim_threshold=args.sim_threshold,
            unknown_policy=args.unknown_policy
        )
        if len(face_registry.people) > 0:
            print(f"[OK] {len(face_registry.people)}명 얼굴 로드.")
    except Exception as e:
        sg.popup_error(f"얼굴 모델/DB 로드 실패: {e}"); window.close(); sys.exit(1)

    writer = None
    out_path = args.output if args.output else "gui_output.mp4"

    try:
        while True:
            event, values = window.read(timeout=1)
            if event in (sg.WIN_CLOSED, "-EXIT-"):
                break

            if event == "-MODE-": mode_pixelate = not mode_pixelate
            if event == "-PLAY-":
                is_playing = not is_playing
                window["-STATUS-"].update(f"상태: {'재생' if is_playing else '정지'}")
            if event == "-SAVE-":
                is_saving = not is_saving
                if is_saving and writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    ensure_dir(out_path)
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
                    if not writer.isOpened():
                        sg.popup_error("저장 파일 생성 실패.")
                        is_saving = False; writer = None
                if not is_saving and writer is not None:
                    writer.release(); writer = None
                window["-STATUS-"].update(f"상태: {'저장중' if is_saving else ('재생' if is_playing else '정지')}")

            if event == "-REWIND-":
                cap.release(); cap = cv2.VideoCapture(args.input)
                is_playing = False
                window["-STATUS-"].update("상태: 처음으로 이동 (정지)")

            draw_box = values["-BOX-"]
            strength = int(values["-STRENGTH-"])
            pad_x = values["-PADX-"] / 100.0
            pad_y = values["-PADY-"] / 100.0
            min_face = int(values["-MINFACE-"])
            neighbors = int(values["-NEIGH-"])

            if is_playing:
                ok, frame = cap.read()
                if not ok:
                    is_playing = False
                    window["-STATUS-"].update("상태: 끝 (정지)")
                else:
                    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Not needed for YOLO
                    rects = detect_faces_yolo(frame) # Use YOLO detector

                    vis = frame.copy()
                    for (x,y,w,h) in rects:
                        x2,y2,w2,h2 = expand_box(x,y,w,h, pad_x, pad_y, vis.shape[1], vis.shape[0])
                        
                        face_img = frame[y:y+h, x:x+w]
                        if face_img.size == 0: continue

                        embedding = face_embedder.get_embedding(face_img)
                        name, _ = face_registry.match(embedding)

                        policy = face_registry.unknown_policy
                        if name:
                            policy = face_registry.get_policy(name)

                        apply_mosaic = (policy == 'force')

                        if apply_mosaic:
                            if mode_pixelate:
                                pixelate_region(vis, x2,y2,w2,h2, strength=strength)
                            else:
                                gaussian_blur_region(vis, x2,y2,w2,h2, strength=strength)
                        
                        if draw_box:
                            color = (0, 255, 0) # green
                            if policy == 'force': color = (0,0,255) # red
                            elif policy == 'exclude': color = (255,0,0) # blue
                            
                            cv2.rectangle(vis, (x2,y2), (x2+w2, y2+h2), color, 2)
                            display_name = name if name else "unknown"
                            cv2.putText(vis, display_name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if is_saving and writer is not None:
                        writer.write(vis)

                    imgbytes = cv2.imencode(".png", vis)[1].tobytes()
                    window["-IMAGE-"].update(data=imgbytes)
    finally:
        if writer is not None: writer.release()
        cap.release(); window.close()
