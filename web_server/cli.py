import sys, argparse
from .pipelines import cmd_scan, cmd_render, cmd_preview, cmd_gui

# ⬇ 분할된 스튜디오 진입점 임포트
from .studio_image import run as image_run
from .studio_video import run as video_run
from .studio_base import run as base_run


def build_parser():
    p = argparse.ArgumentParser(description="Face Mosaic Tool (scan & render & preview & gui)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_opts(sp):
        sp.add_argument("--scale-factor", type=float, default=1.1)
        sp.add_argument("--min-neighbors", type=int, default=5)
        sp.add_argument("--min-face", type=int, default=24)
        sp.add_argument("--pad-x", type=float, default=0.15)
        sp.add_argument("--pad-y", type=float, default=0.20)
        sp.add_argument("--preview", action="store_true")
        sp.add_argument("--preview-stride", type=int, default=15)

    def add_face_recognition_opts(sp):
        sp.add_argument("--db-path", default="face_db", help="얼굴 DB 경로")
        sp.add_argument("--sim-threshold", type=float, default=0.82, help="얼굴 유사도 문턱값 (낮을수록 엄격)")
        sp.add_argument("--unknown-policy", choices=["force", "exclude"], default="force", help="미등록 얼굴 정책")

    s = sub.add_parser("scan", help="얼굴 자동 탐지 후 CSV로 내보내기")
    s.add_argument("-i","--input", required=True)
    s.add_argument("-o","--output-csv", default="faces_auto.csv")
    add_common_opts(s)

    r = sub.add_parser("render", help="CSV(선택) 기반 렌더")
    r.add_argument("-i","--input", required=True)
    r.add_argument("-o","--output", default="mosaic_output.mp4")
    r.add_argument("--csv")
    r.add_argument("--mode", choices=["pixelate","gaussian"], default="pixelate")
    r.add_argument("--strength", type=int, default=18)
    r.add_argument("--draw-box", action="store_true")
    add_common_opts(r)
    add_face_recognition_opts(r)

    pvw = sub.add_parser("preview", help="실시간 프리뷰(키보드/슬라이더, r=저장)")
    pvw.add_argument("-i","--input", required=True)
    pvw.add_argument("-o","--output", default="preview_record.mp4")
    pvw.add_argument("--scale-factor", type=float, default=1.1)
    pvw.add_argument("--min-neighbors", type=int, default=5)
    pvw.add_argument("--min-face", type=int, default=24)
    pvw.add_argument("--pad-x", type=float, default=0.15)
    pvw.add_argument("--pad-y", type=float, default=0.20)
    add_face_recognition_opts(pvw)

    # ⬇ 중복된 sub.add_parser("gui") 제거하고 하나만 둠
    gui = sub.add_parser("gui", help="GUI(재생/일시정지/저장/종료 버튼)")
    gui.add_argument("-i","--input", required=True)
    gui.add_argument("-o","--output", default="gui_output.mp4")
    gui.add_argument("--scale-factor", type=float, default=1.1)
    gui.add_argument("--min-neighbors", type=int, default=5)
    gui.add_argument("--min-face", type=int, default=24)
    gui.add_argument("--pad-x", type=float, default=0.15)
    gui.add_argument("--pad-y", type=float, default=0.20)
    add_face_recognition_opts(gui)

    studio = sub.add_parser("studio", help="Tk 인터랙티브 스튜디오(업로드→구간선택→얼굴선택→수동박스→내보내기)")
    # ⬇ 어떤 스튜디오를 띄울지 선택할 수 있게 옵션 추가
    studio.add_argument("--mode", choices=["image", "video", "base"], default="image")

    return p


# ⬇ 분기용 래퍼 구현
def cmd_studio(mode: str = "image"):
    if mode == "image":
        return image_run()
    elif mode == "video":
        return video_run()
    else:
        return base_run()


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help(); sys.exit(2)
    args = parser.parse_args()
    try:
        if args.cmd == "scan":
            cmd_scan(args)
        elif args.cmd == "render":
            cmd_render(args)
        elif args.cmd == "preview":
            cmd_preview(args)
        elif args.cmd == "gui":
            cmd_gui(args)
        elif args.cmd == "studio":
            # ⬇ studio 모드 전달
            cmd_studio(mode=getattr(args, "mode", "image"))
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단. 안전 종료.")
