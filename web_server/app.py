import os
import smtplib
from email.message import EmailMessage
import requests
from dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_from_directory,
    send_file,
)
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
from requests.exceptions import RequestException  # ★ 누락되었던 부분 추가

load_dotenv()

# ---------------------------
# Flask 앱 / DB 설정
# ---------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
# 이메일/SMTP 설정
app.config["SMTP_HOST"] = os.getenv("SMTP_HOST", "smtp.gmail.com")
app.config["SMTP_PORT"] = int(os.getenv("SMTP_PORT", "587"))
app.config["SMTP_USER"] = os.getenv("SMTP_USER", "")
app.config["SMTP_PASSWORD"] = os.getenv("SMTP_PASSWORD", "")
app.config["SMTP_SENDER"] = os.getenv("SMTP_SENDER", app.config["SMTP_USER"])
app.config["BASE_URL"] = os.getenv("BASE_URL", "http://127.0.0.1:5000")

# MariaDB 연결 설정
app.config["MYSQL_HOST"] = os.getenv("MYSQL_HOST", "127.0.0.1")
app.config["MYSQL_USER"] = os.getenv("MYSQL_USER", "mozik_user")
app.config["MYSQL_PASSWORD"] = os.getenv("MYSQL_PASSWORD", "")
app.config["MYSQL_DB"] = os.getenv("MYSQL_DB", "mozik_db")
# app.config["MYSQL_CURSORCLASS"] = "DictCursor"  # auth.py에서 dict 접근 쓸 거면 주석 해제하면 됨이다.

mysql = MySQL(app)
bcrypt = Bcrypt(app)

# 세션 설정
app.config["SESSION_TYPE"] = "filesystem"

# ---------------------------
# 업로드 폴더 설정
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 모자이크 API 서버 주소
# 같은 Flask 앱의 /api/mosaic 엔드포인트를 호출하는 구조이다.
MOSAIC_API_URL = os.getenv("MOSAIC_API_URL", "http://127.0.0.1:5000/api/mosaic")


# ---------------------------
# 기본 라우트 (홈 = base.html)
# ---------------------------
@app.route("/")
def home():
    return render_template("base.html")


# ---------------------------
# 업로드 / 모자이크 처리 라우트
# ---------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    # 로그인 여부 확인
    if "user_id" not in session:
        return redirect(url_for("login"))

    # GET 요청이면 업로드 폼만 보여준다.
    if request.method == "GET":
        return render_template("upload.html")

    # POST 요청 (폼에서 파일 업로드)
    if "file" not in request.files:
        return "파일이 전송되지 않았음이다.", 400

    file = request.files["file"]

    if file.filename == "":
        return "선택된 파일이 없음이다.", 400

    # 파일 이름 안전하게 변환 후 저장
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(input_path)

    # ---------------------------
    # 모자이크 API 호출
    # ---------------------------
    try:
        with open(input_path, "rb") as f:
            resp = requests.post(
                MOSAIC_API_URL,
                files={"file": f},
                timeout=60,
            )
    except RequestException as e:
        app.logger.exception("모자이크 API 서버 호출 중 예외 발생")
        return f"모자이크 API 서버 호출 중 예외 발생: {e}", 502

    # 정상 응답(200)이 아니면 실패로 처리한다.
    if resp.status_code != 200:
        app.logger.error(
            "모자이크 API 호출 실패: status=%s, body=%s",
            resp.status_code,
            resp.text[:200],
        )
        return (
            f"모자이크 API 호출 실패임이다. "
            f"status={resp.status_code}, body={resp.text}",
            502,
        )

    # 모자이크 API가 이미지 바이너리를 바로 돌려준다고 가정한다.
    out_name = "mosaic_" + filename
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
    with open(output_path, "wb") as out_f:
        out_f.write(resp.content)

    # 작업 기록 DB에 저장
    user_id = session.get("user_id")
    blur_strength = request.form.get("blur_strength", "0")
    
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO job_history (user_id, original_filename, output_filename, blur_strength, status) VALUES (%s, %s, %s, %s, %s)",
            (user_id, filename, out_name, blur_strength, "success"),
        )
        mysql.connection.commit()
        cur.close()
    except Exception as e:
        app.logger.error(f"작업 기록 저장 실패: {e}")

    # 결과 페이지 렌더링 (모자이크된 이미지 보여주기)
    return render_template(
        "result.html",
        image_url=url_for("uploaded_file", filename=out_name),
    )


# ---------------------------
# 내부 모자이크 API (테스트용)
# ---------------------------
@app.route("/api/mosaic", methods=["POST"])
def api_mosaic():
    """
    업로드된 이미지를 받아서 모자이크 처리 후 다시 돌려주는 API이다.
    지금은 테스트용으로 '원본 그대로' 돌려준다.
    나중에 여기 안에 실제 모자이크 처리 코드를 넣으면 된다.
    """
    if "file" not in request.files:
        return "file 필드가 없음이다.", 400

    up_file = request.files["file"]
    if up_file.filename == "":
        return "파일명이 비어 있음이다.", 400

    try:
        # 업로드된 파일을 PIL 이미지로 열기
        img = Image.open(up_file.stream).convert("RGB")

        # TODO: 여기에서 img에 모자이크 처리 로직을 적용하면 됨이다.
        # 예: img = do_mosaic(img)

        # 다시 바이너리로 변환해서 응답
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return f"모자이크 처리 중 오류 발생임이다: {e}", 500


# 업로드된 파일을 직접 서빙하는 라우트
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ---------------------------
# 인증 / 기타 라우트 가져오기
# ---------------------------
def send_reset_email(to_email: str, token: str) -> None:
    """
    비밀번호 재설정 링크를 포함한 이메일을 발송한다.
    SMTP 설정이 비어 있으면 로깅만 수행한다.
    """
    reset_url = f"{app.config['BASE_URL']}/reset-password/{token}"
    subject = "[Mozik] 비밀번호 재설정 안내"
    body = (
        "안녕하세요.\n\n"
        "요청하신 비밀번호 재설정 링크는 아래와 같습니다.\n"
        f"{reset_url}\n\n"
        "본인이 요청하지 않았다면 이 메일을 무시하세요.\n"
        "링크는 1시간 후 만료됩니다.\n"
    )

    if not app.config["SMTP_USER"] or not app.config["SMTP_PASSWORD"]:
        app.logger.warning("SMTP 설정이 비어 있어 이메일을 보내지 못했습니다.")
        return

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = app.config["SMTP_SENDER"]
    message["To"] = to_email
    message.set_content(body)

    try:
        with smtplib.SMTP(app.config["SMTP_HOST"], app.config["SMTP_PORT"]) as smtp:
            smtp.starttls()
            smtp.login(app.config["SMTP_USER"], app.config["SMTP_PASSWORD"])
            smtp.send_message(message)
    except Exception as exc:
        app.logger.exception("비밀번호 재설정 메일 전송 실패: %s", exc)


from auth import *  # auth.py에서 /signup, /login, /logout 등을 정의한다고 가정함이다.


# ---------------------------
# 작업 기록 조회 라우트
# ---------------------------
@app.route("/history")
def history():
    # 로그인 여부 확인
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    user_id = session.get("user_id")
    
    # 사용자의 작업 기록 조회
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT id, original_filename, output_filename, blur_strength, status, created_at FROM job_history WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        jobs = cur.fetchall()
        cur.close()
        
        # 튜플을 딕셔너리 형태로 변환
        job_list = []
        for job in jobs:
            job_list.append({
                "id": job[0],
                "original_filename": job[1],
                "output_filename": job[2],
                "blur_strength": job[3],
                "status": job[4],
                "created_at": job[5],
            })
    except Exception as e:
        app.logger.error(f"작업 기록 조회 실패: {e}")
        job_list = []
    
    return render_template("history.html", jobs=job_list)


# ---------------------------
# 앱 실행 (개발용)
# ---------------------------
if __name__ == "__main__":
    # ⚠ 여기서 app.run 과 MOSAIC_API_URL(127.0.0.1:5000/api/mosaic)을 같이 쓰면
    # 단일 프로세스/단일 스레드일 때 자기 자신을 다시 호출해서 막힐 수 있음이다.
    # 개발할 때는 api_mosaic 를 직접 호출하거나, 포트를 다르게 써도 됨이다.
    app.run(host=os.getenv("FLASK_HOST", "0.0.0.0"), port=int(os.getenv("FLASK_PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
