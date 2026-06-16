from flask import render_template, request, redirect, session, url_for
from datetime import datetime, timedelta
import secrets
from app import app, mysql, bcrypt, send_reset_email


# -----------------------
# 회원가입
# -----------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    # GET 요청이면 회원가입 폼만 보여준다.
    if request.method == "GET":
        return render_template("signup.html")

    # POST 요청이면 회원가입 처리이다.
    email = request.form.get("email", "").strip()
    raw_password = request.form.get("password", "")

    if not email or not raw_password:
        return render_template(
            "signup.html",
            error="이메일과 비밀번호를 모두 입력해주세요.",
        ), 400

    cur = mysql.connection.cursor()

    # 1) 이미 존재하는 이메일인지 먼저 확인
    cur.execute(
        "SELECT id FROM users WHERE email = %s",
        (email,),
    )
    existing = cur.fetchone()
    if existing:
        cur.close()
        return render_template(
            "signup.html",
            error="이미 가입된 이메일입니다.",
        ), 409

    # 2) 비밀번호 해시 생성
    hashed_pw = bcrypt.generate_password_hash(raw_password).decode("utf-8")

    # 3) DB 저장
    try:
        cur.execute(
            "INSERT INTO users (email, password) VALUES (%s, %s)",
            (email, hashed_pw),
        )
        mysql.connection.commit()
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        return render_template(
            "signup.html",
            error=f"회원가입 중 오류가 발생했습니다: {e}",
        ), 500
    finally:
        cur.close()

    # 회원가입 후 로그인 페이지로 이동
    return redirect(url_for("login"))


# -----------------------
# 로그인
# -----------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    # GET 요청이면 로그인 화면 렌더링
    if request.method == "GET":
        return render_template("login.html")

    # POST 요청이면 로그인 처리
    email = request.form.get("email", "").strip()
    raw_password = request.form.get("password", "")

    if not email or not raw_password:
        return render_template(
            "login.html",
            error="이메일과 비밀번호를 모두 입력해주세요.",
        ), 400

    # DB에서 사용자 조회
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT id, password FROM users WHERE email = %s",
        (email,),
    )
    user = cur.fetchone()
    cur.close()

    # user: (id, hashed_password)
    if not user:
        return render_template(
            "login.html",
            error="이메일 또는 비밀번호가 올바르지 않습니다.",
        ), 401

    user_id = user[0]
    hashed_pw = user[1]

    # 비밀번호 검증
    if not bcrypt.check_password_hash(hashed_pw, raw_password):
        return render_template(
            "login.html",
            error="이메일 또는 비밀번호가 올바르지 않습니다.",
        ), 401

    # 로그인 성공 → 세션에 사용자 ID 저장
    session["user_id"] = user_id
    # 로그인 후 업로드 페이지로 바로 이동
    return redirect(url_for("upload"))


# -----------------------
# 로그아웃
# -----------------------
@app.route("/logout")
def logout():
    session.clear()
    # 로그아웃 후 홈으로 이동 (또는 login 으로 바꿔도 됨이다.)
    return redirect(url_for("home"))


# -----------------------
# 비밀번호 찾기 / 재설정
# -----------------------
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html")

    email = request.form.get("email", "").strip()
    if not email:
        return render_template(
            "forgot_password.html",
            error="이메일을 입력해주세요.",
        )

    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
    user = cur.fetchone()

    if not user:
        cur.close()
        return render_template(
            "forgot_password.html",
            success=True,
        )

    user_id = user[0]
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)

    try:
        cur.execute(
            """
            INSERT INTO password_resets (user_id, token, expires_at, used)
            VALUES (%s, %s, %s, 0)
            """,
            (user_id, token, expires_at),
        )
        mysql.connection.commit()
    except Exception as exc:
        mysql.connection.rollback()
        cur.close()
        app.logger.exception("비밀번호 재설정 토큰 저장 실패: %s", exc)
        return render_template(
            "forgot_password.html",
            error="요청을 처리하는 중 오류가 발생했습니다.",
        )
    finally:
        cur.close()

    send_reset_email(email, token)
    return render_template("forgot_password.html", success=True)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    cur = mysql.connection.cursor()
    cur.execute(
        """
        SELECT pr.id, pr.expires_at, pr.used, u.id, u.email
        FROM password_resets pr
        JOIN users u ON pr.user_id = u.id
        WHERE pr.token = %s
        ORDER BY pr.id DESC
        LIMIT 1
        """,
        (token,),
    )
    token_row = cur.fetchone()

    if not token_row:
        cur.close()
        return render_template(
            "reset_password.html",
            invalid=True,
        )

    reset_id, expires_at, used_flag, user_id, email = token_row

    if used_flag or expires_at < datetime.utcnow():
        cur.close()
        return render_template(
            "reset_password.html",
            expired=True,
        )

    if request.method == "GET":
        cur.close()
        return render_template("reset_password.html", token=token, email=email)

    new_password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not new_password or new_password != confirm_password:
        cur.close()
        return render_template(
            "reset_password.html",
            token=token,
            email=email,
            error="비밀번호가 비어 있거나 일치하지 않습니다.",
        )

    hashed_pw = bcrypt.generate_password_hash(new_password).decode("utf-8")

    try:
        cur.execute(
            "UPDATE users SET password = %s WHERE id = %s",
            (hashed_pw, user_id),
        )
        cur.execute(
            "UPDATE password_resets SET used = 1 WHERE id = %s",
            (reset_id,),
        )
        mysql.connection.commit()
    except Exception as exc:
        mysql.connection.rollback()
        cur.close()
        app.logger.exception("비밀번호 재설정 실패: %s", exc)
        return render_template(
            "reset_password.html",
            token=token,
            email=email,
            error="비밀번호를 변경하는 중 문제가 발생했습니다.",
        )
    finally:
        cur.close()

    return render_template("reset_password.html", completed=True)
