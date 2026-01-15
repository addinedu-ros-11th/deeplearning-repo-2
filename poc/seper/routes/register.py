"""등록 라우트"""
from datetime import datetime
import numpy as np
import cv2
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, send_file

from config import DATA_DIR, CAPTURE_DIR
from services.face_service import (
    get_face_app, l2_normalize, serialize_embedding, refresh_gallery
)
from services.video_service import get_latest_jpeg
from utils.database import get_db, get_db_lock

register_bp = Blueprint('register', __name__)


@register_bp.route("/register", methods=["GET", "POST"])
def register():
    """얼굴 등록"""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        serial_number = request.form.get("serial_number", "").strip()
        role = request.form.get("role")
        capture_filename = request.form.get("capture_filename")
        upload_file = request.files.get("image_file")

        if not name or not serial_number or role not in {"employee", "patient"}:
            return render_template("register.html", error="모든 필드를 입력하세요.")

        frame = None
        if upload_file and upload_file.filename:
            data = upload_file.read()
            if data:
                image = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        elif capture_filename:
            capture_path = CAPTURE_DIR / capture_filename
            if capture_path.exists():
                frame = cv2.imread(str(capture_path))
        else:
            return render_template("register.html", error="사진을 찍거나 파일을 업로드하세요.")

        if frame is None:
            return render_template("register.html", error="이미지를 불러오지 못했습니다.")

        face_app = get_face_app()
        faces = face_app.get(frame)
        if not faces:
            return render_template("register.html", error="얼굴을 찾지 못했습니다.")

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        embedding = l2_normalize(face.embedding)
        captured_frame = frame

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{role}_{name}_{timestamp}.jpg"
        image_path = DATA_DIR / filename
        cv2.imwrite(str(image_path), captured_frame)

        registered_at = datetime.now()
        db_lock = get_db_lock()
        with db_lock:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO persons (name, serial_number, role, registered_at, image_path, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    name,
                    serial_number,
                    role,
                    registered_at,
                    str(image_path),
                    serialize_embedding(embedding),
                ),
            )
            cur.close()
            conn.close()

        refresh_gallery()
        return redirect(url_for("index.index"))

    return render_template("register.html")


@register_bp.route("/capture_frame", methods=["POST"])
def capture_frame():
    """프레임 캡처"""
    jpeg_bytes = get_latest_jpeg()
    if not jpeg_bytes:
        return jsonify({"error": "no_frame"}), 400
    
    image = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "decode_failed"}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"capture_{timestamp}.jpg"
    path = CAPTURE_DIR / filename
    cv2.imwrite(str(path), frame)
    
    return jsonify({"filename": filename, "url": url_for("register.serve_capture", filename=filename)})


@register_bp.route("/captures/<path:filename>")
def serve_capture(filename):
    """캡처 이미지 서빙"""
    path = CAPTURE_DIR / filename
    if not path.exists():
        return "Capture not found", 404
    return send_file(path, mimetype="image/jpeg")
