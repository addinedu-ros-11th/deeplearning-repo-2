import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import mysql.connector
import numpy as np
from flask import Flask, Response, redirect, render_template, request, url_for
from insightface.app import FaceAnalysis


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "face_registry"
LOG_DIR = APP_DIR / "data" / "face_logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "face_id",
    "autocommit": True,
}

DB_BOOTSTRAP_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "autocommit": True,
}

MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)
SIM_THRESHOLD = 0.35
LOG_DEDUP_SECONDS = 2
REGISTER_FRAME_COUNT = 5
REGISTER_FRAME_DELAY = 0.2

COLOR_EMPLOYEE = (255, 0, 0)
COLOR_PATIENT = (255, 255, 255)
COLOR_UNKNOWN = (0, 0, 255)

app = Flask(__name__)

_db_lock = threading.Lock()
_camera_lock = threading.Lock()
_camera = None

_gallery_lock = threading.Lock()
_gallery_embeddings = None
_gallery_meta = None
_last_log = {}


class Gallery:
    def __init__(self, embeddings, meta):
        self.embeddings = embeddings
        self.meta = meta


face_app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=DETECTION_SIZE)


def get_db():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    with _db_lock:
        conn = mysql.connector.connect(**DB_BOOTSTRAP_CONFIG)
        cur = conn.cursor()
        cur.execute("CREATE DATABASE IF NOT EXISTS face_id")
        cur.execute("USE face_id")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                serial_number VARCHAR(100) NOT NULL,
                role ENUM('employee', 'patient') NOT NULL,
                registered_at DATETIME NOT NULL,
                image_path VARCHAR(255) NOT NULL,
                embedding LONGBLOB NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                person_id INT NOT NULL,
                name VARCHAR(100) NOT NULL,
                role ENUM('employee', 'patient') NOT NULL,
                similarity FLOAT NOT NULL,
                image_path VARCHAR(255),
                seen_at DATETIME NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_event_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                task VARCHAR(32) NOT NULL,
                label VARCHAR(128),
                score FLOAT,
                source_id VARCHAR(64),
                payload_json TEXT,
                seen_at DATETIME NOT NULL
            )
            """
        )
        try:
            cur.execute("ALTER TABLE recognition_logs ADD COLUMN image_path VARCHAR(255)")
        except mysql.connector.Error:
            pass
        cur.close()
        conn.close()


def get_camera():
    global _camera
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(0)
        return _camera


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def serialize_embedding(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def load_gallery():
    embeddings = []
    meta = []
    with _db_lock:
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM persons")
        for row in cur.fetchall():
            embeddings.append(l2_normalize(deserialize_embedding(row["embedding"])))
            meta.append({
                "id": row["id"],
                "name": row["name"],
                "role": row["role"],
            })
        cur.close()
        conn.close()
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.empty((0, 512), dtype=np.float32)
    return Gallery(embeddings, meta)


def refresh_gallery():
    global _gallery_embeddings, _gallery_meta
    gallery = load_gallery()
    with _gallery_lock:
        _gallery_embeddings = gallery.embeddings
        _gallery_meta = gallery.meta


def get_gallery():
    with _gallery_lock:
        if _gallery_embeddings is None:
            refresh_gallery()
        return _gallery_embeddings, _gallery_meta


def draw_label(frame, bbox, text, color, text_color):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_x1 = x1
    label_y1 = max(0, y1 - th - baseline - 6)
    label_x2 = x1 + tw + 6
    label_y2 = y1

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        frame,
        text,
        (label_x1 + 3, label_y2 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        2,
        cv2.LINE_AA,
    )


def best_match(embedding, embeddings, meta):
    if embeddings.size == 0:
        return None, -1.0
    embedding = l2_normalize(embedding)
    sims = embeddings @ embedding
    idx = int(np.argmax(sims))
    return meta[idx], float(sims[idx])


def maybe_log_recognition(person, similarity, frame, bbox):
    now = datetime.now()
    last_time = _last_log.get(person["id"])
    if last_time and now - last_time < timedelta(seconds=LOG_DEDUP_SECONDS):
        return

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    face_crop = frame[y1:y2, x1:x2]

    image_path = None
    if face_crop.size:
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"log_{person['id']}_{timestamp}.jpg"
        image_path = LOG_DIR / filename
        cv2.imwrite(str(image_path), face_crop)

    with _db_lock:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO recognition_logs (person_id, name, role, similarity, image_path, seen_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                person["id"],
                person["name"],
                person["role"],
                similarity,
                str(image_path) if image_path else None,
                now,
            ),
        )
        cur.close()
        conn.close()

    _last_log[person["id"]] = now


def annotate_frame(frame):
    faces = face_app.get(frame)
    embeddings, meta = get_gallery()

    for face in faces:
        person, similarity = best_match(face.embedding, embeddings, meta)
        if person is None or similarity < SIM_THRESHOLD:
            draw_label(frame, face.bbox, "외부인", COLOR_UNKNOWN, (255, 255, 255))
            continue

        if person["role"] == "employee":
            color = COLOR_EMPLOYEE
            text_color = (255, 255, 255)
            role_label = "직원"
        else:
            color = COLOR_PATIENT
            text_color = (0, 0, 0)
            role_label = "환자"

        label = f"{role_label}: {person['name']}"
        draw_label(frame, face.bbox, label, color, text_color)
        maybe_log_recognition(person, similarity, frame, face.bbox)

    return frame


def generate_frames():
    cam = get_camera()
    while True:
        with _camera_lock:
            ret, frame = cam.read()
        if not ret:
            break
        frame = annotate_frame(frame)
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        serial_number = request.form.get("serial_number", "").strip()
        role = request.form.get("role")

        if not name or not serial_number or role not in {"employee", "patient"}:
            return render_template("register.html", error="모든 필드를 입력하세요.")

        cam = get_camera()
        embeddings = []
        captured_frame = None

        for idx in range(REGISTER_FRAME_COUNT):
            with _camera_lock:
                ret, frame = cam.read()
            if not ret:
                break
            faces = face_app.get(frame)
            if not faces:
                time.sleep(REGISTER_FRAME_DELAY)
                continue
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            embeddings.append(l2_normalize(face.embedding))
            if captured_frame is None:
                captured_frame = frame
            time.sleep(REGISTER_FRAME_DELAY)

        if not embeddings:
            return render_template("register.html", error="얼굴을 찾지 못했습니다.")

        embedding = l2_normalize(np.mean(np.vstack(embeddings), axis=0))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{role}_{name}_{timestamp}.jpg"
        image_path = DATA_DIR / filename
        cv2.imwrite(str(image_path), captured_frame)

        registered_at = datetime.now()
        with _db_lock:
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
        return redirect(url_for("index"))

    return render_template("register.html")


@app.route("/logs")
def logs():
    name = request.args.get("name", "").strip()
    role = request.args.get("role", "")
    start = request.args.get("start", "")
    end = request.args.get("end", "")

    query = "SELECT name, role, similarity, seen_at FROM recognition_logs WHERE 1=1"
    params = []

    if name:
        query += " AND name LIKE %s"
        params.append(f"%{name}%")
    if role in {"employee", "patient"}:
        query += " AND role = %s"
        params.append(role)
    if start:
        query += " AND seen_at >= %s"
        params.append(start)
    if end:
        query += " AND seen_at <= %s"
        params.append(end)

    query += " ORDER BY seen_at DESC LIMIT 200"

    with _db_lock:
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

    return render_template("logs.html", rows=rows)


@app.route("/ai-logs")
def ai_logs():
    label = request.args.get("label", "").strip()
    task = request.args.get("task", "")
    source_id = request.args.get("source_id", "").strip()
    start = request.args.get("start", "")
    end = request.args.get("end", "")

    query = (
        "SELECT task, label, score, source_id, payload_json, seen_at "
        "FROM ai_event_logs WHERE 1=1"
    )
    params = []

    if label:
        query += " AND label LIKE %s"
        params.append(f"%{label}%")
    if task in {"detect", "pose"}:
        query += " AND task = %s"
        params.append(task)
    if source_id:
        query += " AND source_id LIKE %s"
        params.append(f"%{source_id}%")
    if start:
        query += " AND seen_at >= %s"
        params.append(start)
    if end:
        query += " AND seen_at <= %s"
        params.append(end)

    query += " ORDER BY seen_at DESC LIMIT 200"

    with _db_lock:
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

    return render_template("ai_logs.html", rows=rows)


@app.route("/reload")
def reload_gallery():
    refresh_gallery()
    return redirect(url_for("index"))


if __name__ == "__main__":
    init_db()
    refresh_gallery()
    app.run(host="0.0.0.0", port=5000, debug=True)
