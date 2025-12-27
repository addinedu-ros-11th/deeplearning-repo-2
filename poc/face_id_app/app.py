import json
import os
import socket
import struct
import threading
import time
import subprocess
import atexit
import signal
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import cv2
import mysql.connector
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, url_for
from insightface.app import FaceAnalysis


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "face_registry"
LOG_DIR = APP_DIR / "data" / "face_logs"
CLIP_DIR = APP_DIR / "data" / "event_clips"
CAPTURE_DIR = APP_DIR / "data" / "captures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

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

EVENT_UDP_BIND = "0.0.0.0"
EVENT_UDP_PORT = 6001
FALL_LABEL_KEYWORDS = {"fall", "fallen", "lying"}
CLIP_PRE_SECONDS = 4
CLIP_POST_SECONDS = 11
CLIP_COOLDOWN_SECONDS = 8
UNKNOWN_LOG_DEDUP_SECONDS = 5
UNKNOWN_MIN_SECONDS = float(os.getenv("UNKNOWN_MIN_SECONDS", "2.0"))
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
UDP_VIDEO_TARGETS = os.getenv("UDP_VIDEO_TARGETS", "")
UDP_JPEG_QUALITY = int(os.getenv("UDP_JPEG_QUALITY", "80"))
UDP_MAX_DATAGRAM = int(os.getenv("UDP_MAX_DATAGRAM", "1400"))
UDP_FPS = float(os.getenv("UDP_FPS", "0"))

UDP_HEADER_FORMAT = "!IHH"
UDP_HEADER_SIZE = struct.calcsize(UDP_HEADER_FORMAT)

COLOR_EMPLOYEE = (255, 0, 0)
COLOR_PATIENT = (255, 255, 255)
COLOR_UNKNOWN = (0, 0, 255)

LABEL_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
]
_label_font_cache = {}

app = Flask(__name__)

_db_lock = threading.Lock()
_camera_lock = threading.Lock()
_camera = None
_camera_source = None

_gallery_lock = threading.Lock()
_gallery_embeddings = None
_gallery_meta = None
_last_log = {}
_last_unknown_log = None
_unknown_since = None
_clip_recorder = None
_udp_sock = None
_udp_targets = None
_udp_frame_id = 0
_udp_max_payload = None
_udp_frame_interval = 0.0
_udp_next_frame_time = 0.0
_latest_jpeg = None
_latest_raw_jpeg = None
_latest_lock = threading.Lock()
_shutdown_event = threading.Event()


class Gallery:
    def __init__(self, embeddings, meta):
        self.embeddings = embeddings
        self.meta = meta


class ClipRecorder:
    def __init__(self, pre_seconds, post_seconds):
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.buffer = deque()
        self.lock = threading.Lock()
        self.active = None
        self.last_trigger = None

    def add_frame(self, frame):
        now = time.time()
        with self.lock:
            self.buffer.append((now, frame.copy()))
            self._trim_buffer(now)
            if self.active:
                self._write_frame(now, frame)
                if now >= self.active["end_time"]:
                    self._finalize_clip()

    def trigger(self, row_id, label, source_id):
        now = time.time()
        with self.lock:
            if self.active:
                return
            if self.last_trigger and now - self.last_trigger < CLIP_COOLDOWN_SECONDS:
                return
            self.last_trigger = now

            filename = self._build_filename(label, source_id, now)
            path = CLIP_DIR / filename
            self.active = {
                "row_id": row_id,
                "path": path,
                "writer": None,
                "end_time": now + self.post_seconds,
            }

            pre_frames = [f for ts, f in self.buffer if ts >= now - self.pre_seconds]
            for pre_frame in pre_frames:
                self._write_frame(now, pre_frame)

    def _build_filename(self, label, source_id, timestamp):
        safe_label = (label or "event").replace(" ", "_")
        safe_source = (source_id or "cam").replace(" ", "_")
        dt = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        return f"{safe_source}_{safe_label}_{dt}.mp4"

    def _trim_buffer(self, now):
        cutoff = now - (self.pre_seconds + 1)
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def _write_frame(self, _now, frame):
        if not self.active:
            return
        writer = self.active["writer"]
        if writer is None:
            height, width = frame.shape[:2]
            path = str(self.active["path"])
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"avc1"),
                10,
                (width, height),
            )
            if not writer.isOpened():
                writer = cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (width, height),
                )
            if not writer.isOpened():
                print("Failed to open VideoWriter for clip:", path)
                self.active = None
                return
            self.active["writer"] = writer
        writer.write(frame)

    def _finalize_clip(self):
        writer = self.active.get("writer")
        if writer:
            writer.release()
        row_id = self.active.get("row_id")
        path = self.active.get("path")
        self.active = None
        if row_id and path and path.exists() and path.stat().st_size > 0:
            self._update_event_path(row_id, path)
            threading.Thread(target=self._transcode_clip, args=(path,), daemon=True).start()

    def _update_event_path(self, row_id, path):
        with _db_lock:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "UPDATE ai_event_logs SET video_path = %s WHERE id = %s",
                (path.name, row_id),
            )
            cur.close()
            conn.close()

    def _transcode_clip(self, path):
        temp_path = path.with_name(f"{path.stem}_h264{path.suffix}")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0 or not temp_path.exists():
            return False
        os.replace(temp_path, path)
        return True


def parse_udp_targets(value):
    targets = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        host, port_str = item.rsplit(":", 1)
        targets.append((host, int(port_str)))
    return targets


def init_udp_sender():
    global _udp_sock, _udp_targets, _udp_max_payload, _udp_frame_interval, _udp_next_frame_time
    if not UDP_VIDEO_TARGETS:
        return
    _udp_targets = parse_udp_targets(UDP_VIDEO_TARGETS)
    if not _udp_targets:
        return
    _udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    _udp_max_payload = max(1, UDP_MAX_DATAGRAM - UDP_HEADER_SIZE)
    _udp_frame_interval = 1.0 / UDP_FPS if UDP_FPS > 0 else 0.0
    _udp_next_frame_time = time.monotonic()


def udp_send_frame(jpeg_bytes):
    global _udp_frame_id, _udp_next_frame_time
    if _udp_sock is None:
        return

    if _udp_frame_interval:
        now = time.monotonic()
        if now < _udp_next_frame_time:
            return
        _udp_next_frame_time = now + _udp_frame_interval

    total_chunks = (len(jpeg_bytes) + _udp_max_payload - 1) // _udp_max_payload
    for chunk_id in range(total_chunks):
        start = chunk_id * _udp_max_payload
        end = start + _udp_max_payload
        header = struct.pack(UDP_HEADER_FORMAT, _udp_frame_id, chunk_id, total_chunks)
        packet = header + jpeg_bytes[start:end]
        for target in _udp_targets:
            _udp_sock.sendto(packet, target)

    _udp_frame_id = (_udp_frame_id + 1) & 0xFFFFFFFF


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
                video_path VARCHAR(255),
                seen_at DATETIME NOT NULL
            )
            """
        )
        try:
            cur.execute("ALTER TABLE recognition_logs ADD COLUMN image_path VARCHAR(255)")
        except mysql.connector.Error:
            pass
        try:
            cur.execute("ALTER TABLE ai_event_logs ADD COLUMN video_path VARCHAR(255)")
        except mysql.connector.Error:
            pass
        cur.close()
        conn.close()


def get_camera():
    global _camera
    global _camera_source
    with _camera_lock:
        if _camera_source is None:
            _camera_source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(_camera_source)
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

    if text.isascii():
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
        return

    font_size = 18
    font = _label_font_cache.get(font_size)
    if font is None:
        loaded = None
        for path in LABEL_FONT_CANDIDATES:
            if Path(path).exists():
                try:
                    loaded = ImageFont.truetype(path, font_size)
                    break
                except OSError:
                    continue
        if loaded is None:
            loaded = ImageFont.load_default()
        _label_font_cache[font_size] = loaded
        font = loaded

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    text_box = draw.textbbox((0, 0), text, font=font)
    tw = text_box[2] - text_box[0]
    th = text_box[3] - text_box[1]
    label_x1 = x1
    label_y1 = max(0, y1 - th - 8)
    label_x2 = x1 + tw + 8
    label_y2 = y1

    bg_color = (color[2], color[1], color[0])
    fg_color = (text_color[2], text_color[1], text_color[0])
    draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=bg_color)
    draw.text((label_x1 + 4, label_y1 + 2), text, font=font, fill=fg_color)
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


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


def log_unknown_event(similarity, frame, bbox):
    global _last_unknown_log
    if _clip_recorder and _clip_recorder.active:
        return None
    now = datetime.now()
    if _last_unknown_log and now - _last_unknown_log < timedelta(seconds=UNKNOWN_LOG_DEDUP_SECONDS):
        return None

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    face_crop = frame[y1:y2, x1:x2]

    image_path = None
    if face_crop.size:
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"unknown_{timestamp}.jpg"
        image_path = LOG_DIR / filename
        cv2.imwrite(str(image_path), face_crop)

    payload = {
        "label": "unknown",
        "score": float(similarity),
        "bbox": [float(v) for v in bbox],
        "image_path": str(image_path) if image_path else None,
    }

    row_id = None
    with _db_lock:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ai_event_logs (task, label, score, source_id, payload_json, seen_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                "intruder",
                "unknown",
                similarity,
                "face_id",
                json.dumps(payload, ensure_ascii=True),
                now,
            ),
        )
        row_id = cur.lastrowid
        cur.close()
        conn.close()

    _last_unknown_log = now
    return row_id


def annotate_frame(frame):
    global _unknown_since
    faces = face_app.get(frame)
    embeddings, meta = get_gallery()
    now_mono = time.monotonic()
    unknown_seen = False
    unknown_logged = False

    for face in faces:
        person, similarity = best_match(face.embedding, embeddings, meta)
        if person is None or similarity < SIM_THRESHOLD:
            unknown_seen = True
            if _unknown_since is None:
                _unknown_since = now_mono
            draw_label(frame, face.bbox, "외부인", COLOR_UNKNOWN, (255, 255, 255))
            if not unknown_logged and _unknown_since is not None:
                if now_mono - _unknown_since >= UNKNOWN_MIN_SECONDS:
                    row_id = log_unknown_event(similarity, frame, face.bbox)
                    if row_id and _clip_recorder:
                        _clip_recorder.trigger(row_id, "unknown", "face_id")
                    unknown_logged = True
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

    if not unknown_seen:
        _unknown_since = None

    return frame


def camera_loop():
    global _camera
    global _latest_jpeg
    global _latest_raw_jpeg
    cam = get_camera()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    while not _shutdown_event.is_set():
        with _camera_lock:
            ret, frame = cam.read()
        if not ret or frame is None:
            with _camera_lock:
                if cam is _camera:
                    cam.release()
                    _camera = None
                cam = get_camera()
            time.sleep(0.2)
            continue
        ok, raw_buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with _latest_lock:
                _latest_raw_jpeg = raw_buffer.tobytes()
        try:
            frame = annotate_frame(frame)
        except Exception as exc:
            print(f"[camera] annotate_frame failed: {exc}")
        if _clip_recorder:
            _clip_recorder.add_frame(frame)
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue
        jpeg_bytes = buffer.tobytes()
        udp_send_frame(jpeg_bytes)
        with _latest_lock:
            _latest_jpeg = jpeg_bytes


def generate_frames():
    while True:
        with _latest_lock:
            jpeg_bytes = _latest_jpeg
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


def generate_raw_frames():
    while True:
        with _latest_lock:
            jpeg_bytes = _latest_raw_jpeg
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_raw")
def video_feed_raw():
    return Response(generate_raw_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/register", methods=["GET", "POST"])
def register():
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


@app.route("/capture_frame", methods=["POST"])
def capture_frame():
    with _latest_lock:
        jpeg_bytes = _latest_jpeg
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
    return jsonify({"filename": filename, "url": url_for("serve_capture", filename=filename)})


@app.route("/captures/<path:filename>")
def serve_capture(filename):
    path = CAPTURE_DIR / filename
    if not path.exists():
        return "Capture not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/ai-logs")
def ai_logs():
    label = request.args.get("label", "").strip()
    task = request.args.get("task", "")
    source_id = request.args.get("source_id", "").strip()
    start = request.args.get("start", "")
    end = request.args.get("end", "")

    query = (
        "SELECT task, label, score, source_id, payload_json, seen_at, video_path "
        "FROM ai_event_logs WHERE 1=1"
    )
    params = []

    if label:
        query += " AND label LIKE %s"
        params.append(f"%{label}%")
    if task in {"detect", "pose", "intruder"}:
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

    for row in rows:
        video_path = row.get("video_path")
        if video_path and not (CLIP_DIR / video_path).exists():
            row["video_path"] = None

    return render_template("ai_logs.html", rows=rows)


@app.route("/reload")
def reload_gallery():
    refresh_gallery()
    return redirect(url_for("index"))


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    path = CLIP_DIR / filename
    if not path.exists():
        return "Clip not found", 404
    return send_file(path, mimetype="video/mp4")


def is_fall_event(label):
    if not label:
        return False
    lower = label.lower()
    return any(keyword in lower for keyword in FALL_LABEL_KEYWORDS)


def event_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((EVENT_UDP_BIND, EVENT_UDP_PORT))
    sock.settimeout(0.5)
    try:
        while not _shutdown_event.is_set():
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            task = payload.get("task", "")
            source_id = payload.get("source_id") or None
            events = payload.get("events") or []
            if not events:
                continue

            now = datetime.now()
            with _db_lock:
                conn = get_db()
                cur = conn.cursor()
                for event in events:
                    label = event.get("label")
                    score = event.get("score")
                    cur.execute(
                        """
                        INSERT INTO ai_event_logs (task, label, score, source_id, payload_json, seen_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            task,
                            label,
                            score,
                            source_id,
                            json.dumps(event, ensure_ascii=True),
                            now,
                        ),
                    )
                    row_id = cur.lastrowid
                    if is_fall_event(label) and _clip_recorder:
                        _clip_recorder.trigger(row_id, label, source_id)
                cur.close()
                conn.close()
    finally:
        sock.close()


if __name__ == "__main__":
    _clip_recorder = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    listener = threading.Thread(target=event_listener, daemon=True)
    listener.start()
    init_udp_sender()
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    init_db()
    refresh_gallery()
    def _shutdown_cleanup():
        _shutdown_event.set()
        with _camera_lock:
            if _camera is not None:
                _camera.release()
        if _udp_sock is not None:
            _udp_sock.close()
        os._exit(0)
    atexit.register(_shutdown_cleanup)
    signal.signal(signal.SIGINT, lambda *_: _shutdown_cleanup())
    signal.signal(signal.SIGTERM, lambda *_: _shutdown_cleanup())
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
