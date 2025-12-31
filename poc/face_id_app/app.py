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
REPO_ROOT = APP_DIR.parents[1]
DATA_DIR = APP_DIR / "data" / "face_registry"
LOG_DIR = APP_DIR / "data" / "face_logs"
CLIP_DIR = APP_DIR / "data" / "event_clips"
CAPTURE_DIR = APP_DIR / "data" / "captures"
TEST_VIDEO_DIR = APP_DIR / "data" / "test_videos"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
TEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

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
FALL_LABEL_KEYWORDS = {"fall", "fallen", "lying", "laying"}
CLIP_PRE_SECONDS = 4
CLIP_POST_SECONDS = 11
CLIP_COOLDOWN_SECONDS = 8
UNKNOWN_LOG_DEDUP_SECONDS = 5
FALL_LOG_DEDUP_SECONDS = 5
FIRE_SMOKE_LOG_DEDUP_SECONDS = 5
UNKNOWN_MIN_SECONDS = float(os.getenv("UNKNOWN_MIN_SECONDS", "2.0"))
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "auto")
CAMERA_SOURCE_2 = os.getenv("CAMERA_SOURCE_2", "auto")
CAMERA_BY_ID_MATCH = os.getenv("CAMERA_BY_ID_MATCH", "").strip()
CAMERA2_BY_ID_MATCH = os.getenv("CAMERA2_BY_ID_MATCH", "").strip()
FALL_MODEL_PATH = os.getenv(
    "FALL_MODEL_PATH",
    str(REPO_ROOT / "runs" / "train" / "cctv_fall_laying_pose_v8n" / "weights" / "best.pt"),
)
FALL_CONF = float(os.getenv("FALL_CONF", "0.5"))
FALL_FPS = float(os.getenv("FALL_FPS", "5.0"))
FIRE_SMOKE_MODEL_PATH = os.getenv(
    "FIRE_SMOKE_MODEL_PATH",
    str(REPO_ROOT / "runs" / "train" / "fire_smoke_detect_v8s" / "weights" / "best.pt"),
)
FIRE_SMOKE_CONF = float(os.getenv("FIRE_SMOKE_CONF", "0.5"))
FIRE_SMOKE_FPS = float(os.getenv("FIRE_SMOKE_FPS", "5.0"))
UDP_VIDEO_TARGETS = os.getenv("UDP_VIDEO_TARGETS", "")
UDP_JPEG_QUALITY = int(os.getenv("UDP_JPEG_QUALITY", "80"))
UDP_MAX_DATAGRAM = int(os.getenv("UDP_MAX_DATAGRAM", "1400"))
UDP_FPS = float(os.getenv("UDP_FPS", "0"))

UDP_HEADER_FORMAT = "!IHH"
UDP_HEADER_SIZE = struct.calcsize(UDP_HEADER_FORMAT)

CCTV1_FACE_SOURCE_ID = "cctv1_face"
CCTV1_FALL_SOURCE_ID = "cctv1_fall"
CCTV1_FIRE_SOURCE_ID = "cctv1_fire"
CCTV2_FACE_SOURCE_ID = "cctv2_face"
CCTV2_FALL_SOURCE_ID = "cctv2_fall"
CCTV2_FIRE_SOURCE_ID = "cctv2_fire"

CCTV_SOURCE_MAP = {
    "cctv1": [
        CCTV1_FACE_SOURCE_ID,
        CCTV1_FALL_SOURCE_ID,
        CCTV1_FIRE_SOURCE_ID,
        "face_id",
        "fall_cam",
        "fire_smoke_cam",
        None,
    ],
    "cctv2": [
        CCTV2_FACE_SOURCE_ID,
        CCTV2_FALL_SOURCE_ID,
        CCTV2_FIRE_SOURCE_ID,
    ],
}

COLOR_EMPLOYEE = (255, 0, 0)
COLOR_PATIENT = (255, 255, 255)
COLOR_UNKNOWN = (0, 0, 255)

LABEL_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
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
_camera2_lock = threading.Lock()
_camera2 = None
_camera2_source = None

_gallery_lock = threading.Lock()
_gallery_embeddings = None
_gallery_meta = None
_last_log = {}
_last_unknown_log = {}
_last_fall_log = {}
_last_fire_smoke_log = {}
_unknown_since = {}
_clip_recorder = None
_fall_clip_recorder = None
_fire_smoke_clip_recorder = None
_clip_recorder_2 = None
_fall_clip_recorder_2 = None
_fire_smoke_clip_recorder_2 = None
_udp_sock = None
_udp_targets = None
_udp_frame_id = 0
_udp_max_payload = None
_udp_frame_interval = 0.0
_udp_next_frame_time = 0.0
_latest_jpeg = None
_latest_raw_jpeg = None
_latest_raw_jpeg_2 = None
_latest_fall_jpeg = None
_latest_fire_smoke_jpeg = None
_latest_fall_jpeg_2 = None
_latest_fire_smoke_jpeg_2 = None
_latest_lock = threading.Lock()
_last_frame_time = None
_last_frame_time_2 = None
_shutdown_event = threading.Event()
_fall_model = None
_fall_model_error = None
_fall_lock = threading.Lock()
_fire_smoke_model = None
_fire_smoke_model_error = None
_fire_smoke_lock = threading.Lock()
_test_thread = None
_test_stop_event = threading.Event()
_test_latest_jpeg = None
_test_latest_lock = threading.Lock()
_test_status = {"state": "idle", "message": ""}
_test_lock = threading.Lock()
_test_source_url = None
_test_model_path = None
_test_pause = False
_test_seek_seconds = None
_test_duration_seconds = None
_test_position_seconds = 0.0
_test_control_lock = threading.Lock()


TEST_FPS = float(os.getenv("TEST_FPS", "5.0"))
TEST_CONF = float(os.getenv("TEST_CONF", str(FALL_CONF)))
TEST_COOKIE_FILE = TEST_VIDEO_DIR / "cookies.txt"


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
            cur.execute("ALTER TABLE ai_event_logs ADD COLUMN video_path VARCHAR(255)")
        except mysql.connector.Error:
            pass
        cur.close()
        conn.close()


def _video_index(path: Path) -> int:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return int(digits) if digits else 9999


def _list_by_id_nodes():
    by_id_dir = Path("/dev/v4l/by-id")
    if not by_id_dir.exists():
        return []
    nodes = []
    for entry in by_id_dir.iterdir():
        if "video-index0" not in entry.name:
            continue
        nodes.append(entry)
    return sorted(nodes, key=lambda p: p.name)


def _list_video_nodes():
    nodes = [p for p in Path("/dev").glob("video*") if p.name[5:].isdigit()]
    return sorted(nodes, key=_video_index)


def _resolve_by_id_match(match_value):
    if not match_value:
        return None
    match_lower = match_value.lower()
    for entry in _list_by_id_nodes():
        if match_lower in entry.name.lower():
            return str(entry)
    return None


def _resolve_auto_source(fallback_index, exclude_sources):
    exclude_set = {str(source) for source in (exclude_sources or [])}
    by_id_nodes = _list_by_id_nodes()
    by_id_paths = [str(p) for p in by_id_nodes if str(p) not in exclude_set]
    if by_id_paths:
        return by_id_paths[min(fallback_index, len(by_id_paths) - 1)]
    video_nodes = _list_video_nodes()
    video_paths = [str(p) for p in video_nodes if str(p) not in exclude_set]
    if video_paths:
        return video_paths[min(fallback_index, len(video_paths) - 1)]
    return None


def _normalize_camera_source(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if value.lower() == "auto":
        return "auto"
    if value.isdigit():
        return int(value)
    return value


def _expand_exclude_sources(sources):
    expanded = []
    for source in sources:
        if source is None:
            continue
        expanded.append(str(source))
        if isinstance(source, int):
            expanded.append(f"/dev/video{source}")
    return expanded


def resolve_camera_source(primary_value, by_id_match, fallback_index, exclude_sources=None):
    normalized = _normalize_camera_source(primary_value)
    if normalized and normalized != "auto":
        return normalized
    by_id_path = _resolve_by_id_match(by_id_match)
    if by_id_path:
        return by_id_path
    auto_path = _resolve_auto_source(fallback_index, exclude_sources or [])
    if auto_path is not None:
        return auto_path
    return fallback_index if normalized == "auto" else normalized


def get_camera():
    global _camera
    global _camera_source
    with _camera_lock:
        if _camera_source is None:
            _camera_source = resolve_camera_source(CAMERA_SOURCE, CAMERA_BY_ID_MATCH, 0, [])
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(_camera_source)
        return _camera


def get_camera_secondary():
    global _camera2
    global _camera2_source
    with _camera2_lock:
        if _camera2_source is None:
            exclude = _expand_exclude_sources([_camera_source])
            _camera2_source = resolve_camera_source(
                CAMERA_SOURCE_2,
                CAMERA2_BY_ID_MATCH,
                1,
                exclude,
            )
        if _camera2 is None or not _camera2.isOpened():
            _camera2 = cv2.VideoCapture(_camera2_source)
        return _camera2


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


def log_unknown_event(similarity, frame, bbox, source_id, clip_recorder):
    global _last_unknown_log
    if clip_recorder and clip_recorder.active:
        return None
    now = datetime.now()
    source_key = source_id or "unknown"
    last_seen = _last_unknown_log.get(source_key)
    if last_seen and now - last_seen < timedelta(seconds=UNKNOWN_LOG_DEDUP_SECONDS):
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
                source_id,
                json.dumps(payload, ensure_ascii=True),
                now,
            ),
        )
        row_id = cur.lastrowid
        cur.close()
        conn.close()

    _last_unknown_log[source_key] = now
    return row_id


def log_fall_event(label, score, boxes, source_id):
    global _last_fall_log
    now = datetime.now()
    source_key = source_id or "unknown"
    last_seen = _last_fall_log.get(source_key)
    if last_seen and now - last_seen < timedelta(seconds=FALL_LOG_DEDUP_SECONDS):
        return None
    payload = {
        "label": label,
        "score": float(score),
        "boxes": boxes,
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
                "pose",
                label,
                float(score),
                source_id,
                json.dumps(payload, ensure_ascii=True),
                now,
            ),
        )
        row_id = cur.lastrowid
        cur.close()
        conn.close()
    _last_fall_log[source_key] = now
    return row_id


def log_fire_smoke_event(label, score, boxes, source_id):
    global _last_fire_smoke_log
    now = datetime.now()
    source_key = source_id or "unknown"
    last_seen = _last_fire_smoke_log.get(source_key)
    if last_seen and now - last_seen < timedelta(seconds=FIRE_SMOKE_LOG_DEDUP_SECONDS):
        return None
    payload = {
        "label": label,
        "score": float(score),
        "boxes": boxes,
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
                "detect",
                label,
                float(score),
                source_id,
                json.dumps(payload, ensure_ascii=True),
                now,
            ),
        )
        row_id = cur.lastrowid
        cur.close()
        conn.close()
    _last_fire_smoke_log[source_key] = now
    return row_id


def annotate_frame(frame, source_id, clip_recorder):
    global _unknown_since
    faces = face_app.get(frame)
    embeddings, meta = get_gallery()
    now_mono = time.monotonic()
    unknown_seen = False
    unknown_logged = False
    source_key = source_id or "unknown"

    for face in faces:
        person, similarity = best_match(face.embedding, embeddings, meta)
        if person is None or similarity < SIM_THRESHOLD:
            unknown_seen = True
            if _unknown_since.get(source_key) is None:
                _unknown_since[source_key] = now_mono
            draw_label(frame, face.bbox, "외부인", COLOR_UNKNOWN, (255, 255, 255))
            unknown_since_value = _unknown_since.get(source_key)
            if not unknown_logged and unknown_since_value is not None:
                if now_mono - unknown_since_value >= UNKNOWN_MIN_SECONDS:
                    row_id = log_unknown_event(similarity, frame, face.bbox, source_id, clip_recorder)
                    if row_id and clip_recorder:
                        clip_recorder.trigger(row_id, "unknown", source_id)
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
        _unknown_since[source_key] = None

    return frame


def get_fall_model():
    global _fall_model, _fall_model_error
    with _fall_lock:
        if _fall_model is not None or _fall_model_error:
            return _fall_model
        try:
            from ultralytics import YOLO
        except Exception as exc:
            _fall_model_error = exc
            print(f"[fall] ultralytics import failed: {exc}")
            return None
        model_path = Path(FALL_MODEL_PATH)
        if not model_path.exists():
            _fall_model_error = f"missing model: {model_path}"
            print(f"[fall] model not found: {model_path}")
            return None
        _fall_model = YOLO(str(model_path))
        return _fall_model


def get_fire_smoke_model():
    global _fire_smoke_model, _fire_smoke_model_error
    with _fire_smoke_lock:
        if _fire_smoke_model is not None or _fire_smoke_model_error:
            return _fire_smoke_model
        try:
            from ultralytics import YOLO
        except Exception as exc:
            _fire_smoke_model_error = exc
            print(f"[fire_smoke] ultralytics import failed: {exc}")
            return None
        model_path = Path(FIRE_SMOKE_MODEL_PATH)
        if not model_path.exists():
            _fire_smoke_model_error = f"missing model: {model_path}"
            print(f"[fire_smoke] model not found: {model_path}")
            return None
        _fire_smoke_model = YOLO(str(model_path))
        return _fire_smoke_model


def annotate_fall_frame(frame, source_id, clip_recorder):
    model = get_fall_model()
    if model is None:
        return frame
    results = model.predict(frame, conf=FALL_CONF, verbose=False)
    if not results:
        return frame
    res = results[0]
    annotated = res.plot()
    try:
        names = res.names or {}
        fall_found = False
        fall_label = None
        fall_score = None
        fall_boxes = []
        if res.boxes is not None and res.boxes.cls is not None:
            cls_list = res.boxes.cls.tolist()
            conf_list = res.boxes.conf.tolist() if res.boxes.conf is not None else [None] * len(cls_list)
            xyxy_list = res.boxes.xyxy.tolist() if res.boxes.xyxy is not None else []
            for idx, cls_id in enumerate(cls_list):
                label = names.get(int(cls_id), str(int(cls_id)))
                score = conf_list[idx] if idx < len(conf_list) else None
                if idx < len(xyxy_list):
                    x1, y1, x2, y2 = xyxy_list[idx]
                    fall_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                if is_fall_event(label):
                    fall_found = True
                    if fall_score is None or (score is not None and score > fall_score):
                        fall_score = score
                        fall_label = label
        if fall_found:
            cv2.putText(
                annotated,
                "낙상 감지",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if fall_label is None:
                fall_label = "fall"
            if fall_score is None:
                fall_score = 0.0
            row_id = log_fall_event(fall_label, fall_score, fall_boxes, source_id)
            if row_id and clip_recorder:
                clip_recorder.trigger(row_id, fall_label, source_id)
    except Exception as exc:
        print(f"[fall] label overlay failed: {exc}")
    return annotated


def annotate_fire_smoke_frame(frame, source_id, clip_recorder):
    model = get_fire_smoke_model()
    if model is None:
        return frame
    results = model.predict(frame, conf=FIRE_SMOKE_CONF, verbose=False)
    if not results:
        return frame
    res = results[0]
    annotated = res.plot()
    try:
        names = res.names or {}
        fire_smoke_found = False
        fire_smoke_label = None
        fire_smoke_score = None
        fire_smoke_boxes = []
        if res.boxes is not None and res.boxes.cls is not None:
            cls_list = res.boxes.cls.tolist()
            conf_list = res.boxes.conf.tolist() if res.boxes.conf is not None else [None] * len(cls_list)
            xyxy_list = res.boxes.xyxy.tolist() if res.boxes.xyxy is not None else []
            for idx, cls_id in enumerate(cls_list):
                label = names.get(int(cls_id), str(int(cls_id)))
                score = conf_list[idx] if idx < len(conf_list) else None
                if idx < len(xyxy_list):
                    x1, y1, x2, y2 = xyxy_list[idx]
                    fire_smoke_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                if label:
                    fire_smoke_found = True
                    if fire_smoke_score is None or (score is not None and score > fire_smoke_score):
                        fire_smoke_score = score
                        fire_smoke_label = label
        if fire_smoke_found:
            cv2.putText(
                annotated,
                "화재/연기 감지",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if fire_smoke_label is None:
                fire_smoke_label = "fire"
            if fire_smoke_score is None:
                fire_smoke_score = 0.0
            row_id = log_fire_smoke_event(fire_smoke_label, fire_smoke_score, fire_smoke_boxes, source_id)
            if row_id and clip_recorder:
                clip_recorder.trigger(row_id, fire_smoke_label, source_id)
    except Exception as exc:
        print(f"[fire_smoke] label overlay failed: {exc}")
    return annotated
def _set_test_status(state, message=""):
    with _test_lock:
        _test_status["state"] = state
        _test_status["message"] = message


def _get_test_status():
    with _test_lock:
        status = dict(_test_status)
    with _test_control_lock:
        status["paused"] = _test_pause
        status["position_seconds"] = _test_position_seconds
        status["duration_seconds"] = _test_duration_seconds
    return status


def _set_test_pause(paused: bool) -> None:
    global _test_pause
    with _test_control_lock:
        _test_pause = paused


def _request_test_seek(seconds: float) -> None:
    global _test_seek_seconds
    with _test_control_lock:
        _test_seek_seconds = seconds


def _consume_test_seek():
    global _test_seek_seconds
    with _test_control_lock:
        seconds = _test_seek_seconds
        _test_seek_seconds = None
        paused = _test_pause
    return seconds, paused


def _update_test_position(seconds: float) -> None:
    global _test_position_seconds
    with _test_control_lock:
        _test_position_seconds = seconds


def _get_video_stream_source(path_str):
    if not path_str:
        return None, "missing video source"
    path = Path(path_str)
    if path.exists():
        return str(path), ""
    return None, f"file not found: {path_str}"


def _start_test_stream(youtube_url, model_path):
    global _test_thread, _test_source_url, _test_model_path

    if _test_thread and _test_thread.is_alive():
        _stop_test_stream()

    _test_stop_event.clear()
    _test_source_url = youtube_url
    _test_model_path = model_path
    thread = threading.Thread(
        target=_test_loop,
        args=(youtube_url, model_path, TEST_CONF, TEST_FPS),
        daemon=True,
    )
    _test_thread = thread
    thread.start()


def _stop_test_stream():
    global _test_thread
    _test_stop_event.set()
    if _test_thread and _test_thread.is_alive():
        _test_thread.join(timeout=2.0)
    _test_thread = None
    _set_test_status("stopped", "stream stopped")


def _test_loop(youtube_url, model_path, conf, fps):
    global _test_latest_jpeg
    _set_test_status("starting", "loading video source")
    stream_url, err = _get_video_stream_source(youtube_url)
    if not stream_url:
        _set_test_status("error", err)
        return

    try:
        from ultralytics import YOLO
    except Exception as exc:
        _set_test_status("error", f"ultralytics import failed: {exc}")
        return

    model_path = Path(model_path)
    if not model_path.exists():
        _set_test_status("error", f"model not found: {model_path}")
        return

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        _set_test_status("error", "failed to open video source")
        return

    _set_test_status("running", "streaming")
    with _test_control_lock:
        _test_pause = False
        _test_seek_seconds = None
        _test_position_seconds = 0.0
        fps_cap = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps_cap > 0 and frame_count > 0:
            _test_duration_seconds = frame_count / fps_cap
        else:
            _test_duration_seconds = None
        duration_seconds = _test_duration_seconds
    frame_interval = 1.0 / fps if fps > 0 else 0.0
    next_time = time.time()

    while not _test_stop_event.is_set():
        seek_seconds, paused = _consume_test_seek()
        if seek_seconds is not None:
            if duration_seconds and duration_seconds > 0:
                seek_seconds = min(seek_seconds, duration_seconds)
            pos_ok = cap.set(cv2.CAP_PROP_POS_MSEC, seek_seconds * 1000.0)
            if not pos_ok and fps_cap > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_seconds * fps_cap))
            ret, frame = cap.read()
            if ret and frame is not None:
                results = model.predict(frame, conf=conf, verbose=False)
                if results:
                    annotated = results[0].plot()
                else:
                    annotated = frame
                ok, buffer = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY],
                )
                if ok:
                    with _test_latest_lock:
                        _test_latest_jpeg = buffer.tobytes()
                _update_test_position(seek_seconds)
            if paused:
                time.sleep(0.05)
                continue
        if paused:
            time.sleep(0.05)
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            _set_test_status("error", "stream ended or failed to read frame")
            break

        now = time.time()
        if frame_interval > 0 and now < next_time:
            time.sleep(max(0.0, next_time - now))
        next_time = time.time() + frame_interval

        results = model.predict(frame, conf=conf, verbose=False)
        if results:
            annotated = results[0].plot()
        else:
            annotated = frame

        ok, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY],
        )
        if ok:
            with _test_latest_lock:
                _test_latest_jpeg = buffer.tobytes()
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
        _update_test_position(pos_msec / 1000.0)

    cap.release()


def camera_loop():
    global _camera
    global _latest_jpeg
    global _latest_raw_jpeg
    global _last_frame_time
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
                _last_frame_time = time.time()
        try:
            frame = annotate_frame(frame, CCTV1_FACE_SOURCE_ID, _clip_recorder)
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


def camera_loop_secondary():
    global _camera2
    global _latest_raw_jpeg_2
    global _last_frame_time_2
    cam = get_camera_secondary()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    while not _shutdown_event.is_set():
        with _camera2_lock:
            ret, frame = cam.read()
        if not ret or frame is None:
            with _camera2_lock:
                if cam is _camera2:
                    cam.release()
                    _camera2 = None
                cam = get_camera_secondary()
            time.sleep(0.2)
            continue
        ok, raw_buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with _latest_lock:
                _latest_raw_jpeg_2 = raw_buffer.tobytes()
                _last_frame_time_2 = time.time()
        try:
            frame = annotate_frame(frame, CCTV2_FACE_SOURCE_ID, _clip_recorder_2)
        except Exception as exc:
            print(f"[camera2] annotate_frame failed: {exc}")
        if _clip_recorder_2:
            _clip_recorder_2.add_frame(frame)


def fall_loop():
    global _latest_fall_jpeg
    interval = 1.0 / FALL_FPS if FALL_FPS > 0 else 0.0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    while not _shutdown_event.is_set():
        start = time.time()
        with _latest_lock:
            raw_bytes = _latest_raw_jpeg
        if not raw_bytes:
            time.sleep(0.1)
            continue
        image = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if frame is None:
            time.sleep(0.1)
            continue
        try:
            frame = annotate_fall_frame(frame, CCTV1_FALL_SOURCE_ID, _fall_clip_recorder)
        except Exception as exc:
            print(f"[fall] annotate failed: {exc}")
        if _fall_clip_recorder:
            _fall_clip_recorder.add_frame(frame)
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with _latest_lock:
                _latest_fall_jpeg = buffer.tobytes()
        elapsed = time.time() - start
        if interval > 0:
            time.sleep(max(0.0, interval - elapsed))


def fire_smoke_loop():
    global _latest_fire_smoke_jpeg
    interval = 1.0 / FIRE_SMOKE_FPS if FIRE_SMOKE_FPS > 0 else 0.0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    while not _shutdown_event.is_set():
        start = time.time()
        with _latest_lock:
            raw_bytes = _latest_raw_jpeg
        if not raw_bytes:
            time.sleep(0.1)
            continue
        image = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if frame is None:
            time.sleep(0.1)
            continue
        try:
            frame = annotate_fire_smoke_frame(frame, CCTV1_FIRE_SOURCE_ID, _fire_smoke_clip_recorder)
        except Exception as exc:
            print(f"[fire_smoke] annotate failed: {exc}")
        if _fire_smoke_clip_recorder:
            _fire_smoke_clip_recorder.add_frame(frame)
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with _latest_lock:
                _latest_fire_smoke_jpeg = buffer.tobytes()
        elapsed = time.time() - start
        if interval > 0:
            time.sleep(max(0.0, interval - elapsed))


def fall_loop_secondary():
    global _latest_fall_jpeg_2
    interval = 1.0 / FALL_FPS if FALL_FPS > 0 else 0.0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    while not _shutdown_event.is_set():
        start = time.time()
        with _latest_lock:
            raw_bytes = _latest_raw_jpeg_2
        if not raw_bytes:
            time.sleep(0.1)
            continue
        image = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if frame is None:
            time.sleep(0.1)
            continue
        try:
            frame = annotate_fall_frame(frame, CCTV2_FALL_SOURCE_ID, _fall_clip_recorder_2)
        except Exception as exc:
            print(f"[fall2] annotate failed: {exc}")
        if _fall_clip_recorder_2:
            _fall_clip_recorder_2.add_frame(frame)
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with _latest_lock:
                _latest_fall_jpeg_2 = buffer.tobytes()
        elapsed = time.time() - start
        if interval > 0:
            time.sleep(max(0.0, interval - elapsed))


def fire_smoke_loop_secondary():
    global _latest_fire_smoke_jpeg_2
    interval = 1.0 / FIRE_SMOKE_FPS if FIRE_SMOKE_FPS > 0 else 0.0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    while not _shutdown_event.is_set():
        start = time.time()
        with _latest_lock:
            raw_bytes = _latest_raw_jpeg_2
        if not raw_bytes:
            time.sleep(0.1)
            continue
        image = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if frame is None:
            time.sleep(0.1)
            continue
        try:
            frame = annotate_fire_smoke_frame(frame, CCTV2_FIRE_SOURCE_ID, _fire_smoke_clip_recorder_2)
        except Exception as exc:
            print(f"[fire_smoke2] annotate failed: {exc}")
        if _fire_smoke_clip_recorder_2:
            _fire_smoke_clip_recorder_2.add_frame(frame)
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with _latest_lock:
                _latest_fire_smoke_jpeg_2 = buffer.tobytes()
        elapsed = time.time() - start
        if interval > 0:
            time.sleep(max(0.0, interval - elapsed))

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


def generate_raw_frames_secondary():
    while True:
        with _latest_lock:
            jpeg_bytes = _latest_raw_jpeg_2
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


def generate_fall_frames():
    while True:
        with _latest_lock:
            jpeg_bytes = _latest_fall_jpeg
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


def generate_fire_smoke_frames():
    while True:
        with _latest_lock:
            jpeg_bytes = _latest_fire_smoke_jpeg
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


def generate_test_frames():
    while True:
        with _test_latest_lock:
            jpeg_bytes = _test_latest_jpeg
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


def build_event_message(task, label):
    normalized = (label or "").lower()
    if task == "intruder" or normalized in {"unknown", "intruder"}:
        return "외부인이 감지되었습니다."
    if task == "pose" or normalized in FALL_LABEL_KEYWORDS:
        return "넘어짐이 감지되었습니다."
    if task == "detect":
        if "smoke" in normalized or "fire" in normalized:
            return "화재가 감지되었습니다."
        return "화재가 감지되었습니다."
    return "이상행동이 감지되었습니다."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_raw")
def video_feed_raw():
    return Response(generate_raw_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_cctv1")
def video_feed_cctv1():
    return Response(generate_raw_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_cctv2")
def video_feed_cctv2():
    return Response(generate_raw_frames_secondary(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_fall")
def video_feed_fall():
    return Response(generate_fall_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_fire_smoke")
def video_feed_fire_smoke():
    return Response(generate_fire_smoke_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera_status")
def camera_status():
    now = time.time()
    with _latest_lock:
        last1 = _last_frame_time
        last2 = _last_frame_time_2
    return jsonify(
        {
            "cctv1": last1 is not None and now - last1 < 2.0,
            "cctv2": last2 is not None and now - last2 < 2.0,
        }
    )


@app.route("/event-logs")
def event_logs():
    source = request.args.get("source", "").strip()
    limit_str = request.args.get("limit", "20")
    try:
        limit = int(limit_str)
    except ValueError:
        limit = 20
    limit = max(1, min(limit, 100))

    where_clauses = []
    params = []
    if source:
        source_ids = CCTV_SOURCE_MAP.get(source, [source])
        non_null_sources = [sid for sid in source_ids if sid is not None]
        clauses = []
        if non_null_sources:
            placeholders = ", ".join(["%s"] * len(non_null_sources))
            clauses.append(f"source_id IN ({placeholders})")
            params.extend(non_null_sources)
        if any(sid is None for sid in source_ids):
            clauses.append("source_id IS NULL")
        if clauses:
            where_clauses.append("(" + " OR ".join(clauses) + ")")

    query = (
        "SELECT id, task, label, score, source_id, payload_json, seen_at, video_path "
        "FROM ai_event_logs"
    )
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY seen_at DESC LIMIT %s"
    params.append(limit)

    rows = []
    with _db_lock:
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        for row in cur.fetchall():
            seen_at = row.get("seen_at")
            if isinstance(seen_at, datetime):
                seen_at_str = seen_at.strftime("%Y-%m-%d %H:%M:%S")
            else:
                seen_at_str = str(seen_at or "")
            message = build_event_message(row.get("task"), row.get("label"))
            video_path = row.get("video_path")
            clip_url = None
            if video_path and (CLIP_DIR / video_path).exists():
                clip_url = url_for("serve_clip", filename=video_path)
            rows.append(
                {
                    "id": row.get("id"),
                    "message": message,
                    "task": row.get("task"),
                    "label": row.get("label"),
                    "seen_at": seen_at_str,
                    "clip_url": clip_url,
                }
            )
        cur.close()
        conn.close()
    return jsonify(rows)


@app.route("/test")
def test_page():
    status = _get_test_status()
    return render_template(
        "test.html",
        status=status,
        model_path=_test_model_path or FALL_MODEL_PATH,
        video_source=_test_source_url or "",
        download_message="",
        test_conf=TEST_CONF,
        test_fps=TEST_FPS,
    )


@app.route("/test/status")
def test_status():
    return jsonify(_get_test_status())


@app.route("/test/pause", methods=["POST"])
def test_pause():
    _set_test_pause(True)
    _set_test_status("paused", "paused")
    return ("", 204)


@app.route("/test/resume", methods=["POST"])
def test_resume():
    _set_test_pause(False)
    _set_test_status("running", "streaming")
    return ("", 204)


@app.route("/test/seek", methods=["POST"])
def test_seek():
    try:
        seconds = float(request.form.get("seconds", "0"))
    except ValueError:
        seconds = 0.0
    _request_test_seek(max(0.0, seconds))
    return ("", 204)


@app.route("/test/start", methods=["POST"])
def test_start():
    video_source = request.form.get("video_source", "").strip()
    model_path = request.form.get("model_path", "").strip()
    if not video_source or not model_path:
        return render_template(
            "test.html",
            status={"state": "error", "message": "영상 소스와 모델 경로를 입력하세요."},
            model_path=model_path or FALL_MODEL_PATH,
            video_source=video_source,
            download_message="",
            test_conf=TEST_CONF,
            test_fps=TEST_FPS,
        )
    _start_test_stream(
        video_source,
        model_path,
    )
    return redirect(url_for("test_page"))


@app.route("/test/stop", methods=["POST"])
def test_stop():
    _stop_test_stream()
    return redirect(url_for("test_page"))


@app.route("/test/download", methods=["POST"])
def test_download():
    youtube_url = request.form.get("youtube_url", "").strip()
    status = _get_test_status()
    if not youtube_url:
        return render_template(
            "test.html",
            status=status,
            model_path=_test_model_path or FALL_MODEL_PATH,
            video_source=_test_source_url or "",
            download_message="YouTube URL을 입력하세요.",
            test_conf=TEST_CONF,
            test_fps=TEST_FPS,
        )
    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        return render_template(
            "test.html",
            status=status,
            model_path=_test_model_path or FALL_MODEL_PATH,
            video_source=_test_source_url or "",
            download_message="YouTube URL만 다운로드 가능합니다.",
            test_conf=TEST_CONF,
            test_fps=TEST_FPS,
        )

    output_template = str(TEST_VIDEO_DIR / "%(title).200s.%(ext)s")
    cmd = ["yt-dlp", "-f", "best[ext=mp4]/best", "-o", output_template, youtube_url]
    if TEST_COOKIE_FILE.exists():
        cmd.extend(["--cookies", str(TEST_COOKIE_FILE)])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except FileNotFoundError:
        return render_template(
            "test.html",
            status=status,
            model_path=_test_model_path or FALL_MODEL_PATH,
            video_source=_test_source_url or "",
            download_message="yt-dlp가 설치되어 있지 않습니다.",
            test_conf=TEST_CONF,
            test_fps=TEST_FPS,
        )
    except subprocess.TimeoutExpired:
        return render_template(
            "test.html",
            status=status,
            model_path=_test_model_path or FALL_MODEL_PATH,
            video_source=_test_source_url or "",
            download_message="다운로드 시간이 초과되었습니다.",
            test_conf=TEST_CONF,
            test_fps=TEST_FPS,
        )

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        message = err or "다운로드에 실패했습니다."
        return render_template(
            "test.html",
            status=status,
            model_path=_test_model_path or FALL_MODEL_PATH,
            video_source=_test_source_url or "",
            download_message=message,
            test_conf=TEST_CONF,
            test_fps=TEST_FPS,
        )

    return render_template(
        "test.html",
        status=status,
        model_path=_test_model_path or FALL_MODEL_PATH,
        video_source=_test_source_url or "",
        download_message=f"다운로드 완료: {TEST_VIDEO_DIR}",
        test_conf=TEST_CONF,
        test_fps=TEST_FPS,
    )


@app.route("/test_feed")
def test_feed():
    return Response(generate_test_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
    _fall_clip_recorder = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    _fire_smoke_clip_recorder = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    _clip_recorder_2 = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    _fall_clip_recorder_2 = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    _fire_smoke_clip_recorder_2 = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    listener = threading.Thread(target=event_listener, daemon=True)
    listener.start()
    init_udp_sender()
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    camera2_thread = threading.Thread(target=camera_loop_secondary, daemon=True)
    camera2_thread.start()
    fall_thread = threading.Thread(target=fall_loop, daemon=True)
    fall_thread.start()
    fire_smoke_thread = threading.Thread(target=fire_smoke_loop, daemon=True)
    fire_smoke_thread.start()
    fall2_thread = threading.Thread(target=fall_loop_secondary, daemon=True)
    fall2_thread.start()
    fire_smoke2_thread = threading.Thread(target=fire_smoke_loop_secondary, daemon=True)
    fire_smoke2_thread.start()
    init_db()
    refresh_gallery()
    def _shutdown_cleanup():
        _shutdown_event.set()
        with _camera_lock:
            if _camera is not None:
                _camera.release()
        with _camera2_lock:
            if _camera2 is not None:
                _camera2.release()
        if _udp_sock is not None:
            _udp_sock.close()
        os._exit(0)
    atexit.register(_shutdown_cleanup)
    signal.signal(signal.SIGINT, lambda *_: _shutdown_cleanup())
    signal.signal(signal.SIGTERM, lambda *_: _shutdown_cleanup())
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
