"""얼굴 인식 서비스"""
import threading
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis

from config import (
    MODEL_NAME, DETECTION_SIZE, SIM_THRESHOLD,
    LOG_DIR, COLOR_EMPLOYEE, COLOR_PATIENT, COLOR_UNKNOWN,
    LABEL_FONT_CANDIDATES, UNKNOWN_LOG_DEDUP_SECONDS, UNKNOWN_MIN_SECONDS
)
from utils.database import get_db, get_db_lock

# 전역 변수
_gallery_lock = threading.Lock()
_gallery_embeddings = None
_gallery_meta = None
_last_log = {}
_last_unknown_log = {}
_unknown_since = {}
_label_font_cache = {}

# 얼굴 인식 앱 초기화
face_app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=DETECTION_SIZE)


class Gallery:
    """얼굴 갤러리 클래스"""
    def __init__(self, embeddings, meta):
        self.embeddings = embeddings
        self.meta = meta


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """임베딩을 바이트로 직렬화"""
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """바이트를 임베딩으로 역직렬화"""
    return np.frombuffer(blob, dtype=np.float32)


def load_gallery():
    """DB에서 얼굴 갤러리 로드"""
    embeddings = []
    meta = []
    db_lock = get_db_lock()
    with db_lock:
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
    """갤러리 새로고침"""
    global _gallery_embeddings, _gallery_meta
    gallery = load_gallery()
    with _gallery_lock:
        _gallery_embeddings = gallery.embeddings
        _gallery_meta = gallery.meta


def get_gallery():
    """갤러리 반환"""
    with _gallery_lock:
        if _gallery_embeddings is None:
            refresh_gallery()
        return _gallery_embeddings, _gallery_meta


def best_match(embedding, embeddings, meta):
    """가장 유사한 얼굴 찾기"""
    if embeddings.size == 0:
        return None, -1.0
    embedding = l2_normalize(embedding)
    sims = embeddings @ embedding
    idx = int(np.argmax(sims))
    return meta[idx], float(sims[idx])


def draw_label(frame, bbox, text, color, text_color):
    """프레임에 레이블 그리기"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ASCII 텍스트면 OpenCV 사용
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

    # 한글 텍스트는 PIL 사용
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


def log_unknown_event(similarity, frame, bbox, source_id, clip_recorder):
    """외부인 이벤트 로그 기록"""
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
    db_lock = get_db_lock()
    with db_lock:
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


def annotate_frame(frame, source_id, clip_recorder):
    """프레임에 얼굴 인식 주석 추가"""
    global _unknown_since
    faces = face_app.get(frame)
    embeddings, meta = get_gallery()
    now_mono = threading.current_thread()._target
    import time
    now_mono = time.monotonic()
    unknown_seen = False
    unknown_logged = False
    source_key = source_id or "unknown"

    for face in faces:
        person, similarity = best_match(face.embedding, embeddings, meta)
        
        if person is None or similarity < SIM_THRESHOLD:
            # 외부인
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

        # 등록된 사람
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


def get_face_app():
    """얼굴 인식 앱 반환"""
    return face_app
