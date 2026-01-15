"""이벤트 감지 서비스 (낙상, 화재/연기)"""
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    FALL_MODEL_PATH, FALL_CONF, FIRE_SMOKE_MODEL_PATH, FIRE_SMOKE_CONF,
    FALL_LABEL_KEYWORDS, FALL_LOG_DEDUP_SECONDS, FIRE_SMOKE_LOG_DEDUP_SECONDS, LABEL_FONT_CANDIDATES
)
from utils.database import get_db, get_db_lock

# 전역 변수
_fall_model = None
_fall_model_error = None
_fall_lock = threading.Lock()
_fire_smoke_model = None
_fire_smoke_model_error = None
_fire_smoke_lock = threading.Lock()
_last_fall_log = {}
_last_fire_smoke_log = {}
_label_font_cache = {}


def put_korean_text(img, text, pos, font_size=30, color=(0, 0, 255)):
    """OpenCV 이미지에 한글 텍스트 추가"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    ## 한글 깨짐 임시 수정 - opencv 한글 깨짐 -> PIL로 처리
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
    ##
    draw.text(pos, text, font=font, fill=color[::-1])  # BGR to RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_fall_model():
    """낙상 감지 모델 반환"""
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
        
        try:
            _fall_model = YOLO(str(FALL_MODEL_PATH))
        except Exception as exc:
            _fall_model_error = f"failed to load model: {FALL_MODEL_PATH} ({exc})"
            print(f"[fall] model load failed: {FALL_MODEL_PATH}, {exc}")
            return None
            
        return _fall_model


def get_fire_smoke_model():
    """화재/연기 감지 모델 반환"""
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
        from pathlib import Path
        model_path = Path(FIRE_SMOKE_MODEL_PATH)
        if not model_path.exists():
            _fire_smoke_model_error = f"missing model: {model_path}"
            print(f"[fire_smoke] model not found: {model_path}")
            return None
        _fire_smoke_model = YOLO(str(model_path))
        return _fire_smoke_model


def is_fall_event(label):
    """낙상 이벤트 여부 확인"""
    if not label:
        return False
    lower = label.lower()
    return any(keyword in lower for keyword in FALL_LABEL_KEYWORDS)


def check_fall_rule(bbox, keypoints_xy, keypoints_conf):
    """규칙 기반 낙상 감지"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # 골반이 보이지 않으면 판단 불가
    has_hips = False
    if len(keypoints_conf) >= 13:
        if keypoints_conf[11] > 0.4 or keypoints_conf[12] > 0.4:
            has_hips = True

    if not has_hips:
        return False

    # Rule 1: 가로가 세로보다 긴 경우
    if w > h * 0.9:
        return True

    # Rule 2/3: 어깨가 골반보다 낮은 경우
    if len(keypoints_xy) >= 13:
        if (keypoints_conf[5] > 0.5 and keypoints_conf[6] > 0.5 and
            keypoints_conf[11] > 0.5 and keypoints_conf[12] > 0.5):
            shoulder_y = (keypoints_xy[5][1] + keypoints_xy[6][1]) / 2
            hip_y = (keypoints_xy[11][1] + keypoints_xy[12][1]) / 2
            if shoulder_y > hip_y:
                return True

            vertical_dist = abs(shoulder_y - hip_y)
            if h > 0:
                rel_dist = vertical_dist / h
                if rel_dist < 0.2:
                    return True

    return False


def _draw_pose(frame, keypoints, keypoint_scores=None):
    """포즈 그리기"""
    if not keypoints:
        return
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (0, 14), (0, 15), (14, 16), (15, 17),
    ]
    for idx, (x, y) in enumerate(keypoints):
        if keypoint_scores and keypoint_scores[idx] is not None and keypoint_scores[idx] < 0.3:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    for a, b in skeleton:
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        if keypoint_scores:
            if keypoint_scores[a] is not None and keypoint_scores[a] < 0.3:
                continue
            if keypoint_scores[b] is not None and keypoint_scores[b] < 0.3:
                continue
        ax, ay = keypoints[a]
        bx, by = keypoints[b]
        cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (255, 0, 255), 2)


def log_fall_event(label, score, boxes, source_id):
    """낙상 이벤트 로그 기록"""
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
    """화재/연기 이벤트 로그 기록"""
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


def annotate_fall_frame(frame, source_id, clip_recorder):
    """낙상 감지 주석 추가"""
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
        fall_score = 0.0
        fall_boxes = []
        
        kpts_list = None
        kconf_list = None
        if res.keypoints is not None and res.keypoints.xy is not None:
            kpts_list = res.keypoints.xy.tolist()
            if getattr(res.keypoints, "conf", None) is not None:
                kconf_list = res.keypoints.conf.tolist()
        
        if res.boxes is not None and res.boxes.cls is not None:
            cls_list = res.boxes.cls.tolist()
            conf_list = res.boxes.conf.tolist() if res.boxes.conf is not None else [None] * len(cls_list)
            xyxy_list = res.boxes.xyxy.tolist() if res.boxes.xyxy is not None else []
            
            for idx, cls_id in enumerate(cls_list):
                label = names.get(int(cls_id), str(int(cls_id)))
                score = conf_list[idx] if idx < len(conf_list) else None
                
                if idx >= len(xyxy_list):
                    continue
                
                x1, y1, x2, y2 = xyxy_list[idx]
                fall_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 255), 2)
                label_text = f"{label} {score:.2f}" if score is not None else label
                cv2.putText(
                    annotated,
                    label_text,
                    (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
                
                rule_match = False
                if kpts_list and idx < len(kpts_list):
                    kpts = kpts_list[idx]
                    kconf = kconf_list[idx] if kconf_list and idx < len(kconf_list) else None
                    _draw_pose(annotated, kpts, kconf)
                    if kconf is None:
                        kconf = [1.0] * len(kpts)
                    rule_match = check_fall_rule([x1, y1, x2, y2], kpts, kconf)
                
                if rule_match or is_fall_event(label):
                    fall_found = True
                    fall_label = "fall" if rule_match else label
                    if score is not None and score > fall_score:
                        fall_score = score
                        fall_label = "fall" if rule_match else label
                    
                    if rule_match:
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(
                            annotated,
                            f"FALL {score:.2f}" if score is not None else "FALL",
                            (int(x1), max(0, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
        
        if fall_found:
            annotated = put_korean_text(annotated, "낙상 감지", (10, 30), font_size=30, color=(0, 0, 255))
            if fall_label is None:
                fall_label = "fall"
            if fall_score is None:
                fall_score = 0.0
            row_id = log_fall_event(fall_label, fall_score, fall_boxes, source_id)
            if row_id and clip_recorder:
                clip_recorder.trigger(row_id, fall_label, source_id)
    
    except Exception as exc:
        print(f"[fall] rule-based label overlay failed: {exc}")
        import traceback
        traceback.print_exc()
    
    return annotated


def annotate_fire_smoke_frame(frame, source_id, clip_recorder):
    """화재/연기 감지 주석 추가"""
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
            annotated = put_korean_text(annotated, "화재/연기 감지", (10, 30), font_size=30, color=(0, 0, 255))
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
