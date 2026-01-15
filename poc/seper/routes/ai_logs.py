"""AI 로그 라우트"""
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, url_for, send_file

from config import CLIP_DIR, CCTV_SOURCE_MAP, FALL_LABEL_KEYWORDS
from utils.database import get_db, get_db_lock

ai_logs_bp = Blueprint('ai_logs', __name__)


def build_event_message(task, label):
    """이벤트 메시지 생성"""
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


@ai_logs_bp.route("/ai-logs")
def ai_logs():
    """AI 로그 페이지"""
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

    db_lock = get_db_lock()
    with db_lock:
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


@ai_logs_bp.route("/event-logs")
def event_logs():
    """이벤트 로그 API"""
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
    db_lock = get_db_lock()
    with db_lock:
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
                clip_url = url_for("ai_logs.serve_clip", filename=video_path)
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


@ai_logs_bp.route("/clips/<path:filename>")
def serve_clip(filename):
    """클립 비디오 서빙"""
    path = CLIP_DIR / filename
    if not path.exists():
        return "Clip not found", 404
    return send_file(path, mimetype="video/mp4")
