"""테스트 라우트"""
import time
import threading
import subprocess
import cv2
from pathlib import Path
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, Response

from config import (
    FALL_MODEL_PATH, TEST_FPS, TEST_CONF, TEST_VIDEO_DIR, TEST_COOKIE_FILE,
    UDP_JPEG_QUALITY
)
from services.detection_service import check_fall_rule

test_bp = Blueprint('test', __name__)

# 테스트 전역 변수
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


def _set_test_status(state, message=""):
    """테스트 상태 설정"""
    with _test_lock:
        _test_status["state"] = state
        _test_status["message"] = message


def _get_test_status():
    """테스트 상태 반환"""
    with _test_lock:
        status = dict(_test_status)
    with _test_control_lock:
        status["paused"] = _test_pause
        status["position_seconds"] = _test_position_seconds
        status["duration_seconds"] = _test_duration_seconds
    return status


def _set_test_pause(paused: bool):
    """테스트 일시정지"""
    global _test_pause
    with _test_control_lock:
        _test_pause = paused


def _request_test_seek(seconds: float):
    """테스트 탐색 요청"""
    global _test_seek_seconds
    with _test_control_lock:
        _test_seek_seconds = seconds


def _consume_test_seek():
    """테스트 탐색 소비"""
    global _test_seek_seconds
    with _test_control_lock:
        seconds = _test_seek_seconds
        _test_seek_seconds = None
        paused = _test_pause
    return seconds, paused


def _update_test_position(seconds: float):
    """테스트 위치 업데이트"""
    global _test_position_seconds
    with _test_control_lock:
        _test_position_seconds = seconds


def _get_video_stream_source(path_str):
    """비디오 스트림 소스 반환"""
    if not path_str:
        return None, "missing video source"
    path = Path(path_str)
    if path.exists():
        return str(path), ""
    return None, f"file not found: {path_str}"


def _draw_test_fall_result(results, annotated):
    """테스트 낙상 결과 그리기"""
    res = results[0]
    if res.boxes is not None:
        boxes = res.boxes
        keypoints = res.keypoints
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            score = float(box.conf[0])
            
            kpts_xy = []
            kpts_conf = []
            if keypoints is not None and keypoints.xy is not None:
                if idx < len(keypoints.xy):
                    kpts_xy = keypoints.xy[idx].tolist()
                    if keypoints.conf is not None:
                        kpts_conf = keypoints.conf[idx].tolist()
                    else:
                        kpts_conf = [1.0] * len(kpts_xy)

            if check_fall_rule(xyxy, kpts_xy, kpts_conf):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(
                    annotated, 
                    f"FALL (Test) {score:.2f}", 
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 0, 255), 
                    2
                )
    return annotated


def _test_loop(youtube_url, model_path, conf, fps):
    """테스트 루프"""
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
                    _draw_test_fall_result(results, annotated)
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
            _draw_test_fall_result(results, annotated)
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


def _start_test_stream(youtube_url, model_path, conf=None):
    """테스트 스트림 시작"""
    global _test_thread, _test_source_url, _test_model_path

    if _test_thread and _test_thread.is_alive():
        _stop_test_stream()

    if conf is None:
        conf = TEST_CONF

    _test_stop_event.clear()
    _test_source_url = youtube_url
    _test_model_path = model_path
    thread = threading.Thread(
        target=_test_loop,
        args=(youtube_url, model_path, conf, TEST_FPS),
        daemon=True,
    )
    _test_thread = thread
    thread.start()


def _stop_test_stream():
    """테스트 스트림 중지"""
    global _test_thread
    _test_stop_event.set()
    if _test_thread and _test_thread.is_alive():
        _test_thread.join(timeout=2.0)
    _test_thread = None
    _set_test_status("stopped", "stream stopped")


def generate_test_frames():
    """테스트 프레임 생성기"""
    while True:
        with _test_latest_lock:
            jpeg_bytes = _test_latest_jpeg
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


@test_bp.route("/test")
def test_page():
    """테스트 페이지"""
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


@test_bp.route("/test/status")
def test_status():
    """테스트 상태"""
    return jsonify(_get_test_status())


@test_bp.route("/test/pause", methods=["POST"])
def test_pause():
    """테스트 일시정지"""
    _set_test_pause(True)
    _set_test_status("paused", "paused")
    return ("", 204)


@test_bp.route("/test/resume", methods=["POST"])
def test_resume():
    """테스트 재개"""
    _set_test_pause(False)
    _set_test_status("running", "streaming")
    return ("", 204)


@test_bp.route("/test/seek", methods=["POST"])
def test_seek():
    """테스트 탐색"""
    try:
        seconds = float(request.form.get("seconds", "0"))
    except ValueError:
        seconds = 0.0
    _request_test_seek(max(0.0, seconds))
    return ("", 204)


@test_bp.route("/test/start", methods=["POST"])
def test_start():
    """테스트 시작"""
    video_source = request.form.get("video_source", "").strip()
    model_path = request.form.get("model_path", "").strip()
    try:
        test_conf = float(request.form.get("test_conf", str(TEST_CONF)))
    except ValueError:
        test_conf = TEST_CONF

    if not video_source or not model_path:
        return render_template(
            "test.html",
            status={"state": "error", "message": "영상 소스와 모델 경로를 입력하세요."},
            model_path=model_path or FALL_MODEL_PATH,
            video_source=video_source,
            download_message="",
            test_conf=test_conf,
            test_fps=TEST_FPS,
        )
    
    _start_test_stream(video_source, model_path, conf=test_conf)
    return redirect(url_for("test.test_page"))


@test_bp.route("/test/stop", methods=["POST"])
def test_stop():
    """테스트 중지"""
    _stop_test_stream()
    return redirect(url_for("test.test_page"))


@test_bp.route("/test/download", methods=["POST"])
def test_download():
    """YouTube 다운로드"""
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
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300)
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


@test_bp.route("/test_feed")
def test_feed():
    """테스트 비디오 피드"""
    return Response(generate_test_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
