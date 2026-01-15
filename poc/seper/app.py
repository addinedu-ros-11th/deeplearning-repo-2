"""메인 애플리케이션"""
import os
import signal
import threading
import time
import cv2

from flask import Flask

from config import (
    CLIP_PRE_SECONDS, CLIP_POST_SECONDS,
    CCTV1_FACE_SOURCE_ID, CCTV1_FALL_SOURCE_ID, CCTV1_FIRE_SOURCE_ID,
    CCTV2_FACE_SOURCE_ID, CCTV2_FALL_SOURCE_ID, CCTV2_FIRE_SOURCE_ID,
    FALL_FPS, FIRE_SMOKE_FPS, UDP_JPEG_QUALITY
)
from utils.database import init_db
from services.face_service import refresh_gallery, annotate_frame
from services.video_service import (
    ClipRecorder, init_udp_sender, udp_send_frame,
    get_camera, get_camera_secondary, release_cameras,
    set_latest_jpeg, set_latest_raw_jpeg, set_latest_raw_jpeg_2,
    set_latest_fall_jpeg, set_latest_fire_smoke_jpeg
)
from services.detection_service import annotate_fall_frame, annotate_fire_smoke_frame
from routes.index import index_bp
from routes.register import register_bp
from routes.ai_logs import ai_logs_bp
from routes.test import test_bp

# Flask 앱 생성
app = Flask(__name__)

# Blueprint 등록
app.register_blueprint(index_bp)
app.register_blueprint(register_bp)
app.register_blueprint(ai_logs_bp)
app.register_blueprint(test_bp)

# 전역 변수
_shutdown_event = threading.Event()
_clip_recorder = None
_fall_clip_recorder = None
_fire_smoke_clip_recorder = None
_clip_recorder_2 = None
_fall_clip_recorder_2 = None
_fire_smoke_clip_recorder_2 = None


def camera_loop():
    """주 카메라 루프"""
    cam = get_camera()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    
    while not _shutdown_event.is_set():
        ret, frame = cam.read()
        if not ret or frame is None:
            cam = get_camera()
            time.sleep(0.2)
            continue
        
        ok, raw_buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            set_latest_raw_jpeg(raw_buffer.tobytes())
        
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
        set_latest_jpeg(jpeg_bytes)


def camera_loop_secondary():
    """보조 카메라 루프"""
    cam = get_camera_secondary()
    if cam is None:
        print("[camera2] No secondary camera available, thread exiting")
        return
    
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    
    while not _shutdown_event.is_set():
        ret, frame = cam.read()
        if not ret or frame is None:
            cam = get_camera_secondary()
            if cam is None:
                print("[camera2] Camera lost and cannot reconnect, thread exiting")
                return
            time.sleep(0.2)
            continue
        
        ok, raw_buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            set_latest_raw_jpeg_2(raw_buffer.tobytes())
        
        try:
            frame = annotate_frame(frame, CCTV2_FACE_SOURCE_ID, _clip_recorder_2)
        except Exception as exc:
            print(f"[camera2] annotate_frame failed: {exc}")
        
        if _clip_recorder_2:
            _clip_recorder_2.add_frame(frame)


def fall_loop():
    """낙상 감지 루프"""
    import numpy as np
    from services.video_service import get_latest_raw_jpeg
    
    interval = 1.0 / FALL_FPS if FALL_FPS > 0 else 0.0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    
    while not _shutdown_event.is_set():
        start = time.time()
        raw_bytes = get_latest_raw_jpeg()
        
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
            set_latest_fall_jpeg(buffer.tobytes())
        
        elapsed = time.time() - start
        if interval > 0:
            time.sleep(max(0.0, interval - elapsed))


def fire_smoke_loop():
    """화재/연기 감지 루프"""
    import numpy as np
    from services.video_service import get_latest_raw_jpeg
    
    interval = 1.0 / FIRE_SMOKE_FPS if FIRE_SMOKE_FPS > 0 else 0.0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), UDP_JPEG_QUALITY]
    
    while not _shutdown_event.is_set():
        start = time.time()
        raw_bytes = get_latest_raw_jpeg()
        
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
            set_latest_fire_smoke_jpeg(buffer.tobytes())
        
        elapsed = time.time() - start
        if interval > 0:
            time.sleep(max(0.0, interval - elapsed))


def _shutdown_cleanup():
    """종료 정리"""
    print("\n[shutdown] Cleaning up resources...")
    _shutdown_event.set()
    print("[shutdown] Waiting for threads...")
    time.sleep(0.5)
    print("[shutdown] Releasing cameras...")
    release_cameras()
    print("[shutdown] Cleanup complete, forcing exit...")


def _signal_handler(sig, frame):
    """시그널 핸들러"""
    print(f"\n[shutdown] Received signal {sig}")
    _shutdown_cleanup()
    os._exit(0)


if __name__ == "__main__":
    # 클립 레코더 초기화
    _clip_recorder = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    _fall_clip_recorder = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    _fire_smoke_clip_recorder = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    _clip_recorder_2 = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    _fall_clip_recorder_2 = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    _fire_smoke_clip_recorder_2 = ClipRecorder(CLIP_PRE_SECONDS, CLIP_POST_SECONDS)
    
    # UDP 전송 초기화
    init_udp_sender()
    
    # 스레드 시작
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    
    camera2_thread = threading.Thread(target=camera_loop_secondary, daemon=True)
    camera2_thread.start()
    
    fall_thread = threading.Thread(target=fall_loop, daemon=True)
    fall_thread.start()
    
    fire_smoke_thread = threading.Thread(target=fire_smoke_loop, daemon=True)
    fire_smoke_thread.start()
    
    # 데이터베이스 초기화
    init_db()
    refresh_gallery()
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    try:
        print("[startup] Starting Flask server on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[shutdown] KeyboardInterrupt caught")
        _shutdown_cleanup()
        os._exit(0)
    except Exception as e:
        print(f"\n[error] Unexpected error: {e}")
        _shutdown_cleanup()
        os._exit(1)
