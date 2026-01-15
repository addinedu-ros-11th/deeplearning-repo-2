"""메인 인덱스 라우트"""
import time
from flask import Blueprint, Response, render_template, jsonify

from services.video_service import (
    get_latest_jpeg, get_latest_raw_jpeg, get_latest_raw_jpeg_2,
    get_latest_fall_jpeg, get_latest_fire_smoke_jpeg,
    get_camera_status
)

index_bp = Blueprint('index', __name__)


def generate_frames():
    """프레임 생성기"""
    while True:
        jpeg_bytes = get_latest_jpeg()
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


def generate_raw_frames():
    """RAW 프레임 생성기"""
    while True:
        jpeg_bytes = get_latest_raw_jpeg()
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


def generate_raw_frames_secondary():
    """RAW 프레임 생성기 (보조 카메라)"""
    while True:
        jpeg_bytes = get_latest_raw_jpeg_2()
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


def generate_fall_frames():
    """낙상 감지 프레임 생성기"""
    while True:
        jpeg_bytes = get_latest_fall_jpeg()
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


def generate_fire_smoke_frames():
    """화재/연기 감지 프레임 생성기"""
    while True:
        jpeg_bytes = get_latest_fire_smoke_jpeg()
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.05)


@index_bp.route("/")
def index():
    """메인 페이지"""
    return render_template("index.html")


@index_bp.route("/video_feed")
def video_feed():
    """비디오 피드 (얼굴 인식)"""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@index_bp.route("/video_feed_raw")
def video_feed_raw():
    """비디오 피드 (RAW)"""
    return Response(generate_raw_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@index_bp.route("/video_feed_cctv1")
def video_feed_cctv1():
    """CCTV1 비디오 피드"""
    return Response(generate_raw_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@index_bp.route("/video_feed_cctv2")
def video_feed_cctv2():
    """CCTV2 비디오 피드"""
    return Response(generate_raw_frames_secondary(), mimetype="multipart/x-mixed-replace; boundary=frame")


@index_bp.route("/video_feed_fall")
def video_feed_fall():
    """낙상 감지 비디오 피드"""
    return Response(generate_fall_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@index_bp.route("/video_feed_fire_smoke")
def video_feed_fire_smoke():
    """화재/연기 감지 비디오 피드"""
    return Response(generate_fire_smoke_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@index_bp.route("/camera_status")
def camera_status():
    """카메라 상태"""
    return jsonify(get_camera_status())


@index_bp.route("/reload")
def reload_gallery():
    """갤러리 새로고침"""
    from services.face_service import refresh_gallery
    from flask import redirect, url_for
    refresh_gallery()
    return redirect(url_for("index.index"))
