"""비디오 처리 서비스 (카메라, 클립 녹화, UDP 전송)"""
import os
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    CAMERA_SOURCE, CAMERA_SOURCE_2, CAMERA_BY_ID_MATCH, CAMERA2_BY_ID_MATCH,
    CLIP_DIR, CLIP_PRE_SECONDS, CLIP_POST_SECONDS, CLIP_COOLDOWN_SECONDS,
    UDP_VIDEO_TARGETS, UDP_JPEG_QUALITY, UDP_MAX_DATAGRAM, UDP_FPS
)
from utils.database import get_db, get_db_lock

# UDP 헤더 포맷
UDP_HEADER_FORMAT = "!IHH"
UDP_HEADER_SIZE = struct.calcsize(UDP_HEADER_FORMAT)

# 전역 변수
_camera_lock = threading.Lock()
_camera = None
_camera_source = None
_camera2_lock = threading.Lock()
_camera2 = None
_camera2_source = None
_latest_lock = threading.Lock()
_latest_jpeg = None
_latest_raw_jpeg = None
_latest_raw_jpeg_2 = None
_latest_fall_jpeg = None
_latest_fire_smoke_jpeg = None
_latest_fall_jpeg_2 = None
_latest_fire_smoke_jpeg_2 = None
_last_frame_time = None
_last_frame_time_2 = None
_udp_sock = None
_udp_targets = None
_udp_frame_id = 0
_udp_max_payload = None
_udp_frame_interval = 0.0
_udp_next_frame_time = 0.0


class ClipRecorder:
    """비디오 클립 녹화 클래스"""
    def __init__(self, pre_seconds, post_seconds):
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.buffer = deque()
        self.lock = threading.Lock()
        self.active = None
        self.last_trigger = None

    def add_frame(self, frame):
        """프레임 추가"""
        now = time.time()
        with self.lock:
            self.buffer.append((now, frame.copy()))
            self._trim_buffer(now)
            if self.active:
                self._write_frame(now, frame)
                if now >= self.active["end_time"]:
                    self._finalize_clip()

    def trigger(self, row_id, label, source_id):
        """녹화 트리거"""
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
        """파일명 생성"""
        safe_label = (label or "event").replace(" ", "_")
        safe_source = (source_id or "cam").replace(" ", "_")
        dt = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        return f"{safe_source}_{safe_label}_{dt}.mp4"

    def _trim_buffer(self, now):
        """버퍼 정리"""
        cutoff = now - (self.pre_seconds + 1)
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def _write_frame(self, _now, frame):
        """프레임 쓰기"""
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
        """클립 완료 처리"""
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
        """이벤트 경로 업데이트"""
        db_lock = get_db_lock()
        with db_lock:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "UPDATE ai_event_logs SET video_path = %s WHERE id = %s",
                (path.name, row_id),
            )
            cur.close()
            conn.close()

    def _transcode_clip(self, path):
        """클립 트랜스코드"""
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
    """UDP 타겟 파싱"""
    targets = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        host, port_str = item.rsplit(":", 1)
        targets.append((host, int(port_str)))
    return targets


def init_udp_sender():
    """UDP 전송 초기화"""
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
    """UDP로 프레임 전송"""
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


def _video_index(path: Path) -> int:
    """비디오 인덱스 추출"""
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return int(digits) if digits else 9999


def _list_by_id_nodes():
    """by-id 노드 목록"""
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
    """비디오 노드 목록"""
    nodes = [p for p in Path("/dev").glob("video*") if p.name[5:].isdigit()]
    return sorted(nodes, key=_video_index)


def _resolve_by_id_match(match_value):
    """by-id 매칭"""
    if not match_value:
        return None
    match_lower = match_value.lower()
    for entry in _list_by_id_nodes():
        if match_lower in entry.name.lower():
            return str(entry)
    return None


def _resolve_auto_source(fallback_index, exclude_sources):
    """자동 소스 해결"""
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
    """카메라 소스 정규화"""
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
    """제외 소스 확장"""
    expanded = []
    for source in sources:
        if source is None:
            continue
        expanded.append(str(source))
        if isinstance(source, int):
            expanded.append(f"/dev/video{source}")
    return expanded


def resolve_camera_source(primary_value, by_id_match, fallback_index, exclude_sources=None):
    """카메라 소스 해결"""
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
    """주 카메라 반환"""
    global _camera, _camera_source
    with _camera_lock:
        if _camera_source is None:
            _camera_source = resolve_camera_source(CAMERA_SOURCE, CAMERA_BY_ID_MATCH, 0, [])
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(_camera_source)
        return _camera


def get_camera_secondary():
    """보조 카메라 반환"""
    global _camera2, _camera2_source
    with _camera2_lock:
        if _camera2_source is None:
            exclude = _expand_exclude_sources([_camera_source])
            _camera2_source = resolve_camera_source(
                CAMERA_SOURCE_2,
                CAMERA2_BY_ID_MATCH,
                1,
                exclude,
            )
            if _camera2_source is None:
                print("[camera2] No secondary camera found")
                return None
        if _camera2 is None or not _camera2.isOpened():
            _camera2 = cv2.VideoCapture(_camera2_source)
            if not _camera2.isOpened():
                print(f"[camera2] Failed to open camera: {_camera2_source}")
                _camera2 = None
                return None
        return _camera2


def set_latest_jpeg(jpeg_bytes):
    """최신 JPEG 설정"""
    global _latest_jpeg
    with _latest_lock:
        _latest_jpeg = jpeg_bytes


def get_latest_jpeg():
    """최신 JPEG 반환"""
    with _latest_lock:
        return _latest_jpeg


def set_latest_raw_jpeg(jpeg_bytes):
    """최신 RAW JPEG 설정"""
    global _latest_raw_jpeg, _last_frame_time
    with _latest_lock:
        _latest_raw_jpeg = jpeg_bytes
        _last_frame_time = time.time()


def get_latest_raw_jpeg():
    """최신 RAW JPEG 반환"""
    with _latest_lock:
        return _latest_raw_jpeg


def set_latest_raw_jpeg_2(jpeg_bytes):
    """최신 RAW JPEG 2 설정"""
    global _latest_raw_jpeg_2, _last_frame_time_2
    with _latest_lock:
        _latest_raw_jpeg_2 = jpeg_bytes
        _last_frame_time_2 = time.time()


def get_latest_raw_jpeg_2():
    """최신 RAW JPEG 2 반환"""
    with _latest_lock:
        return _latest_raw_jpeg_2


def set_latest_fall_jpeg(jpeg_bytes):
    """최신 낙상 JPEG 설정"""
    global _latest_fall_jpeg
    with _latest_lock:
        _latest_fall_jpeg = jpeg_bytes


def get_latest_fall_jpeg():
    """최신 낙상 JPEG 반환"""
    with _latest_lock:
        return _latest_fall_jpeg


def set_latest_fire_smoke_jpeg(jpeg_bytes):
    """최신 화재 JPEG 설정"""
    global _latest_fire_smoke_jpeg
    with _latest_lock:
        _latest_fire_smoke_jpeg = jpeg_bytes


def get_latest_fire_smoke_jpeg():
    """최신 화재 JPEG 반환"""
    with _latest_lock:
        return _latest_fire_smoke_jpeg


def get_camera_status():
    """카메라 상태 반환"""
    now = time.time()
    with _latest_lock:
        last1 = _last_frame_time
        last2 = _last_frame_time_2
    return {
        "cctv1": last1 is not None and now - last1 < 2.0,
        "cctv2": last2 is not None and now - last2 < 2.0,
    }


def release_cameras():
    """카메라 리소스 해제"""
    global _camera, _camera2
    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None
    with _camera2_lock:
        if _camera2 is not None:
            _camera2.release()
            _camera2 = None
