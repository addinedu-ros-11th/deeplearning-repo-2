"""애플리케이션 설정"""
import os
from pathlib import Path

# 디렉토리 경로
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
DATA_DIR = APP_DIR / "data" / "face_registry"
LOG_DIR = APP_DIR / "data" / "face_logs"
CLIP_DIR = APP_DIR / "data" / "event_clips"
CAPTURE_DIR = APP_DIR / "data" / "captures"
TEST_VIDEO_DIR = APP_DIR / "data" / "test_videos"

# 데이터베이스 설정
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

# 얼굴 인식 모델 설정
MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)
SIM_THRESHOLD = 0.35
LOG_DEDUP_SECONDS = 2
REGISTER_FRAME_COUNT = 5
REGISTER_FRAME_DELAY = 0.2

# 이벤트 UDP 설정
EVENT_UDP_BIND = "0.0.0.0"
EVENT_UDP_PORT = 6001

# 감지 레이블 키워드
FALL_LABEL_KEYWORDS = {"fall", "fallen", "lying", "laying"}

# 비디오 클립 녹화 설정
CLIP_PRE_SECONDS = 4
CLIP_POST_SECONDS = 11
CLIP_COOLDOWN_SECONDS = 8
UNKNOWN_LOG_DEDUP_SECONDS = 5
FALL_LOG_DEDUP_SECONDS = 5
FIRE_SMOKE_LOG_DEDUP_SECONDS = 5
UNKNOWN_MIN_SECONDS = float(os.getenv("UNKNOWN_MIN_SECONDS", "2.0"))

# 카메라 설정
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "auto")
CAMERA_SOURCE_2 = os.getenv("CAMERA_SOURCE_2", "auto")
CAMERA_BY_ID_MATCH = os.getenv("CAMERA_BY_ID_MATCH", "").strip()
CAMERA2_BY_ID_MATCH = os.getenv("CAMERA2_BY_ID_MATCH", "").strip()

# 낙상 감지 모델 설정
FALL_MODEL_PATH = os.getenv(
    "FALL_MODEL_PATH",
    "yolo11n-pose.pt",
)
FALL_CONF = float(os.getenv("FALL_CONF", "0.5"))
FALL_FPS = float(os.getenv("FALL_FPS", "5.0"))

# 화재/연기 감지 모델 설정
FIRE_SMOKE_MODEL_PATH = os.getenv(
    "FIRE_SMOKE_MODEL_PATH",
    str(REPO_ROOT / "runs" / "train" / "fire_smoke_detect_v8s" / "weights" / "best.pt"),
)
FIRE_SMOKE_CONF = float(os.getenv("FIRE_SMOKE_CONF", "0.5"))
FIRE_SMOKE_FPS = float(os.getenv("FIRE_SMOKE_FPS", "5.0"))

# UDP 비디오 전송 설정
UDP_VIDEO_TARGETS = os.getenv("UDP_VIDEO_TARGETS", "")
UDP_JPEG_QUALITY = int(os.getenv("UDP_JPEG_QUALITY", "80"))
UDP_MAX_DATAGRAM = int(os.getenv("UDP_MAX_DATAGRAM", "1400"))
UDP_FPS = float(os.getenv("UDP_FPS", "0"))

# CCTV 소스 ID
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

# UI 색상 설정
COLOR_EMPLOYEE = (255, 0, 0)
COLOR_PATIENT = (255, 255, 255)
COLOR_UNKNOWN = (0, 0, 255)

# 폰트 설정
LABEL_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
]

# 테스트 설정
TEST_FPS = float(os.getenv("TEST_FPS", "5.0"))
TEST_CONF = float(os.getenv("TEST_CONF", str(FALL_CONF)))
TEST_COOKIE_FILE = TEST_VIDEO_DIR / "cookies.txt"
