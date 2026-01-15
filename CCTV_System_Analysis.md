# CCTV 모니터링 시스템 분석 문서

**작성일**: 2026년 1월 5일

---

## 목차
1. [UDP 비디오 전송 및 AI 서버 테스팅](#1-udp-비디오-전송-및-ai-서버-테스팅)
2. [CCTV 모니터링 시스템 (app.py) 분석](#2-cctv-모니터링-시스템-apppy-분석)
3. [시스템 아키텍처 비교](#3-시스템-아키텍처-비교)
4. [실행 가이드](#4-실행-가이드)

---

## 1. UDP 비디오 전송 및 AI 서버 테스팅

### 1.1 시스템 아키텍처

```
┌────────────────────────────────────────────────────────────────┐
│                      Main Server (app.py)                       │
│  - 카메라 캡처 (CCTV1, CCTV2)                                  │
│  - 웹 대시보드 (Flask:5000)                                     │
│  - 데이터베이스 (MySQL)                                         │
│  - 비디오 클립 저장                                             │
│  - UDP 송신: 프레임 → AI 서버 (Port 7000)                      │
│  - UDP 수신: 결과 ← AI 서버 (Port 7001)                        │
└────────────────────────────────────────────────────────────────┘
                           ↕ UDP 통신
┌────────────────────────────────────────────────────────────────┐
│                    AI Server (ai_server.py)                     │
│  - GPU 활용 AI 추론                                            │
│  - 얼굴 인식 (InsightFace buffalo_l)                           │
│  - 낙상 감지 (YOLO v8n)                                        │
│  - 화재/연기 감지 (YOLO v8s)                                   │
│  - UDP 수신: 프레임 ← 메인 서버 (Port 7000)                   │
│  - UDP 송신: 결과 → 메인 서버 (Port 7001)                     │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 UDP 프로토콜 상세

#### 패킷 구조
```python
# 헤더 포맷
UDP_HEADER_FORMAT = "!IHH"  # frame_id, chunk_id, total_chunks
UDP_HEADER_SIZE = 8 bytes

# 패킷 구조
[Header 8B][Payload ~1392B]
```

#### 전송 프로세스
1. **프레임 인코딩**: OpenCV로 JPEG 인코딩 (Quality: 80%)
2. **메타데이터 생성**: 첫 번째 청크에 JSON 메타데이터 포함
   ```json
   {
     "camera_id": "cctv1",
     "timestamp": 1704412800.123,
     "frame_width": 1920,
     "frame_height": 1080
   }
   ```
3. **청킹**: 최대 데이터그램 크기(1400B)에 맞춰 분할
4. **전송**: 각 청크를 순차적으로 UDP로 전송
5. **재조립**: 수신 측에서 모든 청크 수신 후 프레임 재구성

#### 주요 설정값
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `UDP_JPEG_QUALITY` | 80 | JPEG 압축 품질 (0-100) |
| `UDP_MAX_DATAGRAM` | 1400 | 최대 UDP 데이터그램 크기 (bytes) |
| `UDP_FPS` | 0 | 전송 FPS (0=무제한) |
| `UDP_VIDEO_TARGETS` | "" | 대상 서버 목록 (예: "192.168.1.10:7000") |

### 1.3 AI 서버 구성

#### 파일: `ai_server.py` (563 lines)

**주요 기능:**
- **프레임 수신 스레드**: UDP로 비디오 프레임 수신 및 재조립
- **얼굴 인식 스레드**: InsightFace 모델로 얼굴 검출 및 인식
- **낙상 감지 스레드**: YOLO v8n으로 낙상 이벤트 탐지
- **화재/연기 감지 스레드**: YOLO v8s로 화재/연기 탐지
- **결과 전송**: 추론 결과를 메인 서버로 UDP 전송

#### 환경 변수 설정
```bash
# AI 서버 설정
export AI_SERVER_HOST=0.0.0.0
export AI_SERVER_PORT=7000

# 메인 서버 설정 (결과 전송 대상)
export MAIN_SERVER_HOST=127.0.0.1
export MAIN_SERVER_PORT=7001

# 모델 경로
export FALL_MODEL_PATH=../../runs/train/cctv_fall_laying_pose_v11m/weights/best.pt
export FIRE_SMOKE_MODEL_PATH=../../runs/train/fire_smoke_detect_v8s/weights/best.pt

# 신뢰도 임계값
export FALL_CONF=0.5
export FIRE_SMOKE_CONF=0.5
```

#### GPU 가속 지원
```python
# CUDA 우선, CPU 폴백
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_face_app = FaceAnalysis(name=MODEL_NAME, providers=providers)
```

### 1.4 테스트 도구: `test_udp_stream.py`

#### 사용법

**1. 패턴 테스트 (기본)**
```bash
python test_udp_stream.py \
  --host 127.0.0.1 \
  --port 7000 \
  --camera-id test_cam \
  --mode pattern \
  --duration 10
```

**2. 카메라 테스트**
```bash
python test_udp_stream.py \
  --mode camera \
  --camera-index 0 \
  --duration 30
```

**3. 비디오 파일 테스트**
```bash
python test_udp_stream.py \
  --mode video \
  --video-path /path/to/video.mp4 \
  --duration 60
```

#### 테스트 결과 예시
```
[UDP Tester] Initialized
  Target: 127.0.0.1:7000
  Camera ID: test_cam
  Max payload: 1392 bytes

[Frame 0] Sent 3 chunks, JPEG size: 3456 bytes, Shape: (480, 640, 3)
[Frame 1] Sent 3 chunks, JPEG size: 3512 bytes, Shape: (480, 640, 3)
...

[Test] Complete
  Total frames: 100
  Successfully sent: 100
  Success rate: 100.0%
```

### 1.5 프레임 버퍼 관리

#### AI 서버 측
```python
# 카메라별 최신 프레임 저장
_camera_frames = {
    "cctv1": None,  # {frame, timestamp, metadata}
    "cctv2": None,
}

# 미완성 프레임 버퍼 (5초 타임아웃)
_frame_buffers = {}  # frame_id -> {chunks, total_chunks, timestamp, metadata}
```

#### 처리 흐름
1. UDP 패킷 수신
2. 헤더 파싱 (frame_id, chunk_id, total_chunks)
3. 첫 청크인 경우 메타데이터 추출
4. 청크 버퍼에 저장
5. 모든 청크 도착 시 프레임 재조립
6. 디코딩 및 카메라별 버퍼에 저장
7. AI 처리 스레드에서 소비

---

## 2. CCTV 모니터링 시스템 (app.py) 분석

### 2.1 시스템 개요

**파일**: `app.py` (2020 lines)
**프레임워크**: Flask
**데이터베이스**: MySQL

### 2.2 멀티 카메라 지원

#### 카메라 구성
```python
# CCTV1
CCTV1_FACE_SOURCE_ID = "cctv1_face"      # 얼굴 인식용 피드
CCTV1_FALL_SOURCE_ID = "cctv1_fall"      # 낙상 감지용 피드
CCTV1_FIRE_SOURCE_ID = "cctv1_fire"      # 화재 감지용 피드

# CCTV2
CCTV2_FACE_SOURCE_ID = "cctv2_face"
CCTV2_FALL_SOURCE_ID = "cctv2_fall"
CCTV2_FIRE_SOURCE_ID = "cctv2_fire"
```

#### 카메라 소스 매핑
```python
CCTV_SOURCE_MAP = {
    "cctv1": [
        CCTV1_FACE_SOURCE_ID,
        CCTV1_FALL_SOURCE_ID,
        CCTV1_FIRE_SOURCE_ID,
        "face_id",           # 레거시
        "fall_cam",          # 레거시
        "fire_smoke_cam",    # 레거시
        None,
    ],
    "cctv2": [
        CCTV2_FACE_SOURCE_ID,
        CCTV2_FALL_SOURCE_ID,
        CCTV2_FIRE_SOURCE_ID,
    ],
}
```

### 2.3 AI 모델 통합

#### A. 얼굴 인식 (InsightFace)
```python
MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)
SIM_THRESHOLD = 0.35  # 유사도 임계값
```

**주요 기능:**
- 얼굴 검출 및 임베딩 추출
- 갤러리 기반 얼굴 매칭
- 직원/환자 분류
- 미등록 인물 탐지

**색상 코드:**
| 역할 | 색상 (BGR) | 설명 |
|-----|-----------|------|
| Employee | (255, 0, 0) | 직원 - 파란색 |
| Patient | (255, 255, 255) | 환자 - 흰색 |
| Unknown | (0, 0, 255) | 미등록 - 빨간색 |

#### B. 낙상 감지 (YOLO v8n)
```python
FALL_MODEL_PATH = "runs/train/cctv_fall_laying_pose_v11m/weights/best.pt""
FALL_CONF = 0.5
FALL_FPS = 5.0  # 처리 프레임레이트
FALL_LABEL_KEYWORDS = {"fall", "fallen", "lying", "laying"}
```

**감지 로직:**
- 키워드 기반 낙상 이벤트 필터링
- 바운딩 박스 및 신뢰도 추출
- 이벤트 로깅 및 클립 녹화

#### C. 화재/연기 감지 (YOLO v8s)
```python
FIRE_SMOKE_MODEL_PATH = "runs/train/fire_smoke_detect_v8s/weights/best.pt"
FIRE_SMOKE_CONF = 0.5
FIRE_SMOKE_FPS = 5.0
```

**감지 대상:**
- Fire (화재)
- Smoke (연기)

### 2.4 데이터베이스 스키마

#### MySQL 설정
```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "face_id",
    "autocommit": True,
}
```

#### 주요 테이블 (추정)
1. **persons**: 등록된 인물 정보
   - id, name, role (employee/patient)
   
2. **face_logs**: 얼굴 인식 로그
   - timestamp, person_id, camera_id, similarity
   
3. **event_logs**: 이벤트 로그 (낙상, 화재 등)
   - timestamp, event_type, camera_id, clip_path

### 2.5 이벤트 관리 시스템

#### 로그 중복 제거
```python
# 같은 이벤트가 짧은 시간 내 반복 로깅되는 것 방지
LOG_DEDUP_SECONDS = 2              # 얼굴 인식
UNKNOWN_LOG_DEDUP_SECONDS = 5      # 미등록 인물
FALL_LOG_DEDUP_SECONDS = 5         # 낙상 감지
FIRE_SMOKE_LOG_DEDUP_SECONDS = 5   # 화재/연기 감지
```

#### 미등록 인물 추적
```python
UNKNOWN_MIN_SECONDS = 2.0  # 최소 2초 이상 출현 시 로깅
```

#### 비디오 클립 녹화
```python
CLIP_PRE_SECONDS = 4        # 이벤트 전 4초 버퍼
CLIP_POST_SECONDS = 11      # 이벤트 후 11초 녹화
CLIP_COOLDOWN_SECONDS = 8   # 다음 녹화까지 대기 시간
```

**ClipRecorder 클래스:**
- 순환 버퍼로 이벤트 전 영상 보관
- 이벤트 트리거 시 자동 녹화 시작
- 쿨다운 기간으로 중복 녹화 방지
- 저장 경로: `data/event_clips/`

### 2.6 웹 인터페이스 (Flask)

#### 주요 엔드포인트

**비디오 스트림:**
- `/video_feed/<source_id>`: 라이브 비디오 스트림 (MJPEG)
- `/raw_video_feed`: CCTV1 원본 피드
- `/raw_video_feed_2`: CCTV2 원본 피드
- `/fall_video_feed`: CCTV1 낙상 감지 피드
- `/fire_smoke_video_feed`: CCTV1 화재 감지 피드

**관리 페이지:**
- `/`: 메인 대시보드 (4개 화면 모니터링)
- `/register`: 얼굴 등록 페이지
- `/logs`: 이벤트 로그 조회
- `/ai_logs`: AI 처리 로그

**API 엔드포인트:**
- `/api/persons`: 등록 인물 목록
- `/api/logs`: 로그 조회 (JSON)
- `/api/capture`: 스크린샷 캡처
- `/api/test/*`: 모델 테스트 인터페이스

### 2.7 UDP 비디오 브로드캐스팅

#### 설정
```python
UDP_VIDEO_TARGETS = "192.168.1.10:7000,192.168.1.11:7000"  # 다중 타겟 지원
UDP_JPEG_QUALITY = 80
UDP_MAX_DATAGRAM = 1400
UDP_FPS = 0  # 0 = 무제한
```

#### 전송 함수
```python
def udp_send_frame(jpeg_bytes):
    """프레임을 모든 UDP 타겟에 브로드캐스트"""
    # 메타데이터 생성
    metadata = {
        "camera_id": camera_id,
        "timestamp": time.time(),
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0]
    }
    
    # 청킹 및 전송
    for target in _udp_targets:
        for chunk in chunks:
            _udp_sock.sendto(packet, target)
```

### 2.8 글로벌 상태 관리

#### 스레드 안전성
```python
# 락 객체들
_db_lock = threading.Lock()
_camera_lock = threading.Lock()
_camera2_lock = threading.Lock()
_gallery_lock = threading.Lock()
_latest_lock = threading.Lock()
_fall_lock = threading.Lock()
_fire_smoke_lock = threading.Lock()
```

#### 최신 프레임 캐싱
```python
_latest_jpeg = None              # 메인 처리 결과 (얼굴 인식)
_latest_raw_jpeg = None          # CCTV1 원본
_latest_raw_jpeg_2 = None        # CCTV2 원본
_latest_fall_jpeg = None         # CCTV1 낙상 감지 결과
_latest_fire_smoke_jpeg = None   # CCTV1 화재 감지 결과
_latest_fall_jpeg_2 = None       # CCTV2 낙상 감지 결과
_latest_fire_smoke_jpeg_2 = None # CCTV2 화재 감지 결과
```

### 2.9 한글 폰트 지원

```python
LABEL_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
]
```
- PIL로 한글 텍스트 렌더링
- 폰트 캐싱으로 성능 최적화

---

## 3. 시스템 아키텍처 비교

### 3.1 app.py vs app_re.py

| 기능 | app.py (Full) | app_re.py (Simplified) |
|-----|---------------|------------------------|
| **파일 크기** | 2020 lines | 789 lines |
| **카메라 수** | 2 (CCTV1, CCTV2) | 1 |
| **얼굴 인식** | ✅ | ✅ |
| **낙상 감지** | ✅ | ❌ |
| **화재 감지** | ✅ | ❌ |
| **비디오 피드** | 7개 (다중 소스) | 2개 (기본) |
| **이벤트 클립** | 6개 레코더 | 1개 레코더 |
| **테스트 모드** | ✅ (비디오 플레이백) | ❌ |
| **UDP 전송** | ✅ | ✅ |
| **AI 서버 모드** | ✅ | ❌ |

### 3.2 단독 모드 vs AI 서버 모드

#### 단독 모드 (AI_SERVER_ENABLED=false)
```
Camera → Capture → AI Processing (Local) → DB Log → Web UI
```
- 모든 처리를 메인 서버에서 수행
- GPU가 메인 서버에 있어야 함
- 단순한 구조, 낮은 네트워크 부하

#### AI 서버 모드 (AI_SERVER_ENABLED=true)
```
Camera → Capture → UDP Send → AI Server (GPU) → UDP Receive → DB Log → Web UI
```
- AI 처리를 전용 서버로 분산
- GPU 서버 분리 가능
- 확장 가능한 구조
- 네트워크 대역폭 필요

---

## 4. 실행 가이드

### 4.1 사전 준비

#### 1. Python 패키지 설치
```bash
pip install flask opencv-python numpy pillow mysql-connector-python insightface ultralytics
```

#### 2. MySQL 데이터베이스 설정
```bash
mysql -u root -p
CREATE DATABASE face_id;
```

#### 3. 모델 파일 확인
```bash
# 낙상 감지 모델
ls runs/train/cctv_fall_laying_pose_v8n/weights/best.pt

# 화재 감지 모델
ls runs/train/fire_smoke_detect_v8s/weights/best.pt
```

### 4.2 실행 방법

#### 방법 1: 단독 모드 (모든 처리 로컬)
```bash
cd /home/addinedu/Project_ws/deeplearning-repo-2/deeplearning-repo-2/poc/face_id_app

# 환경 변수 설정
export AI_SERVER_ENABLED=false
export CAMERA_SOURCE=0  # 또는 비디오 파일 경로
export CAMERA_SOURCE_2=1

# 실행
python app.py
```

접속: http://localhost:5000

#### 방법 2: AI 서버 모드 (분산 처리)

**터미널 1: AI 서버 시작**
```bash
cd /home/addinedu/Project_ws/deeplearning-repo-2/deeplearning-repo-2/poc/face_id_app

# 환경 변수 설정
export AI_SERVER_HOST=0.0.0.0
export AI_SERVER_PORT=7000
export MAIN_SERVER_HOST=127.0.0.1
export MAIN_SERVER_PORT=7001

# 실행
python ai_server.py
```

**터미널 2: 메인 서버 시작**
```bash
cd /home/addinedu/Project_ws/deeplearning-repo-2/deeplearning-repo-2/poc/face_id_app

# 환경 변수 설정
export AI_SERVER_ENABLED=true
export AI_SERVER_HOST=127.0.0.1
export AI_SERVER_PORT=7000
export MAIN_SERVER_HOST=0.0.0.0
export MAIN_SERVER_PORT=7001
export CAMERA_SOURCE=0
export CAMERA_SOURCE_2=1

# 실행
python main.py
```

접속: http://localhost:5000

#### 방법 3: UDP 테스트
```bash
# AI 서버 시작 (터미널 1)
python ai_server.py

# 테스트 클라이언트 실행 (터미널 2)
python test_udp_stream.py \
  --host 127.0.0.1 \
  --port 7000 \
  --camera-id test_cam \
  --mode pattern \
  --duration 10
```

### 4.3 환경 변수 전체 목록

#### 공통
```bash
# 카메라 소스
export CAMERA_SOURCE=0                    # 카메라 인덱스 또는 비디오 경로
export CAMERA_SOURCE_2=1                  # CCTV2 소스
export CAMERA_BY_ID_MATCH=""              # v4l2 장치 ID 매칭
export CAMERA2_BY_ID_MATCH=""

# 모델 경로
export FALL_MODEL_PATH="runs/train/cctv_fall_laying_pose_v8n/weights/best.pt"
export FIRE_SMOKE_MODEL_PATH="runs/train/fire_smoke_detect_v8s/weights/best.pt"

# 모델 설정
export FALL_CONF=0.5                      # 낙상 감지 신뢰도
export FIRE_SMOKE_CONF=0.5                # 화재 감지 신뢰도
export FALL_FPS=5.0                       # 낙상 감지 FPS
export FIRE_SMOKE_FPS=5.0                 # 화재 감지 FPS

# 기타
export UNKNOWN_MIN_SECONDS=2.0            # 미등록 인물 최소 출현 시간
```

#### UDP 전송 관련
```bash
export UDP_VIDEO_TARGETS="192.168.1.10:7000,192.168.1.11:7000"
export UDP_JPEG_QUALITY=80
export UDP_MAX_DATAGRAM=1400
export UDP_FPS=0
```

#### AI 서버 모드
```bash
export AI_SERVER_ENABLED=true
export AI_SERVER_HOST=127.0.0.1
export AI_SERVER_PORT=7000
export MAIN_SERVER_HOST=127.0.0.1
export MAIN_SERVER_PORT=7001
```

### 4.4 디렉토리 구조

```
poc/face_id_app/
├── app.py                    # 메인 서버 (Full, 2020 lines)
├── main.py                   # 메인 서버 (AI 서버 모드용)
├── ai_server.py              # AI 추론 서버 (563 lines)
├── test_udp_stream.py        # UDP 전송 테스트 도구
├── requirements.txt          # Python 패키지 목록
├── README_AI_SERVER.md       # AI 서버 가이드
├── templates/
│   ├── index.html            # 메인 대시보드
│   ├── register.html         # 얼굴 등록
│   ├── ai_logs.html          # AI 로그 뷰어
│   └── test.html             # 테스트 인터페이스
├── static/
│   └── style.css
└── data/
    ├── face_registry/        # 얼굴 DB
    ├── face_logs/            # 로그 파일
    ├── event_clips/          # 이벤트 비디오 클립
    ├── captures/             # 스크린샷
    └── test_videos/          # 테스트 비디오
```

---

## 5. 트러블슈팅

### 5.1 UDP 전송 문제

#### 증상: 프레임이 AI 서버에 도착하지 않음
```bash
# 방화벽 확인
sudo ufw status
sudo ufw allow 7000/udp
sudo ufw allow 7001/udp

# 네트워크 인터페이스 확인
ip addr show
```

#### 증상: 프레임 손실 발생
- UDP 버퍼 크기 증가:
  ```python
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # 1MB
  ```
- JPEG 품질 낮추기: `UDP_JPEG_QUALITY=60`
- 프레임 전송 속도 제한: `UDP_FPS=10`

### 5.2 AI 모델 로딩 실패

#### 증상: CUDA 오류
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

#### 증상: 모델 파일 없음
- 모델 경로 확인: `echo $FALL_MODEL_PATH`
- 상대 경로 대신 절대 경로 사용

### 5.3 데이터베이스 연결 오류

```bash
# MySQL 서비스 확인
sudo systemctl status mysql

# 데이터베이스 생성
mysql -u root -p1234 -e "CREATE DATABASE IF NOT EXISTS face_id;"

# 권한 확인
mysql -u root -p1234 -e "SHOW GRANTS FOR 'root'@'localhost';"
```

### 5.4 카메라 연결 문제

```bash
# v4l2 장치 목록
v4l2-ctl --list-devices

# 카메라 테스트
ffplay /dev/video0
```

---

## 6. 성능 최적화

### 6.1 GPU 활용

- **InsightFace**: 자동으로 CUDA 사용 (가능 시)
- **YOLO**: `device='cuda:0'` 또는 `device='cpu'` 지정 가능
- **배치 처리**: 여러 프레임을 한 번에 처리 (구현 필요)

### 6.2 프레임 스킵

```python
# FPS 제한으로 처리 부하 감소
FALL_FPS = 5.0           # 초당 5프레임만 처리
FIRE_SMOKE_FPS = 5.0
```

### 6.3 네트워크 최적화

- **UDP 청크 크기**: 1400B (MTU 1500 고려)
- **JPEG 품질**: 80% (대역폭 vs 품질 균형)
- **로컬 네트워크**: 가능한 Gigabit Ethernet 사용

### 6.4 메모리 관리

- **프레임 버퍼**: 5초 타임아웃으로 미완성 프레임 정리
- **클립 버퍼**: 순환 버퍼로 메모리 사용량 제한
- **갤러리 크기**: 등록 인물 수에 비례하여 메모리 사용

---

## 7. 향후 개선 사항

### 7.1 기능 추가
- [ ] 다중 AI 서버 지원 (로드 밸런싱)
- [ ] WebSocket 기반 실시간 알림
- [ ] 모바일 앱 연동
- [ ] 클라우드 백업
- [ ] 통계 대시보드

### 7.2 성능 개선
- [ ] 배치 추론 (여러 프레임 동시 처리)
- [ ] 모델 양자화 (INT8)
- [ ] TensorRT 최적화
- [ ] 비동기 DB 쓰기

### 7.3 안정성 향상
- [ ] 자동 재연결 (카메라, DB, AI 서버)
- [ ] 헬스 체크 엔드포인트
- [ ] 로그 로테이션
- [ ] 에러 알림 시스템

---

## 부록: 코드 스니펫

### A. UDP 프레임 전송 (메인 서버)

```python
def udp_send_frame(jpeg_bytes):
    global _udp_frame_id, _udp_targets, _udp_sock, _udp_max_payload
    
    if not _udp_targets:
        return
    
    # 메타데이터 생성
    metadata = {
        "camera_id": "cctv1",
        "timestamp": time.time(),
        "frame_width": 1920,
        "frame_height": 1080
    }
    meta_json = json.dumps(metadata, ensure_ascii=True).encode('utf-8')
    meta_len = struct.pack("!I", len(meta_json))
    
    # 청킹
    first_chunk_meta_size = 4 + len(meta_json)
    first_chunk_jpeg_size = _udp_max_payload - first_chunk_meta_size
    
    remaining = len(jpeg_bytes) - first_chunk_jpeg_size
    additional_chunks = max(0, (remaining + _udp_max_payload - 1) // _udp_max_payload)
    total_chunks = 1 + additional_chunks
    
    # 첫 번째 청크 전송
    header = struct.pack(UDP_HEADER_FORMAT, _udp_frame_id, 0, total_chunks)
    payload = meta_len + meta_json + jpeg_bytes[:first_chunk_jpeg_size]
    packet = header + payload
    
    for target in _udp_targets:
        _udp_sock.sendto(packet, target)
    
    # 나머지 청크 전송
    offset = first_chunk_jpeg_size
    for chunk_id in range(1, total_chunks):
        start = offset
        end = start + _udp_max_payload
        header = struct.pack(UDP_HEADER_FORMAT, _udp_frame_id, chunk_id, total_chunks)
        packet = header + jpeg_bytes[start:end]
        
        for target in _udp_targets:
            _udp_sock.sendto(packet, target)
        
        offset = end
    
    _udp_frame_id = (_udp_frame_id + 1) & 0xFFFFFFFF
```

### B. UDP 프레임 수신 (AI 서버)

```python
def receive_frames():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((AI_SERVER_HOST, AI_SERVER_PORT))
    sock.settimeout(1.0)
    
    while not _shutdown_event.is_set():
        try:
            data, addr = sock.recvfrom(65535)
            
            # 헤더 파싱
            header = data[:UDP_HEADER_SIZE]
            frame_id, chunk_id, total_chunks = struct.unpack(UDP_HEADER_FORMAT, header)
            payload = data[UDP_HEADER_SIZE:]
            
            # 첫 청크: 메타데이터 추출
            if chunk_id == 0:
                meta_len = struct.unpack("!I", payload[:4])[0]
                meta_json = payload[4:4+meta_len].decode('utf-8')
                meta = json.loads(meta_json)
                jpeg_data = payload[4+meta_len:]
                
                _frame_buffers[frame_id] = {
                    "chunks": {0: jpeg_data},
                    "total_chunks": total_chunks,
                    "timestamp": time.time(),
                    "metadata": meta
                }
            else:
                # 일반 청크
                if frame_id not in _frame_buffers:
                    _frame_buffers[frame_id] = {
                        "chunks": {},
                        "total_chunks": total_chunks,
                        "timestamp": time.time(),
                        "metadata": {}
                    }
                _frame_buffers[frame_id]["chunks"][chunk_id] = payload
            
            # 프레임 완성 확인
            buffer = _frame_buffers[frame_id]
            if len(buffer["chunks"]) == buffer["total_chunks"]:
                # 재조립
                jpeg_bytes = b"".join(buffer["chunks"][i] for i in range(buffer["total_chunks"]))
                
                # 디코딩
                image = np.frombuffer(jpeg_bytes, np.uint8)
                frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    camera_id = buffer["metadata"].get("camera_id", "cctv1")
                    
                    # 프레임 버퍼 업데이트
                    with _camera_locks[camera_id]:
                        _camera_frames[camera_id] = {
                            "frame": frame,
                            "timestamp": time.time(),
                            "metadata": buffer["metadata"]
                        }
                
                # 정리
                del _frame_buffers[frame_id]
            
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[udp] Error: {e}")
    
    sock.close()
```

### C. 얼굴 인식 처리

```python
def process_face_recognition(frame, source_id):
    if _face_app is None or _gallery_embeddings is None:
        return []
    
    faces = _face_app.get(frame)
    results = []
    
    with _gallery_lock:
        embeddings = _gallery_embeddings
        meta = _gallery_meta
    
    for face in faces:
        embedding = l2_normalize(face.embedding)
        
        # 유사도 계산
        person = None
        similarity = -1.0
        if embeddings.size > 0:
            sims = embeddings @ embedding
            idx = int(np.argmax(sims))
            similarity = float(sims[idx])
            if similarity >= SIM_THRESHOLD:
                person = meta[idx]
        
        result = {
            "bbox": [float(v) for v in face.bbox],
            "similarity": similarity,
        }
        
        if person:
            result["person"] = {
                "id": person["id"],
                "name": person["name"],
                "role": person["role"],
            }
        else:
            result["person"] = None
        
        results.append(result)
    
    return results
```

---

## 참고 자료

- **InsightFace**: https://github.com/deepinsight/insightface
- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **Flask**: https://flask.palletsprojects.com/
- **OpenCV**: https://opencv.org/
