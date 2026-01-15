# Face ID App (모듈화 버전)

기존 `face_id_app`의 모든 기능을 유지하면서, 유지보수하기 쉽게 모듈별로 분리한 버전입니다.

## 📁 프로젝트 구조

```
seper/
├── app.py                      # 메인 애플리케이션 (Flask 앱 초기화)
├── config.py                   # 전역 설정 및 상수
├── requirements.txt            # Python 패키지 의존성
├── README.md                   # 이 파일
│
├── routes/                     # 라우트 (메뉴별로 분리)
│   ├── index.py               # 메인 페이지, 비디오 피드
│   ├── register.py            # 얼굴 등록
│   ├── ai_logs.py             # AI 로그 조회
│   └── test.py                # 테스트 기능
│
├── services/                   # 비즈니스 로직
│   ├── face_service.py        # 얼굴 인식 처리
│   ├── video_service.py       # 카메라, 클립 녹화, UDP 전송
│   └── detection_service.py   # 낙상/화재 감지
│
├── utils/                      # 유틸리티
│   └── database.py            # DB 연결 및 초기화
│
├── templates/                  # HTML 템플릿
│   ├── index.html
│   ├── register.html
│   ├── ai_logs.html
│   └── test.html
│
├── static/                     # CSS, JS 등
│   └── style.css
│
└── data/                       # 데이터 저장소
    ├── face_registry/         # 등록된 얼굴 이미지
    ├── face_logs/             # 외부인 감지 로그
    ├── event_clips/           # 이벤트 비디오 클립
    ├── captures/              # 캡처 이미지
    └── test_videos/           # 테스트용 비디오
```

## 🎯 주요 개선사항

### 1. **메뉴별 라우트 분리**
- **index.py**: 메인 페이지, 카메라 피드
- **register.py**: 얼굴 등록 기능
- **ai_logs.py**: AI 로그 조회
- **test.py**: 테스트 및 YouTube 다운로드

### 2. **서비스 레이어 분리**
- **face_service.py**: 얼굴 인식, 갤러리 관리
- **video_service.py**: 카메라 제어, 클립 녹화, UDP 전송
- **detection_service.py**: 낙상 감지, 화재 감지

### 3. **설정 중앙화**
- **config.py**: 모든 설정값을 한 곳에서 관리

### 4. **유틸리티 모듈**
- **database.py**: DB 연결 및 초기화 로직 분리

## 🚀 실행 방법

### 1. 의존성 설치
```bash
cd /home/addinedu/Project_ws/deeplearning-repo-2/deeplearning-repo-2/poc/seper
pip install -r requirements.txt
```

### 2. MySQL 데이터베이스 준비
```bash
# MySQL이 실행 중인지 확인
sudo systemctl status mysql

# 필요시 시작
sudo systemctl start mysql
```

### 3. 애플리케이션 실행
```bash
python app.py
```

### 4. 브라우저에서 접속
```
http://localhost:5000
```

## 🔧 환경 변수 설정

필요에 따라 환경 변수로 설정을 변경할 수 있습니다:

```bash
# 카메라 소스
export CAMERA_SOURCE="auto"              # 또는 "/dev/video0"
export CAMERA_SOURCE_2="auto"

# 낙상 감지
export FALL_MODEL_PATH="yolo11n-pose.pt"
export FALL_CONF="0.5"
export FALL_FPS="5.0"

# 화재/연기 감지
export FIRE_SMOKE_MODEL_PATH="runs/train/fire_smoke_detect_v8s/weights/best.pt"
export FIRE_SMOKE_CONF="0.5"
export FIRE_SMOKE_FPS="5.0"

# UDP 비디오 전송
export UDP_VIDEO_TARGETS="192.168.1.100:6000,192.168.1.101:6000"
export UDP_JPEG_QUALITY="80"
export UDP_FPS="10"

# 외부인 감지
export UNKNOWN_MIN_SECONDS="2.0"
```

## 📝 주요 엔드포인트

### 메인 페이지
- `GET /` - 메인 대시보드
- `GET /video_feed` - 얼굴 인식 비디오 피드
- `GET /video_feed_fall` - 낙상 감지 비디오 피드
- `GET /video_feed_fire_smoke` - 화재/연기 감지 비디오 피드
- `GET /camera_status` - 카메라 상태 확인

### 얼굴 등록
- `GET /register` - 등록 페이지
- `POST /register` - 얼굴 등록 처리
- `POST /capture_frame` - 현재 프레임 캡처
- `GET /reload` - 갤러리 새로고침

### AI 로그
- `GET /ai-logs` - 로그 페이지 (필터링 가능)
- `GET /event-logs` - 로그 API (JSON)
- `GET /clips/<filename>` - 이벤트 클립 재생

### 테스트
- `GET /test` - 테스트 페이지
- `POST /test/start` - 테스트 시작
- `POST /test/stop` - 테스트 중지
- `POST /test/pause` - 일시정지
- `POST /test/resume` - 재개
- `POST /test/seek` - 탐색
- `POST /test/download` - YouTube 다운로드
- `GET /test_feed` - 테스트 비디오 피드

## 🔍 기존 버전과의 차이점

| 항목 | 기존 (face_id_app) | 신규 (seper) |
|------|-------------------|--------------|
| 파일 수 | 1개 (app.py, 2237줄) | 13개 (모듈별 분리) |
| 라우트 관리 | 한 파일에 모두 | Blueprint로 메뉴별 분리 |
| 비즈니스 로직 | app.py에 섞임 | services/ 폴더에 분리 |
| 설정 관리 | 파일 상단에 분산 | config.py에 중앙화 |
| 유지보수성 | 어려움 | 쉬움 |
| 테스트 용이성 | 어려움 | 모듈별 독립 테스트 가능 |
| 코드 재사용 | 제한적 | 서비스 함수 재사용 가능 |

## 🛠️ 개발 가이드

### 새로운 기능 추가 시

1. **새로운 라우트 추가**
   ```python
   # routes/new_feature.py
   from flask import Blueprint
   
   new_feature_bp = Blueprint('new_feature', __name__)
   
   @new_feature_bp.route('/new-feature')
   def new_feature():
       return render_template('new_feature.html')
   ```
   
   ```python
   # app.py에 등록
   from routes.new_feature import new_feature_bp
   app.register_blueprint(new_feature_bp)
   ```

2. **새로운 서비스 로직 추가**
   ```python
   # services/new_service.py
   def new_function():
       # 비즈니스 로직
       pass
   ```

3. **설정 추가**
   ```python
   # config.py
   NEW_SETTING = os.getenv("NEW_SETTING", "default_value")
   ```

### 코드 수정 가이드

- **설정 변경**: `config.py` 수정
- **DB 스키마 변경**: `utils/database.py`의 `init_db()` 수정
- **얼굴 인식 로직**: `services/face_service.py` 수정
- **비디오 처리**: `services/video_service.py` 수정
- **감지 로직**: `services/detection_service.py` 수정
- **UI 변경**: `templates/` 또는 `static/` 수정
- **라우트 추가/수정**: `routes/` 폴더 수정

## 📦 기존 버전 비교

기존 `face_id_app/app.py`와 동일한 기능을 제공하지만, 다음과 같이 개선되었습니다:

✅ **유지보수성**: 각 기능이 독립된 파일로 분리
✅ **확장성**: 새로운 기능 추가가 용이
✅ **가독성**: 코드 구조가 명확
✅ **테스트**: 모듈별 단위 테스트 가능
✅ **협업**: 여러 개발자가 동시 작업 가능

## 🐛 문제 해결

### 카메라 인식 안 됨
```bash
# 카메라 장치 확인
ls -l /dev/video*
ls -l /dev/v4l/by-id/

# 환경 변수로 카메라 지정
export CAMERA_SOURCE="/dev/video0"
```

### DB 연결 오류
```bash
# MySQL 연결 정보 확인
mysql -u root -p

# config.py에서 DB 설정 확인
```

### 모델 로드 실패
```bash
# 낙상 감지 모델 (자동 다운로드됨)
export FALL_MODEL_PATH="yolo11n-pose.pt"

# 화재 감지 모델 (직접 학습한 모델 필요)
export FIRE_SMOKE_MODEL_PATH="runs/train/fire_smoke_detect_v8s/weights/best.pt"
```

## 📄 라이선스

기존 face_id_app과 동일

## 👥 기여

이 모듈화 버전은 기존 코드를 유지보수하기 쉽게 재구성한 것입니다.
