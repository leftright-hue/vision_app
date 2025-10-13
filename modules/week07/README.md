# Week 7: 행동인식 (Action Recognition)

## 📚 개요
비디오에서 사람이나 객체의 행동을 자동으로 분류하는 기술을 학습합니다.

## 🎯 학습 목표
- 행동인식 개념과 주요 아키텍처 이해
- MediaPipe를 활용한 실시간 행동 인식 구현
- Google Video Intelligence API를 통한 클라우드 기반 비디오 분석
- 실전 응용: 운동 카운터, 제스처 인식, 이상 행동 감지

## 📂 파일 구조
```
week07/
├── __init__.py                        # 모듈 초기화
├── action_recognition_module.py       # 메인 모듈
├── action_recognition_realtime.py     # MediaPipe & Google API 구현
├── action_helpers.py                  # 헬퍼 함수
├── labs/
│   ├── lab01_video_basics.py         # 비디오 처리 기초
│   ├── lab02_temporal_features.py    # 시간적 특징 추출
│   ├── lab03_action_classification.py # 행동 분류
│   ├── lab04_realtime_recognition.py  # 실시간 인식
│   └── lab05_practical_apps.py       # 실전 응용
└── README.md                          # 이 문서
```

## 🔧 설치 방법

### 기본 패키지
```bash
pip install opencv-python numpy pillow matplotlib
```

### MediaPipe (Open Source)
```bash
pip install mediapipe
```

### Google Cloud Video Intelligence API
```bash
pip install google-cloud-videointelligence

# API 키 설정
export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
```

### 선택적 패키지 (고급 기능)
```bash
pip install transformers torch  # HuggingFace 모델
```

## 🚀 실행 방법

### Streamlit 앱 실행
```bash
# 전체 앱 실행
streamlit run app.py

# Week 7 테스트
streamlit run test_week7_action.py
```

### Lab 파일 실행
```bash
cd modules/week07/labs
python lab04_realtime_recognition.py  # 웹캠 실시간 인식
```

## 📋 주요 기능

### 1. MediaPipe (Open Source)
- **Pose Detection**: 33개 신체 랜드마크 추적
- **Hand Tracking**: 21개 손 랜드마크 추적
- **Holistic**: 통합 추적 (Pose + Hand + Face)
- **운동 카운팅**: 스쿼트, 푸시업 등 자동 카운트
- **제스처 인식**: 손동작 기반 명령

### 2. Google Video Intelligence API (Cloud)
- **Label Detection**: 400+ 사전 정의된 행동 레이블
- **Shot Change Detection**: 장면 전환 감지
- **Person Detection**: 사람 감지 및 추적
- **Object Tracking**: 객체 추적
- **Explicit Content Detection**: 부적절 콘텐츠 감지

### 3. 비디오 처리 기초
- **프레임 추출**: 비디오를 이미지 시퀀스로 변환
- **Optical Flow**: 프레임 간 움직임 계산
- **시각화**: 움직임을 색상으로 표현

### 4. 사전훈련 모델
- **VideoMAE**: Masked Autoencoding 기반
- **TimeSformer**: Transformer 아키텍처
- **X-CLIP**: CLIP 기반 비디오 모델

## 💡 활용 예시

### MediaPipe 운동 카운터
```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 비디오 처리
cap = cv2.VideoCapture('exercise_video.mp4')
counter = 0
state = "up"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pose 감지
    results = pose.process(frame)

    if results.pose_landmarks:
        # 무릎 각도 계산
        angle = calculate_angle(
            results.pose_landmarks.landmark[23],  # HIP
            results.pose_landmarks.landmark[25],  # KNEE
            results.pose_landmarks.landmark[27]   # ANKLE
        )

        # 스쿼트 카운팅
        if angle < 90 and state == "up":
            state = "down"
        elif angle > 160 and state == "down":
            counter += 1
            state = "up"
```

### Google Video Intelligence API
```python
from google.cloud import videointelligence

client = videointelligence.VideoIntelligenceServiceClient()

# 비디오 분석
operation = client.annotate_video(
    request={
        "features": [
            videointelligence.Feature.LABEL_DETECTION,
            videointelligence.Feature.PERSON_DETECTION
        ],
        "input_uri": "gs://bucket/video.mp4"
    }
)

# 결과 대기
result = operation.result(timeout=180)

# 레이블 출력
for annotation in result.annotation_results:
    for label in annotation.segment_label_annotations:
        print(f"Label: {label.entity.description}")
        print(f"Confidence: {label.segments[0].confidence}")
```

## 🎯 실습 과제

### 과제 1: MediaPipe 운동 트레이너
- 푸시업, 스쿼트, 런지 카운터 구현
- 올바른 자세 피드백 제공
- 운동 기록 저장

### 과제 2: Google API 비디오 요약
- 비디오 업로드 및 분석
- 주요 행동/객체 타임라인 생성
- 하이라이트 구간 추출

### 과제 3: 제스처 기반 컨트롤러
- MediaPipe Hands로 제스처 인식
- 5가지 이상 제스처 분류
- 실시간 명령 실행

## 📊 성능 비교

| 기능 | MediaPipe | Google Video Intelligence |
|------|-----------|-------------------------|
| **실시간 처리** | ✅ 가능 (30+ FPS) | ❌ 배치 처리 |
| **오프라인 동작** | ✅ 가능 | ❌ 인터넷 필요 |
| **비용** | 무료 | 유료 (월 1000분 무료) |
| **정확도** | 중상 | 상 |
| **커스터마이징** | ✅ 가능 | ❌ 제한적 |
| **행동 종류** | 제한적 (직접 구현) | 400+ 사전 정의 |

## 🔍 트러블슈팅

### MediaPipe 설치 오류
```bash
# Windows에서 설치 실패 시
pip install mediapipe --no-deps
pip install opencv-python numpy protobuf

# Mac M1/M2
pip install mediapipe-silicon
```

### Google Cloud 인증 오류
```bash
# 서비스 계정 키 생성
# 1. Google Cloud Console 접속
# 2. IAM & Admin > Service Accounts
# 3. Create Service Account
# 4. Create Key (JSON)

# 환경 변수 설정
export GOOGLE_APPLICATION_CREDENTIALS="key.json"
```

### 메모리 부족
- 비디오 해상도 축소 (480p 권장)
- 프레임 샘플링 (매 5프레임)
- 배치 크기 감소

## 📚 참고 자료

### 논문
- [C3D: Learning Spatiotemporal Features](https://arxiv.org/abs/1412.0767)
- [Two-Stream Convolutional Networks](https://arxiv.org/abs/1406.2199)
- [VideoMAE: Masked Autoencoders](https://arxiv.org/abs/2203.12602)

### 문서
- [MediaPipe 공식 문서](https://google.github.io/mediapipe/)
- [Google Video Intelligence API](https://cloud.google.com/video-intelligence/docs)
- [OpenCV 비디오 처리](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### 데이터셋
- [Kinetics-400](https://www.deepmind.com/open-source/kinetics)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)

## 🤝 기여하기
이슈나 개선사항이 있으면 GitHub 이슈를 생성해주세요.

---

**작성일**: 2024년 1월
**버전**: 1.0.0
**라이선스**: MIT