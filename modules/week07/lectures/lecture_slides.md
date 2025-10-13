# Week 7: 행동인식 (Action Recognition)
## 🎬 실시간 비디오 분석과 행동 인식

---

# 목차

1. **행동인식 개요** - 기본 개념과 응용 분야
2. **핵심 기술** - 주요 아키텍처와 알고리즘
3. **MediaPipe 실습** - Open Source 실시간 인식
4. **Google Video Intelligence API** - Cloud 기반 분석
5. **실습 및 데모** - Hands-on Labs
6. **프로젝트 과제** - 실전 응용

---

# Part 1: 행동인식 개요
## 📚 Action Recognition Introduction

---

## 행동인식이란?

### 정의
> **비디오에서 사람이나 객체의 행동을 자동으로 분류하는 컴퓨터 비전 기술**

### 핵심 차이점

| 구분 | 이미지 분류 | 행동 인식 |
|------|------------|-----------|
| **입력** | 단일 이미지 | 비디오 시퀀스 |
| **차원** | 2D (H×W×C) | 3D (T×H×W×C) |
| **정보** | 공간(Spatial) | 시공간(Spatiotemporal) |
| **예시** | "사람이 있다" | "사람이 뛰고 있다" |

💡 **핵심**: 시간적 변화를 이해하는 것이 관건!

---

## 응용 분야

### 1. 🏥 헬스케어 & 피트니스
- 운동 자세 교정
- 재활 모니터링
- 낙상 감지 (노인 케어)

### 2. 🔒 보안 & 감시
- 이상 행동 감지
- 폭력/싸움 감지
- 무단 침입 감지

### 3. 🎮 엔터테인먼트
- 제스처 기반 게임
- AR/VR 인터랙션
- 스포츠 분석

### 4. 🚗 자율주행
- 보행자 행동 예측
- 교통 상황 인식

---

## 기술적 도전과제

### 주요 과제

1. **높은 연산량**
   - 비디오 = 이미지 × 시간
   - 실시간 처리 어려움

2. **시간적 모호성**
   - 같은 행동, 다른 속도
   - 부분 가림, 각도 변화

3. **데이터 라벨링**
   - 시작/끝 지점 애매
   - 주관적 판단 개입

4. **메모리 제약**
   - 긴 비디오 처리
   - 모바일 환경 제한

---

# Part 2: 핵심 기술
## 🏗️ Core Technologies

---

## 주요 아키텍처 진화

```
2014: Two-Stream Networks (공간+시간 분리)
  ↓
2015: 3D CNN (C3D) (3차원 컨볼루션)
  ↓
2017: I3D (Inflated 3D) (2D→3D 확장)
  ↓
2021: TimeSformer (Transformer 적용)
  ↓
2022: VideoMAE (자기지도학습)
  ↓
2024: MediaPipe + Cloud APIs (실용화)
```

---

## 1. 3D CNN (C3D)

### 개념
- 2D Conv를 3D로 확장
- 시간 차원 추가 처리

### 구조
```python
# 2D Convolution (이미지)
Conv2D(filters=64, kernel_size=(3,3))

# 3D Convolution (비디오)
Conv3D(filters=64, kernel_size=(3,3,3))
#                              ↑ 시간축
```

### 장단점
- ✅ 직관적, 구현 간단
- ✅ End-to-end 학습
- ❌ 메모리 사용량 많음
- ❌ 계산 비용 높음

---

## 2. Two-Stream Networks

### 아키텍처

```
입력 비디오
    ├── RGB 프레임 → Spatial CNN → 외형 특징
    │                               ↓
    └── Optical Flow → Temporal CNN → 움직임 특징
                                    ↓
                              Late Fusion → 행동 분류
```

### Optical Flow란?
- 연속 프레임 간 픽셀 이동 벡터
- 움직임 패턴 캡처

### 시각화
- 🔴 빨강: 오른쪽 이동
- 🔵 파랑: 왼쪽 이동
- 🟢 초록: 아래 이동
- 🟡 노랑: 위 이동

---

## 3. Video Transformers

### TimeSformer (2021)
```python
# Divided Space-Time Attention
# 1. Temporal Attention (시간축)
# 2. Spatial Attention (공간축)
# 효율적 계산, 장거리 의존성
```

### VideoMAE (2022)
```python
# Masked Autoencoding
# 1. 90% 패치 마스킹
# 2. 복원 학습
# 3. 적은 데이터로 고성능
```

### 장점
- ✅ 장거리 시간 관계 포착
- ✅ Pre-training 효과적
- ❌ 계산 비용 여전히 높음

---

# Part 3: MediaPipe 실습
## 🔧 Open Source Real-time Recognition

---

## MediaPipe 소개

### Google의 오픈소스 ML 프레임워크

```python
# 설치
pip install mediapipe opencv-python

# 주요 솔루션
- Pose: 33개 신체 랜드마크
- Hands: 21개 손 랜드마크
- Face: 468개 얼굴 랜드마크
- Holistic: 통합 추적
```

### 특징
- ✅ **실시간**: 30+ FPS (CPU)
- ✅ **오프라인**: 인터넷 불필요
- ✅ **무료**: 완전 무료
- ✅ **크로스플랫폼**: Win/Mac/Linux/Mobile

---

## Lab Demo 1: 운동 카운터
### 🏋️ Exercise Counter with MediaPipe

```bash
# 실행 방법
cd modules/week07/labs
python lab06_mediapipe_google_demo.py
# 옵션 1 선택: MediaPipe 실시간 포즈 감지
```

### 핵심 코드
```python
import mediapipe as mp
import cv2

# 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 비디오 처리
cap = cv2.VideoCapture(0)  # 웹캠

while cap.isOpened():
    ret, frame = cap.read()

    # 포즈 감지
    results = pose.process(frame)

    if results.pose_landmarks:
        # 무릎 각도 계산
        angle = calculate_angle(hip, knee, ankle)

        # 스쿼트 카운팅
        if angle < 90:
            state = "down"
        elif angle > 160 and state == "down":
            counter += 1
            state = "up"
```

---

## 실시간 데모: 스쿼트 카운터

### 동작 원리

```
1. 신체 랜드마크 검출 (33개 포인트)
   ↓
2. 관절 각도 계산 (Hip-Knee-Ankle)
   ↓
3. 상태 머신
   - UP 상태: 각도 > 160°
   - DOWN 상태: 각도 < 90°
   ↓
4. 상태 전환 시 카운트 증가
```

### 주요 랜드마크
- 23: LEFT_HIP (왼쪽 엉덩이)
- 25: LEFT_KNEE (왼쪽 무릎)
- 27: LEFT_ANKLE (왼쪽 발목)

---

## Lab Demo 2: 제스처 인식
### 👋 Hand Gesture Recognition

```python
# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# 제스처 분류
def classify_gesture(landmarks):
    # 손가락 상태 확인
    fingers_up = count_fingers(landmarks)

    if fingers_up == [0,0,0,0,0]:
        return "주먹 ✊"
    elif fingers_up == [0,1,1,0,0]:
        return "V 사인 ✌️"
    elif fingers_up == [1,1,1,1,1]:
        return "보 ✋"
    # ... 더 많은 제스처
```

### 응용
- 프레젠테이션 컨트롤
- 스마트 홈 제어
- 수화 번역

---

## MediaPipe 성능 최적화

### 프레임 샘플링
```python
frame_counter = 0
SAMPLE_RATE = 5  # 매 5프레임마다 처리

if frame_counter % SAMPLE_RATE == 0:
    # MediaPipe 처리
    results = pose.process(frame)
```

### 해상도 다운스케일
```python
# 원본: 1920x1080 → 처리: 640x480
small_frame = cv2.resize(frame, (640, 480))
results = pose.process(small_frame)
```

### 신뢰도 임계값
```python
pose = mp_pose.Pose(
    min_detection_confidence=0.7,  # 높이면 정확도↑ 속도↓
    min_tracking_confidence=0.5
)
```

---

# Part 4: Google Video Intelligence API
## ☁️ Cloud-based Video Analysis

---

## Google Video Intelligence 개요

### 클라우드 기반 비디오 분석 서비스

```bash
# 설치
pip install google-cloud-videointelligence

# 인증 설정
export GOOGLE_APPLICATION_CREDENTIALS="key.json"
```

### 주요 기능
- **Label Detection**: 400+ 행동/객체 레이블
- **Shot Change**: 장면 전환 감지
- **Person Detection**: 사람 추적
- **Object Tracking**: 객체 추적
- **Face Detection**: 얼굴 감지
- **Speech Transcription**: 음성→텍스트
- **Explicit Content**: 부적절 콘텐츠

---

## Lab Demo 3: Cloud API 분석
### ☁️ Video Analysis with Google API

```bash
# 실행 방법
cd modules/week07/labs
python lab06_mediapipe_google_demo.py
# 옵션 2 선택: Google Video Intelligence API
```

### 핵심 코드
```python
from google.cloud import videointelligence

# 클라이언트 초기화
client = videointelligence.VideoIntelligenceServiceClient()

# 분석 요청
operation = client.annotate_video(
    request={
        "features": [
            videointelligence.Feature.LABEL_DETECTION,
            videointelligence.Feature.PERSON_DETECTION
        ],
        "input_uri": "gs://bucket/video.mp4"  # 또는 input_content
    }
)

# 결과 대기 (비동기)
result = operation.result(timeout=180)

# 레이블 추출
for label in result.annotation_results[0].segment_label_annotations:
    print(f"{label.entity.description}: {label.segments[0].confidence:.1%}")
```

---

## API 응답 예시

### Label Detection 결과
```json
{
  "entity": {
    "entityId": "/m/04rlf",
    "description": "running",
    "languageCode": "en-US"
  },
  "segments": [
    {
      "segment": {
        "startTimeOffset": "1.5s",
        "endTimeOffset": "4.2s"
      },
      "confidence": 0.92
    }
  ]
}
```

### 감지 가능한 행동들
- Walking, Running, Jumping
- Dancing, Swimming, Cycling
- Playing sports (축구, 농구 등)
- Cooking, Eating, Drinking
- Fighting, Falling
- 400+ 더 많은 행동...

---

## 가격 및 성능

### 요금제 (2024년 기준)
- **무료**: 월 1,000분
- **유료**: $0.10/분 (Label Detection)
- **추가 기능**: $0.05~0.15/분

### 성능 벤치마크
| 메트릭 | 값 |
|--------|-----|
| 정확도 | 85-95% |
| 처리 시간 | 30초~2분/분 |
| 동시 처리 | 100+ 비디오 |
| 최대 크기 | 50GB/파일 |

---

# Part 5: 실습 가이드
## 🧪 Hands-on Lab Sessions

---

## Lab 구조 및 실행 방법

### 📁 파일 구조
```
week07/labs/
├── lab01_video_basics.py          # 비디오 기초
├── lab02_temporal_features.py     # Optical Flow
├── lab03_action_classification.py # 분류 모델
├── lab04_realtime_recognition.py  # 실시간 인식
├── lab05_practical_apps.py        # 응용 예제
└── lab06_mediapipe_google_demo.py # 통합 데모 ⭐
```

### 실행 순서
```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 기초 실습
python lab01_video_basics.py

# 3. MediaPipe 실습
python lab06_mediapipe_google_demo.py
# 옵션 1 선택

# 4. Google API 실습
python lab06_mediapipe_google_demo.py
# 옵션 2 선택
```

---

## Streamlit 웹 앱 실행

### 메인 앱에서 Week 7 실행
```bash
# 프로젝트 루트에서
streamlit run app.py
# Week 7 선택 → 각 탭 탐색
```

### Week 7만 단독 실행
```bash
streamlit run test_week7_action.py
```

### 실시간 모듈만 실행
```bash
cd modules/week07
streamlit run action_recognition_realtime.py
```

---

## 실습 1: 비디오 처리 기초

### 목표
- 비디오를 프레임으로 분해
- Optical Flow 계산
- 움직임 시각화

### 코드 예제
```python
import cv2
import numpy as np

# 비디오 읽기
cap = cv2.VideoCapture('video.mp4')

# 프레임 추출
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Optical Flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, next_gray, None,
    0.5, 3, 15, 3, 5, 1.2, 0
)

# 시각화
magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
```

---

## 실습 2: MediaPipe 운동 앱

### 구현할 기능
1. 실시간 포즈 검출
2. 운동별 카운터 (푸시업, 스쿼트, 런지)
3. 자세 피드백
4. 운동 기록 저장

### 확장 아이디어
```python
class ExerciseTracker:
    def __init__(self):
        self.exercises = {
            'pushup': PushupCounter(),
            'squat': SquatCounter(),
            'lunge': LungeCounter()
        }
        self.history = []

    def process_frame(self, landmarks):
        # 현재 운동 감지
        exercise = self.detect_exercise(landmarks)

        # 카운트 업데이트
        if exercise:
            count = self.exercises[exercise].update(landmarks)

        # 피드백 생성
        feedback = self.generate_feedback(landmarks, exercise)

        return count, feedback
```

---

## 실습 3: Google API 비디오 요약

### 목표
- 비디오 업로드 및 분석
- 주요 장면 추출
- 행동 타임라인 생성
- 하이라이트 생성

### 구현 코드
```python
def create_video_summary(video_path):
    # 1. 비디오 분석
    results = analyze_video(video_path)

    # 2. 중요 구간 추출
    highlights = []
    for label in results.labels:
        if label.confidence > 0.8:
            highlights.append({
                'action': label.description,
                'start': label.start_time,
                'end': label.end_time,
                'confidence': label.confidence
            })

    # 3. 타임라인 생성
    timeline = create_timeline(highlights)

    # 4. 요약 비디오 생성
    create_highlight_reel(video_path, highlights)

    return timeline, highlights
```

---

# Part 6: 비교 분석
## 📊 MediaPipe vs Google Video Intelligence

---

## 상세 비교표

| 특성 | MediaPipe | Google Video Intelligence |
|------|-----------|--------------------------|
| **처리 방식** | 실시간 스트리밍 | 배치 처리 |
| **속도** | 30+ FPS | 1-2분/비디오 |
| **정확도** | 85-90% | 90-95% |
| **비용** | 무료 | $0.10/분 |
| **인터넷** | 불필요 | 필수 |
| **커스터마이징** | 완전 가능 | 제한적 |
| **행동 종류** | 직접 구현 | 400+ 사전정의 |
| **플랫폼** | 모든 플랫폼 | 클라우드만 |
| **확장성** | 제한적 | 무제한 |
| **개인정보** | 로컬 처리 | 클라우드 전송 |

---

## 선택 가이드

### MediaPipe 적합한 경우
✅ 실시간 처리 필요
✅ 오프라인 환경
✅ 비용 민감
✅ 커스터마이징 필요
✅ 프로토타입 개발
✅ 모바일 앱

### Google API 적합한 경우
✅ 대량 비디오 처리
✅ 높은 정확도 필요
✅ 다양한 행동 인식
✅ 기업 환경
✅ 자동화 시스템
✅ 콘텐츠 모더레이션

---

## 하이브리드 접근법

### 최적의 조합
```
실시간 모니터링 → MediaPipe
    ↓
이상 감지 시
    ↓
상세 분석 → Google API
    ↓
결과 저장 및 알림
```

### 구현 예시
```python
class HybridActionRecognizer:
    def __init__(self):
        self.mediapipe = MediaPipeProcessor()
        self.google_api = GoogleVideoAPI()

    def process(self, video_stream):
        # 1차: MediaPipe로 실시간 모니터링
        alerts = self.mediapipe.detect_anomalies(video_stream)

        if alerts:
            # 2차: Google API로 상세 분석
            detailed = self.google_api.analyze(video_stream)
            self.handle_alert(detailed)
```

---

# Part 7: 프로젝트 과제
## 📝 Final Projects

---

## 과제 1: 홈 트레이닝 앱

### 요구사항
- MediaPipe 사용
- 3가지 이상 운동 지원
- 실시간 자세 피드백
- 운동 기록 저장
- 그래프 시각화

### 평가 기준
- 정확도 (30%)
- UI/UX (20%)
- 기능 완성도 (30%)
- 코드 품질 (20%)

### 제출물
```
home_trainer/
├── app.py              # Streamlit 앱
├── pose_detector.py    # MediaPipe 처리
├── exercise_counter.py # 운동 카운터
├── feedback.py         # 피드백 생성
├── database.py         # 기록 저장
└── README.md          # 사용 설명서
```

---

## 과제 2: 보안 카메라 시스템

### 요구사항
- Google Video Intelligence API 활용
- 이상 행동 감지 (폭력, 낙상 등)
- 실시간 알림
- 비디오 하이라이트 생성
- 대시보드 구현

### 구현 힌트
```python
class SecurityMonitor:
    def __init__(self):
        self.alert_labels = [
            'fighting', 'falling',
            'running', 'violence'
        ]

    async def monitor(self, video_feed):
        # 주기적 분석
        results = await self.analyze_async(video_feed)

        # 위험 행동 체크
        for label in results:
            if label.name in self.alert_labels:
                await self.send_alert(label)

        # 하이라이트 생성
        if self.has_incidents(results):
            self.create_highlight(results)
```

---

## 과제 3: 스포츠 분석 도구

### 목표
- 선수 동작 분석
- 팀 전술 분석
- 통계 생성

### 기능 구현
1. **선수 추적**: Person Detection
2. **동작 분석**: Pose Estimation
3. **공 추적**: Object Tracking
4. **하이라이트**: 골/중요 장면
5. **통계**: 이동 거리, 속도

### 기술 스택
- MediaPipe: 개인 동작
- Google API: 전체 분석
- OpenCV: 영상 처리
- Matplotlib: 시각화

---

# Part 8: 추가 리소스
## 📚 Additional Resources

---

## 학습 자료

### 📖 필수 논문
1. [Two-Stream Networks (2014)](https://arxiv.org/abs/1406.2199)
2. [C3D (2015)](https://arxiv.org/abs/1412.0767)
3. [I3D (2017)](https://arxiv.org/abs/1705.07750)
4. [TimeSformer (2021)](https://arxiv.org/abs/2102.05095)
5. [VideoMAE (2022)](https://arxiv.org/abs/2203.12602)

### 🎥 데이터셋
- **Kinetics-400**: 400개 행동, 30만 비디오
- **UCF-101**: 101개 행동, 1.3만 비디오
- **HMDB-51**: 51개 행동, 7천 비디오
- **AVA**: 80개 원자 행동

---

## 유용한 도구 및 라이브러리

### 프레임워크
```python
# MediaPipe
pip install mediapipe

# MMAction2 (행동인식 전문)
pip install mmaction2

# PaddleVideo
pip install paddlevideo

# Detectron2
pip install detectron2
```

### 시각화 도구
- **Weights & Biases**: 실험 추적
- **Tensorboard**: 학습 모니터링
- **Streamlit**: 웹 앱 개발
- **Gradio**: 빠른 데모

---

## FAQ & 트러블슈팅

### Q1: MediaPipe가 느려요
```python
# 해결책
1. 해상도 낮추기 (480p)
2. 프레임 샘플링 (매 3-5 프레임)
3. 신뢰도 임계값 높이기
4. GPU 사용 (가능한 경우)
```

### Q2: Google API 비용이 걱정돼요
```python
# 절약 팁
1. 저해상도 업로드
2. 필요한 feature만 선택
3. 비디오 자르기 (중요 부분만)
4. 캐싱 활용
```

### Q3: 정확도가 낮아요
```python
# 개선 방법
1. 조명 개선
2. 카메라 각도 조정
3. 배경 단순화
4. 여러 모델 앙상블
```

---

## 다음 주 예고

### Week 8: 생성 모델 (Generative Models)

- **GAN**: 이미지 생성
- **Diffusion Models**: Stable Diffusion
- **VAE**: 변분 오토인코더
- **실습**: 이미지 생성 앱 만들기

### 준비사항
```bash
# 패키지 미리 설치
pip install diffusers transformers
pip install torch torchvision
```

---

# 마무리
## 🎯 Key Takeaways

---

## 핵심 정리

### ✅ 배운 내용
1. **행동인식 개념**: 시공간 정보 처리
2. **주요 아키텍처**: 3D CNN, Two-Stream, Transformer
3. **MediaPipe**: 실시간 오픈소스 솔루션
4. **Google API**: 클라우드 기반 분석
5. **실전 응용**: 운동, 보안, 제스처

### 💡 기억할 점
- **적절한 도구 선택**: 실시간 vs 정확도
- **비용 고려**: 오픈소스 vs 클라우드
- **프라이버시**: 로컬 vs 클라우드
- **확장성**: 프로토타입 vs 프로덕션

---

## 실습 체크리스트

### 완료했나요?
- [ ] MediaPipe 설치 및 테스트
- [ ] 운동 카운터 구현
- [ ] Google API 설정
- [ ] 비디오 분석 실행
- [ ] Lab 파일 실습
- [ ] Streamlit 앱 실행

### 도전 과제
- [ ] 새로운 운동 추가
- [ ] 제스처 5개 인식
- [ ] 비디오 요약 생성
- [ ] 하이브리드 시스템 구현

---

## 질문 & 토론

### 💬 생각해볼 문제들

1. **윤리적 고려사항**
   - CCTV 행동 인식의 프라이버시 문제는?
   - 동의 없는 분석의 한계는?

2. **기술적 도전**
   - 실시간 vs 정확도 트레이드오프?
   - 엣지 vs 클라우드 선택 기준?

3. **미래 전망**
   - 행동 인식의 다음 단계는?
   - AI 규제가 미칠 영향은?

---

# 감사합니다! 🙏

## 다음 주에 만나요!

### 연락처
- 📧 Email: instructor@example.com
- 💬 Slack: #week7-action-recognition
- 📝 과제 제출: GitHub Classroom

### Office Hours
- 화요일 14:00-16:00
- 목요일 14:00-16:00
- Zoom 링크는 LMS 참조

---

## 부록: 빠른 참조 코드

### MediaPipe 빠른 시작
```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = pose.process(frame)
    # 처리...
    cv2.imshow('Pose', frame)
    if cv2.waitKey(1) == 27: break
```

### Google API 빠른 시작
```python
from google.cloud import videointelligence

client = videointelligence.VideoIntelligenceServiceClient()
operation = client.annotate_video(
    request={"features": [videointelligence.Feature.LABEL_DETECTION],
             "input_uri": "gs://bucket/video.mp4"})
result = operation.result()
```