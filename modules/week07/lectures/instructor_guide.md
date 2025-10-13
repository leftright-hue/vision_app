# Week 7: 행동인식 - 강사 가이드
## Instructor's Guide for Action Recognition

---

## 📋 강의 개요

### 시간 배분 (90분)
- **도입** (10분): 행동인식 개요 및 응용 분야
- **이론** (20분): 핵심 기술 및 아키텍처
- **MediaPipe 실습** (25분): Open Source 데모
- **Google API 실습** (25분): Cloud 데모
- **비교 및 정리** (10분): Q&A 포함

### 학습 목표
1. 행동인식의 개념과 응용 분야 이해
2. MediaPipe를 활용한 실시간 처리 구현
3. Google Video Intelligence API 활용법 습득
4. 두 접근법의 장단점 비교 및 선택 기준 이해

---

## 🎯 핵심 포인트

### 강조할 내용
1. **시공간 정보의 중요성**: 이미지 vs 비디오의 차이
2. **실시간 vs 정확도 트레이드오프**: 적절한 도구 선택
3. **비용 고려사항**: Open Source vs Cloud API
4. **프라이버시**: 로컬 처리 vs 클라우드 처리

### 주의사항
- MediaPipe 설치 문제 대비 (특히 Windows)
- Google API 키 설정 복잡성
- 실시간 웹캠 데모 백업 플랜 준비

---

## 🔧 사전 준비

### 환경 설정 체크리스트
```bash
# 1. 필수 패키지 설치 확인
pip install mediapipe opencv-python numpy
pip install google-cloud-videointelligence
pip install streamlit matplotlib

# 2. 테스트 실행
python modules/week07/labs/lab06_mediapipe_google_demo.py

# 3. Streamlit 앱 테스트
streamlit run test_week7_action.py
```

### 샘플 비디오 준비
- 운동 비디오 (스쿼트, 푸시업)
- 제스처 비디오 (손동작)
- 일반 행동 비디오 (걷기, 뛰기)

---

## 📚 강의 진행 가이드

### Part 1: 도입 (10분)

#### 시작 질문
"유튜브나 틱톡이 어떻게 운동 동작을 자동으로 인식할까요?"

#### 핵심 메시지
- 행동인식 = 시간 + 공간 정보
- 실생활 응용 예시 (피트니스 앱, 보안 카메라)

#### 데모
```python
# 간단한 프레임 차이 시각화
import cv2
import numpy as np

# 두 프레임 간 차이 보여주기
diff = cv2.absdiff(frame1, frame2)
cv2.imshow('Motion', diff)
```

---

### Part 2: MediaPipe 실습 (25분)

#### 라이브 코딩 순서

1. **기본 포즈 감지** (5분)
```python
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# 웹캠에서 포즈 감지
```

2. **랜드마크 시각화** (5분)
```python
# 33개 랜드마크 표시
mp_drawing.draw_landmarks(image, results.pose_landmarks)
```

3. **운동 카운터 구현** (10분)
```python
# 각도 계산 및 카운팅 로직
angle = calculate_angle(hip, knee, ankle)
if angle < 90:
    counter += 1
```

4. **실시간 데모** (5분)
- 학생 자원자와 함께 스쿼트 카운팅

#### 트러블슈팅
- 웹캠 없을 경우: 녹화된 비디오 사용
- MediaPipe 설치 실패: 시뮬레이션 모드 사용

---

### Part 3: Google API 실습 (25분)

#### 설정 단계별 설명 (10분)

1. **Google Cloud Console 접속**
   - 프로젝트 생성 화면 공유
   - API 활성화 과정 시연

2. **인증 키 생성**
   - 서비스 계정 생성
   - JSON 키 다운로드
   - 환경변수 설정

3. **API 호출 데모** (10분)
```python
from google.cloud import videointelligence

# 비디오 분석 요청
operation = client.annotate_video(
    request={"features": [...], "input_uri": "..."}
)
```

4. **결과 해석** (5분)
   - 레이블 신뢰도 이해
   - 시간 구간 해석
   - JSON 응답 구조

#### 시뮬레이션 모드
API 키 없는 학생을 위한 모의 결과 제공

---

## 💻 실습 운영

### Lab 파일 실행 순서

1. **개별 실습** (15분)
```bash
cd modules/week07/labs
python lab06_mediapipe_google_demo.py
```

2. **Streamlit 웹 앱** (10분)
```bash
streamlit run test_week7_action.py
```

### 페어 프로그래밍
- 2명씩 팀 구성
- 한 명은 MediaPipe, 한 명은 Google API
- 결과 비교 및 토론

---

## 🎓 평가 및 과제

### 형성평가 (수업 중)
- MediaPipe로 간단한 제스처 인식
- Google API 레이블 해석

### 과제 옵션

#### 초급 과제
"MediaPipe로 3가지 운동 카운터 만들기"

#### 중급 과제
"Google API로 비디오 하이라이트 생성"

#### 고급 과제
"MediaPipe + Google API 하이브리드 시스템"

### 평가 기준
- 코드 완성도 (40%)
- 정확도 (30%)
- 창의성 (20%)
- 문서화 (10%)

---

## 🚨 일반적인 문제 해결

### 1. MediaPipe 설치 오류
```bash
# Windows
pip install mediapipe --no-deps
pip install opencv-python numpy protobuf

# Mac M1/M2
pip install mediapipe-silicon
```

### 2. Google API 인증 실패
- 환경변수 확인: `echo $GOOGLE_APPLICATION_CREDENTIALS`
- 키 파일 경로 절대경로 사용
- API 활성화 여부 확인

### 3. 웹캠 접근 오류
- 권한 설정 확인
- 다른 프로그램에서 사용 중인지 확인
- 가상 카메라 사용 고려

### 4. 메모리 부족
- 비디오 해상도 감소
- 프레임 샘플링 증가
- 처리 구간 제한

---

## 📊 강의 효과 측정

### 즉각적 피드백
- Mentimeter 실시간 퀴즈
- Kahoot 게임형 평가
- Slido Q&A

### 이해도 체크 질문
1. "MediaPipe와 Google API 중 어떤 걸 선택하시겠습니까? 왜?"
2. "실시간 피트니스 앱을 만든다면 어떤 기술을?"
3. "프라이버시가 중요한 경우 어떤 접근법을?"

---

## 🎯 학습 성과 지표

### 성공 기준
- [ ] 80% 이상 학생이 MediaPipe 데모 실행
- [ ] 70% 이상 학생이 운동 카운터 구현
- [ ] 60% 이상 학생이 Google API 이해
- [ ] 전체 학생이 두 방식 차이 설명 가능

### 후속 조치
- 어려워하는 학생 개별 지도
- 추가 자료 제공
- Office Hour 활용 권장

---

## 📚 추가 자료

### 심화 학습
- [MediaPipe 공식 튜토리얼](https://google.github.io/mediapipe/)
- [Google Colab 노트북](https://colab.research.google.com)
- [Kaggle 행동인식 대회](https://www.kaggle.com/competitions)

### 참고 도서
- "Deep Learning for Computer Vision" - Rajalingappaa Shanmugamani
- "Hands-On Computer Vision with TensorFlow 2" - Benjamin Planche

### 온라인 강의
- Coursera: "Deep Learning Specialization"
- Fast.ai: "Practical Deep Learning"

---

## 💡 Teaching Tips

### Do's ✅
- 실제 데모 많이 보여주기
- 학생 참여 유도 (웹캠 사용)
- 실패 사례도 보여주고 디버깅
- 비용 관련 현실적 조언
- 윤리적 이슈 언급

### Don'ts ❌
- 이론만 장시간 설명
- 설치 문제로 시간 낭비
- 복잡한 수식 강조
- 완벽한 결과만 보여주기
- API 키 공개

---

## 📝 체크리스트

### 강의 전
- [ ] 모든 데모 코드 테스트
- [ ] 백업 비디오 준비
- [ ] API 키 설정 확인
- [ ] 프레젠테이션 슬라이드 확인
- [ ] 네트워크 연결 확인

### 강의 중
- [ ] 시간 배분 준수
- [ ] 실습 진행 상황 확인
- [ ] 질문 시간 확보
- [ ] 핵심 개념 반복

### 강의 후
- [ ] 과제 공지
- [ ] 추가 자료 업로드
- [ ] 피드백 수집
- [ ] 다음 주 예고

---

## 🔗 유용한 링크 모음

### 코드 저장소
- GitHub: `https://github.com/your-course/week07`
- Colab: 실습 노트북

### 문서
- [강의 슬라이드](lecture_slides.md)
- [학생용 README](../README.md)
- [Lab 가이드](../labs/)

### 지원
- Slack: #week7-support
- Email: instructor@course.edu
- Office Hours: Tue/Thu 2-4 PM

---

## 마무리

이 가이드는 Week 7 행동인식 강의를 효과적으로 진행하기 위한 참고 자료입니다.
학생들의 수준과 관심사에 따라 유연하게 조정하시기 바랍니다.

**Remember**: The goal is not perfection, but understanding! 🎯

---

*Last Updated: 2024년 1월*
*Version: 1.0*