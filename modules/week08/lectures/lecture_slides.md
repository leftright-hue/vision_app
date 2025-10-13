# Week 8: 고급 감정 인식 (Advanced Emotion Recognition)

## 강의 슬라이드

---

# 📚 8주차 학습 목표

## 오늘 배울 내용

1. **감정 인식의 기초와 발전**
2. **VAD 3차원 감정 모델**
3. **멀티모달 API 활용 (Gemini + GPT-4o)**
4. **시계열 감정 변화 추적**

---

# Part 1: 감정 인식 기초

## 🎭 감정 인식이란?

### 정의
> "인간의 감정 상태를 얼굴 표정, 음성, 텍스트 등에서 자동으로 인식하는 기술"

### 왜 중요한가?
- **Human-AI Interaction**: 감정 인식 챗봇, 가상 비서
- **멘탈 헬스**: 우울증, 불안 감지
- **마케팅**: 광고 반응 분석
- **교육**: 학습자 몰입도 측정
- **보안**: 스트레스, 거짓말 감지

---

## 📊 감정 이론의 발전

### Ekman의 6가지 기본 감정 (1970s)

```
┌─────────────────────────────────────────┐
│  Paul Ekman - 문화 보편적 감정          │
├─────────────────────────────────────────┤
│  😊 Happy (행복)                        │
│  😢 Sad (슬픔)                          │
│  😠 Angry (분노)                        │
│  😨 Fear (공포)                         │
│  😲 Surprise (놀람)                     │
│  🤢 Disgust (혐오)                      │
└─────────────────────────────────────────┘
```

**특징**:
- 문화권을 초월한 보편적 표정
- 얼굴 근육 움직임(FACS) 기반
- 이산적(discrete) 감정 모델

**장점**: 명확하고 구분이 쉬움
**단점**: 복잡한 감정 표현 불가 (예: 질투, 자부심)

---

### Plutchik의 감정 바퀴 (1980)

```
            기쁨 (Joy)
               ↑
      사랑 ←   │   → 낙관
   (기쁨+신뢰)  │  (기쁨+예상)
               │
신뢰 ────────────────── 예상
   ←                    →
경외       중립       공격성
(신뢰+두려움)       (분노+예상)
   ←                    →
두려움 ────────────────── 분노
               │
      복종 ←   │   → 경멸
   (두려움+슬픔) │  (분노+혐오)
               ↓
            슬픔 (Sadness)
```

**핵심 아이디어**:
- 8가지 기본 감정 + 강도 변화
- 복합 감정 = 기본 감정의 조합
- 예: 사랑 = 기쁨 + 신뢰

**장점**: 복잡한 감정 표현 가능
**단점**: 여전히 이산적 모델의 한계

---

## 🎯 VAD 3차원 감정 모델

### Russell의 Circumplex Model (1980)

```
         각성 (Arousal)
              ↑
              │   긴장
              │
분노 ─────────┼───────── 흥분
      │       │       │
      │       │       │
부정 ─┼───────0───────┼─ 긍정
(Valence)     │    (Valence)
      │       │       │
      │       │       │
슬픔 ─────────┼───────── 평온
              │
              │   이완
              ↓
```

**3가지 차원**:
1. **Valence (원자가)**: 긍정 ↔ 부정 (-1.0 ~ 1.0)
2. **Arousal (각성)**: 차분 ↔ 흥분 (-1.0 ~ 1.0)
3. **Dominance (지배)**: 복종 ↔ 지배 (-1.0 ~ 1.0)

---

## 📐 VAD 모델의 장점

### 1. 연속적 표현

```
이산적 모델:
┌────┬────┬────┬────┬────┐
│행복│슬픔│분노│공포│놀람│
└────┴────┴────┴────┴────┘

연속적 모델 (VAD):
┌─────────────────────────┐
│  ●        ●      ●      │  무한한 감정 상태 표현 가능
│     ●  ●     ●     ●    │
│  ●     ●        ●    ●  │
└─────────────────────────┘
```

### 2. 감정 유사도 계산

```python
# 유클리드 거리 기반 유사도
def similarity(vad1, vad2):
    distance = sqrt((v1-v2)² + (a1-a2)² + (d1-d2)²)
    max_distance = sqrt(3 * 2²)  # sqrt(12) = 3.46
    return 1.0 - (distance / max_distance)
```

### 3. 세밀한 감정 구분

```
Happy (행복):
  Valence: +0.8 (긍정적)
  Arousal: +0.5 (중간 각성)
  Dominance: +0.6 (약간 지배적)

Excited (흥분):
  Valence: +0.7 (긍정적)
  Arousal: +0.9 (매우 높은 각성)
  Dominance: +0.5 (중립)

Calm (평온):
  Valence: +0.3 (약간 긍정)
  Arousal: -0.5 (낮은 각성)
  Dominance: +0.2 (약간 지배적)
```

---

## 🎨 기본 감정의 VAD 좌표

### 7가지 감정 매핑

| 감정 | Valence | Arousal | Dominance |
|-----|---------|---------|-----------|
| Happy | +0.8 | +0.5 | +0.6 |
| Sad | -0.7 | -0.6 | -0.5 |
| Angry | -0.5 | +0.7 | +0.8 |
| Fear | -0.6 | +0.7 | -0.6 |
| Surprise | +0.2 | +0.8 | 0.0 |
| Disgust | -0.6 | +0.4 | +0.3 |
| Neutral | 0.0 | 0.0 | 0.0 |

### 3D 공간 시각화

```
           Dominance
              ↑
              │
          Angry
            ●
           /│\
          / │ \
         /  │  \
    Fear   │   Happy
      ●    │    ●
       \   │   /
        \  │  /
         \ │ /
          \│/
           ●
         Sad
```

---

# Part 2: 멀티모달 API 활용

## 🤖 Google Gemini API

### Gemini 2.5 Pro 특징

```
┌─────────────────────────────────────┐
│  Google Gemini 2.5 Pro              │
├─────────────────────────────────────┤
│  ✅ 멀티모달 입력 (텍스트+이미지)   │
│  ✅ 복잡한 추론 능력                │
│  ✅ 긴 컨텍스트 (2M 토큰)          │
│  ✅ 구조화된 JSON 출력              │
│  ✅ 무료 tier 제공                  │
└─────────────────────────────────────┘
```

### API 키 발급 방법

1. [ai.google.dev](https://ai.google.dev) 접속
2. Google 계정 로그인
3. "Get API key" 클릭
4. 새 프로젝트 생성 또는 선택
5. API 키 복사 (형식: `AIza...`)

**무료 할당량**:
- 분당 60건 요청
- 일일 1,500건 요청
- 신용카드 불필요

---

## 🔧 Gemini API 실전 코드

### 기본 감정 인식

```python
import google.generativeai as genai
from PIL import Image

# API 설정
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-2.5-pro')

# 이미지 로드
image = Image.open("face.jpg")

# 프롬프트 구성
prompt = '''이미지 속 사람의 감정을 분석하고 다음 JSON 형식으로만 반환하세요.
다른 설명 없이 JSON만 출력해주세요:
{
  "happy": 0.0, "sad": 0.0, "angry": 0.0, "fear": 0.0,
  "surprise": 0.0, "disgust": 0.0, "neutral": 0.0
}'''

# 감정 분석
response = model.generate_content([prompt, image])
result = json.loads(response.text)

print(result)
# {'happy': 0.8, 'sad': 0.1, 'angry': 0.0, ...}
```

---

## 🚀 OpenAI GPT-4o API

### GPT-4o Vision 특징

```
┌─────────────────────────────────────┐
│  OpenAI GPT-4o                      │
├─────────────────────────────────────┤
│  ✅ 고품질 이미지 이해              │
│  ✅ 상세한 설명 생성                │
│  ✅ Base64 인코딩 지원              │
│  ✅ 스트리밍 응답 가능              │
│  ⚠️  유료 (신용카드 필요)           │
└─────────────────────────────────────┘
```

### API 키 발급

1. [platform.openai.com](https://platform.openai.com) 접속
2. 계정 생성 및 로그인
3. API Keys → Create new secret key
4. 키 복사 (형식: `sk-...`)
5. 결제 정보 등록 ($5 최소)

**비용**:
- GPT-4o: $2.50 / 1M tokens (입력)
- GPT-4o: $10.00 / 1M tokens (출력)

---

## 🔑 GPT-4o Vision 코드

### Base64 인코딩 방식

```python
import base64
from openai import OpenAI
from PIL import Image
import io

# API 클라이언트
client = OpenAI(api_key="YOUR_API_KEY")

# 이미지를 Base64로 인코딩
def image_to_base64(image: Image.Image) -> str:
    # RGBA → RGB 변환
    if image.mode in ('RGBA', 'LA', 'P'):
        rgb = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        rgb.paste(image, mask=image.split()[-1])
        image = rgb

    # Base64 인코딩
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# 감정 분석
image = Image.open("face.jpg")
image_base64 = image_to_base64(image)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "감정을 JSON으로 분석하세요"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }}
        ]
    }],
    max_tokens=500
)

result = response.choices[0].message.content
```

---

## 🎯 3-tier Fallback 패턴

### 안정적인 API 통합

```
┌─────────────────────────────────────┐
│  Tier 1: Google Gemini              │
│  - 빠른 응답 (~1-2초)               │
│  - 무료                              │
│  - 우선 사용                         │
└─────────────────────────────────────┘
              ↓ (실패 시)
┌─────────────────────────────────────┐
│  Tier 2: OpenAI GPT-4o              │
│  - 고품질 분석                       │
│  - 유료 (비용 발생)                  │
│  - 백업 API                          │
└─────────────────────────────────────┘
              ↓ (실패 시)
┌─────────────────────────────────────┐
│  Tier 3: Simulation                 │
│  - 랜덤 감정 생성                    │
│  - 테스트/데모용                     │
│  - 최종 fallback                     │
└─────────────────────────────────────┘
```

### 구현 코드

```python
class EmotionHelper:
    def __init__(self):
        self.mode = None
        self._initialize_apis()

    def _initialize_apis(self):
        # Tier 1: Gemini 시도
        if self._try_gemini():
            self.mode = 'gemini'
            return

        # Tier 2: OpenAI 시도
        if self._try_openai():
            self.mode = 'openai'
            return

        # Tier 3: Simulation
        self.mode = 'simulation'

    def analyze_basic_emotion(self, image):
        if self.mode == 'gemini':
            return self._analyze_with_gemini(image)
        elif self.mode == 'openai':
            return self._analyze_with_openai(image)
        else:
            return self._simulate_emotion()
```

---

# Part 3: 멀티모달 분석

## 🎨 이미지 + 텍스트 통합

### 왜 멀티모달인가?

```
이미지만 분석:
┌──────────────────┐
│  😊 웃는 얼굴     │  → "happy" (90%)
└──────────────────┘

텍스트 컨텍스트:
"오늘 시험에 떨어졌어요..."

통합 분석:
┌──────────────────┐
│  😊 억지 미소     │  → "sad" (80%)
└──────────────────┘
```

**핵심**: 얼굴 표정만으로는 감정을 완전히 이해할 수 없다!

---

## 📊 멀티모달 분석 과정

### 3단계 분석

```
Step 1: 이미지 단독 분석
┌───────────────────────────────────┐
│  Image → API → Base Emotions      │
│  {'happy': 0.6, 'sad': 0.2, ...} │
└───────────────────────────────────┘

Step 2: 이미지 + 텍스트 통합 분석
┌───────────────────────────────────┐
│  Image + Text → API → Combined    │
│  {'happy': 0.2, 'sad': 0.7, ...} │
└───────────────────────────────────┘

Step 3: 차이 분석
┌───────────────────────────────────┐
│  Combined - Image Only = Δ        │
│  {'happy': -0.4, 'sad': +0.5}    │
└───────────────────────────────────┘
```

---

## 🔍 감정 불일치 감지

### 불일치 임계값

```python
def detect_conflict(image_emotions, combined_emotions,
                   threshold=0.3):
    """
    이미지와 통합 분석 간 감정 불일치 감지
    """
    # 지배적 감정 찾기
    dominant_image = max(image_emotions.items(),
                        key=lambda x: x[1])[0]
    dominant_combined = max(combined_emotions.items(),
                          key=lambda x: x[1])[0]

    # 감정이 다르고 신뢰도 차이가 큰 경우
    if dominant_image != dominant_combined:
        diff = abs(image_emotions[dominant_image] -
                  combined_emotions.get(dominant_image, 0))

        if diff > threshold:
            return True, f"불일치 감지: {dominant_image} → {dominant_combined}"

    return False, None
```

---

## 💡 멀티모달 활용 사례

### 1. SNS 감정 분석

```
게시물:
┌─────────────────────────────────┐
│  📷 이미지: 웃는 얼굴             │
│  📝 캡션: "드디어 합격했어요!"   │
└─────────────────────────────────┘
       ↓
결과: happy (95%) ✅ 일치
```

### 2. 억지 미소 감지

```
게시물:
┌─────────────────────────────────┐
│  📷 이미지: 웃는 표정             │
│  📝 캡션: "오늘 해고당했습니다"  │
└─────────────────────────────────┘
       ↓
결과: sad (85%) ⚠️ 불일치
```

### 3. 뉴스 기사 분석

```
기사:
┌─────────────────────────────────┐
│  📷 이미지: 정치인 표정           │
│  📝 텍스트: 스캔들 관련 기사     │
└─────────────────────────────────┘
       ↓
결과: anger (70%) ⚠️ 불일치
```

---

# Part 4: 시계열 감정 분석

## 📈 시계열 분석이란?

### 정의
> "시간에 따른 감정 변화를 추적하고 패턴을 찾는 분석"

### 응용 분야

```
┌─────────────────────────────────────────┐
│  1. 비디오 분석                         │
│     - 영화/드라마의 감정 흐름           │
│     - 인터뷰 중 감정 변화               │
├─────────────────────────────────────────┤
│  2. 실시간 모니터링                     │
│     - 온라인 수업 몰입도                │
│     - 운전자 피로도 감지                │
├─────────────────────────────────────────┤
│  3. 멘탈 헬스                           │
│     - 우울증 패턴 추적                  │
│     - 치료 효과 모니터링                │
└─────────────────────────────────────────┘
```

---

## 🔢 시계열 분석 구성 요소

### 1. 프레임별 감정 추출

```
비디오 → 프레임 추출 → 감정 분석
┌─────┬─────┬─────┬─────┬─────┐
│ F1  │ F2  │ F3  │ F4  │ F5  │
└─────┴─────┴─────┴─────┴─────┘
  ↓     ↓     ↓     ↓     ↓
happy  happy  sad   sad   angry
0.8    0.7    0.6   0.8   0.7
```

### 2. 트렌드 분석

```python
def get_trend(emotion_values):
    """
    선형 회귀로 트렌드 계산
    """
    x = np.arange(len(emotion_values))
    slope = np.polyfit(x, emotion_values, 1)[0]

    if slope > 0.05:
        return 'increasing'  # ↑ 상승
    elif slope < -0.05:
        return 'decreasing'  # ↓ 하락
    else:
        return 'stable'      # → 안정
```

---

## 📊 변화점 감지

### Change Point Detection

```
감정 변화 그래프:

Happy
1.0 │     ●●●
    │    ●   ●
0.8 │   ●     ●
    │  ●       ●●●
0.6 │ ●           ●
    │●             ●
0.4 │               ●●●  ← 변화점!
    │                  ●
0.2 │                   ●●●
0.0 └───────────────────────→
    0  2  4  6  8 10 12 14  Time
```

### 알고리즘

```python
def detect_change_points(history, threshold=0.3):
    """
    프레임 간 감정 변화가 큰 지점 탐지
    """
    change_points = []

    for i in range(1, len(history)):
        # 각 감정별 변화량 계산
        max_change = max(
            abs(history[i]['emotions'][e] -
                history[i-1]['emotions'][e])
            for e in history[i]['emotions']
        )

        # 임계값 초과 시 변화점으로 기록
        if max_change > threshold:
            change_points.append(i)

    return change_points
```

---

## 📉 시계열 시각화

### Matplotlib 타임라인

```python
import matplotlib.pyplot as plt

def visualize_timeline(history):
    """
    감정 변화를 타임라인으로 시각화
    """
    # 감정별 데이터 추출
    emotions = ['happy', 'sad', 'angry', 'fear']
    data = {e: [] for e in emotions}

    for frame in history:
        for emotion in emotions:
            data[emotion].append(frame['emotions'][emotion])

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 6))

    for emotion in emotions:
        ax.plot(data[emotion], label=emotion.capitalize(),
               marker='o', linewidth=2)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Confidence')
    ax.set_title('Emotion Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
```

---

## 💾 CSV 데이터 저장

### 구조화된 저장

```python
def export_to_csv(history, output_path):
    """
    시계열 데이터를 CSV로 저장
    """
    import pandas as pd

    # 데이터 구조화
    rows = []
    for i, frame in enumerate(history):
        row = {'frame': i, 'timestamp': frame['timestamp']}
        row.update(frame['emotions'])
        rows.append(row)

    # DataFrame 생성 및 저장
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8')
```

**출력 예시**:
```csv
frame,timestamp,happy,sad,angry,fear,surprise,disgust,neutral
0,0,0.8,0.1,0.0,0.0,0.0,0.0,0.1
1,1,0.7,0.2,0.0,0.0,0.0,0.0,0.1
2,2,0.5,0.3,0.1,0.0,0.0,0.0,0.1
```

---

# 실습 시간

## 🧪 Lab 01: 기본 감정 인식

### 목표
- Gemini API를 사용한 기본 감정 인식
- 단일 이미지 및 배치 처리
- JSON 결과 출력

### 실습 코드

```python
#!/usr/bin/env python
"""
Lab 01: 기본 감정 인식
"""

import argparse
from PIL import Image
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from modules.week08.emotion_helpers import EmotionHelper

def analyze_single_image(helper, image_path, verbose=True):
    """단일 이미지 감정 분석"""
    if verbose:
        print(f"📷 이미지 분석: {image_path}")

    image = Image.open(image_path)
    result = helper.analyze_basic_emotion(image)

    if verbose:
        # 상위 3개 감정 표시
        sorted_emotions = sorted(result.items(),
                                key=lambda x: x[1],
                                reverse=True)

        print("\n🏆 Top 3 감정:")
        for i, (emotion, score) in enumerate(sorted_emotions[:3], 1):
            bar = "█" * int(score * 30)
            print(f"  {i}. {emotion.capitalize():<10} "
                  f"{bar} {score:.2%}")

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Lab 01: 기본 감정 인식"
    )
    parser.add_argument("--input", required=True,
                       help="입력 이미지 파일 경로")
    parser.add_argument("--output", help="JSON 저장 경로")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # EmotionHelper 초기화
    print("🤖 감정 인식 시스템 초기화...")
    helper = EmotionHelper()
    print(f"✅ 초기화 완료: {helper.mode} 모드\n")

    # 분석 실행
    result = analyze_single_image(helper, args.input,
                                  verbose=not args.quiet)

    # JSON 저장
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 결과 저장: {args.output}")

if __name__ == "__main__":
    main()
```

### 사용법

```bash
# 단일 이미지 분석
python lab01_basic_emotion.py --input face.jpg

# JSON 저장
python lab01_basic_emotion.py --input face.jpg --output result.json

# 최소 출력
python lab01_basic_emotion.py --input face.jpg --quiet
```

---

## 🧪 Lab 02: VAD 모델 분석

### 목표
- VAD 좌표 계산
- 3D 공간 시각화
- 감정 유사도 분석

### 핵심 코드

```python
#!/usr/bin/env python
"""
Lab 02: VAD 3차원 감정 모델
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from modules.week08.emotion_helpers import (
    EmotionHelper, VADModel
)

def analyze_vad(helper, image_path):
    """VAD 분석"""
    # 감정 분석
    image = Image.open(image_path)
    emotions = helper.analyze_basic_emotion(image)

    # 지배적 감정
    dominant = max(emotions.items(), key=lambda x: x[1])
    emotion_name, confidence = dominant

    # VAD 좌표 계산
    vad = VADModel.emotion_to_vad(emotion_name)

    print(f"\n🎯 감정 분석 결과:")
    print(f"  주요 감정: {emotion_name.upper()}")
    print(f"  신뢰도: {confidence:.2%}")
    print(f"\n📊 VAD 좌표:")
    print(f"  Valence: {vad[0]:+.2f}")
    print(f"  Arousal: {vad[1]:+.2f}")
    print(f"  Dominance: {vad[2]:+.2f}")

    return emotion_name, vad

def find_similar_emotions(target_emotion, top_n=3):
    """유사 감정 찾기"""
    target_vad = VADModel.emotion_to_vad(target_emotion)

    similarities = []
    for emotion in VADModel.EMOTION_VAD_MAP.keys():
        if emotion != target_emotion:
            emotion_vad = VADModel.emotion_to_vad(emotion)
            similarity = VADModel.calculate_similarity(
                target_vad, emotion_vad
            )
            similarities.append((emotion, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def plot_vad_3d(emotions_vad, output_path, highlight=None):
    """3D 시각화"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for emotion, (v, a, d) in emotions_vad.items():
        if highlight and emotion == highlight:
            ax.scatter(v, a, d, c='red', s=300,
                      marker='*', label=f'{emotion.upper()} (분석)',
                      edgecolors='darkred', linewidths=2,
                      zorder=5)
        else:
            ax.scatter(v, a, d, s=100, alpha=0.6,
                      label=emotion.capitalize())

        ax.text(v, a, d, f'  {emotion}', fontsize=9)

    ax.set_xlabel('Valence (긍정 ↔ 부정)', fontsize=11)
    ax.set_ylabel('Arousal (차분 ↔ 흥분)', fontsize=11)
    ax.set_zlabel('Dominance (복종 ↔ 지배)', fontsize=11)
    ax.set_title('VAD 3차원 감정 공간', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 3D 플롯 저장: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Lab 02: VAD 모델 분석"
    )
    parser.add_argument("--input", help="입력 이미지")
    parser.add_argument("--plot", help="3D 플롯 저장 경로")
    parser.add_argument("--similarity-matrix",
                       action="store_true",
                       help="유사도 매트릭스 생성")

    args = parser.parse_args()

    if args.input:
        # 감정 분석
        helper = EmotionHelper()
        emotion_name, vad = analyze_vad(helper, args.input)

        # 유사 감정 찾기
        print("\n🔍 유사한 감정:")
        similar = find_similar_emotions(emotion_name, 3)
        for i, (emotion, score) in enumerate(similar, 1):
            print(f"  {i}. {emotion.capitalize()}: {score:.2%}")

        # 3D 시각화
        if args.plot:
            emotions_vad = {
                emotion: VADModel.emotion_to_vad(emotion)
                for emotion in VADModel.EMOTION_VAD_MAP.keys()
            }
            plot_vad_3d(emotions_vad, args.plot,
                       highlight=emotion_name)

if __name__ == "__main__":
    main()
```

### 사용법

```bash
# VAD 분석
python lab02_vad_model.py --input face.jpg

# 3D 시각화
python lab02_vad_model.py --input face.jpg --plot vad_3d.png

# 유사도 매트릭스
python lab02_vad_model.py --similarity-matrix --output matrix.png
```

---

## 🧪 Lab 03: 멀티모달 분석

### 목표
- 이미지 + 텍스트 통합 분석
- 감정 불일치 감지
- 컨텍스트 영향 분석

### 핵심 코드

```python
#!/usr/bin/env python
"""
Lab 03: 멀티모달 감정 분석
"""

def analyze_multimodal(helper, image_path, text):
    """멀티모달 분석"""
    image = Image.open(image_path)
    result = helper.analyze_multimodal(image, text)

    # 결과 비교
    image_only = result['image_only']
    combined = result['combined']
    difference = result['difference']

    dominant_image = max(image_only.items(),
                        key=lambda x: x[1])[0]
    dominant_combined = max(combined.items(),
                           key=lambda x: x[1])[0]

    print("\n📊 멀티모달 분석 결과")
    print("=" * 50)
    print(f"\n🖼️  이미지만: {dominant_image.upper()}")
    print(f"🎨 통합 분석: {dominant_combined.upper()}")

    # 차이 분석
    print("\n🔍 텍스트 영향:")
    significant = [(e, d) for e, d in difference.items()
                  if abs(d) > 0.05]
    for emotion, diff in significant:
        direction = "↑" if diff > 0 else "↓"
        print(f"  {emotion.capitalize()}: "
              f"{diff:+.2%} {direction}")

    return result

def detect_conflict(result, threshold=0.3):
    """불일치 감지"""
    image_only = result['image_only']
    combined = result['combined']

    dominant_image = max(image_only.items(),
                        key=lambda x: x[1])[0]
    dominant_combined = max(combined.items(),
                           key=lambda x: x[1])[0]

    if dominant_image != dominant_combined:
        diff = abs(image_only[dominant_image] -
                  combined.get(dominant_image, 0))

        if diff > threshold:
            return True, (f"불일치 감지: "
                         f"{dominant_image} → {dominant_combined}")

    return False, None

def main():
    parser = argparse.ArgumentParser(
        description="Lab 03: 멀티모달 분석"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--detect-conflict",
                       action="store_true")
    parser.add_argument("--threshold", type=float,
                       default=0.3)

    args = parser.parse_args()

    # 분석
    helper = EmotionHelper()
    result = analyze_multimodal(helper, args.input,
                                args.text)

    # 불일치 감지
    if args.detect_conflict:
        has_conflict, msg = detect_conflict(result,
                                           args.threshold)
        if has_conflict:
            print(f"\n🚨 {msg}")
        else:
            print("\n✅ 감정 일치")

if __name__ == "__main__":
    main()
```

### 사용법

```bash
# 멀티모달 분석
python lab03_multimodal.py \
  --input face.jpg \
  --text "오늘 시험에 떨어졌어요"

# 불일치 감지
python lab03_multimodal.py \
  --input face.jpg \
  --text "드디어 합격했습니다" \
  --detect-conflict
```

---

## 🧪 Lab 04: 시계열 분석

### 목표
- 여러 이미지 시계열 분석
- 비디오 프레임 추출
- 변화점 감지 및 CSV 저장

### 핵심 코드 (간략)

```python
#!/usr/bin/env python
"""
Lab 04: 시계열 감정 분석
"""

def analyze_timeseries(helper, images):
    """시계열 분석"""
    from modules.week08.emotion_helpers import EmotionTimeSeries

    timeseries = EmotionTimeSeries(window_size=len(images))

    for i, image in enumerate(images):
        emotions = helper.analyze_basic_emotion(image)
        timeseries.add_frame(emotions, timestamp=i)

    # 요약
    summary = timeseries.get_summary()
    print(f"\n📊 분석 요약:")
    print(f"  프레임 수: {summary['total_frames']}")
    print(f"  지배적 감정: {summary['dominant_emotion'].upper()}")
    print(f"  평균 신뢰도: {summary['avg_confidence']:.2%}")

    # 트렌드
    print("\n📈 감정 트렌드:")
    for emotion in ['happy', 'sad', 'angry', 'fear']:
        trend = timeseries.get_trend(emotion)
        symbols = {'increasing': '↑', 'decreasing': '↓',
                  'stable': '→'}
        print(f"  {emotion.capitalize()}: {symbols[trend]}")

    # 변화점
    changes = timeseries.detect_change_points()
    if changes:
        print(f"\n⚠️  변화점: {len(changes)}개 발견")
        print(f"  프레임: {changes}")

    return timeseries
```

### 사용법

```bash
# 여러 이미지 분석
python lab04_timeseries.py --images img1.jpg img2.jpg img3.jpg

# 디렉토리 분석
python lab04_timeseries.py --input-dir frames/

# 비디오 분석
python lab04_timeseries.py --video video.mp4 --sample-rate 30

# CSV 저장
python lab04_timeseries.py --images *.jpg --csv results.csv
```

---

## 🧪 Lab 05: API 성능 비교

### 목표
- Gemini vs GPT-4o vs Simulation 비교
- 속도, 일관성, 비용 측정
- 벤치마크 결과 출력

### 핵심 코드 (간략)

```python
#!/usr/bin/env python
"""
Lab 05: API 성능 비교 벤치마크
"""

import time

def benchmark_api(api_mode, image, runs=3):
    """API 벤치마크"""
    helper = EmotionHelper()
    helper.mode = api_mode

    times = []
    results = []

    for _ in range(runs):
        start = time.time()
        result = helper.analyze_basic_emotion(image)
        elapsed = time.time() - start

        times.append(elapsed)
        results.append(result)

    # 통계
    return {
        'mode': api_mode,
        'avg_time': statistics.mean(times),
        'min_time': min(times),
        'max_time': max(times),
        'consistency': calculate_consistency(results),
        'cost_per_1k': API_COSTS[api_mode]['per_1k_images']
    }

def print_comparison_table(benchmarks):
    """비교 테이블 출력"""
    print("\n📊 API 성능 비교")
    print("=" * 70)
    print(f"{'API':<20} {'평균 시간':<12} {'일관성':<10} "
          f"{'비용(1K)':<12}")
    print("-" * 70)

    for bench in benchmarks:
        print(f"{bench['name']:<20} "
              f"{bench['avg_time']:<12.3f}초 "
              f"{bench['consistency']:<10.2%} "
              f"${bench['cost_per_1k']:<11.4f}")
```

### 사용법

```bash
# 단일 이미지 벤치마크
python lab05_comparison.py --input face.jpg --runs 5

# 모든 API 비교
python lab05_comparison.py --input face.jpg

# 특정 API만 테스트
python lab05_comparison.py --input face.jpg --modes gemini openai
```

---

# 🎯 핵심 정리

## 오늘 배운 내용

✅ **감정 인식 기초**
- Ekman 6가지 → Plutchik 복합 → VAD 3차원
- 연속적 감정 모델의 장점

✅ **VAD 3차원 모델**
- Valence, Arousal, Dominance
- 감정 유사도 계산
- 세밀한 감정 구분

✅ **멀티모달 API**
- Google Gemini vs OpenAI GPT-4o
- 3-tier Fallback 패턴
- 이미지 + 텍스트 통합

✅ **시계열 분석**
- 프레임별 감정 추출
- 트렌드 분석 (선형 회귀)
- 변화점 감지

---

## 💡 실전 팁

### API 선택 가이드

```
┌─────────────────────────────────────┐
│  속도 중요 → Simulation (테스트용)   │
│  품질 중요 → Gemini (무료)          │
│  최고 품질 → GPT-4o (유료)          │
│  비용 절약 → 3-tier Fallback        │
└─────────────────────────────────────┘
```

### 성능 최적화

1. **이미지 크기 조정**: API 비용 절감
2. **배치 처리**: 여러 이미지 동시 처리
3. **캐싱**: 동일 이미지 재분석 방지
4. **비동기 처리**: 병렬 API 호출

---

## 🚀 다음 단계

### 추가 학습 주제

1. **오디오 감정 인식**
   - 음성 톤, 피치, 템포 분석
   - OpenSmile 라이브러리

2. **실시간 감정 인식**
   - 웹캠 스트리밍
   - 실시간 처리 최적화

3. **Transformer 모델**
   - BERT for Emotion Classification
   - Fine-tuning 기법

4. **감정 생성**
   - Text-to-Emotion
   - Emotion-conditional Image Generation

---

## 📚 참고 자료

### 논문
- Russell (1980): A Circumplex Model of Affect
- Ekman & Friesen (1971): Constants across cultures
- Mehrabian (1996): PAD Emotion Scales

### API 문서
- [Google Gemini](https://ai.google.dev)
- [OpenAI GPT-4o](https://platform.openai.com/docs)
- [Hugging Face Transformers](https://huggingface.co/docs)

### 라이브러리
- `google-generativeai`: Gemini API
- `openai`: GPT-4o API
- `transformers`: 사전훈련 모델
- `matplotlib`: 데이터 시각화

---

## ❓ Q&A

### 자주 묻는 질문

**Q1: Gemini와 GPT-4o 중 어느 것이 더 좋나요?**
A: 무료로 사용하려면 Gemini, 최고 품질이 필요하면 GPT-4o를 추천합니다.

**Q2: VAD 모델이 이산 모델보다 항상 좋나요?**
A: 아닙니다. 명확한 분류가 필요한 경우 이산 모델이 더 직관적일 수 있습니다.

**Q3: 시계열 분석에 최소 몇 개의 프레임이 필요한가요?**
A: 최소 3-5개 이상을 권장하며, 트렌드 분석은 10개 이상이 좋습니다.

**Q4: 멀티모달 분석이 항상 더 정확한가요?**
A: 텍스트가 관련성이 높을 때만 도움이 됩니다. 무관한 텍스트는 오히려 방해가 될 수 있습니다.

---

# 🎉 수고하셨습니다!

Week 8 강의를 마치며, 고급 감정 인식의 핵심 개념과 실전 구현을 모두 다뤘습니다.

**다음 주 예고**: 최종 프로젝트 및 포트폴리오 구축! 🚀
