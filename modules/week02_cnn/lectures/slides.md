# 🖼️ Week 2: 디지털 이미지 기초와 CNN (Convolutional Neural Networks)

## 📌 학습 목표

이번 주차에서는 디지털 이미지의 기본 개념부터 CNN의 원리까지 체계적으로 학습합니다.

**핵심 학습 내용:**
- 🎨 디지털 이미지의 구조와 표현 방식
- 🔍 이미지 필터링과 Convolution 연산
- 🧠 CNN의 핵심 구성 요소와 작동 원리
- 🤗 HuggingFace 생태계를 통한 사전훈련 모델 활용

---

## 1. 디지털 이미지의 구조와 표현

### 1.1 픽셀과 이미지 배열

디지털 이미지는 수많은 작은 점들(픽셀)의 집합으로 구성됩니다. 각 픽셀은 특정 위치의 색상과 밝기 정보를 담고 있습니다.

#### 그레이스케일 이미지
- 각 픽셀은 0(검정)부터 255(흰색)까지의 단일 값
- 2차원 배열로 표현 (높이 × 너비)

```python
import numpy as np

# 5x5 그레이스케일 이미지 예제
grayscale_image = np.array([
    [0,   50,  100, 150, 200],
    [10,  60,  110, 160, 210],
    [20,  70,  120, 170, 220],
    [30,  80,  130, 180, 230],
    [40,  90,  140, 190, 255]
])
```

#### 컬러 이미지
- 각 픽셀은 RGB 3개 채널의 조합
- 3차원 배열로 표현 (높이 × 너비 × 채널)

```python
# 2x2 RGB 컬러 이미지 예제
color_image = np.array([
    [[255, 0, 0], [0, 255, 0]],    # 빨강, 초록
    [[0, 0, 255], [255, 255, 255]]  # 파랑, 흰색
])
```

### 1.2 색상 공간 (Color Spaces)

이미지는 다양한 색상 공간으로 표현될 수 있으며, 각각 특정 용도에 최적화되어 있습니다.

#### RGB (Red, Green, Blue)
- 가장 일반적인 색상 표현 방식
- 각 채널당 0-255 값 (8비트)
- 디스플레이 장치에 최적화
- **주요 용도**:
  - 웹/앱 디스플레이
  - 디지털 사진 저장
  - 컴퓨터 그래픽스
  - 이미지 파일 포맷 (JPEG, PNG)

#### HSV (Hue, Saturation, Value)
- H: 색상 (0-360도)
- S: 채도 (0-100%)
- V: 명도 (0-100%)
- 색상 기반 필터링에 유용
- **주요 용도**:
  - 특정 색상 검출 (예: 피부색, 신호등)
  - 색상 기반 객체 추적
  - 이미지 편집 소프트웨어
  - 색상 보정 작업

#### LAB 색상 공간
- L: 밝기 (Lightness)
- A: 녹색-빨강 축
- B: 파랑-노랑 축
- 인간의 시각 인지와 더 유사한 색상 표현
- **주요 용도**:
  - 색상 차이 계산
  - 프린터 색상 관리
  - 산업용 색상 매칭
  - 의료 영상 분석
  - Adobe Photoshop 색상 조정

### 1.3 이미지의 메타데이터와 속성

디지털 이미지는 픽셀 데이터 외에도 다양한 메타데이터를 포함합니다:

#### 기본 속성
- **해상도**: 이미지의 가로×세로 픽셀 수
- **비트 깊이**: 각 픽셀당 사용되는 비트 수
- **색상 깊이**: 표현 가능한 색상의 수
- **파일 크기**: 저장에 필요한 메모리 용량

#### EXIF 정보
- 촬영 일시, 카메라 모델, 노출 설정 등
- GPS 위치 정보 (위치 서비스 활성화 시)
- 렌즈 정보, ISO 설정, 셔터 속도 등

### 1.4 이미지 압축과 파일 형식

#### 무손실 압축 형식
- **PNG**: 투명도 지원, 무손실 압축
- **GIF**: 256색 제한, 애니메이션 지원
- **BMP**: 압축 없음, 큰 파일 크기

#### 손실 압축 형식
- **JPEG**: 높은 압축률, 사진에 적합
- **WebP**: 구글이 개발한 차세대 웹 이미지 포맷

### 1.5 이미지 처리의 기본 연산

디지털 이미지 처리에서는 다양한 수학적 연산을 통해 이미지를 변환하고 분석합니다.

#### 픽셀별 연산
- **밝기 조정**: 각 픽셀 값에 상수를 더하거나 곱함
- **대비 조정**: 픽셀 값의 분포를 조절
- **감마 보정**: 비선형 변환을 통한 밝기 조정

#### 공간적 연산
- **필터링**: 주변 픽셀들을 고려한 변환
- **기하학적 변환**: 회전, 크기 조정, 이동
- **모폴로지 연산**: 구조적 요소를 사용한 변환

```python
# 간단한 픽셀 연산 예제
import numpy as np

# 원본 이미지 (3x3 예제)
original = np.array([
    [100, 150, 200],
    [120, 180, 220],
    [140, 160, 180]
])

# 밝기 증가
brighter = original + 50

# 대비 증가
higher_contrast = original * 1.5

print("원본:", original)
print("밝게:", brighter)
print("대비 높게:", higher_contrast)
```

---

## 2. 이미지 필터링과 Convolution 연산

### 2.1 이미지 필터링의 개념

이미지 필터링은 디지털 이미지 처리의 핵심 기법으로, 특정 목적에 따라 이미지의 특징을 강화하거나 노이즈를 제거하는 과정입니다.

#### 필터링의 목적
1. **노이즈 제거**: 이미지에서 원치 않는 잡음 제거
2. **특징 강화**: 엣지, 코너 등 중요한 특징 부각
3. **블러링**: 이미지를 부드럽게 만들어 세부 사항 제거
4. **샤프닝**: 이미지의 선명도 증가

### 2.2 Convolution 연산의 수학적 정의

Convolution은 두 함수의 합성을 나타내는 수학 연산으로, 이미지 처리에서는 다음과 같이 정의됩니다:

```
(f * g)(x, y) = ΣΣ f(m, n) × g(x-m, y-n)
```

여기서:
- f: 입력 이미지
- g: 필터(커널)
- *: convolution 연산자

#### 실제 구현 과정
1. **필터 배치**: 필터를 이미지의 특정 위치에 배치
2. **요소별 곱셈**: 겹치는 부분의 값들을 서로 곱함
3. **합계 계산**: 모든 곱셈 결과를 합산
4. **결과 저장**: 계산된 값을 출력 이미지의 해당 위치에 저장
5. **반복**: 이미지 전체에 대해 과정 반복

```python
# 상세한 convolution 구현 예제
import numpy as np

def convolution_2d(image, kernel):
    """
    2D convolution 연산 구현
    """
    # 이미지와 커널의 크기 획득
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # 출력 이미지 크기 계산
    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1
    
    # 출력 이미지 초기화
    output = np.zeros((output_height, output_width))
    
    # Convolution 연산 수행
    for i in range(output_height):
        for j in range(output_width):
            # 현재 위치에서 커널 크기만큼 영역 추출
            region = image[i:i+kernel_height, j:j+kernel_width]
            
            # 요소별 곱셈 후 합계 계산
            output[i, j] = np.sum(region * kernel)
    
    return output

# 사용 예제
test_image = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9]
])

# 엣지 검출 필터
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

result = convolution_2d(test_image, edge_kernel)
print("Convolution 결과:")
print(result)
```

### 2.3 주요 필터 유형과 응용

#### 블러 필터 (Blur Filters)

블러 필터는 이미지를 부드럽게 만들어 노이즈를 제거하거나 세부 사항을 감소시킵니다.

**1. 박스 필터 (Box Filter)**
```python
box_filter = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])
```

**2. 가우시안 필터 (Gaussian Filter)**
```python
# 가우시안 필터 (표준편차 σ=1)
gaussian_filter = np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
])
```

#### 엣지 검출 필터 (Edge Detection Filters)

엣지 검출은 이미지에서 밝기가 급격하게 변하는 부분을 찾는 작업입니다.

**1. Sobel 필터**
```python
# 수직 엣지 검출
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# 수평 엣지 검출
sobel_horizontal = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])
```

**2. 라플라시안 필터 (Laplacian Filter)**
```python
laplacian = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])
```

#### 샤프닝 필터 (Sharpening Filters)

샤프닝은 이미지의 경계를 강화하여 선명도를 높입니다.

```python
# 기본 샤프닝 필터
sharpening = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# 강한 샤프닝 필터
strong_sharpening = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])
```

---

## 3. Convolutional Neural Network (CNN) 기초

### 3.1 CNN의 개념과 필요성

Convolutional Neural Network(합성곱 신경망)는 이미지 인식과 처리에 특화된 딥러닝 모델입니다. 전통적인 완전연결 신경망과 달리, CNN은 이미지의 공간적 특성을 보존하면서 학습할 수 있습니다.

#### CNN이 필요한 이유

**1. 매개변수 수의 문제**
- 일반 신경망으로 1000x1000 컬러 이미지를 처리하려면 3,000,000개의 입력 노드가 필요
- 첫 번째 은닉층이 1000개 노드라면 3억 개의 가중치 필요
- 메모리와 계산 비용이 기하급수적으로 증가

**2. 공간적 정보의 손실**
- 이미지를 1차원으로 펼치면 픽셀 간의 공간적 관계 정보 손실
- 인접한 픽셀들의 상관관계 무시
- 이미지의 구조적 특성을 활용하기 어려움

**3. 일반화 성능의 한계**
- 같은 객체가 다른 위치에 있으면 완전히 다른 패턴으로 인식
- 회전, 크기 변화에 매우 민감
- 과적합(overfitting) 문제 발생하기 쉬움

### 3.2 CNN의 핵심 구성 요소

CNN은 크게 네 가지 주요 구성 요소로 이루어져 있습니다: 합성곱층(Convolutional Layer), 활성화 함수(Activation Function), 풀링층(Pooling Layer), 그리고 완전연결층(Fully Connected Layer)입니다.

#### 합성곱층 (Convolutional Layer)

합성곱층은 CNN의 핵심 구성 요소로, 필터(커널)를 사용하여 입력 이미지의 특징을 추출합니다.

**필터(Filter/Kernel)의 작동 원리:**
- 필터는 작은 크기의 행렬 (예: 3×3, 5×5)
- 입력 이미지 위를 슬라이딩하면서 합성곱 연산 수행
- 각 위치에서 요소별 곱셈 후 합계 계산

```python
# 합성곱 연산 예제
import numpy as np

# 입력 이미지 (5x5)
input_image = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
])

# 엣지 검출 필터 (3x3)
edge_filter = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# 합성곱 연산 (한 위치에서)
# 좌상단 3x3 영역에 필터 적용
region = input_image[0:3, 0:3]
result = np.sum(region * edge_filter)
print(f"합성곱 결과: {result}")
```

**스트라이드(Stride)와 패딩(Padding):**
- **스트라이드**: 필터가 이동하는 간격
- **패딩**: 입력 이미지 경계에 추가하는 픽셀들
- 출력 크기 계산: (입력크기 - 필터크기 + 2×패딩) / 스트라이드 + 1

#### 활성화 함수 (Activation Function)

합성곱 연산 후 비선형성을 도입하기 위해 활성화 함수를 적용합니다.

**주요 활성화 함수들:**

1. **ReLU (Rectified Linear Unit)**:
   ```
   f(x) = max(0, x)
   ```
   - 음수 값을 0으로, 양수 값은 그대로 유지
   - 계산이 간단하고 기울기 소실 문제 완화

2. **Leaky ReLU**:
   ```
   f(x) = max(0.01x, x)
   ```
   - 음수 영역에서도 작은 기울기 유지

3. **Sigmoid**:
   ```
   f(x) = 1 / (1 + e^(-x))
   ```
   - 출력이 0과 1 사이로 제한
   - 기울기 소실 문제가 있어 깊은 네트워크에 부적합

#### 풀링층 (Pooling Layer)

풀링층은 특징 맵의 크기를 줄이고 중요한 특징을 추출하는 역할을 합니다.

**주요 풀링 기법:**

1. **최대 풀링 (Max Pooling)**:
   ```python
   # 2x2 최대 풀링 예제
   input_feature = np.array([
       [1, 3, 2, 4],
       [5, 6, 7, 8],
       [1, 2, 3, 4],
       [5, 6, 7, 8]
   ])
   
   # 2x2 영역에서 최대값 선택
   # 결과: [[6, 8], [6, 8]]
   ```

2. **평균 풀링 (Average Pooling)**:
   - 풀링 영역의 평균값 계산
   - 더 부드러운 다운샘플링

3. **전역 풀링 (Global Pooling)**:
   - 전체 특징 맵을 하나의 값으로 압축
   - 완전연결층의 매개변수 수 대폭 감소

### 3.3 CNN의 작동 과정

CNN의 전체적인 작동 과정은 다음과 같이 요약할 수 있습니다:

1. **특징 추출 단계**:
   - 합성곱층에서 다양한 필터를 사용하여 특징 추출
   - 활성화 함수로 비선형성 도입
   - 풀링층에서 크기 축소 및 중요 특징 선별

2. **분류 단계**:
   - 완전연결층에서 추출된 특징들을 조합
   - 최종 출력층에서 확률 분포 생성

```python
# CNN 구조 예시 (의사코드)
def CNN_forward(input_image):
    # 첫 번째 합성곱 블록
    conv1 = convolution(input_image, filters_32_3x3)
    relu1 = ReLU(conv1)
    pool1 = max_pooling(relu1, 2x2)
    
    # 두 번째 합성곱 블록
    conv2 = convolution(pool1, filters_64_3x3)
    relu2 = ReLU(conv2)
    pool2 = max_pooling(relu2, 2x2)
    
    # 분류를 위한 완전연결층
    flattened = flatten(pool2)
    fc1 = fully_connected(flattened, 128_units)
    relu3 = ReLU(fc1)
    output = fully_connected(relu3, num_classes)
    
    return softmax(output)
```

### 3.4 CNN의 학습 과정

CNN은 역전파 알고리즘(Backpropagation)을 통해 학습합니다.

#### 순전파 (Forward Propagation)
1. 입력 이미지가 네트워크를 통과하며 예측값 생성
2. 손실 함수로 실제값과 예측값의 차이 계산

#### 역전파 (Backward Propagation)
1. 출력층부터 입력층까지 거꾸로 진행하며 기울기 계산
2. 각 층의 가중치를 기울기 방향으로 업데이트
3. 학습률(Learning Rate)로 업데이트 정도 조절

#### 손실 함수와 최적화
**분류 문제의 손실 함수:**
- **교차 엔트로피 (Cross Entropy)**:
  ```
  Loss = -Σ y_true × log(y_pred)
  ```
  
**최적화 알고리즘:**
- **SGD (Stochastic Gradient Descent)**
- **Adam**: 적응적 학습률과 모멘텀을 결합한 최적화 기법
- **RMSprop**: 기울기 제곱의 이동평균을 사용한 적응적 최적화

---

## 4. CNN 아키텍처의 발전

### 4.1 주요 CNN 아키텍처

#### LeNet-5 (1998)
- 최초의 성공적인 CNN 아키텍처
- 손글씨 숫자 인식(MNIST)에 사용
- 7개 레이어 구조

#### AlexNet (2012)
- ImageNet 대회 우승
- GPU 병렬 처리 활용
- ReLU 활성화 함수 도입
- Dropout 정규화 사용

#### VGGNet (2014)
- 작은 3×3 필터만 사용
- 깊은 네트워크 구조 (16-19층)
- 단순하고 일관된 아키텍처

#### ResNet (2015)
- Residual Connection 도입
- 매우 깊은 네트워크 가능 (152층+)
- 기울기 소실 문제 해결

```python
# ResNet의 Residual Block 구현 예제
def residual_block(x, filters):
    """
    ResNet의 기본 Residual Block
    """
    # 메인 경로
    conv1 = Conv2D(filters, (3, 3), padding='same')(x)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)
    
    conv2 = Conv2D(filters, (3, 3), padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    
    # Skip Connection
    add = Add()([x, bn2])
    output = ReLU()(add)
    
    return output
```

---

## 5. HuggingFace 생태계 활용

### 5.1 HuggingFace 소개

HuggingFace는 최신 AI 모델을 쉽게 사용할 수 있게 만드는 플랫폼입니다. 수천 개의 사전훈련 모델과 데이터셋을 제공합니다.

#### 주요 특징
- **Model Hub**: 40만개 이상의 사전훈련 모델
- **Datasets**: 5만개 이상의 데이터셋
- **Transformers**: 통합 API로 모든 모델 사용 가능
- **무료 추론 API**: 많은 모델을 무료로 테스트 가능

### 5.2 HuggingFace Transformers 설치 및 설정

```bash
# 필수 패키지 설치
pip install transformers
pip install torch torchvision
pip install pillow
```

### 5.3 이미지 분류 실습

```python
from transformers import pipeline
from PIL import Image

# 이미지 분류 파이프라인 생성
classifier = pipeline("image-classification", 
                      model="google/vit-base-patch16-224")

# 이미지 로드
image = Image.open("sample_image.jpg")

# 분류 수행
results = classifier(image)

# 결과 출력
for result in results[:5]:
    print(f"{result['label']}: {result['score']:.4f}")
```

### 5.4 객체 검출 실습

```python
from transformers import pipeline

# 객체 검출 파이프라인
detector = pipeline("object-detection",
                    model="facebook/detr-resnet-50")

# 이미지에서 객체 검출
image = Image.open("street_scene.jpg")
results = detector(image)

# 검출된 객체 표시
for obj in results:
    print(f"Object: {obj['label']}")
    print(f"Confidence: {obj['score']:.2f}")
    print(f"Location: {obj['box']}")
    print("---")
```

### 5.5 이미지 세그멘테이션

```python
from transformers import pipeline

# 세그멘테이션 파이프라인
segmenter = pipeline("image-segmentation",
                     model="facebook/detr-resnet-50-panoptic")

# 이미지 세그멘테이션
image = Image.open("complex_scene.jpg")
segments = segmenter(image)

# 세그먼트 정보 출력
for segment in segments:
    print(f"Label: {segment['label']}")
    print(f"Score: {segment['score']:.2f}")
```

---

## 6. 프로젝트: 이미지 분석 통합 시스템

### 6.1 프로젝트 개요

**통합 이미지 분석 시스템**은 Week 2에서 학습한 모든 내용을 결합한 실습 프로젝트입니다.

#### 주요 기능
- **기본 분석**: 이미지 속성, 히스토그램 분석
- **필터링**: 다양한 컨볼루션 필터 적용
- **AI 분석**: HuggingFace 모델을 활용한 이미지 분류 및 객체 검출
- **엣지 검출**: Sobel, Laplacian, Canny 엣지 검출
- **통합 분석**: 모든 분석을 한 번에 수행

### 6.2 시스템 아키텍처

```python
class IntegratedImageAnalysisSystem:
    """통합 이미지 분석 시스템"""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.initialize_models()    # AI 모델 초기화
        self.initialize_filters()   # 이미지 필터 초기화
```

#### 모듈 구성
- **AI 모델**: ViT, DETR 등 최신 Transformer 모델
- **이미지 필터**: Blur, Gaussian, Edge Detection, Sharpen, Emboss
- **분석 도구**: 히스토그램, 속성 분석, 엣지 검출

### 6.3 핵심 기능 구현

#### 다중 모드 분석 시스템

```python
def run_streamlit_app(self):
    """Streamlit 웹 앱 실행"""
    # 분석 모드 선택
    analysis_mode = st.selectbox(
        "분석 모드",
        ["기본 분석", "필터링", "AI 분석", "엣지 검출", "통합 분석"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # 모드별 분석 실행
        if analysis_mode == "기본 분석":
            self.basic_analysis(image)
        elif analysis_mode == "AI 분석":
            self.ai_analysis_mode(image)
        elif analysis_mode == "통합 분석":
            self.integrated_analysis(image)
```

#### AI 분석 모듈

```python
def ai_analysis_mode(self, image):
    """AI 기반 이미지 분석"""
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🏷️ 이미지 분류"):
            results = self.models['classifier'](image)
            for result in results[:5]:
                st.write(f"**{result['label']}**: {result['score']:.2%}")

    with col2:
        if st.button("🎯 객체 검출"):
            results = self.models['detector'](image)
            img_with_boxes = self.draw_detection_boxes(image, results)
            st.image(img_with_boxes)
```

#### 다중 엣지 검출

```python
def detect_edges_multi(self, image):
    """다양한 엣지 검출 방법 적용"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    edges = {
        'Sobel X': cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
        'Sobel Y': cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3),
        'Laplacian': cv2.Laplacian(gray, cv2.CV_64F),
        'Canny': cv2.Canny(gray, 50, 150)
    }
    return edges
```

### 6.4 실행 및 사용법

1. **환경 설정**
   ```bash
   pip install streamlit torch transformers pillow opencv-python
   ```

2. **앱 실행**
   ```bash
   streamlit run 06_integrated_project.py
   ```

3. **사용 방법**
   - 이미지 업로드
   - 분석 모드 선택
   - 각 모드별 분석 실행
   - 결과 확인 및 비교

---

