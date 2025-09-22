# Week 2: CNN 원리 + Hugging Face 생태계

## 학습 목표
이번 강의를 통해 학습자는 다음 내용을 이해하고 실습할 수 있습니다:
- Convolutional Neural Network(CNN)의 핵심 구성 요소와 작동 원리
- CNN에서의 Backpropagation 과정과 학습 메커니즘
- 주요 CNN 아키텍처의 발전 과정 (LeNet → ResNet)
- Hugging Face 생태계의 구조와 활용 방법
- 사전훈련 모델을 활용한 실제 AI 서비스 구축

## 1. Convolutional Neural Network (CNN) 기초

### 1.1 CNN의 핵심 아이디어

Convolutional Neural Network는 인간의 시각 시스템에서 영감을 받아 개발된 딥러닝 아키텍처입니다. CNN의 핵심 아이디어는 다음과 같습니다:

#### 지역적 연결성 (Local Connectivity)
- 각 뉴런은 입력의 일부 영역만 연결
- 전체 이미지를 한 번에 보는 대신 작은 패치(patch) 단위로 처리
- 계산 효율성과 학습 안정성 향상

#### 가중치 공유 (Weight Sharing)
- 동일한 필터가 이미지의 모든 위치에 적용
- 파라미터 수를 크게 줄여 학습 효율성 증대
- 변환 불변성(translation invariance) 학습

#### 계층적 특징 추출 (Hierarchical Feature Extraction)
- 낮은 레이어: 엣지, 모서리 등 기본 특징
- 중간 레이어: 모양, 패턴 등 복합 특징
- 높은 레이어: 객체, 장면 등 고수준 특징

### 1.2 Convolution 연산의 수학적 기초

#### 1D Convolution
1차원 신호에서의 convolution은 다음과 같이 정의됩니다:

```
(f * g)(t) = ∫ f(τ)g(t-τ)dτ
```

이산 신호의 경우:
```
(f * g)[n] = Σ f[k]g[n-k]
```

#### 2D Convolution
이미지 처리에서 사용되는 2D convolution:

```
(I * K)(x,y) = Σ Σ I(i,j)K(x-i,y-j)
```

여기서:
- I: 입력 이미지
- K: 커널(필터)
- (x,y): 출력 위치
- (i,j): 입력 위치

#### Convolution의 주요 특성
1. **선형성**: (af + bg) * h = a(f * h) + b(g * h)
2. **시변성**: f(t) * g(t) = g(t) * f(t)
3. **결합성**: (f * g) * h = f * (g * h)

### 1.3 다양한 Convolution 필터

#### 엣지 검출 필터
```
Sobel X:    Sobel Y:
[-1  0  1]  [-1 -2 -1]
[-2  0  2]  [ 0  0  0]
[-1  0  1]  [ 1  2  1]
```

#### 블러 필터
```
평균 블러:
[1/9  1/9  1/9]
[1/9  1/9  1/9]
[1/9  1/9  1/9]

가우시안 블러:
[1/16  2/16  1/16]
[2/16  4/16  2/16]
[1/16  2/16  1/16]
```

#### 샤프닝 필터
```
[ 0 -1  0]
[-1  5 -1]
[ 0 -1  0]
```

### 1.4 Padding과 Stride

#### Padding
- 입력 이미지 주변에 추가 픽셀을 삽입
- 출력 크기를 조절하고 경계 정보 보존
- **Same Padding**: 출력 크기를 입력과 동일하게 유지
- **Valid Padding**: 패딩 없이 convolution 수행

#### Stride
- 필터가 이동하는 간격
- Stride = 1: 한 픽셀씩 이동
- Stride = 2: 두 픽셀씩 이동
- 출력 크기 계산: (W - F + 2P) / S + 1

## 2. CNN의 핵심 구성 요소

### 2.1 Convolution Layer

#### 다중 채널 Convolution
RGB 이미지의 경우 3채널 입력을 처리:

```
입력: (H, W, 3) → 커널: (F, F, 3, K) → 출력: (H', W', K)
```

여기서:
- H, W: 입력 이미지의 높이, 너비
- F: 필터 크기
- K: 출력 채널 수

#### 1x1 Convolution
- 채널 수 조절을 위한 효율적인 방법
- 비선형성 추가
- 계산량 감소

### 2.2 Pooling Layer

#### Max Pooling
가장 일반적인 pooling 방식:
```
출력 = max(입력 윈도우)
```

#### Average Pooling
평균값을 사용:
```
출력 = mean(입력 윈도우)
```

#### Pooling의 역할
1. **차원 축소**: 계산량 감소
2. **과적합 방지**: 일반화 성능 향상
3. **변환 불변성**: 작은 변형에 대한 강건성

### 2.3 Activation Functions

#### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
```
- 가장 널리 사용되는 활성화 함수
- 계산이 간단하고 그래디언트 소실 문제 완화

#### Leaky ReLU
```
f(x) = max(αx, x), α < 1
```
- ReLU의 개선 버전
- 음수 입력에 대해서도 작은 그래디언트 제공

#### ELU (Exponential Linear Unit)
```
f(x) = x if x > 0 else α(e^x - 1)
```
- 부드러운 음수 영역
- 더 나은 일반화 성능

### 2.4 Batch Normalization

#### 배치 정규화의 목적
- 내부 공변량 이동(internal covariate shift) 해결
- 학습 속도 향상
- 그래디언트 소실/폭발 문제 완화

#### 배치 정규화 과정
1. 미니배치 평균 계산: μ = (1/m)Σx
2. 미니배치 분산 계산: σ² = (1/m)Σ(x-μ)²
3. 정규화: x̂ = (x-μ)/√(σ²+ε)
4. 스케일링 및 이동: y = γx̂ + β

## 3. CNN에서의 Backpropagation

### 3.1 Convolution Layer의 역전파

#### 그래디언트 계산
Convolution layer의 역전파는 다음과 같이 계산됩니다:

```
∂L/∂X = ∂L/∂Y * rot180(K)
∂L/∂K = ∂L/∂Y * X
```

여기서:
- L: 손실 함수
- X: 입력
- Y: 출력
- K: 커널
- rot180: 180도 회전

#### 다중 채널의 경우
```
∂L/∂X[:,:,c] = Σ ∂L/∂Y[:,:,k] * rot180(K[:,:,c,k])
∂L/∂K[:,:,c,k] = ∂L/∂Y[:,:,k] * X[:,:,c]
```

### 3.2 Pooling Layer의 역전파

#### Max Pooling 역전파
- 최대값이 선택된 위치에만 그래디언트 전달
- 다른 위치는 0

#### Average Pooling 역전파
- 모든 위치에 동일한 그래디언트 분배

### 3.3 전체 네트워크의 역전파

#### 체인 룰 적용
```
∂L/∂θ = ∂L/∂y * ∂y/∂θ
```

각 레이어의 그래디언트는 다음 레이어의 그래디언트와 연결됩니다.

## 4. 주요 CNN 아키텍처

### 4.1 LeNet-5 (1998)

#### 구조
```
입력 (32×32×1)
↓
Conv1 (6 filters, 5×5) → ReLU → MaxPool (2×2)
↓
Conv2 (16 filters, 5×5) → ReLU → MaxPool (2×2)
↓
Conv3 (120 filters, 5×5) → ReLU
↓
Flatten
↓
FC1 (84 neurons) → ReLU
↓
FC2 (10 neurons) → Softmax
```

#### 특징
- 최초의 성공적인 CNN
- MNIST 데이터셋에서 우수한 성능
- 약 60,000개의 파라미터

### 4.2 AlexNet (2012)

#### 구조
```
입력 (227×227×3)
↓
Conv1 (96 filters, 11×11, stride=4) → ReLU → MaxPool (3×3, stride=2)
↓
Conv2 (256 filters, 5×5, padding=2) → ReLU → MaxPool (3×3, stride=2)
↓
Conv3 (384 filters, 3×3, padding=1) → ReLU
↓
Conv4 (384 filters, 3×3, padding=1) → ReLU
↓
Conv5 (256 filters, 3×3, padding=1) → ReLU → MaxPool (3×3, stride=2)
↓
Flatten
↓
FC1 (4096 neurons) → ReLU → Dropout
↓
FC2 (4096 neurons) → ReLU → Dropout
↓
FC3 (1000 neurons) → Softmax
```

#### 혁신적 특징
- ReLU 활성화 함수 도입
- Dropout 정규화 기법
- GPU 가속 활용
- ImageNet에서 혁신적인 성능

### 4.3 VGGNet (2014)

#### 구조
- 3×3 convolution 필터만 사용
- 깊은 네트워크 구조 (16-19 layers)
- 단순하고 일관된 설계

#### VGG-16 구조
```
입력 (224×224×3)
↓
Conv1: 2×[Conv(64, 3×3)] → MaxPool
↓
Conv2: 2×[Conv(128, 3×3)] → MaxPool
↓
Conv3: 3×[Conv(256, 3×3)] → MaxPool
↓
Conv4: 3×[Conv(512, 3×3)] → MaxPool
↓
Conv5: 3×[Conv(512, 3×3)] → MaxPool
↓
Flatten
↓
FC1 (4096) → ReLU → Dropout
↓
FC2 (4096) → ReLU → Dropout
↓
FC3 (1000) → Softmax
```

### 4.4 ResNet (2015)

#### Residual Connection
```
F(x) = H(x) - x
H(x) = F(x) + x
```

#### ResNet-50 구조
```
입력 (224×224×3)
↓
Conv1 (7×7, 64, stride=2) → BN → ReLU → MaxPool
↓
Conv2_x: 3×[Bottleneck(64, 64, 256)]
↓
Conv3_x: 4×[Bottleneck(128, 128, 512)]
↓
Conv4_x: 6×[Bottleneck(256, 256, 1024)]
↓
Conv5_x: 3×[Bottleneck(512, 512, 2048)]
↓
Global Average Pooling
↓
FC (1000) → Softmax
```

#### Bottleneck Block
```
입력
↓
Conv1×1 (64) → BN → ReLU
↓
Conv3×3 (64) → BN → ReLU
↓
Conv1×1 (256) → BN
↓
+ (Residual Connection)
↓
ReLU
```

## 5. Hugging Face 생태계

### 5.1 Hugging Face 소개

Hugging Face는 자연어 처리와 머신러닝을 위한 오픈소스 플랫폼입니다.

#### 주요 구성 요소
1. **Transformers**: 사전훈련 모델 라이브러리
2. **Datasets**: 데이터셋 관리 도구
3. **Tokenizers**: 텍스트 토큰화 도구
4. **Accelerate**: 분산 학습 도구
5. **Hub**: 모델 및 데이터셋 공유 플랫폼

### 5.2 Transformers 라이브러리

#### 주요 모델 아키텍처
- **BERT**: Bidirectional Encoder Representations from Transformers
- **GPT**: Generative Pre-trained Transformer
- **T5**: Text-to-Text Transfer Transformer
- **CLIP**: Contrastive Language-Image Pre-training
- **ViT**: Vision Transformer

#### 모델 로드 및 사용
```python
from transformers import AutoTokenizer, AutoModel

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 텍스트 토큰화
inputs = tokenizer("Hello, world!", return_tensors="pt")

# 모델 추론
outputs = model(**inputs)
```

### 5.3 Hugging Face Hub

#### 모델 공유
- 사전훈련 모델 업로드
- 모델 카드 작성
- 버전 관리
- 커뮤니티 피드백

#### 데이터셋 공유
- 다양한 데이터셋 제공
- 데이터셋 버전 관리
- 데이터셋 카드 작성

### 5.4 Serverless Inference API

#### API 사용법
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/모델명"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(API_URL, headers=headers, data=데이터)
result = response.json()
```

#### 지원하는 태스크
- 텍스트 분류
- 토큰 분류
- 질문 답변
- 텍스트 생성
- 이미지 분류
- 객체 탐지
- 이미지 세그멘테이션

## 6. 사전훈련 모델 활용

### 6.1 CLIP (Contrastive Language-Image Pre-training)

#### CLIP의 핵심 아이디어
- 텍스트와 이미지를 동일한 임베딩 공간에 매핑
- 대조 학습(contrastive learning)을 통한 학습
- 제로샷(zero-shot) 분류 가능

#### CLIP 사용 예제
```python
from transformers import CLIPProcessor, CLIPModel

# 모델 로드
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 텍스트와 이미지 준비
texts = ["a photo of a cat", "a photo of a dog"]
image = load_image("cat.jpg")

# 전처리
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 추론
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
```

### 6.2 Vision Transformer (ViT)

#### ViT의 구조
1. **이미지 패칭**: 이미지를 고정 크기 패치로 분할
2. **패치 임베딩**: 각 패치를 임베딩 벡터로 변환
3. **위치 인코딩**: 위치 정보 추가
4. **Transformer 인코더**: Self-attention과 MLP 블록
5. **분류 헤드**: 최종 분류

#### ViT 사용 예제
```python
from transformers import ViTImageProcessor, ViTForImageClassification

# 모델 로드
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# 이미지 전처리
inputs = processor(images=image, return_tensors="pt")

# 추론
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
```

### 6.3 BERT (Bidirectional Encoder Representations from Transformers)

#### BERT의 특징
- 양방향 컨텍스트 이해
- 마스킹된 언어 모델링(MLM)
- 다음 문장 예측(NSP)

#### BERT 사용 예제
```python
from transformers import BertTokenizer, BertModel

# 모델 로드
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 텍스트 토큰화
inputs = tokenizer("Hello, world!", return_tensors="pt")

# 추론
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
pooled_output = outputs.pooler_output
```

## 7. Gradio를 통한 웹 서비스 구축

### 7.1 Gradio 소개

Gradio는 머신러닝 모델을 위한 웹 인터페이스를 쉽게 만들 수 있는 Python 라이브러리입니다.

#### 주요 특징
- 간단한 API
- 다양한 입력/출력 타입 지원
- 자동 UI 생성
- Hugging Face Spaces와 통합

### 7.2 기본 Gradio 앱 생성

```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)

demo.launch()
```

### 7.3 이미지 분류 앱 예제

```python
import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# 모델 로드
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

def classify_image(image):
    # 이미지 전처리
    inputs = processor(images=image, return_tensors="pt")
    
    # 추론
    outputs = model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(dim=-1)
    
    # 상위 5개 결과
    top5_probs, top5_indices = probs.topk(5)
    
    results = []
    for prob, idx in zip(top5_probs[0], top5_indices[0]):
        label = model.config.id2label[idx.item()]
        results.append(f"{label}: {prob:.4f}")
    
    return "\n".join(results)

# Gradio 인터페이스 생성
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="분류 결과"),
    title="이미지 분류기",
    description="이미지를 업로드하면 AI가 무엇인지 분류해드립니다."
)

demo.launch()
```

### 7.4 Hugging Face Spaces 배포

#### Space 생성 과정
1. Hugging Face Hub에서 새 Space 생성
2. Gradio 앱 코드 작성
3. requirements.txt 파일 생성
4. README.md 작성
5. 자동 배포

#### app.py 예제
```python
import gradio as gr
from transformers import pipeline

# 파이프라인 로드
classifier = pipeline("image-classification")

def classify(image):
    result = classifier(image)
    return {r["label"]: r["score"] for r in result}

# Gradio 앱 생성
demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="이미지 분류기"
)

demo.launch()
```

## 8. 실습 및 프로젝트

### 8.1 CNN 수동 구현 실습

#### Convolution 연산 구현
```python
import numpy as np

def manual_convolution_2d(input_tensor, kernel, stride=1, padding=0):
    # 패딩 적용
    if padding > 0:
        padded_input = np.pad(input_tensor, ((padding, padding), (padding, padding)))
    else:
        padded_input = input_tensor
    
    # 출력 크기 계산
    h_out = (padded_input.shape[0] - kernel.shape[0]) // stride + 1
    w_out = (padded_input.shape[1] - kernel.shape[1]) // stride + 1
    
    # 출력 초기화
    output = np.zeros((h_out, w_out))
    
    # Convolution 연산
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + kernel.shape[0]
            w_end = w_start + kernel.shape[1]
            
            window = padded_input[h_start:h_end, w_start:w_end]
            output[i, j] = np.sum(window * kernel)
    
    return output
```

### 8.2 Hugging Face 모델 테스트

#### 다양한 모델 비교
```python
import time
from transformers import pipeline

# 모델들 로드
models = {
    "ViT": pipeline("image-classification", model="google/vit-base-patch16-224"),
    "ResNet": pipeline("image-classification", model="microsoft/resnet-50"),
    "EfficientNet": pipeline("image-classification", model="google/efficientnet-b0")
}

# 성능 비교
for name, model in models.items():
    start_time = time.time()
    result = model(image)
    end_time = time.time()
    
    print(f"{name}: {end_time - start_time:.3f}초")
    print(f"예측: {result[0]['label']} ({result[0]['score']:.4f})")
```

### 8.3 종합 프로젝트: 멀티모달 이미지 검색 시스템

#### 프로젝트 개요
- CLIP을 활용한 텍스트-이미지 검색
- 사용자가 자연어로 이미지 검색
- 검색 결과 시각화

#### 구현 단계
1. 이미지 데이터셋 준비
2. CLIP 모델로 이미지 임베딩 생성
3. 텍스트 쿼리 임베딩 생성
4. 유사도 계산 및 정렬
5. Gradio 인터페이스 구축

## 9. 성능 최적화 및 모범 사례

### 9.1 모델 성능 최적화

#### 배치 처리
- 여러 이미지를 한 번에 처리
- GPU 메모리 효율적 활용
- 처리 속도 향상

#### 모델 양자화
- FP16 (반정밀도) 사용
- INT8 양자화
- 모델 크기 및 추론 속도 개선

#### 모델 압축
- 지식 증류(Knowledge Distillation)
- 프루닝(Pruning)
- 모델 아키텍처 검색(NAS)

### 9.2 메모리 최적화

#### 그래디언트 체크포인팅
- 메모리 사용량 감소
- 학습 속도와 메모리 트레이드오프

#### 혼합 정밀도 학습
- FP16과 FP32 혼합 사용
- 메모리 효율성과 수치 안정성 균형

### 9.3 배포 최적화

#### 모델 서빙
- ONNX 변환
- TensorRT 최적화
- TorchScript 사용

#### 웹 서비스 최적화
- 비동기 처리
- 캐싱 전략
- 로드 밸런싱

## 10. 향후 발전 방향

### 10.1 Vision Transformer의 발전

#### Swin Transformer
- 계층적 윈도우 어텐션
- 효율적인 계산 복잡도
- 다양한 스케일 처리

#### DeiT (Data-efficient image Transformers)
- 지식 증류를 통한 효율적 학습
- 적은 데이터로도 우수한 성능

### 10.2 멀티모달 AI의 발전

#### CLIP의 확장
- 더 큰 모델과 데이터셋
- 다양한 도메인 적용
- 제로샷 성능 향상

#### 새로운 멀티모달 모델
- DALL-E, Stable Diffusion
- 텍스트-이미지 생성
- 창의적 AI 응용

### 10.3 효율적인 AI

#### 모델 경량화
- MobileNet, EfficientNet
- 엣지 디바이스 최적화
- 실시간 추론

#### 자동화된 머신러닝
- AutoML
- 신경망 아키텍처 검색
- 하이퍼파라미터 최적화

## 결론

이번 강의를 통해 CNN의 핵심 원리와 Hugging Face 생태계의 활용 방법을 학습했습니다. 

### 주요 학습 내용
1. **CNN의 수학적 기초**: Convolution 연산, Backpropagation
2. **CNN 아키텍처의 발전**: LeNet부터 ResNet까지
3. **Hugging Face 생태계**: Transformers, Hub, API
4. **실제 응용**: Gradio를 통한 웹 서비스 구축

### 다음 단계
- 더 깊은 CNN 아키텍처 학습 (DenseNet, EfficientNet 등)
- Vision Transformer 심화 학습
- 멀티모달 AI 모델 활용
- 실제 프로젝트 구현 및 배포

### 실습 과제
1. CNN 수동 구현 및 시각화
2. Hugging Face 모델 성능 비교
3. Gradio 앱 개발 및 배포
4. 멀티모달 검색 시스템 구축

이러한 기초를 바탕으로 더 고급 딥러닝 기술과 실제 AI 서비스 개발에 도전해보시기 바랍니다.
