# Week 4: Vision Transformer + 최신 모델 비교

## 강의 슬라이드

---

# 📚 4주차 학습 목표

## 오늘 배울 내용

1. **Self-Attention 메커니즘의 이해**
2. **Vision Transformer (ViT) 아키텍처**
3. **DINO와 자기지도학습**
4. **SAM과 범용 세그멘테이션**
5. **멀티모달 모델 벤치마크**

---

# Part 1: Self-Attention 메커니즘

## 🧠 Attention is All You Need

### 핵심 아이디어
> "모든 위치가 다른 모든 위치를 참조할 수 있다"

### CNN vs Attention
- **CNN**: Local receptive field → 점진적 확장
- **Attention**: Global receptive field from the start

### 수식 표현
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

---

## 📐 Self-Attention 계산 과정

### Step 1: Query, Key, Value 생성
```python
# 입력 X에서 Q, K, V 계산
Q = X @ W_q  # Query
K = X @ W_k  # Key  
V = X @ W_v  # Value
```

### Step 2: Attention Score 계산
```python
# 유사도 계산
scores = Q @ K.T / sqrt(d_k)
attention_weights = softmax(scores)
```

### Step 3: Weighted Sum
```python
# 가중합 계산
output = attention_weights @ V
```

---

## 🎯 Multi-Head Attention

### 왜 Multi-Head인가?

```
Single Head: 하나의 관점
Multi-Head: 여러 관점에서 동시에 학습
```

### 병렬 처리의 장점
- **Head 1**: 텍스처 패턴 학습
- **Head 2**: 엣지 정보 포착
- **Head 3**: 색상 관계 모델링
- **Head N**: 고수준 의미 정보

### 구현
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.heads = n_heads
        self.d_k = d_model // n_heads
        
    def forward(self, x):
        # Split into multiple heads
        # Apply attention per head
        # Concatenate results
```

---

# Part 2: Vision Transformer (ViT)

## 🖼️ 이미지를 시퀀스로

### 핵심 혁신
"An Image is Worth 16x16 Words"

### 패치 분할 과정
```
Original: 224×224×3
    ↓
Patches: 14×14 patches of 16×16×3
    ↓
Flatten: 196 patches × 768 dims
    ↓
Add [CLS] token: 197 × 768
```

### 위치 인코딩의 중요성
- 패치의 공간적 위치 정보 보존
- Learnable vs Sinusoidal encoding

---

## 🏗️ ViT 아키텍처

### 전체 구조
```
Input Image
    ↓
Patch Embedding + Position Encoding
    ↓
Transformer Encoder × L
    ↓
MLP Head
    ↓
Class Prediction
```

### 레이어 구성
```python
class ViTBlock(nn.Module):
    def __init__(self):
        self.norm1 = LayerNorm()
        self.attn = MultiHeadAttention()
        self.norm2 = LayerNorm()
        self.mlp = MLP()
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## 📊 ViT vs CNN 성능 비교

### 데이터 효율성

| Dataset Size | CNN (ResNet) | ViT-B/16 |
|-------------|--------------|----------|
| 10K images | 70% | 55% |
| 100K images | 78% | 73% |
| 1M images | 82% | 84% |
| 14M images | 84% | 88% |

### Key Insights
- **Small data**: CNN이 우세 (inductive bias)
- **Large data**: ViT가 우세 (flexibility)
- **Pre-training**: ViT에 필수적

---

## 🚀 ViT 변형 모델들

### Model Zoo

```
ViT-Tiny:   12 layers, 192 dim, 3 heads
ViT-Small:  12 layers, 384 dim, 6 heads  
ViT-Base:   12 layers, 768 dim, 12 heads
ViT-Large:  24 layers, 1024 dim, 16 heads
ViT-Huge:   32 layers, 1280 dim, 16 heads
```

### 효율성 개선
- **DeiT**: Data-efficient training
- **Swin Transformer**: Hierarchical architecture
- **CvT**: Convolutional vision transformer
- **PVT**: Pyramid vision transformer

---

# Part 3: DINO - 자기지도학습

## 🦖 DINO란?

### Self-DIstillation with NO labels

```
Teacher Model → Predictions
      ↓            ↓
   Momentum     Knowledge
    Update      Transfer
      ↓            ↓
Student Model → Predictions
```

### 핵심 특징
- 라벨 없이 학습
- Teacher-Student framework
- Vision Transformer 백본
- 범용 특징 학습

---

## 📚 DINO 학습 과정

### Knowledge Distillation
```python
# Teacher: Momentum updated
teacher_params = momentum * teacher_params + 
                (1 - momentum) * student_params

# Student: Gradient updated
loss = cross_entropy(student_output, 
                    teacher_output.detach())
```

### Multi-crop Strategy
- **Global views**: 2개 (224×224)
- **Local views**: 8개 (96×96)
- 다양한 스케일에서 일관된 표현 학습

---

## 🎯 DINOv2 개선사항

### 주요 업그레이드
1. **더 큰 데이터셋**: 142M 이미지
2. **개선된 아키텍처**: ViT-g (1.1B params)
3. **더 나은 증강**: Advanced augmentations
4. **Self-supervised objectives**: Multiple tasks

### 성능 향상
```
Task           | DINOv1 | DINOv2
---------------|--------|--------
ImageNet       | 82.8%  | 86.3%
Segmentation   | 45.1   | 49.2 mIoU
Depth          | 0.417  | 0.356 RMSE
Retrieval      | 89.5%  | 94.2%
```

---

## 💡 DINO 활용 사례

### 1. Feature Extraction
```python
features = dino_model.forward_features(image)
# Shape: [B, 768] for ViT-B
```

### 2. Semantic Segmentation
```python
patch_features = get_patch_features(image)
clusters = kmeans.fit_predict(patch_features)
```

### 3. Image Retrieval
```python
query_features = extract_features(query_image)
similarities = compute_similarity(query_features, 
                                 database_features)
```

### 4. Few-shot Learning
- 적은 샘플로도 높은 성능
- Pre-trained features 활용

---

# Part 4: SAM - Segment Anything

## ✂️ SAM의 혁신

### Promptable Segmentation
> "어떤 프롬프트든, 어떤 객체든 세그멘테이션"

### 핵심 컴포넌트
```
Image Encoder (ViT)
      ↓
Prompt Encoder
      ↓
Mask Decoder
      ↓
Segmentation Masks
```

### 프롬프트 타입
- **Points**: Positive/Negative clicks
- **Boxes**: Bounding rectangles
- **Masks**: Rough masks to refine
- **Text**: Natural language (SAM 2)

---

## 🏗️ SAM 아키텍처

### Image Encoder
```python
# ViT-H backbone
image_encoder = ViT(
    img_size=1024,
    patch_size=16,
    embed_dim=1280,
    depth=32,
    n_heads=16
)
```

### Prompt Encoder
```python
# Handles different prompt types
def encode_prompt(prompt_type, prompt_data):
    if prompt_type == "point":
        return embed_points(prompt_data)
    elif prompt_type == "box":
        return embed_box(prompt_data)
    # ...
```

### Mask Decoder
- Transformer-based decoder
- Outputs multiple masks with scores
- IoU prediction head

---

## 📊 SAM 데이터셋 - SA-1B

### 규모
- **11M 이미지**
- **1.1B 마스크**
- **평균 100 마스크/이미지**

### Data Engine
```
Stage 1: Model-Assisted Manual
    ↓
Stage 2: Semi-Automatic
    ↓  
Stage 3: Fully Automatic
```

### 품질 메트릭
- IoU with ground truth: 94.6%
- 인간 평가자와 일치율: 89%

---

## 🚀 SAM 응용 분야

### 1. Interactive Segmentation
```python
predictor.set_image(image)
masks = predictor.predict(
    point_coords=[[x, y]],
    point_labels=[1],  # 1: foreground
)
```

### 2. Everything Mode
```python
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
# Returns all possible masks
```

### 3. Video Segmentation (SAM 2)
- Temporal consistency
- Object tracking
- Real-time processing

### 4. 3D Segmentation
- Point cloud segmentation
- Medical imaging
- Robotics applications

---

# Part 5: 멀티모달 모델 비교

## 🏆 주요 경쟁자들

### API 기반 모델

| Model | Company | Strengths | Weaknesses |
|-------|---------|-----------|------------|
| **Gemini Vision** | Google | Speed, Multilingual | Limited control |
| **GPT-4V** | OpenAI | Reasoning, Accuracy | Cost, Speed |
| **Claude Vision** | Anthropic | Safety, Analysis | Availability |
| **Llama Vision** | Meta | Open source | Deployment complexity |

### 오픈소스 모델
- **BLIP-2**: Bootstrapped language-image pretraining
- **LLaVA**: Large language and vision assistant
- **MiniGPT-4**: Lightweight multimodal model

---

## 📈 성능 벤치마크

### Image Captioning (COCO)

```
Model      | BLEU-4 | METEOR | CIDEr
-----------|--------|--------|-------
GPT-4V     | 40.3   | 31.2   | 138.2
Gemini     | 39.7   | 30.8   | 136.5
BLIP-2     | 38.1   | 29.9   | 133.7
LLaVA      | 36.2   | 28.7   | 128.3
```

### Visual Question Answering (VQA v2)

```
Model      | Accuracy | Yes/No | Number | Other
-----------|----------|--------|--------|-------
GPT-4V     | 82.1%    | 95.2%  | 61.3%  | 73.8%
Gemini     | 81.3%    | 94.7%  | 59.8%  | 72.1%
Claude     | 80.7%    | 94.1%  | 58.2%  | 71.5%
```

---

## 💰 비용 분석

### API Pricing (per 1K images)

```python
pricing = {
    'gpt4v': {
        'input': 0.01,    # per image
        'output': 0.03    # per 1K tokens
    },
    'gemini': {
        'input': 0.0025,  # per image
        'output': 0.01    # per 1K tokens
    },
    'claude': {
        'input': 0.008,   # per image
        'output': 0.024   # per 1K tokens
    }
}
```

### 비용 최적화 전략
1. **Caching**: 반복 쿼리 캐싱
2. **Batching**: 배치 처리로 효율 개선
3. **Model Selection**: 태스크별 최적 모델
4. **Hybrid Approach**: 오픈소스 + API 조합

---

## 🎯 모델 선택 가이드

### Decision Tree
```
요구사항 분석
├─ 실시간 처리 필요?
│  ├─ Yes → 로컬 모델 (ViT, DINO, SAM)
│  └─ No → API 가능
├─ 높은 정확도 필수?
│  ├─ Yes → GPT-4V, Gemini
│  └─ No → 오픈소스 모델
├─ 비용 민감?
│  ├─ Yes → 오픈소스 우선
│  └─ No → 최고 성능 모델
└─ 커스터마이징 필요?
   ├─ Yes → 오픈소스 (fine-tuning)
   └─ No → API 사용
```

---

# 실습 및 과제

## 🧪 Lab 4: 통합 실습

### 실습 목표
1. ViT 어텐션 시각화
2. DINOv2로 특징 추출
3. SAM으로 세그멘테이션
4. 멀티모달 API 벤치마크

### 단계별 가이드
```python
# Step 1: ViT 어텐션
attention_maps = vit_model.get_attention()
visualize_attention(attention_maps)

# Step 2: DINO 특징
features = dino_model.extract_features(image)
similar_images = find_similar(features)

# Step 3: SAM 세그멘테이션
masks = sam_model.segment(image, prompts)

# Step 4: API 벤치마크
results = benchmark_apis(image, task)
```

---

## 📝 Assignment 4: 멀티모달 벤치마크 앱

### 요구사항
1. **ViT 구현 및 분석**
2. **DINOv2 활용 시스템**
3. **SAM 통합 인터페이스**
4. **API 모델 벤치마크**
5. **통합 대시보드**

### 평가 기준
- 구현 완성도: 40%
- 성능 분석: 25%
- UI/UX: 15%
- 문서화: 10%
- 창의성: 10%

### 제출물
- GitHub 저장소
- Hugging Face Space 배포
- 벤치마크 리포트 (PDF)
- 5분 발표 자료

---

## 🔑 핵심 정리

### 오늘 배운 내용

✅ **Self-Attention**: 전역적 관계 모델링
✅ **Vision Transformer**: 이미지를 시퀀스로
✅ **DINO**: 라벨 없는 학습의 힘
✅ **SAM**: 범용 세그멘테이션
✅ **멀티모달 벤치마크**: 최적 모델 선택

### Key Takeaways
1. ViT는 대규모 데이터에서 CNN을 능가
2. 자기지도학습이 만드는 범용 특징
3. Promptable 모델의 미래
4. 태스크별 최적 모델이 다름

---

## 🚀 다음 주 예고

### Week 5: 프로젝트 발표 및 마무리

**준비사항**:
- 최종 프로젝트 완성
- 발표 자료 준비
- 코드 정리 및 문서화

**발표 내용**:
- 프로젝트 아이디어 및 목표
- 기술적 구현 상세
- 실험 결과 및 분석
- 데모 시연
- Q&A

---

## 💬 Q&A

### 자주 묻는 질문

**Q1: ViT가 항상 CNN보다 좋나요?**
- 데이터가 충분하면 ViT
- 작은 데이터셋은 CNN
- Hybrid 모델도 좋은 선택

**Q2: DINO vs Supervised Learning?**
- DINO: 범용성, 라벨 불필요
- Supervised: 특정 태스크 최적화
- 둘 다 사용하는 것이 최선

**Q3: SAM의 한계는?**
- 투명 객체 어려움
- 매우 작은 객체 제한
- 텍스트 프롬프트 미지원 (SAM 1)

---

## 📚 참고 자료

### 핵심 논문
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
3. [Emerging Properties in Self-Supervised ViT](https://arxiv.org/abs/2104.14294)
4. [DINOv2](https://arxiv.org/abs/2304.07193)
5. [Segment Anything](https://arxiv.org/abs/2304.02643)

### 구현 리소스
- [timm library](https://github.com/rwightman/pytorch-image-models)
- [DINOv2 official](https://github.com/facebookresearch/dinov2)
- [SAM official](https://github.com/facebookresearch/segment-anything)
- [Transformers library](https://huggingface.co/transformers)

### 튜토리얼
- [ViT from Scratch](https://github.com/lucidrains/vit-pytorch)
- [DINO Tutorial](https://github.com/facebookresearch/dino)
- [SAM Demo](https://segment-anything.com/demo)

---

# Thank You! 🙏

## 수고하셨습니다!

### 연락처
- 이메일: newmind68@hs.ac.kr
- 오피스 아워: 수요일 14:00-16:00

### 온라인 리소스
- 강의 자료: [Course GitHub]
- 질문 포럼: [Course Discord]
- 과제 제출: [Assignment Portal]

### 다음 주 준비
- 프로젝트 최종 점검
- 발표 리허설
- 동료 평가 준비

---