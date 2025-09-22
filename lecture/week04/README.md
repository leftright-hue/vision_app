# Week 04: Vision Transformer + 최신 모델 비교

## 📚 이번 주 학습 목표

### 이론 학습 목표
1. **Self-Attention 메커니즘 완벽 이해**
   - Query, Key, Value의 역할
   - Multi-Head Attention의 필요성
   - Position Encoding의 중요성

2. **Vision Transformer (ViT) 아키텍처 마스터**
   - CNN vs Transformer 비교
   - Patch Embedding 과정
   - Classification Token의 역할

3. **자기지도학습 (Self-Supervised Learning)**
   - DINO의 원리와 혁신성
   - Teacher-Student 프레임워크
   - Vision Foundation Models

4. **최신 멀티모달 모델 비교**
   - Gemini vs GPT-4V vs Llama Vision
   - 각 모델의 장단점과 사용 시나리오
   - 실전 선택 가이드

### 실습 목표
- Vision Transformer를 PyTorch로 직접 구현
- DINOv2로 강력한 이미지 특징 추출
- SAM으로 제로샷 세그멘테이션 체험
- 멀티모달 모델 성능 벤치마크
- **최종 프로젝트**: 종합 벤치마크 앱 구축

---

## 🎯 핵심 개념

### 1. Attention is All You Need (in Vision)

#### 1.1 Self-Attention의 핵심
```python
"""
Self-Attention이란?
입력의 모든 위치가 다른 모든 위치를 참조하여
관계를 학습하는 메커니즘
"""

# 핵심 수식
Attention(Q, K, V) = softmax(QK^T / √d_k) V

# Where:
# Q (Query): 무엇을 찾고 있는가?
# K (Key): 무엇을 제공할 수 있는가?
# V (Value): 실제 정보는 무엇인가?
```

#### 1.2 왜 Vision에 Transformer인가?

| 특성 | CNN | Vision Transformer |
|------|-----|-------------------|
| **Receptive Field** | 제한적 (커널 크기) | 전역적 (전체 이미지) |
| **Inductive Bias** | 강함 (locality, translation) | 약함 (더 많은 데이터 필요) |
| **계산 복잡도** | O(n) | O(n²) |
| **병렬화** | 제한적 | 매우 높음 |
| **해석가능성** | 어려움 | Attention Map 시각화 가능 |

### 2. Vision Transformer (ViT) 아키텍처

#### 2.1 이미지를 시퀀스로 변환
```python
"""
ViT의 핵심 아이디어:
이미지를 패치로 나누어 시퀀스처럼 처리
"""

# 이미지 → 패치 → 토큰
Image (224×224×3) 
→ Patches (16×16×3) × 196
→ Linear Projection (768-dim) × 196
→ + Position Embedding
→ + [CLS] Token
→ Transformer Encoder
```

#### 2.2 ViT의 주요 구성 요소
1. **Patch Embedding**: 이미지를 고정 크기 패치로 분할
2. **Position Embedding**: 패치의 위치 정보 인코딩
3. **[CLS] Token**: 전체 이미지 표현을 위한 특별 토큰
4. **Transformer Encoder**: Multi-Head Self-Attention + FFN
5. **MLP Head**: 최종 분류를 위한 헤드

### 3. DINO: Self-Supervised Vision Transformers

#### 3.1 자기지도학습의 혁명
```python
"""
DINO (Self-DIstillation with NO labels)
레이블 없이 강력한 visual representation 학습
"""

# Teacher-Student Framework
Teacher Network → Momentum Update → EMA
      ↑                               ↓
   Gradient                      Predictions
      ↑                               ↓
Student Network ← Loss ← Cross-Entropy
```

#### 3.2 DINO의 놀라운 특성
- **Semantic Segmentation**: 학습하지 않았는데도 객체 분할
- **Object Discovery**: 자동으로 객체 경계 발견
- **Fine-grained Features**: 매우 세밀한 특징 추출
- **Transfer Learning**: 다양한 태스크에 뛰어난 전이

### 4. SAM (Segment Anything Model)

#### 4.1 제로샷 세그멘테이션
```python
"""
SAM의 3가지 프롬프팅 방식:
1. Point Prompts: 클릭으로 객체 선택
2. Box Prompts: 바운딩 박스로 영역 지정
3. Mask Prompts: 기존 마스크 개선
"""
```

#### 4.2 SAM의 구조
- **Image Encoder**: ViT-based (Heavy)
- **Prompt Encoder**: 경량 (Light)
- **Mask Decoder**: 경량 (Light)
- **Data Engine**: 10억+ 마스크로 학습

---

## 💻 실습 환경 설정

### 필요한 라이브러리 설치
```bash
# 기본 라이브러리
pip install torch torchvision transformers
pip install numpy pandas matplotlib seaborn

# Vision Transformer 관련
pip install timm  # PyTorch Image Models
pip install einops  # 텐서 연산 라이브러리

# DINO & SAM
pip install git+https://github.com/facebookresearch/dino.git
pip install segment-anything

# 멀티모달 API
pip install google-generativeai
pip install together
pip install openai

# 유틸리티
pip install gradio
pip install opencv-python
pip install scikit-image
```

### GPU 설정 확인
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## 🔬 이론 파트 1: Self-Attention 메커니즘

### 1. Attention의 수학적 기초

#### 1.1 Scaled Dot-Product Attention
```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention
    
    Args:
        Q: Query tensor [batch, seq_len, d_k]
        K: Key tensor [batch, seq_len, d_k]
        V: Value tensor [batch, seq_len, d_v]
        mask: Optional mask [batch, seq_len, seq_len]
    
    Returns:
        output: Attention output [batch, seq_len, d_v]
        attention_weights: Attention weights [batch, seq_len, seq_len]
    """
    d_k = Q.size(-1)
    
    # 1. Q와 K의 dot product 계산
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]
    
    # 2. Scaling (gradient vanishing 방지)
    scores = scores / math.sqrt(d_k)
    
    # 3. Mask 적용 (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 4. Softmax로 attention weights 계산
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. Value에 weights 적용
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

#### 1.2 Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 구현
    여러 개의 attention head를 병렬로 계산
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. Linear projections in batch
        Q = self.W_q(query)  # [batch, seq_len, d_model]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Reshape for multi-head
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # [batch, n_heads, seq_len, d_k]
        
        # 3. Apply attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 5. Final linear projection
        output = self.W_o(attn_output)
        
        return output, attn_weights
```

### 2. Position Encoding

#### 2.1 Sinusoidal Position Encoding
```python
def get_sinusoidal_encoding(seq_len, d_model):
    """
    Sinusoidal position encoding
    """
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                         -(math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
```

#### 2.2 Learnable Position Encoding
```python
class LearnablePositionEncoding(nn.Module):
    """
    학습 가능한 position encoding (ViT에서 사용)
    """
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.position_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model)
        )
    
    def forward(self, x):
        return x + self.position_embedding
```

---

## 🔬 이론 파트 2: Vision Transformer 구현

### 1. 완전한 ViT 구현

```python
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 구현
    An Image is Worth 16x16 Words 논문 기반
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        
        # 이미지와 패치 정보
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.pool = pool
        
        # Patch Embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        
        # Position Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Dropout
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        # Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        # 1. Patch Embedding
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape
        
        # 2. Add CLS Token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add Position Embedding
        x += self.pos_embedding[:, :(num_patches + 1)]
        x = self.dropout(x)
        
        # 4. Transformer Encoder
        x = self.transformer(x)
        
        # 5. Classification
        if self.pool == 'cls':
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)  # Global average pooling
        
        return self.mlp_head(x)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Stack
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Block
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dim_head, dropout)
        
        # Feed-Forward Network
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x):
        # Self-Attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # FFN with residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Module
    """
    def __init__(self, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    Feed-Forward Network
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
```

### 2. ViT 사용 예제

```python
def test_vit():
    """
    Vision Transformer 테스트
    """
    # 모델 생성
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        channels=3
    )
    
    # 입력 이미지 (배치)
    img = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    preds = model(img)
    print(f"Input shape: {img.shape}")
    print(f"Output shape: {preds.shape}")
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
```

---

## 🔬 이론 파트 3: DINO와 자기지도학습

### 1. DINO 원리 이해

```python
class DINOLoss(nn.Module):
    """
    DINO Loss: Self-Distillation with NO labels
    Teacher-Student 프레임워크 기반
    """
    def __init__(self, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1))
        
    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs
        """
        student_out = student_output / self.student_temp
        teacher_out = teacher_output / self.teacher_temp
        
        # Center the teacher output
        teacher_out = teacher_out - self.center
        
        # Softmax
        student_softmax = F.log_softmax(student_out, dim=-1)
        teacher_softmax = F.softmax(teacher_out, dim=-1)
        
        # Cross-entropy loss
        loss = -torch.sum(teacher_softmax * student_softmax, dim=-1).mean()
        
        # Update center
        self.update_center(teacher_output)
        
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        EMA update of center
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)
```

### 2. DINO의 놀라운 특성 시각화

```python
def visualize_dino_attention():
    """
    DINO의 attention map 시각화
    객체 경계를 자동으로 발견하는 능력 시연
    """
    # 사전훈련된 DINO 모델 로드
    from transformers import ViTModel
    
    model = ViTModel.from_pretrained('facebook/dino-vitb16')
    
    # Attention 추출 및 시각화 코드
    # ...
```

---

## 🛠️ 실습 1: Vision Transformer 활용

### timm 라이브러리로 사전훈련 ViT 사용
```python
import timm
from PIL import Image
import torch

def use_pretrained_vit():
    """
    사전훈련된 ViT 모델 사용
    """
    # 1. 모델 로드
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    
    # 2. 데이터 전처리
    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**config)
    
    # 3. 이미지 준비
    img = Image.open('sample.jpg')
    img_tensor = transform(img).unsqueeze(0)
    
    # 4. 예측
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        top5_prob, top5_idx = probs.topk(5)
    
    # 5. 결과 출력
    for i in range(5):
        print(f"Class {top5_idx[0][i]}: {top5_prob[0][i]:.2%}")
    
    return model
```

---

## 🛠️ 실습 2: DINOv2 특징 추출

### Hugging Face를 사용한 DINOv2
```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import numpy as np

class DINOv2FeatureExtractor:
    """
    DINOv2를 사용한 강력한 이미지 특징 추출
    """
    def __init__(self, model_name='facebook/dinov2-base'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델과 프로세서 로드
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, image_path):
        """
        이미지에서 특징 추출
        """
        # 이미지 로드 및 전처리
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 특징 추출
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # CLS token features
            cls_features = outputs.last_hidden_state[:, 0]
            
            # Patch features
            patch_features = outputs.last_hidden_state[:, 1:]
            
            # Global average pooling
            global_features = patch_features.mean(dim=1)
        
        return {
            'cls_features': cls_features.cpu().numpy(),
            'global_features': global_features.cpu().numpy(),
            'patch_features': patch_features.cpu().numpy()
        }
    
    def compute_similarity(self, features1, features2):
        """
        두 이미지의 유사도 계산
        """
        # Cosine similarity
        feat1 = features1['global_features']
        feat2 = features2['global_features']
        
        similarity = np.dot(feat1, feat2.T) / (
            np.linalg.norm(feat1) * np.linalg.norm(feat2)
        )
        
        return similarity.item()
    
    def find_similar_regions(self, image_path1, image_path2):
        """
        두 이미지에서 유사한 영역 찾기
        """
        features1 = self.extract_features(image_path1)
        features2 = self.extract_features(image_path2)
        
        # Patch-level similarity
        patch_sim = np.dot(
            features1['patch_features'][0], 
            features2['patch_features'][0].T
        )
        
        return patch_sim

# 사용 예제
def demo_dinov2():
    extractor = DINOv2FeatureExtractor()
    
    # 특징 추출
    features = extractor.extract_features('image.jpg')
    print(f"CLS features shape: {features['cls_features'].shape}")
    print(f"Global features shape: {features['global_features'].shape}")
    
    # 유사도 계산
    features1 = extractor.extract_features('image1.jpg')
    features2 = extractor.extract_features('image2.jpg')
    similarity = extractor.compute_similarity(features1, features2)
    print(f"Similarity: {similarity:.3f}")
```

---

## 🛠️ 실습 3: SAM 세그멘테이션

### Segment Anything Model 활용
```python
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

class SAMSegmentation:
    """
    SAM을 사용한 제로샷 세그멘테이션
    """
    def __init__(self, model_type='vit_h', checkpoint_path='sam_vit_h.pth'):
        # SAM 모델 로드
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Predictor (프롬프트 기반 세그멘테이션)
        self.predictor = SamPredictor(self.sam)
        
        # Automatic Mask Generator (자동 세그멘테이션)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
    def segment_with_points(self, image_path, points, labels):
        """
        포인트 프롬프트로 세그멘테이션
        
        Args:
            image_path: 이미지 경로
            points: 포인트 좌표 [(x1, y1), (x2, y2), ...]
            labels: 포인트 레이블 [1, 0, ...] (1: 포함, 0: 제외)
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predictor 설정
        self.predictor.set_image(image)
        
        # 포인트로 예측
        points_np = np.array(points)
        labels_np = np.array(labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=True
        )
        
        # 최고 점수 마스크 선택
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        return best_mask, scores[best_mask_idx]
    
    def segment_with_box(self, image_path, box):
        """
        바운딩 박스로 세그멘테이션
        
        Args:
            image_path: 이미지 경로
            box: [x1, y1, x2, y2] 형태의 바운딩 박스
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predictor 설정
        self.predictor.set_image(image)
        
        # 박스로 예측
        box_np = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            box=box_np,
            multimask_output=True
        )
        
        # 최고 점수 마스크 선택
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        return best_mask, scores[best_mask_idx]
    
    def automatic_segmentation(self, image_path):
        """
        자동으로 모든 객체 세그멘테이션
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 자동 세그멘테이션
        masks = self.mask_generator.generate(image)
        
        # 크기순으로 정렬
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    def visualize_masks(self, image_path, masks):
        """
        세그멘테이션 결과 시각화
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # 각 마스크를 다른 색으로 오버레이
        for i, mask_data in enumerate(masks[:10]):  # 상위 10개만
            mask = mask_data['segmentation']
            color = np.random.random(3)
            
            # 마스크 오버레이
            h, w = mask.shape
            mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, 3)
            plt.imshow(mask_img, alpha=0.4)
        
        plt.axis('off')
        plt.title(f"Found {len(masks)} objects")
        plt.show()

# 사용 예제
def demo_sam():
    sam = SAMSegmentation()
    
    # 1. 포인트로 세그멘테이션
    mask, score = sam.segment_with_points(
        'image.jpg',
        points=[(100, 200), (150, 250)],
        labels=[1, 1]  # 모두 포함
    )
    print(f"Point segmentation score: {score:.3f}")
    
    # 2. 박스로 세그멘테이션
    mask, score = sam.segment_with_box(
        'image.jpg',
        box=[50, 50, 200, 200]
    )
    print(f"Box segmentation score: {score:.3f}")
    
    # 3. 자동 세그멘테이션
    masks = sam.automatic_segmentation('image.jpg')
    print(f"Found {len(masks)} objects")
    
    # 시각화
    sam.visualize_masks('image.jpg', masks)
```

---

## 🛠️ 실습 4: 멀티모달 모델 벤치마크

### Gemini vs Llama Vision 비교
```python
import time
import pandas as pd
from typing import Dict, List
import google.generativeai as genai
import together

class MultimodalBenchmark:
    """
    최신 멀티모달 모델 성능 비교
    """
    def __init__(self):
        # API 설정
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        together.api_key = os.getenv('TOGETHER_API_KEY')
        
        self.models = {
            'gemini': genai.GenerativeModel('gemini-1.5-flash'),
            'llama': 'meta-llama/Llama-3.2-11B-Vision-Instruct'
        }
        
        self.results = []
    
    def benchmark_image_understanding(self, image_path, tasks):
        """
        이미지 이해 능력 벤치마크
        
        Tasks:
        - Caption: 이미지 설명 생성
        - Q&A: 질문 답변
        - OCR: 텍스트 추출
        - Object Detection: 객체 탐지
        """
        results = {}
        
        for task_name, prompt in tasks.items():
            # Gemini 테스트
            start = time.time()
            gemini_response = self.test_gemini(image_path, prompt)
            gemini_time = time.time() - start
            
            # Llama Vision 테스트
            start = time.time()
            llama_response = self.test_llama(image_path, prompt)
            llama_time = time.time() - start
            
            results[task_name] = {
                'gemini': {
                    'response': gemini_response,
                    'time': gemini_time
                },
                'llama': {
                    'response': llama_response,
                    'time': llama_time
                }
            }
        
        return results
    
    def test_gemini(self, image_path, prompt):
        """Gemini 테스트"""
        try:
            image = Image.open(image_path)
            response = self.models['gemini'].generate_content([prompt, image])
            return response.text
        except Exception as e:
            return f"Error: {e}"
    
    def test_llama(self, image_path, prompt):
        """Llama Vision 테스트"""
        try:
            # 이미지를 base64로 인코딩
            import base64
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()
            
            response = together.Complete.create(
                model=self.models['llama'],
                prompt=f"<image>{image_base64}</image>\n{prompt}",
                max_tokens=512
            )
            
            return response['output']['choices'][0]['text']
        except Exception as e:
            return f"Error: {e}"
    
    def compare_accuracy(self, ground_truth, predictions):
        """
        정확도 비교 (human evaluation needed)
        """
        # 실제로는 human evaluation이나 자동 메트릭 필요
        pass
    
    def generate_report(self):
        """
        벤치마크 리포트 생성
        """
        df = pd.DataFrame(self.results)
        
        # 평균 응답 시간
        avg_times = df.groupby('model')['response_time'].mean()
        
        print("="*50)
        print("MULTIMODAL MODEL BENCHMARK REPORT")
        print("="*50)
        print("\nAverage Response Times:")
        print(avg_times)
        
        return df

# 사용 예제
def run_benchmark():
    benchmark = MultimodalBenchmark()
    
    # 테스트 태스크 정의
    tasks = {
        'caption': "Describe this image in detail.",
        'qa': "What is the main subject of this image?",
        'ocr': "Extract all text from this image.",
        'objects': "List all objects visible in this image."
    }
    
    # 벤치마크 실행
    results = benchmark.benchmark_image_understanding('test.jpg', tasks)
    
    # 결과 분석
    for task, model_results in results.items():
        print(f"\n{task.upper()}:")
        for model, data in model_results.items():
            print(f"  {model}: {data['time']:.2f}s")
            print(f"    Response: {data['response'][:100]}...")
    
    # 리포트 생성
    report = benchmark.generate_report()
```

---

## 🎯 통합 프로젝트: 멀티모달 벤치마크 앱

### 완전한 벤치마크 시스템 구현
```python
import gradio as gr
import torch
from transformers import AutoModel, AutoImageProcessor
import timm
import numpy as np
from PIL import Image
import time

class ComprehensiveBenchmarkApp:
    """
    Vision Transformer 모델과 API 종합 벤치마크 앱
    """
    def __init__(self):
        self.models = {}
        self.load_models()
        self.benchmark_results = []
    
    def load_models(self):
        """모든 모델 로드"""
        # ViT
        self.models['vit'] = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.models['vit'].eval()
        
        # DINOv2
        self.models['dino_processor'] = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.models['dino'] = AutoModel.from_pretrained('facebook/dinov2-base')
        
        # API 모델들
        # Gemini, Llama 등...
        
        print("All models loaded successfully!")
    
    def benchmark_classification(self, image):
        """분류 성능 벤치마크"""
        results = {}
        
        # ViT 테스트
        start = time.time()
        vit_preds = self.classify_with_vit(image)
        vit_time = time.time() - start
        results['ViT'] = {
            'predictions': vit_preds,
            'time': vit_time
        }
        
        return results
    
    def benchmark_feature_extraction(self, image):
        """특징 추출 성능 벤치마크"""
        results = {}
        
        # DINOv2 테스트
        start = time.time()
        dino_features = self.extract_with_dino(image)
        dino_time = time.time() - start
        results['DINOv2'] = {
            'features_shape': dino_features.shape,
            'time': dino_time
        }
        
        return results
    
    def benchmark_segmentation(self, image):
        """세그멘테이션 성능 벤치마크"""
        # SAM 테스트
        pass
    
    def classify_with_vit(self, image):
        """ViT로 분류"""
        # 전처리
        config = timm.data.resolve_data_config({}, model=self.models['vit'])
        transform = timm.data.create_transform(**config)
        img_tensor = transform(image).unsqueeze(0)
        
        # 예측
        with torch.no_grad():
            output = self.models['vit'](img_tensor)
            probs = torch.softmax(output, dim=1)
            top5_prob, top5_idx = probs.topk(5)
        
        return [(idx.item(), prob.item()) for idx, prob in zip(top5_idx[0], top5_prob[0])]
    
    def extract_with_dino(self, image):
        """DINOv2로 특징 추출"""
        inputs = self.models['dino_processor'](images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.models['dino'](inputs['pixel_values'])
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features.numpy()
    
    def create_comparison_plot(self, results):
        """결과 비교 시각화"""
        import matplotlib.pyplot as plt
        
        models = list(results.keys())
        times = [r['time'] for r in results.values()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, times)
        plt.xlabel('Model')
        plt.ylabel('Response Time (s)')
        plt.title('Model Performance Comparison')
        
        return plt.gcf()

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    app = ComprehensiveBenchmarkApp()
    
    with gr.Blocks(title="Vision Model Benchmark") as interface:
        gr.Markdown("""
        # 🔬 Vision Transformer & Multimodal Model Benchmark
        ### Compare ViT, DINOv2, SAM, and various multimodal APIs
        """)
        
        with gr.Tab("Classification Benchmark"):
            with gr.Row():
                input_image = gr.Image(label="Input Image", type="pil")
                output_results = gr.JSON(label="Classification Results")
            
            classify_btn = gr.Button("Run Classification Benchmark")
            classify_btn.click(
                app.benchmark_classification,
                inputs=[input_image],
                outputs=[output_results]
            )
        
        with gr.Tab("Feature Extraction"):
            with gr.Row():
                feat_image = gr.Image(label="Input Image", type="pil")
                feat_results = gr.JSON(label="Feature Extraction Results")
            
            feature_btn = gr.Button("Run Feature Extraction Benchmark")
            feature_btn.click(
                app.benchmark_feature_extraction,
                inputs=[feat_image],
                outputs=[feat_results]
            )
        
        with gr.Tab("Segmentation"):
            with gr.Row():
                seg_image = gr.Image(label="Input Image", type="pil")
                seg_output = gr.Image(label="Segmentation Result")
            
            seg_btn = gr.Button("Run Segmentation")
        
        with gr.Tab("Model Comparison"):
            gr.Markdown("""
            ### Head-to-Head Model Comparison
            Compare all models on the same image
            """)
            
            comp_image = gr.Image(label="Test Image", type="pil")
            comp_btn = gr.Button("Compare All Models")
            comp_output = gr.Plot(label="Performance Comparison")
    
    return interface

# 앱 실행
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)
```

---

## 📊 실습 5: 최적 모델 선택 가이드

### 모델 선택 의사결정 트리
```python
def select_optimal_model(requirements):
    """
    요구사항에 따른 최적 모델 선택
    
    Requirements:
    - task: 'classification', 'detection', 'segmentation', 'caption'
    - speed: 'realtime', 'fast', 'normal'
    - accuracy: 'high', 'medium', 'low'
    - deployment: 'cloud', 'edge', 'mobile'
    """
    
    recommendations = {
        'classification': {
            'realtime': 'MobileViT or EfficientNet',
            'fast': 'ViT-Small or DeiT',
            'normal': 'ViT-Base or Swin Transformer'
        },
        'detection': {
            'realtime': 'YOLO or EfficientDet',
            'fast': 'DETR',
            'normal': 'Mask R-CNN with ViT backbone'
        },
        'segmentation': {
            'realtime': 'MobileNetV3 + DeepLab',
            'fast': 'SegFormer',
            'normal': 'SAM or Mask2Former'
        },
        'caption': {
            'realtime': 'CLIP + GPT-2',
            'fast': 'BLIP',
            'normal': 'Gemini or GPT-4V'
        }
    }
    
    task = requirements.get('task', 'classification')
    speed = requirements.get('speed', 'normal')
    
    recommendation = recommendations.get(task, {}).get(speed, 'ViT-Base')
    
    return recommendation

# 사용 예제
requirements = {
    'task': 'segmentation',
    'speed': 'fast',
    'accuracy': 'high',
    'deployment': 'cloud'
}

model = select_optimal_model(requirements)
print(f"Recommended model: {model}")
```

---

## 📝 과제

### Assignment 4: Vision Transformer 마스터하기

#### 과제 목표
1. ViT를 처음부터 구현하고 학습시키기
2. DINOv2와 SAM을 활용한 응용 프로그램 개발
3. 멀티모달 모델 성능 비교 리포트 작성
4. 통합 벤치마크 앱 구축 및 배포

#### 평가 기준
- **구현 정확도 (30%)**: ViT 구현의 정확성
- **응용 창의성 (25%)**: DINOv2/SAM 활용 방법
- **성능 분석 (25%)**: 벤치마크 결과 분석
- **코드 품질 (10%)**: 구조화, 문서화
- **배포 완성도 (10%)**: Gradio/HF Space 배포

#### 제출 요구사항
1. 소스 코드 (GitHub)
2. 학습된 모델 체크포인트
3. 성능 비교 리포트 (PDF)
4. 배포된 앱 URL
5. 데모 비디오

---

## 🎓 학습 정리

### 핵심 개념 복습
1. **Self-Attention**: 모든 위치 간의 관계를 학습
2. **Vision Transformer**: 이미지를 패치 시퀀스로 처리
3. **DINO**: 레이블 없는 자기지도학습
4. **SAM**: 프롬프트 기반 범용 세그멘테이션

### 실습 체크리스트
- [ ] ViT 구현 및 학습
- [ ] DINOv2로 특징 추출
- [ ] SAM으로 세그멘테이션
- [ ] 멀티모달 모델 비교
- [ ] 벤치마크 앱 개발
- [ ] 결과 분석 및 시각화

### 다음 주 예습
- YOLO 시리즈 발전사
- One-stage vs Two-stage Detectors
- 실시간 객체 탐지

---

## 🔗 참고 자료

### 논문
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Segment Anything](https://arxiv.org/abs/2304.02643)

### 튜토리얼
- [Vision Transformer from Scratch](https://github.com/lucidrains/vit-pytorch)
- [DINOv2 Official Repository](https://github.com/facebookresearch/dinov2)
- [SAM Demo](https://segment-anything.com/)

### 코드 저장소
- [Week 4 완전한 코드](https://github.com/your-repo/week04)
- [사전훈련 체크포인트](https://drive.google.com/your-checkpoints)

---

## 💡 트러블슈팅

### 자주 발생하는 문제와 해결법

#### 1. ViT 학습 시 메모리 부족
```python
# 해결법: Gradient Accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. DINOv2 느린 추론 속도
```python
# 해결법: 배치 처리 및 mixed precision
with torch.cuda.amp.autocast():
    features = model(batch_images)
```

#### 3. SAM 세그멘테이션 품질 문제
```python
# 해결법: 여러 프롬프트 조합
points = [(x1, y1), (x2, y2)]  # 여러 포인트
boxes = [x1, y1, x2, y2]  # 바운딩 박스도 추가
masks = sam.predict(points=points, boxes=boxes)
```

---

**이번 주 학습을 완료하신 것을 축하합니다! 🎉**

Vision Transformer의 세계에 오신 것을 환영합니다. 다음 주에는 실시간 객체 탐지의 최신 기술을 탐구해보겠습니다.