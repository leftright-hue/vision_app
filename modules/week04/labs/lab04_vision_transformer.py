#!/usr/bin/env python3
"""
Week 4 Lab: Vision Transformer 실습
Vision Transformer (ViT) 구현과 Self-Attention 메커니즘 분석

이 실습에서는:
1. ViT 아키텍처 완전 구현
2. Self-Attention 시각화
3. 패치 임베딩 분석
4. CNN vs ViT 성능 비교
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTModel, ViTImageProcessor
import time
import warnings
warnings.filterwarnings('ignore')

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {DEVICE}")

class PatchEmbedding(nn.Module):
    """
    이미지를 패치로 분할하고 임베딩하는 클래스
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # 컨볼루션으로 패치 임베딩 구현 (더 효율적)
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 패치 임베딩
        x = self.projection(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2)        # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)   # [B, num_patches, embed_dim]
        
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention 구현
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V를 한 번에 계산하기 위한 선형 레이어
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        
        # Q, K, V 계산
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention 점수 계산
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Value와 가중합
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        if return_attention:
            return x, attn
        return x

class TransformerBlock(nn.Module):
    """
    Transformer 블록 (Self-Attention + MLP)
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, return_attention=False):
        # Self-Attention with residual connection
        if return_attention:
            attn_output, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_output
        else:
            x = x + self.attn(self.norm1(x))
            attn_weights = None
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x

class VisionTransformer(nn.Module):
    """
    완전한 Vision Transformer 구현
    """
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # 1. 패치 임베딩
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        
        # 2. 클래스 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. 위치 임베딩
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # 4. Transformer 블록들
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # 5. 정규화 및 분류 헤드
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 가중치 초기화
        self.init_weights()
    
    def init_weights(self):
        """가중치 초기화"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 분류 헤드 초기화
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
    
    def forward(self, x, return_attention=False):
        B = x.shape[0]
        
        # 1. 패치 임베딩
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # 2. 클래스 토큰 추가
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # 3. 위치 임베딩 추가
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # 4. Transformer 블록들 통과
        attention_weights = []
        for i, block in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:  # 마지막 블록의 attention만 반환
                x, attn = block(x, return_attention=True)
                attention_weights.append(attn)
            else:
                x = block(x)
        
        # 5. 정규화
        x = self.norm(x)
        
        # 6. 분류 (CLS 토큰 사용)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        if return_attention:
            return logits, attention_weights[0] if attention_weights else None
        return logits

class ViTAnalyzer:
    """
    ViT 분석 및 시각화 도구
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def visualize_patches(self, image, patch_size=16):
        """이미지 패치 분할 시각화"""
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image.squeeze())
        
        img_array = np.array(image)
        H, W = img_array.shape[:2]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 원본 이미지
        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title('원본 이미지')
        axes[0, 0].axis('off')
        
        # 패치 그리드 표시
        axes[0, 1].imshow(img_array)
        for i in range(0, H, patch_size):
            axes[0, 1].axhline(y=i, color='red', linewidth=1)
        for j in range(0, W, patch_size):
            axes[0, 1].axvline(x=j, color='red', linewidth=1)
        axes[0, 1].set_title(f'패치 그리드 ({patch_size}x{patch_size})')
        axes[0, 1].axis('off')
        
        # 개별 패치들 샘플
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # 몇 개 패치만 선택해서 표시
        sample_patches = []
        positions = [(2, 2), (2, 8), (8, 2), (8, 8)]  # 4개 코너 근처
        
        for idx, (i, j) in enumerate(positions):
            if i < num_patches_h and j < num_patches_w:
                patch = img_array[i*patch_size:(i+1)*patch_size, 
                                j*patch_size:(j+1)*patch_size]
                sample_patches.append(patch)
        
        # 패치들을 하나의 이미지로 합치기
        if sample_patches:
            patch_grid = np.concatenate([
                np.concatenate(sample_patches[:2], axis=1),
                np.concatenate(sample_patches[2:], axis=1)
            ], axis=0)
            
            axes[1, 0].imshow(patch_grid)
            axes[1, 0].set_title('샘플 패치들')
            axes[1, 0].axis('off')
        
        # 패치 임베딩 차원 정보
        embed_info = f"""패치 정보:
        - 이미지 크기: {H}x{W}
        - 패치 크기: {patch_size}x{patch_size}
        - 패치 개수: {num_patches_h}x{num_patches_w} = {num_patches_h * num_patches_w}
        - 패치 차원: {3 * patch_size * patch_size}
        - 임베딩 차원: {self.model.embed_dim}"""
        
        axes[1, 1].text(0.1, 0.5, embed_info, fontsize=12, 
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('패치 임베딩 정보')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_attention(self, image, layer_idx=-1, head_idx=0):
        """Attention 가중치 시각화"""
        # 이미지 전처리
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if isinstance(image, Image.Image):
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        else:
            input_tensor = image.to(DEVICE)
        
        # Attention 가중치 추출
        with torch.no_grad():
            _, attention_weights = self.model(input_tensor, return_attention=True)
        
        if attention_weights is None:
            print("Attention 가중치를 가져올 수 없습니다.")
            return
        
        # 특정 헤드의 attention 가중치 선택
        attn = attention_weights[0, head_idx].cpu().numpy()  # [seq_len, seq_len]
        
        # CLS 토큰에 대한 attention (첫 번째 행)
        cls_attention = attn[0, 1:]  # CLS 토큰이 다른 패치들에 주는 attention
        
        # Attention을 이미지 형태로 reshape
        grid_size = int(np.sqrt(len(cls_attention)))
        attention_map = cls_attention.reshape(grid_size, grid_size)
        
        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 원본 이미지
        if isinstance(image, Image.Image):
            axes[0, 0].imshow(image)
        else:
            # 정규화 해제
            img_denorm = input_tensor.squeeze().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img_denorm * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            axes[0, 0].imshow(img_denorm.permute(1, 2, 0))
        
        axes[0, 0].set_title('원본 이미지')
        axes[0, 0].axis('off')
        
        # Attention 히트맵
        im1 = axes[0, 1].imshow(attention_map, cmap='hot', interpolation='nearest')
        axes[0, 1].set_title(f'CLS Token Attention (Head {head_idx})')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Attention을 원본 이미지 크기로 업샘플링
        from scipy.ndimage import zoom
        attention_upsampled = zoom(attention_map, (224/grid_size, 224/grid_size), order=1)
        
        # 원본 이미지와 오버레이
        axes[0, 2].imshow(image if isinstance(image, Image.Image) else 
                         img_denorm.permute(1, 2, 0), alpha=0.7)
        im2 = axes[0, 2].imshow(attention_upsampled, cmap='hot', alpha=0.5)
        axes[0, 2].set_title('Attention Overlay')
        axes[0, 2].axis('off')
        
        # 전체 attention 행렬
        im3 = axes[1, 0].imshow(attn, cmap='Blues')
        axes[1, 0].set_title('전체 Attention Matrix')
        axes[1, 0].set_xlabel('Key Position')
        axes[1, 0].set_ylabel('Query Position')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 패치별 attention 분포
        patch_attention_avg = np.mean(attn[1:, 1:], axis=0)  # 패치들 간의 평균 attention
        patch_attention_2d = patch_attention_avg.reshape(grid_size, grid_size)
        
        im4 = axes[1, 1].imshow(patch_attention_2d, cmap='viridis')
        axes[1, 1].set_title('패치 간 평균 Attention')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Attention 통계
        stats_text = f"""Attention 통계:
        - 헤드 개수: {attention_weights.shape[1]}
        - 시퀀스 길이: {attn.shape[0]}
        - 패치 그리드: {grid_size}x{grid_size}
        - CLS attention 평균: {cls_attention.mean():.4f}
        - CLS attention 표준편차: {cls_attention.std():.4f}
        - 최대 attention: {cls_attention.max():.4f}
        - 최소 attention: {cls_attention.min():.4f}"""
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10,
                        verticalalignment='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Attention 분석')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_positional_encoding(self):
        """위치 임베딩 분석"""
        pos_embed = self.model.pos_embed.data.squeeze().cpu().numpy()  # [num_patches+1, embed_dim]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 위치 임베딩 히트맵
        im1 = axes[0, 0].imshow(pos_embed.T, cmap='RdBu', aspect='auto')
        axes[0, 0].set_title('위치 임베딩 히트맵')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Embedding Dimension')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # CLS 토큰 vs 패치 위치 임베딩 비교
        cls_embed = pos_embed[0]  # CLS 토큰 임베딩
        patch_embeds = pos_embed[1:]  # 패치 임베딩들
        
        # 유사도 계산
        similarities = np.dot(patch_embeds, cls_embed) / (
            np.linalg.norm(patch_embeds, axis=1) * np.linalg.norm(cls_embed)
        )
        
        grid_size = int(np.sqrt(len(similarities)))
        similarity_map = similarities.reshape(grid_size, grid_size)
        
        im2 = axes[0, 1].imshow(similarity_map, cmap='coolwarm')
        axes[0, 1].set_title('CLS 토큰과의 위치 임베딩 유사도')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 인접 패치 간 유사도
        adjacent_similarities = []
        for i in range(len(patch_embeds) - 1):
            sim = np.dot(patch_embeds[i], patch_embeds[i+1]) / (
                np.linalg.norm(patch_embeds[i]) * np.linalg.norm(patch_embeds[i+1])
            )
            adjacent_similarities.append(sim)
        
        axes[1, 0].plot(adjacent_similarities)
        axes[1, 0].set_title('인접 패치 간 위치 임베딩 유사도')
        axes[1, 0].set_xlabel('패치 인덱스')
        axes[1, 0].set_ylabel('유사도')
        axes[1, 0].grid(True)
        
        # 위치 임베딩 통계
        stats_text = f"""위치 임베딩 통계:
        - 총 위치 개수: {pos_embed.shape[0]}
        - 임베딩 차원: {pos_embed.shape[1]}
        - CLS 임베딩 norm: {np.linalg.norm(cls_embed):.4f}
        - 패치 임베딩 평균 norm: {np.mean([np.linalg.norm(p) for p in patch_embeds]):.4f}
        - 최대 유사도: {similarities.max():.4f}
        - 최소 유사도: {similarities.min():.4f}
        - 평균 인접 유사도: {np.mean(adjacent_similarities):.4f}"""
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12,
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('위치 임베딩 분석')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

def compare_cnn_vs_vit():
    """CNN과 ViT 성능 비교"""
    print("🔥 CNN vs ViT 성능 비교 실험")
    print("=" * 50)
    
    # 모델 로드
    try:
        # ResNet-50 (CNN)
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True).to(DEVICE)
        resnet.eval()
        
        # ViT-Base (Transformer)
        vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(DEVICE)
        vit_model.eval()
        
        # 커스텀 ViT
        custom_vit = VisionTransformer(
            img_size=224, patch_size=16, num_classes=1000,
            embed_dim=768, depth=12, num_heads=12
        ).to(DEVICE)
        custom_vit.eval()
        
        print("✅ 모든 모델이 성공적으로 로드되었습니다.")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 테스트 이미지 생성
    test_images = torch.randn(10, 3, 224, 224).to(DEVICE)
    
    # 성능 측정 함수
    def measure_performance(model, inputs, model_name, num_runs=50):
        model.eval()
        times = []
        
        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(inputs[:1])
        
        # 실제 측정
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(inputs)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # ms
        
        return {
            'model': model_name,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    # 성능 측정
    results = []
    
    print("\n📊 추론 시간 측정 중...")
    
    # ResNet-50
    resnet_result = measure_performance(resnet, test_images, "ResNet-50")
    results.append(resnet_result)
    print(f"ResNet-50: {resnet_result['avg_time']:.2f}ms ± {resnet_result['std_time']:.2f}ms")
    
    # ViT (HuggingFace)
    vit_result = measure_performance(vit_model, test_images, "ViT-Base (HF)")
    results.append(vit_result)
    print(f"ViT-Base (HF): {vit_result['avg_time']:.2f}ms ± {vit_result['std_time']:.2f}ms")
    
    # Custom ViT
    custom_vit_result = measure_performance(custom_vit, test_images, "Custom ViT")
    results.append(custom_vit_result)
    print(f"Custom ViT: {custom_vit_result['avg_time']:.2f}ms ± {custom_vit_result['std_time']:.2f}ms")
    
    # 메모리 사용량 측정 (CUDA 사용 시)
    if torch.cuda.is_available():
        print("\n💾 메모리 사용량 측정 중...")
        
        def measure_memory(model, inputs, model_name):
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(inputs[:1])  # 단일 이미지로 측정
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return memory_mb
        
        for result in results:
            if result['model'] == "ResNet-50":
                result['memory'] = measure_memory(resnet, test_images, "ResNet-50")
            elif result['model'] == "ViT-Base (HF)":
                result['memory'] = measure_memory(vit_model, test_images, "ViT-Base (HF)")
            elif result['model'] == "Custom ViT":
                result['memory'] = measure_memory(custom_vit, test_images, "Custom ViT")
            
            print(f"{result['model']}: {result['memory']:.2f}MB")
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    # 추론 시간 비교
    plt.subplot(1, 3, 1)
    models = [r['model'] for r in results]
    times = [r['avg_time'] for r in results]
    errors = [r['std_time'] for r in results]
    
    bars = plt.bar(models, times, yerr=errors, capsize=5, alpha=0.7, 
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('추론 시간 비교')
    plt.ylabel('시간 (ms)')
    plt.xticks(rotation=45)
    
    # 막대 위에 값 표시
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{time_val:.1f}ms', ha='center', va='bottom')
    
    # 메모리 사용량 비교 (CUDA 사용 시)
    if torch.cuda.is_available() and 'memory' in results[0]:
        plt.subplot(1, 3, 2)
        memories = [r['memory'] for r in results]
        bars = plt.bar(models, memories, alpha=0.7, 
                      color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('메모리 사용량 비교')
        plt.ylabel('메모리 (MB)')
        plt.xticks(rotation=45)
        
        for bar, mem_val in zip(bars, memories):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{mem_val:.1f}MB', ha='center', va='bottom')
    
    # 모델 파라미터 수 비교
    plt.subplot(1, 3, 3)
    param_counts = []
    
    for result in results:
        if result['model'] == "ResNet-50":
            params = sum(p.numel() for p in resnet.parameters()) / 1e6
        elif result['model'] == "ViT-Base (HF)":
            params = sum(p.numel() for p in vit_model.parameters()) / 1e6
        elif result['model'] == "Custom ViT":
            params = sum(p.numel() for p in custom_vit.parameters()) / 1e6
        param_counts.append(params)
    
    bars = plt.bar(models, param_counts, alpha=0.7,
                  color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('모델 파라미터 수')
    plt.ylabel('파라미터 (M)')
    plt.xticks(rotation=45)
    
    for bar, param_val in zip(bars, param_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{param_val:.1f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 결과 요약
    print("\n📋 성능 비교 요약")
    print("=" * 50)
    for result in results:
        print(f"{result['model']:15s}: {result['avg_time']:6.2f}ms", end="")
        if 'memory' in result:
            print(f", {result['memory']:6.2f}MB", end="")
        print()
    
    return results

def main():
    """메인 실습 함수"""
    print("🤖 Week 4: Vision Transformer 실습")
    print("=" * 50)
    
    # 1. 커스텀 ViT 모델 생성
    print("\n1️⃣ Vision Transformer 모델 생성")
    vit_model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12
    ).to(DEVICE)
    
    print(f"✅ ViT 모델 생성 완료")
    print(f"   - 파라미터 수: {sum(p.numel() for p in vit_model.parameters()) / 1e6:.1f}M")
    print(f"   - 패치 개수: {vit_model.num_patches}")
    print(f"   - 임베딩 차원: {vit_model.embed_dim}")
    
    # 2. 분석 도구 생성
    analyzer = ViTAnalyzer(vit_model)
    
    # 3. 테스트 이미지 로드 (또는 생성)
    print("\n2️⃣ 테스트 이미지 준비")
    try:
        # 샘플 이미지 생성 (실제로는 실제 이미지를 사용하는 것이 좋습니다)
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # 간단한 패턴 그리기
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        
        # 체크보드 패턴
        for i in range(0, 224, 32):
            for j in range(0, 224, 32):
                if (i//32 + j//32) % 2 == 0:
                    draw.rectangle([i, j, i+32, j+32], fill='black')
        
        # 중앙에 원 그리기
        draw.ellipse([80, 80, 144, 144], fill='red')
        
        print("✅ 테스트 이미지 생성 완료")
        
    except Exception as e:
        print(f"❌ 이미지 생성 실패: {e}")
        return
    
    # 4. 패치 분할 시각화
    print("\n3️⃣ 패치 분할 시각화")
    analyzer.visualize_patches(test_image)
    
    # 5. Attention 시각화
    print("\n4️⃣ Self-Attention 시각화")
    analyzer.visualize_attention(test_image, head_idx=0)
    
    # 6. 위치 임베딩 분석
    print("\n5️⃣ 위치 임베딩 분석")
    analyzer.analyze_positional_encoding()
    
    # 7. CNN vs ViT 성능 비교
    print("\n6️⃣ CNN vs ViT 성능 비교")
    comparison_results = compare_cnn_vs_vit()
    
    print("\n🎉 모든 실습이 완료되었습니다!")
    print("\n📚 추가 실험 아이디어:")
    print("   - 다양한 패치 크기 실험 (8x8, 32x32)")
    print("   - 다른 헤드의 Attention 패턴 분석")
    print("   - 실제 이미지 데이터셋으로 테스트")
    print("   - Fine-tuning 실험")

if __name__ == "__main__":
    main()

