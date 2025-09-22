"""
Vision Transformer (ViT) 구현
교육용 순수 PyTorch 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """이미지를 패치로 분할하고 임베딩"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 패치 추출 및 선형 투영
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 이미지 텐서
        Returns:
            [B, N, D] 패치 임베딩 (N = 패치 개수, D = 임베딩 차원)
        """
        return self.projection(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention 메커니즘"""
    
    def __init__(
        self,
        embed_dim: int = 768,
        n_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V 투영
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        # 출력 투영
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] 입력 텐서
            mask: 선택적 어텐션 마스크
        Returns:
            [B, N, D] 출력 텐서
        """
        B, N, D = x.shape
        
        # Q, K, V 계산
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 어텐션 스코어 계산
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 어텐션 적용
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        
        # 출력 투영
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer 인코더 블록"""
    
    def __init__(
        self,
        embed_dim: int = 768,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # MLP
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] 입력 텐서
        Returns:
            [B, N, D] 출력 텐서
        """
        # Self-Attention with residual
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) 전체 모델"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        representation_size: Optional[int] = None,
    ):
        super().__init__()
        
        # 패치 임베딩
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # 위치 임베딩
        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer 인코더
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # 최종 Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 분류 헤드
        if representation_size is not None:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
            self.head = nn.Linear(representation_size, num_classes)
        else:
            self.pre_logits = nn.Identity()
            self.head = nn.Linear(embed_dim, num_classes)
        
        # 가중치 초기화
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        # 위치 임베딩 초기화
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 선형층 초기화
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """특징 추출"""
        B = x.shape[0]
        
        # 패치 임베딩
        x = self.patch_embed(x)  # [B, N, D]
        
        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        
        # 위치 임베딩 추가
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer 인코더
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # CLS 토큰 반환
        return x[:, 0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """전체 forward pass"""
        # 특징 추출
        x = self.forward_features(x)
        
        # 분류
        x = self.pre_logits(x)
        x = self.head(x)
        
        return x


class ViTWithAttentionVisualization(VisionTransformer):
    """어텐션 시각화가 가능한 ViT"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = []
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: 입력 이미지
            return_attention: 어텐션 가중치 반환 여부
        """
        if not return_attention:
            return super().forward(x)
        
        # 어텐션 가중치 저장 모드
        self.attention_weights = []
        B = x.shape[0]
        
        # 패치 임베딩
        x = self.patch_embed(x)
        
        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 위치 임베딩
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # 각 블록의 어텐션 가중치 저장
        for block in self.blocks:
            # 어텐션 계산을 위한 커스텀 forward
            residual = x
            x = block.norm1(x)
            
            # 어텐션 가중치 추출
            B, N, D = x.shape
            qkv = block.attn.qkv(x).reshape(B, N, 3, block.attn.n_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = F.softmax(attn, dim=-1)
            self.attention_weights.append(attn)
            
            # 나머지 forward
            attn_out = attn @ v
            attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
            attn_out = block.attn.proj(attn_out)
            attn_out = block.attn.dropout(attn_out)
            
            x = residual + attn_out
            x = x + block.mlp(block.norm2(x))
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.pre_logits(x)
        x = self.head(x)
        
        return x, self.attention_weights


def create_vit_model(
    model_size: str = 'base',
    img_size: int = 224,
    num_classes: int = 1000,
    pretrained: bool = False
) -> VisionTransformer:
    """
    ViT 모델 생성
    
    Args:
        model_size: 'tiny', 'small', 'base', 'large', 'huge' 중 선택
        img_size: 입력 이미지 크기
        num_classes: 분류 클래스 수
        pretrained: 사전훈련 가중치 사용 여부
    """
    configs = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'n_heads': 3,
            'patch_size': 16
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'n_heads': 6,
            'patch_size': 16
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'n_heads': 12,
            'patch_size': 16
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'n_heads': 16,
            'patch_size': 16
        },
        'huge': {
            'embed_dim': 1280,
            'depth': 32,
            'n_heads': 16,
            'patch_size': 14
        }
    }
    
    config = configs[model_size]
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=config['patch_size'],
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        n_heads=config['n_heads']
    )
    
    if pretrained:
        # 실제 환경에서는 사전훈련 가중치 로드
        print(f"Note: Pretrained weights for ViT-{model_size} would be loaded here")
        # state_dict = torch.hub.load_state_dict_from_url(...)
        # model.load_state_dict(state_dict)
    
    return model


if __name__ == "__main__":
    # 모델 생성 테스트
    model = create_vit_model('base', num_classes=10)
    
    # 더미 입력
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 어텐션 시각화 모델 테스트
    viz_model = ViTWithAttentionVisualization(
        img_size=224,
        patch_size=16,
        num_classes=10,
        embed_dim=768,
        depth=12,
        n_heads=12
    )
    
    output, attention_weights = viz_model(x, return_attention=True)
    print(f"\nAttention weights captured: {len(attention_weights)} layers")
    print(f"Attention shape per layer: {attention_weights[0].shape}")