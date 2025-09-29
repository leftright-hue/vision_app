"""
Transfer Learning Helper Functions
Transfer Learning 관련 유틸리티 함수들
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.models as models
from typing import List, Dict, Any


class TransferLearningHelper:
    """Transfer Learning 헬퍼 클래스"""

    def get_transfer_learning_code(self, model_name: str, num_classes: int, method: str) -> str:
        """Transfer Learning 코드 생성"""

        if method == "Feature Extraction (빠름)":
            code = f"""
import torch
import torch.nn as nn
import torchvision.models as models

# 1. 사전 훈련된 {model_name} 모델 로드
model = models.{model_name.lower()}(pretrained=True)

# 2. 모든 레이어 동결 (Feature Extraction)
for param in model.parameters():
    param.requires_grad = False

# 3. 마지막 레이어만 교체
if '{model_name}' == 'ResNet50':
    model.fc = nn.Linear(model.fc.in_features, {num_classes})
elif '{model_name}' == 'VGG16':
    model.classifier[-1] = nn.Linear(4096, {num_classes})

# 4. 옵티마이저 설정 (마지막 레이어만)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
"""
        elif method == "Fine-tuning (정확함)":
            code = f"""
import torch
import torch.nn as nn
import torchvision.models as models

# 1. 사전 훈련된 {model_name} 모델 로드
model = models.{model_name.lower()}(pretrained=True)

# 2. 일부 레이어만 동결 (Fine-tuning)
for param in model.parameters():
    param.requires_grad = False

# 마지막 몇 개 레이어는 학습 가능하게
for param in list(model.parameters())[-10:]:
    param.requires_grad = True

# 3. 마지막 레이어 교체
if '{model_name}' == 'ResNet50':
    model.fc = nn.Linear(model.fc.in_features, {num_classes})

# 4. 옵티마이저 설정 (다른 학습률)
optimizer = torch.optim.Adam([
    {{'params': model.fc.parameters(), 'lr': 0.001}},
    {{'params': list(model.parameters())[:-10], 'lr': 0.0001}}
])
"""
        else:  # 전체 학습
            code = f"""
import torch
import torch.nn as nn
import torchvision.models as models

# 1. 사전 훈련된 {model_name} 모델 로드
model = models.{model_name.lower()}(pretrained=True)

# 2. 모든 레이어 학습 가능
for param in model.parameters():
    param.requires_grad = True

# 3. 마지막 레이어 교체
if '{model_name}' == 'ResNet50':
    model.fc = nn.Linear(model.fc.in_features, {num_classes})

# 4. 옵티마이저 설정 (전체 모델)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
"""
        return code

    def visualize_features(self, image, model_name: str, layer_choice: str):
        """특징 맵 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'{model_name} - {layer_choice} 특징 시각화', fontsize=16)

        for i, ax in enumerate(axes.flat):
            # 시뮬레이션: 랜덤 특징 맵 생성
            feature_map = np.random.randn(224, 224)
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Feature Map {i+1}')
            ax.axis('off')

        plt.tight_layout()
        return fig

    def get_model_metrics(self, models: List[str]) -> pd.DataFrame:
        """모델 성능 메트릭 생성"""
        metrics_data = []

        for model in models:
            # 시뮬레이션 데이터
            metrics_data.append({
                'Model': model,
                'Top-1 Accuracy': np.random.uniform(85, 95),
                'Top-5 Accuracy': np.random.uniform(95, 99),
                'Inference Time (ms)': np.random.uniform(10, 50),
                'Model Size (MB)': np.random.uniform(20, 150),
                'FLOPs (G)': np.random.uniform(1, 10)
            })

        return pd.DataFrame(metrics_data)

    def create_performance_chart(self, models: List[str]):
        """모델 성능 비교 차트"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 정확도 비교
        accuracies = [np.random.uniform(85, 95) for _ in models]
        axes[0].bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('모델별 정확도 비교')
        axes[0].set_ylim(80, 100)

        # 속도 비교
        speeds = [np.random.uniform(10, 50) for _ in models]
        axes[1].bar(models, speeds, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[1].set_ylabel('Inference Time (ms)')
        axes[1].set_title('모델별 추론 속도')

        plt.tight_layout()
        return fig

    def plot_learning_curves(self):
        """학습 곡선 그리기"""
        epochs = range(1, 21)

        # 시뮬레이션 데이터
        train_loss = [1.5 * np.exp(-0.2 * i) + 0.1 + np.random.normal(0, 0.02) for i in epochs]
        val_loss = [1.5 * np.exp(-0.15 * i) + 0.15 + np.random.normal(0, 0.03) for i in epochs]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('학습 곡선 (Learning Curves)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 최적점 표시
        min_val_idx = np.argmin(val_loss)
        ax.plot(epochs[min_val_idx], val_loss[min_val_idx], 'go', markersize=10)
        ax.annotate('Best Model', xy=(epochs[min_val_idx], val_loss[min_val_idx]),
                    xytext=(epochs[min_val_idx]+2, val_loss[min_val_idx]+0.1),
                    arrowprops=dict(arrowstyle='->', color='green'))

        return fig

    def create_confusion_matrix(self, num_classes: int):
        """혼동 행렬 생성"""
        # 시뮬레이션 혼동 행렬 데이터
        confusion_matrix = np.random.randint(0, 100, (num_classes, num_classes))
        np.fill_diagonal(confusion_matrix, np.random.randint(80, 100, num_classes))

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(confusion_matrix, cmap='Blues')

        # 레이블 추가
        classes = [f'Class {i}' for i in range(num_classes)]
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        # 값 표시
        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, confusion_matrix[i, j],
                             ha="center", va="center", color="black" if confusion_matrix[i, j] < 50 else "white")

        ax.set_title('혼동 행렬 (Confusion Matrix)')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.colorbar(im)

        return fig

    def visualize_feature_space(self):
        """특징 공간 시각화 (t-SNE)"""
        # 시뮬레이션 데이터
        np.random.seed(42)
        n_samples = 300
        n_classes = 5

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        for i in range(n_classes):
            # 클러스터 생성
            center = np.random.randn(2) * 3
            points = np.random.randn(n_samples // n_classes, 2) + center
            ax.scatter(points[:, 0], points[:, 1], c=colors[i], label=f'Class {i+1}',
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        ax.set_title('t-SNE 특징 공간 시각화')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return fig