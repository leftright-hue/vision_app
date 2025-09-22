"""
CNN 향상된 시각화 및 설명 시스템
Week 2: CNN 원리 + Hugging Face 생태계

이 파일은 CNN의 각 단계를 시각화하고 상세한 설명을 제공합니다.
각 이미지에는 캡션, 주석, 화살표 등이 포함되어 이해를 돕습니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedCNNVisualization:
    """CNN 작동 원리를 상세한 설명과 함께 시각화하는 클래스"""

    def __init__(self):
        self.fig_size = (20, 12)
        self.korean_font_setup()

    def korean_font_setup(self):
        """한글 폰트 설정"""
        import matplotlib.font_manager as fm
        import os

        # Windows에서 한글 폰트 설정
        if os.name == 'nt':
            font_path = "C:/Windows/Fonts/malgun.ttf"
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()

        # 마이너스 기호 표시 설정
        plt.rcParams['axes.unicode_minus'] = False

    def create_sample_digit(self):
        """MNIST 스타일 숫자 이미지 생성"""
        image = np.zeros((28, 28))

        # 숫자 7 그리기
        image[5:8, 8:22] = 1  # 상단 가로선
        image[7:20, 19:22] = 1  # 대각선

        # 노이즈 추가
        noise = np.random.normal(0, 0.1, (28, 28))
        image = np.clip(image + noise, 0, 1)

        return image

    def visualize_convolution_process(self):
        """Convolution 과정을 단계별로 시각화"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('🔍 CNN Convolution 과정 상세 시각화', fontsize=16, fontweight='bold')

        # 1. 입력 이미지
        image = self.create_sample_digit()
        ax = axes[0, 0]
        im1 = ax.imshow(image, cmap='gray')
        ax.set_title('1️⃣ 입력 이미지 (28x28)', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.15, '원본 숫자 이미지\n픽셀값: 0(검정)~1(흰색)',
                transform=ax.transAxes, ha='center', fontsize=10, color='blue')

        # 2. 커널/필터
        kernel_edge = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]])
        ax = axes[0, 1]
        im2 = ax.imshow(kernel_edge, cmap='RdBu', vmin=-8, vmax=8)
        ax.set_title('2️⃣ Edge Detection 커널 (3x3)', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.15, '엣지 검출 필터\n중앙: +8, 주변: -1',
                transform=ax.transAxes, ha='center', fontsize=10, color='blue')

        # 커널 값 표시
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{kernel_edge[i, j]:.0f}',
                       ha='center', va='center', color='white', fontweight='bold')

        # 3. Convolution 연산 과정
        ax = axes[0, 2]
        ax.imshow(image, cmap='gray', alpha=0.3)

        # 슬라이딩 윈도우 표시
        rect = patches.Rectangle((5, 5), 3, 3, linewidth=3,
                                edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # 화살표 추가
        arrow = FancyArrowPatch((7, 10), (15, 10),
                               connectionstyle="arc3,rad=0.3",
                               arrowstyle='->', mutation_scale=20,
                               color='red', linewidth=2)
        ax.add_patch(arrow)

        ax.set_title('3️⃣ 슬라이딩 윈도우', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.15, '커널이 이미지를 훑으며\n특징을 추출',
                transform=ax.transAxes, ha='center', fontsize=10, color='blue')

        # 4. Feature Map 결과
        feature_map = cv2.filter2D(image, -1, kernel_edge)
        ax = axes[0, 3]
        im4 = ax.imshow(feature_map, cmap='hot')
        ax.set_title('4️⃣ Feature Map (26x26)', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.15, '엣지가 강조된 특징 맵\n밝은 부분 = 강한 엣지',
                transform=ax.transAxes, ha='center', fontsize=10, color='blue')

        # 5. ReLU 활성화
        relu_output = np.maximum(feature_map, 0)
        ax = axes[1, 0]
        im5 = ax.imshow(relu_output, cmap='hot')
        ax.set_title('5️⃣ ReLU 활성화', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.15, '음수 값 제거\nmax(0, x) 적용',
                transform=ax.transAxes, ha='center', fontsize=10, color='blue')

        # 6. Max Pooling 과정
        ax = axes[1, 1]
        ax.imshow(relu_output, cmap='hot', alpha=0.3)

        # Pooling 영역 표시
        pool_rect = patches.Rectangle((4, 4), 2, 2, linewidth=2,
                                     edgecolor='blue', facecolor='blue', alpha=0.2)
        ax.add_patch(pool_rect)
        ax.text(5, 5, 'Max', ha='center', va='center', fontweight='bold', color='blue')

        ax.set_title('6️⃣ Max Pooling (2x2)', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.15, '2x2 영역의 최대값 선택\n크기 축소 & 특징 보존',
                transform=ax.transAxes, ha='center', fontsize=10, color='blue')

        # 7. Pooled Feature Map
        pooled = F.max_pool2d(torch.tensor(relu_output).unsqueeze(0).unsqueeze(0), 2).squeeze().numpy()
        ax = axes[1, 2]
        im7 = ax.imshow(pooled, cmap='hot')
        ax.set_title('7️⃣ Pooled Output (13x13)', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.15, '크기는 줄고\n중요 특징은 유지',
                transform=ax.transAxes, ha='center', fontsize=10, color='blue')

        # 8. 최종 분류
        ax = axes[1, 3]
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        logits = np.random.randn(10) * 2
        probabilities = np.exp(logits) / np.sum(np.exp(logits))  # Softmax 수동 구현
        probabilities[7] = 0.8  # 7일 확률을 높게 설정
        probabilities = probabilities / probabilities.sum()

        bars = ax.bar(classes, probabilities, color='steelblue')
        bars[7].set_color('green')
        ax.set_title('8️⃣ 최종 예측 결과', fontsize=12, fontweight='bold')
        ax.set_ylabel('확률')
        ax.set_ylim(0, 1)
        ax.text(0.5, -0.15, f'예측: 숫자 7 (확률: {probabilities[7]:.1%})',
                transform=ax.transAxes, ha='center', fontsize=10, color='green', fontweight='bold')

        # 컬러바 추가
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        plt.colorbar(im4, ax=axes[0, 3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig

    def visualize_kernel_effects_with_explanations(self):
        """다양한 커널의 효과를 설명과 함께 시각화"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('🎯 다양한 커널(필터)의 효과와 용도', fontsize=16, fontweight='bold')

        # 샘플 이미지 생성
        image = self.create_complex_sample_image()

        # 다양한 커널과 설명
        kernels_info = {
            'Original': {
                'kernel': None,
                'description': '원본 이미지\n처리 전 상태',
                'use_case': '기준점'
            },
            'Vertical Edge': {
                'kernel': np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]),
                'description': '수직 엣지 검출\nSobel X 필터',
                'use_case': '세로선 감지'
            },
            'Horizontal Edge': {
                'kernel': np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]]),
                'description': '수평 엣지 검출\nSobel Y 필터',
                'use_case': '가로선 감지'
            },
            'Laplacian': {
                'kernel': np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]]),
                'description': '라플라시안\n모든 방향 엣지',
                'use_case': '윤곽선 검출'
            },
            'Blur': {
                'kernel': np.ones((3, 3)) / 9,
                'description': '블러(흐림)\n평균 필터',
                'use_case': '노이즈 제거'
            },
            'Sharpen': {
                'kernel': np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]]),
                'description': '선명화\n엣지 강조',
                'use_case': '이미지 선명도↑'
            },
            'Emboss': {
                'kernel': np.array([[-2, -1, 0],
                                   [-1, 1, 1],
                                   [0, 1, 2]]),
                'description': '엠보싱 효과\n3D 질감',
                'use_case': '입체감 부여'
            },
            'Identity': {
                'kernel': np.array([[0, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]]),
                'description': '항등 필터\n변화 없음',
                'use_case': '테스트용'
            }
        }

        # 커널 효과 시각화
        for idx, (name, info) in enumerate(kernels_info.items()):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]

            if info['kernel'] is None:
                result = image
            else:
                result = cv2.filter2D(image, -1, info['kernel'])

            im = ax.imshow(result, cmap='gray')
            ax.set_title(f'{name}', fontsize=11, fontweight='bold')
            ax.axis('off')

            # 설명 텍스트 박스
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8)
            ax.text(0.5, -0.08, info['description'],
                   transform=ax.transAxes, ha='center', fontsize=9, bbox=bbox_props)

            # 용도 표시
            ax.text(0.5, -0.18, f"용도: {info['use_case']}",
                   transform=ax.transAxes, ha='center', fontsize=8,
                   color='darkgreen', style='italic')

        # 커널 값 시각화 (마지막 4개 위치에)
        for idx, (name, info) in enumerate(list(kernels_info.items())[1:5]):
            ax = axes[2, idx]

            if info['kernel'] is not None:
                im = ax.imshow(info['kernel'], cmap='RdBu', vmin=-2, vmax=2)
                ax.set_title(f'{name} 커널', fontsize=10)

                # 커널 값 표시
                for i in range(3):
                    for j in range(3):
                        value = info['kernel'][i, j]
                        color = 'white' if abs(value) > 0.5 else 'black'
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                               color=color, fontsize=8, fontweight='bold')

                ax.set_xticks([])
                ax.set_yticks([])

        # 나머지 빈 공간 제거
        for idx in range(8, 12):
            if idx < 12:
                row = idx // 4
                col = idx % 4
                if row < 3 and col < 4:
                    axes[row, col].axis('off')

        plt.tight_layout()
        return fig

    def create_complex_sample_image(self, size=64):
        """복잡한 패턴이 있는 샘플 이미지 생성"""
        image = np.zeros((size, size))

        # 다양한 패턴 추가
        # 원
        center = size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        image[mask] = 0.8

        # 사각형
        image[10:20, 10:20] = 0.6
        image[40:50, 40:50] = 0.4

        # 대각선
        for i in range(min(size, 30)):
            image[i, i] = 1.0
            image[i, size-1-i] = 0.7

        # 가로선과 세로선
        image[size//3, :] = 0.5
        image[:, size//3] = 0.5

        return image

    def visualize_feature_map_progression(self):
        """CNN 레이어를 통과하며 변화하는 Feature Map 시각화"""
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('📈 CNN 레이어별 Feature Map 변화 과정', fontsize=16, fontweight='bold')

        # 간단한 CNN 모델 생성
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
                self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
                self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)

            def forward_with_features(self, x):
                features = []

                # Layer 1
                x = F.relu(self.conv1(x))
                features.append(('Conv1 + ReLU', x.clone()))
                x = self.pool(x)
                features.append(('After Pool1', x.clone()))

                # Layer 2
                x = F.relu(self.conv2(x))
                features.append(('Conv2 + ReLU', x.clone()))
                x = self.pool(x)
                features.append(('After Pool2', x.clone()))

                # Layer 3
                x = F.relu(self.conv3(x))
                features.append(('Conv3 + ReLU', x.clone()))

                return features

        model = SimpleCNN()

        # 입력 이미지 생성
        input_image = torch.tensor(self.create_sample_digit()).unsqueeze(0).unsqueeze(0).float()

        # Feature maps 추출
        with torch.no_grad():
            features = model.forward_with_features(input_image)

        # 시각화
        gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)

        # 원본 이미지
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(input_image.squeeze(), cmap='gray')
        ax.set_title('입력 이미지', fontsize=10, fontweight='bold')
        ax.axis('off')

        # 각 레이어의 설명 추가
        layer_descriptions = {
            'Conv1 + ReLU': '첫 번째 층:\n기본 엣지와 선 감지\n8개 필터',
            'After Pool1': '풀링 후:\n크기 감소\n주요 특징 유지',
            'Conv2 + ReLU': '두 번째 층:\n복잡한 패턴 감지\n16개 필터',
            'After Pool2': '풀링 후:\n더 추상적인 특징\n공간 정보 압축',
            'Conv3 + ReLU': '세 번째 층:\n고수준 특징\n32개 필터'
        }

        # Feature maps 시각화
        for idx, (layer_name, feature_map) in enumerate(features):
            feature_map = feature_map.squeeze(0)
            num_channels = min(6, feature_map.shape[0])

            # 레이어 설명
            row = idx // 2 + 1 if idx > 0 else 0
            col_start = (idx % 2) * 3 + 1 if idx > 0 else 1

            for ch in range(num_channels):
                if idx == 0:  # Conv1
                    if ch < 5:  # gs는 6열이므로 최대 5개만
                        ax = fig.add_subplot(gs[0, ch + 1])
                        ax.imshow(feature_map[ch], cmap='viridis')
                        if ch == 0:
                            ax.set_title(f'{layer_name}\nCh {ch+1}', fontsize=8, fontweight='bold')
                        else:
                            ax.set_title(f'Ch {ch+1}', fontsize=8)
                        ax.axis('off')
                else:
                    row = (idx - 1) // 2 + 1
                    col = (idx - 1) % 2 * 3 + (ch % 3)
                    if row < 3 and col < 6:  # 범위 체크
                        ax = fig.add_subplot(gs[row, col])
                        ax.imshow(feature_map[ch], cmap='viridis')
                        if ch == 0:
                            ax.set_title(f'{layer_name}\nCh {ch+1}', fontsize=8, fontweight='bold')
                        else:
                            ax.set_title(f'Ch {ch+1}', fontsize=8)
                        ax.axis('off')

            # 설명 텍스트 추가
            if idx < len(features) - 1:
                text_col = 3 if idx % 2 == 0 else 0
                text_row = (idx // 2) + 1 if idx > 0 else 0

                # text_col + 3이 6을 넘지 않도록 조정
                if text_col + 3 < 6:
                    ax_text = fig.add_subplot(gs[text_row, text_col + 2])
                    ax_text.axis('off')
                    ax_text.text(0.5, 0.5, layer_descriptions[layer_name],
                               transform=ax_text.transAxes, ha='center', va='center',
                               fontsize=9, bbox=dict(boxstyle="round,pad=0.5",
                                                   facecolor='lightblue', alpha=0.7))

        # 화살표와 흐름 표시
        fig.text(0.5, 0.02, '➡️ 깊은 레이어로 갈수록: 단순한 특징(엣지) → 복잡한 특징(패턴) → 추상적 특징(개념)',
                ha='center', fontsize=11, color='darkred', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.3))

        plt.tight_layout()
        return fig

    def create_architecture_comparison_with_annotations(self):
        """CNN 아키텍처 비교를 주석과 함께 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('🏗️ CNN 아키텍처 발전 과정과 특징', fontsize=16, fontweight='bold')

        # LeNet-5 (1998)
        ax = axes[0, 0]
        ax.set_title('LeNet-5 (1998)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')

        # LeNet-5 레이어 그리기
        layers_lenet = [
            {'name': 'Input\n32x32', 'x': 0.5, 'width': 0.8, 'color': 'lightgray'},
            {'name': 'Conv1\n6@28x28', 'x': 2, 'width': 0.8, 'color': 'lightblue'},
            {'name': 'Pool1\n6@14x14', 'x': 3.5, 'width': 0.6, 'color': 'lightgreen'},
            {'name': 'Conv2\n16@10x10', 'x': 5, 'width': 0.8, 'color': 'lightblue'},
            {'name': 'Pool2\n16@5x5', 'x': 6.5, 'width': 0.6, 'color': 'lightgreen'},
            {'name': 'FC\n120', 'x': 8, 'width': 0.5, 'color': 'lightyellow'},
            {'name': 'Output\n10', 'x': 9.5, 'width': 0.3, 'color': 'lightcoral'}
        ]

        for layer in layers_lenet:
            rect = FancyBboxPatch((layer['x'] - layer['width']/2, 1),
                                 layer['width'], 2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=layer['color'],
                                 edgecolor='black',
                                 linewidth=2)
            ax.add_patch(rect)
            ax.text(layer['x'], 2, layer['name'], ha='center', va='center', fontsize=9)

        # 화살표 추가
        for i in range(len(layers_lenet) - 1):
            arrow = FancyArrowPatch((layers_lenet[i]['x'] + layers_lenet[i]['width']/2, 2),
                                  (layers_lenet[i+1]['x'] - layers_lenet[i+1]['width']/2, 2),
                                  arrowstyle='->', mutation_scale=15, color='darkgray')
            ax.add_patch(arrow)

        # 특징 설명
        ax.text(5, 0.5, '✅ 최초의 성공적인 CNN\n✅ 손글씨 숫자 인식\n✅ 약 6만개 파라미터',
               ha='center', fontsize=9, bbox=dict(boxstyle="round", facecolor='wheat'))

        # AlexNet (2012)
        ax = axes[0, 1]
        ax.set_title('AlexNet (2012)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')

        # AlexNet 특징
        ax.text(5, 3, '📊 AlexNet 혁신', ha='center', fontsize=11, fontweight='bold')
        innovations = [
            '• 8개 레이어 (5 Conv + 3 FC)',
            '• ReLU 활성화 함수 도입',
            '• Dropout 정규화',
            '• GPU 병렬 처리',
            '• 6천만개 파라미터',
            '• ImageNet 우승'
        ]
        for i, text in enumerate(innovations):
            ax.text(5, 2.5 - i*0.3, text, ha='center', fontsize=9)

        # VGGNet (2014)
        ax = axes[1, 0]
        ax.set_title('VGGNet (2014)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')

        # VGGNet 특징
        ax.text(5, 3, '🔧 VGGNet 특징', ha='center', fontsize=11, fontweight='bold')
        features = [
            '• 3x3 작은 필터만 사용',
            '• 깊이 증가 (16-19층)',
            '• 단순하고 균일한 구조',
            '• 1.38억개 파라미터',
            '• 전이학습 기반 모델'
        ]
        for i, text in enumerate(features):
            ax.text(5, 2.5 - i*0.3, text, ha='center', fontsize=9)

        # ResNet (2015)
        ax = axes[1, 1]
        ax.set_title('ResNet (2015)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')

        # Skip Connection 시각화
        ax.text(5, 4, '🚀 ResNet 혁명', ha='center', fontsize=11, fontweight='bold')

        # Skip connection 다이어그램
        rect1 = FancyBboxPatch((2, 2.5), 1, 0.5, boxstyle="round,pad=0.05",
                              facecolor='lightblue', edgecolor='black')
        rect2 = FancyBboxPatch((4, 2.5), 1, 0.5, boxstyle="round,pad=0.05",
                              facecolor='lightblue', edgecolor='black')
        rect3 = FancyBboxPatch((6, 2.5), 1, 0.5, boxstyle="round,pad=0.05",
                              facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        # Skip connection 화살표
        arrow1 = FancyArrowPatch((2.5, 3.2), (6.5, 3.2),
                               connectionstyle="arc3,rad=0.3",
                               arrowstyle='->', mutation_scale=15,
                               color='red', linewidth=2)
        ax.add_patch(arrow1)
        ax.text(4.5, 3.5, 'Skip Connection', ha='center', fontsize=9, color='red')

        # 특징 설명
        features = [
            '• Residual Learning',
            '• 152층까지 확장 가능',
            '• Vanishing Gradient 해결',
            '• 현재도 널리 사용'
        ]
        for i, text in enumerate(features):
            ax.text(5, 1.8 - i*0.3, text, ha='center', fontsize=9)

        plt.tight_layout()
        return fig

def main():
    """메인 실행 함수"""
    print("🎨 CNN 향상된 시각화 시스템")
    print("=" * 50)

    visualizer = EnhancedCNNVisualization()

    # 1. Convolution 과정 상세 시각화
    print("\n1. Convolution 과정 시각화 생성 중...")
    fig1 = visualizer.visualize_convolution_process()
    fig1.savefig('cnn_convolution_process.png', dpi=150, bbox_inches='tight')
    print("   ✅ cnn_convolution_process.png 저장 완료")

    # 2. 커널 효과와 설명
    print("\n2. 커널 효과 시각화 생성 중...")
    fig2 = visualizer.visualize_kernel_effects_with_explanations()
    fig2.savefig('cnn_kernel_effects.png', dpi=150, bbox_inches='tight')
    print("   ✅ cnn_kernel_effects.png 저장 완료")

    # 3. Feature Map 진행 과정
    print("\n3. Feature Map 변화 시각화 생성 중...")
    fig3 = visualizer.visualize_feature_map_progression()
    fig3.savefig('cnn_feature_progression.png', dpi=150, bbox_inches='tight')
    print("   ✅ cnn_feature_progression.png 저장 완료")

    # 4. 아키텍처 비교
    print("\n4. CNN 아키텍처 비교 시각화 생성 중...")
    fig4 = visualizer.create_architecture_comparison_with_annotations()
    fig4.savefig('cnn_architecture_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ cnn_architecture_comparison.png 저장 완료")

    print("\n" + "=" * 50)
    print("🎉 모든 시각화 완료!")
    print("\n📚 생성된 시각화 파일:")
    print("   1. cnn_convolution_process.png - Convolution 8단계 과정")
    print("   2. cnn_kernel_effects.png - 다양한 커널의 효과와 용도")
    print("   3. cnn_feature_progression.png - 레이어별 특징 변화")
    print("   4. cnn_architecture_comparison.png - CNN 발전 역사")

    print("\n💡 각 이미지는 다음을 포함합니다:")
    print("   - 단계별 설명과 캡션")
    print("   - 시각적 주석과 화살표")
    print("   - 실제 사용 사례")
    print("   - 핵심 개념 하이라이트")

    # 모든 그래프 표시
    plt.show()

if __name__ == "__main__":
    main()