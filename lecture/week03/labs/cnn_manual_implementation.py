"""
CNN 수동 구현 예제
Week 2: CNN 원리 + Hugging Face 생태계

이 파일은 CNN의 핵심 구성 요소들을 수동으로 구현하여
내부 작동 원리를 이해하기 위한 예제입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

class ManualCNN:
    """CNN 구성 요소들을 수동으로 구현한 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {self.device}")
    
    def manual_convolution_2d(self, input_tensor, kernel, stride=1, padding=0):
        """
        2D Convolution 연산을 수동으로 구현
        
        Args:
            input_tensor: 입력 텐서 (batch, channels, height, width)
            kernel: 커널 텐서 (out_channels, in_channels, kernel_height, kernel_width)
            stride: 스트라이드
            padding: 패딩
        
        Returns:
            output_tensor: 출력 텐서
        """
        batch_size, in_channels, in_height, in_width = input_tensor.shape
        out_channels, _, kernel_height, kernel_width = kernel.shape
        
        # 패딩 적용
        if padding > 0:
            padded_input = torch.zeros(batch_size, in_channels, 
                                     in_height + 2*padding, in_width + 2*padding)
            padded_input[:, :, padding:padding+in_height, padding:padding+in_width] = input_tensor
        else:
            padded_input = input_tensor
        
        # 출력 크기 계산
        out_height = (in_height + 2*padding - kernel_height) // stride + 1
        out_width = (in_width + 2*padding - kernel_width) // stride + 1
        
        # 출력 텐서 초기화
        output = torch.zeros(batch_size, out_channels, out_height, out_width)
        
        # Convolution 연산 수행
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # 현재 윈도우의 시작 위치
                        h_start = oh * stride
                        w_start = ow * stride
                        h_end = h_start + kernel_height
                        w_end = w_start + kernel_width
                        
                        # 윈도우 추출
                        window = padded_input[b, :, h_start:h_end, w_start:w_end]
                        
                        # 커널과 윈도우의 곱셈 및 합계
                        output[b, oc, oh, ow] = torch.sum(window * kernel[oc])
        
        return output
    
    def manual_max_pooling_2d(self, input_tensor, kernel_size=2, stride=2):
        """
        2D Max Pooling을 수동으로 구현
        
        Args:
            input_tensor: 입력 텐서
            kernel_size: 풀링 커널 크기
            stride: 스트라이드
        
        Returns:
            output_tensor: 출력 텐서
        """
        batch_size, channels, in_height, in_width = input_tensor.shape
        
        # 출력 크기 계산
        out_height = (in_height - kernel_size) // stride + 1
        out_width = (in_width - kernel_size) // stride + 1
        
        # 출력 텐서 초기화
        output = torch.zeros(batch_size, channels, out_height, out_width)
        
        # Max Pooling 연산 수행
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # 현재 윈도우의 시작 위치
                        h_start = oh * stride
                        w_start = ow * stride
                        h_end = h_start + kernel_size
                        w_end = w_start + kernel_size
                        
                        # 윈도우 추출
                        window = input_tensor[b, c, h_start:h_end, w_start:w_end]
                        
                        # 최대값 계산
                        output[b, c, oh, ow] = torch.max(window)
        
        return output
    
    def manual_relu(self, input_tensor):
        """
        ReLU 활성화 함수를 수동으로 구현
        
        Args:
            input_tensor: 입력 텐서
        
        Returns:
            output_tensor: ReLU 적용된 텐서
        """
        return torch.maximum(input_tensor, torch.tensor(0.0))
    
    def manual_flatten(self, input_tensor):
        """
        텐서를 평면화하는 함수
        
        Args:
            input_tensor: 입력 텐서 (batch, channels, height, width)
        
        Returns:
            flattened_tensor: 평면화된 텐서 (batch, channels*height*width)
        """
        batch_size = input_tensor.shape[0]
        return input_tensor.view(batch_size, -1)
    
    def manual_linear(self, input_tensor, weight, bias):
        """
        선형 레이어를 수동으로 구현
        
        Args:
            input_tensor: 입력 텐서
            weight: 가중치 행렬
            bias: 편향 벡터
        
        Returns:
            output_tensor: 출력 텐서
        """
        return torch.matmul(input_tensor, weight.T) + bias

class CNNArchitectureComparison:
    """다양한 CNN 아키텍처 비교 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_lenet5(self):
        """LeNet-5 아키텍처 구현"""
        class LeNet5(nn.Module):
            def __init__(self):
                super(LeNet5, self).__init__()
                # Convolutional layers
                self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 28x28 → 28x28
                self.conv2 = nn.Conv2d(6, 16, 5)            # 28x28 → 24x24
                self.conv3 = nn.Conv2d(16, 120, 5)          # 24x24 → 20x20
                
                # Pooling layer
                self.pool = nn.MaxPool2d(2, 2)
                
                # Fully connected layers
                self.fc1 = nn.Linear(120 * 5 * 5, 84)
                self.fc2 = nn.Linear(84, 10)
                
            def forward(self, x):
                # Convolutional layers
                x = self.pool(F.relu(self.conv1(x)))  # 28x28 → 14x14
                x = self.pool(F.relu(self.conv2(x)))  # 14x14 → 7x7
                x = F.relu(self.conv3(x))             # 7x7 → 3x3
                
                # Flatten
                x = x.view(-1, 120 * 5 * 5)
                
                # Fully connected layers
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                
                return x
        
        return LeNet5()
    
    def create_alexnet(self):
        """AlexNet 아키텍처 구현 (간소화 버전)"""
        class AlexNet(nn.Module):
            def __init__(self, num_classes=1000):
                super(AlexNet, self).__init__()
                
                # Convolutional layers
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                
                # Fully connected layers
                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, num_classes),
                )
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), 256 * 6 * 6)
                x = self.classifier(x)
                return x
        
        return AlexNet()
    
    def compare_architectures(self):
        """다양한 CNN 아키텍처 비교"""
        print("=== CNN 아키텍처 비교 ===")
        
        # 모델 생성
        lenet5 = self.create_lenet5()
        alexnet = self.create_alexnet()
        
        # 파라미터 수 계산
        lenet5_params = sum(p.numel() for p in lenet5.parameters())
        alexnet_params = sum(p.numel() for p in alexnet.parameters())
        
        print(f"LeNet-5 파라미터 수: {lenet5_params:,}")
        print(f"AlexNet 파라미터 수: {alexnet_params:,}")
        print(f"비율: {alexnet_params/lenet5_params:.1f}배")
        
        # 레이어 수 비교
        lenet5_layers = len(list(lenet5.modules()))
        alexnet_layers = len(list(alexnet.modules()))
        
        print(f"LeNet-5 레이어 수: {lenet5_layers}")
        print(f"AlexNet 레이어 수: {alexnet_layers}")
        
        return lenet5, alexnet

class ConvolutionVisualization:
    """Convolution 연산 시각화 클래스"""
    
    def __init__(self):
        self.fig_size = (15, 10)
    
    def create_sample_image(self, size=28):
        """샘플 이미지 생성"""
        # 간단한 패턴 생성
        image = np.zeros((size, size))
        
        # 원 그리기
        center = size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        image[mask] = 1
        
        # 사각형 그리기
        image[5:10, 5:10] = 0.5
        
        return image
    
    def create_kernels(self):
        """다양한 커널 생성"""
        kernels = {
            'edge_detection': np.array([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ]),
            'blur': np.array([
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9]
            ]),
            'sharpen': np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]),
            'sobel_x': np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]),
            'sobel_y': np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ])
        }
        return kernels
    
    def visualize_convolution_effects(self):
        """다양한 커널의 효과 시각화"""
        print("=== Convolution 효과 시각화 ===")
        
        # 샘플 이미지 생성
        image = self.create_sample_image()
        kernels = self.create_kernels()
        
        # 결과 저장
        results = {'original': image}
        
        # 각 커널 적용
        for kernel_name, kernel in kernels.items():
            # Convolution 적용
            result = cv2.filter2D(image, -1, kernel)
            results[kernel_name] = result
        
        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        
        # 원본 이미지
        axes[0].imshow(results['original'], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 커널 효과들
        plot_idx = 1
        for kernel_name, result in results.items():
            if kernel_name != 'original':
                axes[plot_idx].imshow(result, cmap='gray')
                axes[plot_idx].set_title(f'{kernel_name.replace("_", " ").title()}')
                axes[plot_idx].axis('off')
                plot_idx += 1
        
        plt.tight_layout()
        plt.show()
        
        return results

def main():
    """메인 실행 함수"""
    print("🔧 CNN 수동 구현 예제")
    print("=" * 50)
    
    # 1. 수동 CNN 구현
    print("\n1. 수동 CNN 구현")
    manual_cnn = ManualCNN()
    
    # 샘플 입력 생성
    input_tensor = torch.randn(1, 1, 28, 28)
    kernel = torch.randn(6, 1, 5, 5)
    
    # 수동 Convolution 테스트
    output = manual_cnn.manual_convolution_2d(input_tensor, kernel, padding=2)
    print(f"입력 크기: {input_tensor.shape}")
    print(f"커널 크기: {kernel.shape}")
    print(f"출력 크기: {output.shape}")
    
    # PyTorch Convolution과 비교
    conv_layer = nn.Conv2d(1, 6, 5, padding=2, bias=False)
    conv_layer.weight.data = kernel
    pytorch_output = conv_layer(input_tensor)
    
    # 결과 비교
    diff = torch.abs(output - pytorch_output).max()
    print(f"수동 구현과 PyTorch 구현 차이: {diff:.6f}")
    
    # 2. 아키텍처 비교
    print("\n2. CNN 아키텍처 비교")
    arch_comparison = CNNArchitectureComparison()
    lenet5, alexnet = arch_comparison.compare_architectures()
    
    # 3. Convolution 효과 시각화
    print("\n3. Convolution 효과 시각화")
    viz = ConvolutionVisualization()
    results = viz.visualize_convolution_effects()
    
    print("\n✅ CNN 수동 구현 예제 완료!")
    print("\n📋 학습 포인트:")
    print("- Convolution 연산의 수학적 원리")
    print("- 다양한 커널의 효과 이해")
    print("- CNN 아키텍처의 발전 과정")
    print("- 수동 구현과 라이브러리 구현의 차이")

if __name__ == "__main__":
    main()
