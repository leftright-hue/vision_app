"""
Transfer Learning 구현 예제
Week 3: 딥러닝 영상처리

이 파일은 PyTorch를 사용한 전이학습의 완전한 구현을 보여줍니다.
Feature Extraction과 Fine-tuning 두 가지 방법을 모두 다룹니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from typing import Dict, Tuple, Optional
from pathlib import Path


class TransferLearningModel:
    """
    전이학습을 위한 종합 클래스
    ResNet, VGG, EfficientNet 등 다양한 백본 지원
    """
    
    def __init__(
        self, 
        model_name: str = 'resnet50',
        num_classes: int = 10,
        feature_extract: bool = True,
        use_pretrained: bool = True
    ):
        """
        Args:
            model_name: 사용할 모델 ('resnet50', 'vgg16', 'efficientnet_b0' 등)
            num_classes: 출력 클래스 수
            feature_extract: True면 Feature Extraction, False면 Fine-tuning
            use_pretrained: 사전훈련된 가중치 사용 여부
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        
        # 모델 초기화
        self.model = self._initialize_model(model_name, num_classes, feature_extract, use_pretrained)
        
        # 디바이스 설정
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 학습 기록
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def _initialize_model(
        self, 
        model_name: str, 
        num_classes: int, 
        feature_extract: bool,
        use_pretrained: bool
    ) -> nn.Module:
        """
        다양한 사전훈련 모델 초기화
        """
        model = None
        input_size = 224  # 기본 입력 크기
        
        if model_name == "resnet50":
            model = models.resnet50(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
            input_size = 224
            
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        self.input_size = input_size
        return model
    
    def _set_parameter_requires_grad(self, model: nn.Module, feature_extract: bool):
        """
        Feature Extraction 모드에서 파라미터 동결
        """
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
    
    def unfreeze_layers(self, num_layers: int):
        """
        마지막 n개 레이어를 언프리즈 (Progressive Unfreezing)
        """
        # 모든 파라미터 리스트
        params = list(self.model.parameters())
        
        # 마지막 n개 레이어 언프리즈
        for param in params[-num_layers:]:
            param.requires_grad = True
        
        # 학습 가능한 파라미터 수 출력
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        print(f"Unfrozen {num_layers} layers")
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def get_optimizer(self, lr: float = 0.001) -> optim.Optimizer:
        """
        학습 가능한 파라미터만 옵티마이저에 전달
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        
        # Discriminative Learning Rates
        if self.feature_extract:
            # Feature Extraction: 단일 학습률
            optimizer = optim.Adam(params_to_update, lr=lr)
        else:
            # Fine-tuning: 레이어별 다른 학습률
            backbone_params = []
            classifier_params = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'fc' in name or 'classifier' in name:
                        classifier_params.append(param)
                    else:
                        backbone_params.append(param)
            
            optimizer = optim.Adam([
                {'params': backbone_params, 'lr': lr * 0.1},  # 백본: 작은 학습률
                {'params': classifier_params, 'lr': lr}        # 분류기: 큰 학습률
            ])
        
        return optimizer
    
    def train_model(
        self, 
        dataloaders: Dict[str, DataLoader],
        num_epochs: int = 25,
        learning_rate: float = 0.001
    ) -> nn.Module:
        """
        모델 학습 메인 함수
        """
        since = time.time()
        
        # 옵티마이저와 스케줄러 설정
        optimizer = self.get_optimizer(learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 손실 함수
        criterion = nn.CrossEntropyLoss()
        
        # 최고 성능 모델 저장
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        # 데이터셋 크기
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # 각 phase (train/val)에 대해
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                # 배치별 처리
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 그래디언트 초기화
                    optimizer.zero_grad()
                    
                    # 순전파
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # 역전파 (학습 시에만)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # 통계
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
                
                # 에폭 통계
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # 기록 저장
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc.cpu().numpy())
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc.cpu().numpy())
                
                # 최고 모델 저장
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        # 최고 성능 모델 로드
        self.model.load_state_dict(best_model_wts)
        return self.model
    
    def plot_training_history(self):
        """
        학습 과정 시각화
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss 플롯
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy 플롯
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, image_tensor: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        단일 이미지 예측
        """
        self.model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.item(), probabilities.squeeze()
    
    def save_model(self, path: str):
        """
        모델 저장
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'feature_extract': self.feature_extract,
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        모델 로드
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    input_size: int = 224
) -> Dict[str, DataLoader]:
    """
    데이터 로더 생성
    """
    # 데이터 변환 정의
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 데이터셋 생성
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    
    # 데이터 로더 생성
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'val']
    }
    
    class_names = image_datasets['train'].classes
    
    return dataloaders, class_names


def compare_transfer_learning_strategies():
    """
    Feature Extraction vs Fine-tuning 비교 실험
    """
    results = {}
    
    # 실험 설정
    strategies = [
        {'name': 'Feature Extraction', 'feature_extract': True},
        {'name': 'Fine-tuning (All)', 'feature_extract': False},
        {'name': 'Progressive Unfreezing', 'feature_extract': True, 'unfreeze': 10}
    ]
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Testing: {strategy['name']}")
        print('='*50)
        
        # 모델 초기화
        model = TransferLearningModel(
            model_name='resnet50',
            num_classes=10,
            feature_extract=strategy.get('feature_extract', True)
        )
        
        # Progressive Unfreezing
        if 'unfreeze' in strategy:
            model.unfreeze_layers(strategy['unfreeze'])
        
        # 학습 (간단한 예제를 위해 에폭 수를 줄임)
        # dataloaders, _ = create_data_loaders('path/to/data')
        # trained_model = model.train_model(dataloaders, num_epochs=5)
        
        # 결과 저장
        results[strategy['name']] = {
            'trainable_params': sum(p.numel() for p in model.model.parameters() if p.requires_grad),
            'total_params': sum(p.numel() for p in model.model.parameters())
        }
    
    # 결과 출력
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for name, stats in results.items():
        trainable = stats['trainable_params']
        total = stats['total_params']
        percentage = 100 * trainable / total
        
        print(f"\n{name}:")
        print(f"  Trainable: {trainable:,} / {total:,} ({percentage:.2f}%)")


def visualize_feature_extraction():
    """
    Feature Extraction 과정 시각화
    """
    import torch.nn.functional as F
    
    # 모델 로드
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # 특징 추출용 훅
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # 중간 레이어에 훅 등록
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4.register_forward_hook(get_activation('layer4'))
    
    # 더미 입력
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 순전파
    output = model(dummy_input)
    
    # 특징 맵 시각화
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    for idx, (name, activation) in enumerate(activations.items()):
        ax = axes[idx // 2, idx % 2]
        
        # 첫 번째 채널의 특징 맵 시각화
        feature_map = activation[0, 0].cpu().numpy()
        
        im = ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'{name} - Shape: {activation.shape}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Feature Maps at Different Layers')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Transfer Learning Implementation")
    print("="*50)
    
    # 1. 모델 초기화 예제
    print("\n1. Model Initialization Examples:")
    
    # Feature Extraction
    fe_model = TransferLearningModel(
        model_name='resnet50',
        num_classes=10,
        feature_extract=True
    )
    print(f"Feature Extraction Model Created")
    
    # Fine-tuning
    ft_model = TransferLearningModel(
        model_name='resnet50',
        num_classes=10,
        feature_extract=False
    )
    print(f"Fine-tuning Model Created")
    
    # 2. 전략 비교
    print("\n2. Comparing Transfer Learning Strategies:")
    compare_transfer_learning_strategies()
    
    # 3. 특징 추출 시각화 (선택적)
    # visualize_feature_extraction()
    
    print("\n" + "="*50)
    print("Transfer Learning Implementation Complete!")