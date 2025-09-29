#!/usr/bin/env python3
"""
Week 4 Lab: DINOv2와 SAM 데모
자기지도학습 모델 DINOv2와 Segment Anything Model (SAM) 활용 실습

이 실습에서는:
1. DINOv2로 고품질 특징 추출
2. SAM으로 객체 세그멘테이션
3. 자기지도학습 vs 지도학습 비교
4. 멀티모달 특징 분석
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {DEVICE}")

class DINOv2Analyzer:
    """
    DINOv2 모델을 사용한 특징 추출 및 분석
    """
    def __init__(self, model_name='dinov2_vitb14'):
        """
        DINOv2 모델 초기화
        
        Args:
            model_name: 사용할 DINOv2 모델 ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        """
        self.model_name = model_name
        self.model = None
        self.load_model()
        
    def load_model(self):
        """DINOv2 모델 로드"""
        try:
            print(f"🔄 {self.model_name} 모델 로딩 중...")
            
            # PyTorch Hub에서 DINOv2 모델 로드
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            self.model.eval().to(DEVICE)
            
            print(f"✅ {self.model_name} 모델 로드 완료")
            print(f"   - 파라미터 수: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("💡 인터넷 연결을 확인하거나 다른 모델명을 시도해보세요.")
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        if isinstance(image, str):
            # URL 또는 파일 경로에서 이미지 로드
            if image.startswith('http'):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # DINOv2 전처리
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0).to(DEVICE)
    
    def extract_features(self, image, return_attention=False):
        """특징 추출"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            if return_attention:
                # Attention 가중치도 함께 추출
                features = self.model.forward_features(input_tensor)
                # 마지막 블록의 attention 가져오기 (구현에 따라 다를 수 있음)
                attention = None  # DINOv2에서는 직접 attention을 가져오기 어려울 수 있음
                return features, attention
            else:
                features = self.model(input_tensor)  # CLS 토큰 특징
                return features
    
    def visualize_features(self, images, labels=None):
        """여러 이미지의 특징을 시각화"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return
        
        print("🔍 특징 추출 중...")
        features_list = []
        
        for i, image in enumerate(images):
            features = self.extract_features(image)
            if features is not None:
                features_list.append(features.cpu().numpy().flatten())
                print(f"   이미지 {i+1}/{len(images)} 처리 완료")
        
        if not features_list:
            print("❌ 특징 추출 실패")
            return
        
        features_array = np.array(features_list)
        
        # PCA로 차원 축소
        print("📊 PCA 차원 축소 중...")
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features_array)
        
        # t-SNE로 2D 시각화
        print("🎨 t-SNE 시각화 중...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(images)-1))
        features_2d = tsne.fit_transform(features_pca)
        
        # 시각화
        plt.figure(figsize=(15, 5))
        
        # PCA 설명 분산
        plt.subplot(1, 3, 1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('주성분 개수')
        plt.ylabel('누적 설명 분산 비율')
        plt.title('PCA 설명 분산')
        plt.grid(True)
        
        # t-SNE 시각화
        plt.subplot(1, 3, 2)
        if labels is not None:
            unique_labels = list(set(labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = np.array(labels) == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[color], label=label, alpha=0.7)
            plt.legend()
        else:
            plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7)
        
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('DINOv2 특징 t-SNE 시각화')
        
        # 특징 통계
        plt.subplot(1, 3, 3)
        feature_stats = {
            '평균': np.mean(features_array),
            '표준편차': np.std(features_array),
            '최대값': np.max(features_array),
            '최소값': np.min(features_array),
            '특징 차원': features_array.shape[1]
        }
        
        stats_text = '\n'.join([f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' 
                               for k, v in feature_stats.items()])
        plt.text(0.1, 0.5, stats_text, fontsize=12, 
                verticalalignment='center', transform=plt.gca().transAxes)
        plt.title('특징 통계')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return features_array, features_2d
    
    def compare_with_supervised(self, images):
        """자기지도학습 vs 지도학습 특징 비교"""
        if self.model is None:
            print("❌ DINOv2 모델이 로드되지 않았습니다.")
            return
        
        try:
            # 지도학습 모델 (ResNet) 로드
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            resnet.fc = torch.nn.Identity()  # 분류 헤드 제거
            resnet.eval().to(DEVICE)
            
            print("🔄 특징 추출 비교 중...")
            
            dinov2_features = []
            resnet_features = []
            
            for i, image in enumerate(images):
                # DINOv2 특징
                dino_feat = self.extract_features(image)
                if dino_feat is not None:
                    dinov2_features.append(dino_feat.cpu().numpy().flatten())
                
                # ResNet 특징
                input_tensor = self.preprocess_image(image)
                with torch.no_grad():
                    resnet_feat = resnet(input_tensor)
                    resnet_features.append(resnet_feat.cpu().numpy().flatten())
                
                print(f"   이미지 {i+1}/{len(images)} 처리 완료")
            
            # 특징 비교 시각화
            dinov2_array = np.array(dinov2_features)
            resnet_array = np.array(resnet_features)
            
            # PCA 적용
            pca_dino = PCA(n_components=2)
            pca_resnet = PCA(n_components=2)
            
            dino_2d = pca_dino.fit_transform(dinov2_array)
            resnet_2d = pca_resnet.fit_transform(resnet_array)
            
            # 시각화
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.scatter(dino_2d[:, 0], dino_2d[:, 1], alpha=0.7, label='DINOv2')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('DINOv2 특징 (PCA)')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.scatter(resnet_2d[:, 0], resnet_2d[:, 1], alpha=0.7, color='orange', label='ResNet')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('ResNet 특징 (PCA)')
            plt.legend()
            
            # 특징 통계 비교
            plt.subplot(1, 3, 3)
            comparison_stats = f"""특징 비교:
            
DINOv2:
- 차원: {dinov2_array.shape[1]}
- 평균: {np.mean(dinov2_array):.4f}
- 표준편차: {np.std(dinov2_array):.4f}

ResNet:
- 차원: {resnet_array.shape[1]}
- 평균: {np.mean(resnet_array):.4f}
- 표준편차: {np.std(resnet_array):.4f}

설명 분산 (PC1+PC2):
- DINOv2: {sum(pca_dino.explained_variance_ratio_):.3f}
- ResNet: {sum(pca_resnet.explained_variance_ratio_):.3f}"""
            
            plt.text(0.1, 0.5, comparison_stats, fontsize=10,
                    verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('특징 비교')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            return dinov2_array, resnet_array
            
        except Exception as e:
            print(f"❌ 비교 실험 실패: {e}")

class SAMDemo:
    """
    Segment Anything Model (SAM) 데모
    """
    def __init__(self):
        """SAM 모델 초기화"""
        self.model = None
        self.predictor = None
        self.load_model()
    
    def load_model(self):
        """SAM 모델 로드"""
        try:
            print("🔄 SAM 모델 로딩 중...")
            
            # SAM 모델 로드 (실제 구현에서는 segment-anything 패키지 필요)
            # pip install git+https://github.com/facebookresearch/segment-anything.git
            
            # 여기서는 시뮬레이션으로 대체
            print("💡 SAM 모델 시뮬레이션 모드")
            print("   실제 사용을 위해서는 segment-anything 패키지 설치 필요")
            
        except Exception as e:
            print(f"❌ SAM 모델 로드 실패: {e}")
    
    def segment_image(self, image, points=None, boxes=None):
        """이미지 세그멘테이션 (시뮬레이션)"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image.squeeze())
        
        # 실제 SAM 구현 대신 간단한 시뮬레이션
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # 가짜 세그멘테이션 마스크 생성
        mask = np.zeros((h, w), dtype=bool)
        
        if points is not None:
            # 포인트 주변에 원형 마스크 생성
            for point in points:
                x, y = point
                y_coords, x_coords = np.ogrid[:h, :w]
                mask_circle = (x_coords - x)**2 + (y_coords - y)**2 <= 50**2
                mask = mask | mask_circle
        else:
            # 중앙에 임의의 마스크 생성
            center_x, center_y = w//2, h//2
            y_coords, x_coords = np.ogrid[:h, :w]
            mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= (min(w, h)//4)**2
        
        return mask
    
    def interactive_segmentation_demo(self, image):
        """대화형 세그멘테이션 데모"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        img_array = np.array(image)
        
        # 시뮬레이션: 여러 포인트에서 세그멘테이션
        points = [(100, 100), (200, 150), (150, 200)]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 원본 이미지
        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title('원본 이미지')
        axes[0, 0].axis('off')
        
        # 포인트 표시
        axes[0, 1].imshow(img_array)
        for i, (x, y) in enumerate(points):
            axes[0, 1].plot(x, y, 'ro', markersize=10)
            axes[0, 1].text(x+10, y-10, f'P{i+1}', color='red', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('클릭 포인트')
        axes[0, 1].axis('off')
        
        # 세그멘테이션 결과
        mask = self.segment_image(image, points)
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('세그멘테이션 마스크')
        axes[1, 0].axis('off')
        
        # 오버레이
        overlay = img_array.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[1, 1].imshow(overlay.astype(np.uint8))
        axes[1, 1].set_title('마스크 오버레이')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return mask

def create_sample_images():
    """샘플 이미지 생성"""
    images = []
    labels = []
    
    # 다양한 패턴의 이미지 생성
    patterns = [
        ('체크보드', 'checkerboard'),
        ('원형', 'circle'),
        ('줄무늬', 'stripes'),
        ('그라디언트', 'gradient'),
        ('노이즈', 'noise')
    ]
    
    for name, pattern in patterns:
        img = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(img)
        
        if pattern == 'checkerboard':
            for i in range(0, 224, 32):
                for j in range(0, 224, 32):
                    if (i//32 + j//32) % 2 == 0:
                        draw.rectangle([i, j, i+32, j+32], fill='black')
        
        elif pattern == 'circle':
            draw.ellipse([50, 50, 174, 174], fill='blue')
            draw.ellipse([80, 80, 144, 144], fill='red')
        
        elif pattern == 'stripes':
            for i in range(0, 224, 20):
                if (i//20) % 2 == 0:
                    draw.rectangle([0, i, 224, i+20], fill='green')
        
        elif pattern == 'gradient':
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(224):
                img_array[:, i] = [i, 255-i, 128]
            img = Image.fromarray(img_array)
        
        elif pattern == 'noise':
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
        
        images.append(img)
        labels.append(name)
    
    return images, labels

def multimodal_benchmark():
    """멀티모달 API 벤치마크 시뮬레이션"""
    print("🚀 멀티모달 API 성능 비교")
    print("=" * 50)
    
    # API 성능 시뮬레이션 데이터
    api_performance = {
        'Gemini Vision': {
            'response_time': np.random.normal(1200, 200, 10),  # ms
            'accuracy': 0.95,
            'cost_per_1k': 0.0,  # 무료
            'rate_limit': '60 RPM',
            'features': ['이미지 분석', '텍스트 추출', 'OCR', '객체 인식']
        },
        'GPT-4V': {
            'response_time': np.random.normal(2100, 300, 10),
            'accuracy': 0.92,
            'cost_per_1k': 0.01,
            'rate_limit': '100 RPM',
            'features': ['이미지 분석', '상세 설명', '추론', '창작']
        },
        'Llama Vision': {
            'response_time': np.random.normal(1800, 250, 10),
            'accuracy': 0.88,
            'cost_per_1k': 0.0,  # 3개월 무료
            'rate_limit': '50 RPM',
            'features': ['오픈소스', '커스터마이징', '로컬 실행']
        },
        'Claude Vision': {
            'response_time': np.random.normal(1600, 180, 10),
            'accuracy': 0.90,
            'cost_per_1k': 0.008,
            'rate_limit': '50 RPM',
            'features': ['안전성', '윤리적 AI', '긴 컨텍스트']
        }
    }
    
    # 성능 비교 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 응답 시간 비교
    apis = list(api_performance.keys())
    response_times = [np.mean(data['response_time']) for data in api_performance.values()]
    response_stds = [np.std(data['response_time']) for data in api_performance.values()]
    
    bars1 = axes[0, 0].bar(apis, response_times, yerr=response_stds, capsize=5, alpha=0.7)
    axes[0, 0].set_title('평균 응답 시간')
    axes[0, 0].set_ylabel('시간 (ms)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    for bar, time_val in zip(bars1, response_times):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                       f'{time_val:.0f}ms', ha='center', va='bottom')
    
    # 정확도 비교
    accuracies = [data['accuracy'] for data in api_performance.values()]
    bars2 = axes[0, 1].bar(apis, accuracies, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('정확도 비교')
    axes[0, 1].set_ylabel('정확도')
    axes[0, 1].set_ylim(0.8, 1.0)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar, acc_val in zip(bars2, accuracies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{acc_val:.2f}', ha='center', va='bottom')
    
    # 비용 비교
    costs = [data['cost_per_1k'] for data in api_performance.values()]
    bars3 = axes[1, 0].bar(apis, costs, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('1K 요청당 비용')
    axes[1, 0].set_ylabel('비용 ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    for bar, cost_val in zip(bars3, costs):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                       f'${cost_val:.3f}' if cost_val > 0 else 'Free', ha='center', va='bottom')
    
    # 종합 점수 (가중 평균)
    # 점수 = (1/응답시간) * 0.3 + 정확도 * 0.4 + (1/비용+0.001) * 0.3
    composite_scores = []
    for api, data in api_performance.items():
        time_score = 1 / (np.mean(data['response_time']) / 1000)  # 초당 처리량
        acc_score = data['accuracy']
        cost_score = 1 / (data['cost_per_1k'] + 0.001)  # 비용 역수
        
        composite = time_score * 0.3 + acc_score * 0.4 + (cost_score / 1000) * 0.3
        composite_scores.append(composite)
    
    bars4 = axes[1, 1].bar(apis, composite_scores, alpha=0.7, color='gold')
    axes[1, 1].set_title('종합 성능 점수')
    axes[1, 1].set_ylabel('점수')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for bar, score_val in zip(bars4, composite_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score_val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 상세 비교 테이블
    print("\n📊 API 상세 비교")
    print("-" * 80)
    print(f"{'API':<15} {'응답시간(ms)':<12} {'정확도':<8} {'비용/1K':<10} {'제한':<10}")
    print("-" * 80)
    
    for api, data in api_performance.items():
        avg_time = np.mean(data['response_time'])
        cost_str = f"${data['cost_per_1k']:.3f}" if data['cost_per_1k'] > 0 else "Free"
        print(f"{api:<15} {avg_time:<12.0f} {data['accuracy']:<8.2f} {cost_str:<10} {data['rate_limit']:<10}")
    
    # 선택 가이드
    print("\n🎯 API 선택 가이드")
    print("-" * 50)
    print("💰 비용 최우선: Gemini Vision 또는 Llama Vision")
    print("🎯 정확도 최우선: Gemini Vision")
    print("⚡ 속도 최우선: Gemini Vision")
    print("🔧 커스터마이징: Llama Vision")
    print("🛡️ 안전성: Claude Vision")
    
    return api_performance

def main():
    """메인 실습 함수"""
    print("🤖 Week 4: DINOv2 & SAM 데모")
    print("=" * 50)
    
    # 1. 샘플 이미지 생성
    print("\n1️⃣ 샘플 이미지 생성")
    sample_images, labels = create_sample_images()
    print(f"✅ {len(sample_images)}개 샘플 이미지 생성 완료")
    
    # 샘플 이미지 표시
    fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 3))
    for i, (img, label) in enumerate(zip(sample_images, labels)):
        axes[i].imshow(img)
        axes[i].set_title(label)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
    # 2. DINOv2 분석
    print("\n2️⃣ DINOv2 특징 추출 및 분석")
    dinov2_analyzer = DINOv2Analyzer('dinov2_vitb14')
    
    if dinov2_analyzer.model is not None:
        # 특징 시각화
        features_array, features_2d = dinov2_analyzer.visualize_features(sample_images, labels)
        
        # 자기지도학습 vs 지도학습 비교
        print("\n3️⃣ 자기지도학습 vs 지도학습 비교")
        dinov2_analyzer.compare_with_supervised(sample_images[:3])  # 처음 3개 이미지만 사용
    
    # 4. SAM 데모
    print("\n4️⃣ SAM 세그멘테이션 데모")
    sam_demo = SAMDemo()
    
    # 첫 번째 샘플 이미지로 세그멘테이션 데모
    mask = sam_demo.interactive_segmentation_demo(sample_images[1])
    
    # 5. 멀티모달 API 벤치마크
    print("\n5️⃣ 멀티모달 API 성능 비교")
    api_performance = multimodal_benchmark()
    
    print("\n🎉 모든 데모가 완료되었습니다!")
    print("\n📚 추가 실험 아이디어:")
    print("   - 실제 이미지 데이터셋으로 DINOv2 특징 분석")
    print("   - SAM을 활용한 자동 라벨링 시스템 구축")
    print("   - 다양한 DINOv2 모델 크기 비교 (ViT-S, B, L, G)")
    print("   - 실제 API를 사용한 멀티모달 성능 테스트")

if __name__ == "__main__":
    main()
