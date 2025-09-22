"""
05. HuggingFace 생태계 활용
Week 2: 디지털 이미지 기초와 CNN

HuggingFace의 사전훈련 모델을 활용한 이미지 분석 실습
"""

from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os
import torch

class HuggingFaceModels:
    """HuggingFace 모델 활용 실습 클래스"""

    def __init__(self):
        self.setup_korean_font()
        print("🤗 HuggingFace 모델 초기화 중...")
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"사용 디바이스: {'GPU' if self.device == 0 else 'CPU'}")

    def setup_korean_font(self):
        """한글 폰트 설정"""
        if os.name == 'nt':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

    def create_sample_image(self):
        """테스트용 샘플 이미지 생성"""
        # 간단한 도형이 있는 이미지 생성
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255

        # 빨간 사각형
        img[50:100, 50:100] = [255, 0, 0]

        # 파란 원 (근사)
        center = (150, 150)
        radius = 30
        y, x = np.ogrid[:224, :224]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img[mask] = [0, 0, 255]

        # 초록 삼각형 (근사)
        for i in range(30):
            for j in range(i):
                if 180+i < 224 and 50+j < 224:
                    img[180+i, 50+j] = [0, 255, 0]
                if 180+i < 224 and 50-j >= 0:
                    img[180+i, 50-j] = [0, 255, 0]

        return Image.fromarray(img)

    def download_sample_image(self, url=None):
        """인터넷에서 샘플 이미지 다운로드"""
        if url is None:
            # 기본 샘플 이미지 URL (고양이 이미지)
            url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"

        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            return img
        except:
            print("이미지 다운로드 실패, 로컬 샘플 이미지 생성")
            return self.create_sample_image()

    def demonstrate_image_classification(self):
        """5.3 이미지 분류 실습"""
        print("\n=== 5.3 이미지 분류 실습 ===")

        # 분류 파이프라인 생성
        classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device=self.device
        )

        # 샘플 이미지 준비
        images = {
            "샘플 이미지": self.create_sample_image(),
            "다운로드 이미지": self.download_sample_image()
        }

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (name, img) in enumerate(images.items()):
            # 분류 수행
            results = classifier(img)

            # 이미지 표시
            axes[idx].imshow(img)
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].axis('off')

            # 상위 3개 결과 표시
            result_text = "예측 결과:\n"
            for i, result in enumerate(results[:3]):
                result_text += f"{i+1}. {result['label']}: {result['score']:.2%}\n"

            axes[idx].text(0.5, -0.15, result_text,
                          transform=axes[idx].transAxes,
                          ha='center', va='top', fontsize=10,
                          bbox=dict(boxstyle="round", facecolor='lightyellow'))

        plt.tight_layout()
        plt.savefig('05_image_classification.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 결과 출력
        print("\n📊 이미지 분류 결과:")
        for name, img in images.items():
            print(f"\n{name}:")
            results = classifier(img)
            for result in results[:3]:
                print(f"  - {result['label']}: {result['score']:.2%}")

    def demonstrate_object_detection(self):
        """5.4 객체 검출 실습"""
        print("\n=== 5.4 객체 검출 실습 ===")

        # 객체 검출 파이프라인
        detector = pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            device=self.device
        )

        # 샘플 이미지
        img = self.download_sample_image()

        # 객체 검출 수행
        results = detector(img)

        # 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)

        # 검출된 객체 박스 그리기
        img_width, img_height = img.size

        for obj in results:
            # 바운딩 박스 좌표
            box = obj['box']
            xmin = box['xmin']
            ymin = box['ymin']
            xmax = box['xmax']
            ymax = box['ymax']

            # 박스 그리기
            rect = plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            # 레이블 표시
            label = f"{obj['label']}: {obj['score']:.2f}"
            ax.text(xmin, ymin - 5, label,
                   bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7),
                   fontsize=10, fontweight='bold')

        ax.set_title('객체 검출 결과', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig('05_object_detection.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 결과 출력
        print("\n🎯 검출된 객체:")
        for obj in results:
            print(f"  - {obj['label']}: {obj['score']:.2%}")
            print(f"    위치: ({obj['box']['xmin']:.0f}, {obj['box']['ymin']:.0f}) "
                  f"- ({obj['box']['xmax']:.0f}, {obj['box']['ymax']:.0f})")

    def demonstrate_image_segmentation(self):
        """5.4 이미지 세그멘테이션 실습"""
        print("\n=== 5.4 이미지 세그멘테이션 실습 ===")

        # 세그멘테이션 파이프라인
        segmenter = pipeline(
            "image-segmentation",
            model="facebook/detr-resnet-50-panoptic",
            device=self.device
        )

        # 샘플 이미지
        img = self.download_sample_image()

        # 세그멘테이션 수행
        results = segmenter(img)

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 원본 이미지
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('원본 이미지', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 세그멘테이션 결과
        if results:
            # 첫 번째 세그멘테이션 마스크
            if len(results) > 0:
                mask1 = results[0]['mask']
                axes[0, 1].imshow(mask1, cmap='viridis')
                axes[0, 1].set_title(f"세그멘트 1: {results[0].get('label', 'Unknown')}",
                                   fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')

            # 두 번째 세그멘테이션 마스크 (있는 경우)
            if len(results) > 1:
                mask2 = results[1]['mask']
                axes[1, 0].imshow(mask2, cmap='plasma')
                axes[1, 0].set_title(f"세그멘트 2: {results[1].get('label', 'Unknown')}",
                                   fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')

            # 모든 마스크 오버레이
            combined_mask = np.zeros_like(np.array(img))
            for i, result in enumerate(results[:3]):  # 최대 3개 세그멘트
                mask = np.array(result['mask'])
                if len(mask.shape) == 2:
                    mask = np.stack([mask] * 3, axis=2)
                combined_mask += mask * (i + 1) * 50

            axes[1, 1].imshow(np.array(img))
            axes[1, 1].imshow(combined_mask, alpha=0.5)
            axes[1, 1].set_title('통합 세그멘테이션 결과', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('05_image_segmentation.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 결과 출력
        print("\n🎯 세그멘테이션 결과:")
        for i, result in enumerate(results):
            label = result.get('label', f'Segment {i+1}')
            print(f"  - {label}: 마스크 크기 {np.array(result['mask']).shape}")

    def demonstrate_feature_extraction(self):
        """이미지 특징 추출 실습"""
        print("\n=== 이미지 특징 추출 실습 ===")

        from transformers import AutoFeatureExtractor, AutoModel

        # 모델과 특징 추출기 로드
        model_name = "google/vit-base-patch16-224"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # 샘플 이미지
        img = self.create_sample_image()

        # 특징 추출
        inputs = feature_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # 특징 벡터
        features = outputs.last_hidden_state
        pooled_features = outputs.pooler_output

        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 원본 이미지
        axes[0].imshow(img)
        axes[0].set_title('원본 이미지', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 특징 맵 (첫 번째 패치들)
        feature_map = features[0, 1:, :].reshape(14, 14, -1)
        axes[1].imshow(feature_map[:, :, 0], cmap='viridis')
        axes[1].set_title('특징 맵 (14x14 패치)', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # 풀링된 특징 벡터
        axes[2].bar(range(min(50, pooled_features.shape[1])),
                   pooled_features[0, :50].numpy())
        axes[2].set_title('특징 벡터 (처음 50차원)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('차원')
        axes[2].set_ylabel('값')

        plt.tight_layout()
        plt.savefig('05_feature_extraction.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n📐 특징 추출 결과:")
        print(f"  - 특징 맵 크기: {features.shape}")
        print(f"  - 풀링된 특징 크기: {pooled_features.shape}")
        print(f"  - 특징 벡터 차원: {pooled_features.shape[1]}")

    def compare_models(self):
        """다양한 모델 비교"""
        print("\n=== 모델 성능 비교 ===")

        # 비교할 모델들
        models = {
            "ViT": "google/vit-base-patch16-224",
            "ResNet": "microsoft/resnet-50",
            "EfficientNet": "timm/efficientnet_b0"
        }

        # 샘플 이미지
        img = self.download_sample_image()

        results_dict = {}

        print("\n🔄 모델별 예측 수행 중...")
        for model_name, model_id in models.items():
            try:
                classifier = pipeline(
                    "image-classification",
                    model=model_id,
                    device=self.device
                )
                results = classifier(img)
                results_dict[model_name] = results[:3]
                print(f"  ✅ {model_name} 완료")
            except Exception as e:
                print(f"  ❌ {model_name} 실패: {e}")
                results_dict[model_name] = []

        # 결과 비교 출력
        print("\n📊 모델별 예측 결과 비교:")
        for model_name, results in results_dict.items():
            print(f"\n{model_name}:")
            for result in results:
                print(f"  - {result['label']}: {result['score']:.2%}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🤗 05. HuggingFace 생태계 활용")
    print("=" * 60)

    hf_models = HuggingFaceModels()

    # 5.3 이미지 분류
    hf_models.demonstrate_image_classification()

    # 5.4 객체 검출
    hf_models.demonstrate_object_detection()

    # 5.5 이미지 세그멘테이션
    hf_models.demonstrate_image_segmentation()

    # 특징 추출
    hf_models.demonstrate_feature_extraction()

    # 모델 비교
    hf_models.compare_models()

    print("\n" + "=" * 60)
    print("✅ 05. HuggingFace 모델 실습 완료!")
    print("생성된 파일:")
    print("  - 05_image_classification.png")
    print("  - 05_object_detection.png")
    print("  - 05_image_segmentation.png")
    print("  - 05_feature_extraction.png")
    print("=" * 60)

if __name__ == "__main__":
    main()