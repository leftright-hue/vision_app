"""
DINOv2를 활용한 특징 추출
자기지도학습 Vision Transformer의 활용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import requests
from io import BytesIO
from transformers import AutoImageProcessor, AutoModel
import gradio as gr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


class DINOv2FeatureExtractor:
    """DINOv2를 활용한 특징 추출기"""
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = None
    ):
        """
        Args:
            model_name: DINOv2 모델 이름
            device: 연산 디바이스
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델과 프로세서 로드
        print(f"Loading DINOv2 model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 특징 차원
        self.feature_dim = self.model.config.hidden_size
        
    def extract_features(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
        layer: str = 'last',
        pool: str = 'cls'
    ) -> torch.Tensor:
        """
        이미지에서 특징 추출
        
        Args:
            images: 입력 이미지(들)
            layer: 추출할 레이어 ('last', 'penultimate', 또는 레이어 번호)
            pool: 풀링 방법 ('cls', 'mean', 'max')
        
        Returns:
            특징 벡터 [B, D]
        """
        # 이미지 전처리
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list):
            inputs = self.processor(images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
        else:
            pixel_values = images.to(self.device)
        
        # 특징 추출
        with torch.no_grad():
            outputs = self.model(pixel_values, output_hidden_states=True)
        
        # 레이어 선택
        if layer == 'last':
            features = outputs.last_hidden_state
        elif layer == 'penultimate':
            features = outputs.hidden_states[-2]
        elif isinstance(layer, int):
            features = outputs.hidden_states[layer]
        else:
            features = outputs.last_hidden_state
        
        # 풀링
        if pool == 'cls':
            # CLS 토큰 사용
            features = features[:, 0]
        elif pool == 'mean':
            # 평균 풀링 (CLS 토큰 제외)
            features = features[:, 1:].mean(dim=1)
        elif pool == 'max':
            # 최대 풀링 (CLS 토큰 제외)
            features = features[:, 1:].max(dim=1)[0]
        else:
            features = features[:, 0]
        
        return features
    
    def extract_patch_features(
        self,
        images: Union[Image.Image, List[Image.Image]],
        reshape: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int]]]:
        """
        패치별 특징 추출 (공간 정보 포함)
        
        Args:
            images: 입력 이미지
            reshape: 2D 그리드로 재구성할지 여부
        
        Returns:
            패치 특징 [B, N, D] 또는 [B, H, W, D]
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        inputs = self.processor(images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values)
            patch_features = outputs.last_hidden_state[:, 1:]  # CLS 토큰 제외
        
        if reshape:
            B, N, D = patch_features.shape
            H = W = int(N ** 0.5)
            patch_features = patch_features.reshape(B, H, W, D)
            return patch_features, (H, W)
        
        return patch_features
    
    def compute_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        metric: str = 'cosine'
    ) -> torch.Tensor:
        """
        특징 간 유사도 계산
        
        Args:
            features1: 첫 번째 특징 벡터
            features2: 두 번째 특징 벡터
            metric: 유사도 메트릭 ('cosine', 'euclidean', 'dot')
        """
        if metric == 'cosine':
            features1 = F.normalize(features1, p=2, dim=-1)
            features2 = F.normalize(features2, p=2, dim=-1)
            similarity = torch.matmul(features1, features2.T)
        elif metric == 'euclidean':
            similarity = -torch.cdist(features1, features2, p=2)
        elif metric == 'dot':
            similarity = torch.matmul(features1, features2.T)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity


class DINOv2Applications:
    """DINOv2를 활용한 다양한 응용"""
    
    def __init__(self, feature_extractor: DINOv2FeatureExtractor):
        self.extractor = feature_extractor
        
    def semantic_segmentation(
        self,
        image: Image.Image,
        n_clusters: int = 5
    ) -> np.ndarray:
        """
        비지도 의미론적 분할
        
        Args:
            image: 입력 이미지
            n_clusters: 클러스터 수
        
        Returns:
            세그멘테이션 맵
        """
        # 패치 특징 추출
        patch_features, (H, W) = self.extractor.extract_patch_features(image, reshape=True)
        patch_features = patch_features.squeeze(0).cpu().numpy()  # [H, W, D]
        
        # K-means 클러스터링
        features_flat = patch_features.reshape(-1, patch_features.shape[-1])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_flat)
        
        # 세그멘테이션 맵 생성
        segmentation_map = labels.reshape(H, W)
        
        return segmentation_map
    
    def image_retrieval(
        self,
        query_image: Image.Image,
        database_images: List[Image.Image],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        이미지 검색
        
        Args:
            query_image: 쿼리 이미지
            database_images: 데이터베이스 이미지들
            top_k: 상위 K개 결과
        
        Returns:
            (인덱스, 유사도) 튜플 리스트
        """
        # 쿼리 특징 추출
        query_features = self.extractor.extract_features(query_image)
        
        # 데이터베이스 특징 추출
        db_features = []
        for img in database_images:
            features = self.extractor.extract_features(img)
            db_features.append(features)
        db_features = torch.stack(db_features).squeeze()
        
        # 유사도 계산
        similarities = self.extractor.compute_similarity(
            query_features,
            db_features,
            metric='cosine'
        ).squeeze()
        
        # Top-K 선택
        top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(database_images)))
        
        results = [(idx.item(), val.item()) for idx, val in zip(top_k_indices, top_k_values)]
        
        return results
    
    def visual_correspondence(
        self,
        image1: Image.Image,
        image2: Image.Image,
        n_points: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        이미지 간 시각적 대응점 찾기
        
        Args:
            image1: 첫 번째 이미지
            image2: 두 번째 이미지
            n_points: 대응점 수
        
        Returns:
            (이미지1 포인트, 이미지2 포인트, 매칭 스코어)
        """
        # 패치 특징 추출
        features1, (H1, W1) = self.extractor.extract_patch_features(image1, reshape=True)
        features2, (H2, W2) = self.extractor.extract_patch_features(image2, reshape=True)
        
        features1 = features1.squeeze(0).reshape(-1, features1.shape[-1])  # [N1, D]
        features2 = features2.squeeze(0).reshape(-1, features2.shape[-1])  # [N2, D]
        
        # 유사도 행렬 계산
        similarity_matrix = self.extractor.compute_similarity(
            features1,
            features2,
            metric='cosine'
        )
        
        # 상위 대응점 찾기
        scores, indices = torch.topk(similarity_matrix.flatten(), n_points)
        
        # 좌표 변환
        indices_2d = torch.stack([indices // similarity_matrix.shape[1], 
                                  indices % similarity_matrix.shape[1]], dim=1)
        
        points1 = []
        points2 = []
        match_scores = []
        
        for i in range(n_points):
            idx1 = indices_2d[i, 0].item()
            idx2 = indices_2d[i, 1].item()
            
            # 패치 인덱스를 픽셀 좌표로 변환
            y1, x1 = idx1 // W1, idx1 % W1
            y2, x2 = idx2 // W2, idx2 % W2
            
            # 이미지 크기에 맞게 스케일링
            h1, w1 = image1.size
            h2, w2 = image2.size
            
            points1.append([x1 * w1 // W1, y1 * h1 // H1])
            points2.append([x2 * w2 // W2, y2 * h2 // H2])
            match_scores.append(scores[i].item())
        
        return np.array(points1), np.array(points2), np.array(match_scores)
    
    def visualize_features(
        self,
        image: Image.Image,
        method: str = 'pca'
    ) -> Image.Image:
        """
        특징 시각화
        
        Args:
            image: 입력 이미지
            method: 시각화 방법 ('pca', 'attention')
        
        Returns:
            시각화된 이미지
        """
        # 패치 특징 추출
        patch_features, (H, W) = self.extractor.extract_patch_features(image, reshape=True)
        patch_features = patch_features.squeeze(0).cpu().numpy()
        
        if method == 'pca':
            # PCA를 사용한 차원 축소
            features_flat = patch_features.reshape(-1, patch_features.shape[-1])
            pca = PCA(n_components=3)
            features_pca = pca.fit_transform(features_flat)
            
            # RGB 이미지로 변환
            features_rgb = features_pca.reshape(H, W, 3)
            features_rgb = (features_rgb - features_rgb.min()) / (features_rgb.max() - features_rgb.min())
            features_rgb = (features_rgb * 255).astype(np.uint8)
            
            # 원본 크기로 리사이즈
            from PIL import Image as PILImage
            visualization = PILImage.fromarray(features_rgb)
            visualization = visualization.resize(image.size, PILImage.LANCZOS)
            
        elif method == 'attention':
            # 자기 어텐션 맵 계산
            features_flat = patch_features.reshape(-1, patch_features.shape[-1])
            features_norm = features_flat / np.linalg.norm(features_flat, axis=-1, keepdims=True)
            attention_map = np.matmul(features_norm, features_norm.T)
            
            # 평균 어텐션
            avg_attention = attention_map.mean(axis=0).reshape(H, W)
            
            # 히트맵 생성
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(avg_attention, cmap='hot')
            plt.title('Attention Heatmap')
            plt.colorbar()
            plt.axis('off')
            
            plt.tight_layout()
            
            # 이미지로 변환
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            visualization = Image.open(buf)
            plt.close()
        else:
            visualization = image
        
        return visualization


def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # DINOv2 초기화
    extractor = DINOv2FeatureExtractor()
    apps = DINOv2Applications(extractor)
    
    def process_segmentation(image, n_clusters):
        """세그멘테이션 처리"""
        segmap = apps.semantic_segmentation(image, n_clusters)
        
        # 컬러맵 적용
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(segmap, cmap='tab20')
        plt.title(f'Segmentation (K={n_clusters})')
        plt.axis('off')
        
        plt.tight_layout()
        
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        result = Image.open(buf)
        plt.close()
        
        return result
    
    def process_visualization(image, method):
        """특징 시각화 처리"""
        return apps.visualize_features(image, method)
    
    with gr.Blocks(title="DINOv2 Feature Extraction") as app:
        gr.Markdown("# 🦖 DINOv2 Feature Extraction Demo")
        
        with gr.Tab("Semantic Segmentation"):
            with gr.Row():
                with gr.Column():
                    seg_image = gr.Image(type="pil", label="Input Image")
                    seg_clusters = gr.Slider(
                        minimum=2,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of Clusters"
                    )
                    seg_button = gr.Button("Segment")
                
                seg_output = gr.Image(label="Segmentation Result")
            
            seg_button.click(
                process_segmentation,
                inputs=[seg_image, seg_clusters],
                outputs=seg_output
            )
        
        with gr.Tab("Feature Visualization"):
            with gr.Row():
                with gr.Column():
                    viz_image = gr.Image(type="pil", label="Input Image")
                    viz_method = gr.Radio(
                        choices=["pca", "attention"],
                        value="pca",
                        label="Visualization Method"
                    )
                    viz_button = gr.Button("Visualize")
                
                viz_output = gr.Image(label="Visualization")
            
            viz_button.click(
                process_visualization,
                inputs=[viz_image, viz_method],
                outputs=viz_output
            )
        
        gr.Markdown("""
        ## About DINOv2
        
        DINOv2는 자기지도학습으로 훈련된 Vision Transformer입니다.
        - 라벨 없이 학습됨
        - 범용적인 시각 특징 제공
        - 다양한 다운스트림 태스크에 활용 가능
        
        ### Applications
        1. **Semantic Segmentation**: 비지도 클러스터링
        2. **Image Retrieval**: 유사 이미지 검색
        3. **Visual Correspondence**: 대응점 찾기
        4. **Feature Visualization**: 특징 시각화
        """)
    
    return app


if __name__ == "__main__":
    # 테스트
    print("Initializing DINOv2...")
    extractor = DINOv2FeatureExtractor()
    apps = DINOv2Applications(extractor)
    
    # 더미 이미지 생성
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # 특징 추출 테스트
    features = extractor.extract_features(test_image)
    print(f"Feature shape: {features.shape}")
    
    # 패치 특징 추출 테스트
    patch_features, (H, W) = extractor.extract_patch_features(test_image)
    print(f"Patch features shape: {patch_features.shape}")
    print(f"Grid size: {H}x{W}")
    
    # Gradio 앱 실행 (실제 환경에서)
    # app = create_gradio_interface()
    # app.launch()
    
    print("\nDINOv2 feature extraction setup complete!")