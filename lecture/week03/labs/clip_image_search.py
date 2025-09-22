"""
CLIP을 활용한 이미지 검색 시스템
Week 3: 딥러닝 영상처리

Hugging Face의 CLIP 모델을 사용하여 자연어 기반 이미지 검색을 구현합니다.
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import os
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """검색 결과를 담는 데이터 클래스"""
    path: str
    score: float
    metadata: Optional[Dict] = None
    
    def __repr__(self):
        return f"SearchResult(path='{Path(self.path).name}', score={self.score:.3f})"


class CLIPImageSearchEngine:
    """
    CLIP 기반 이미지 검색 엔진
    
    주요 기능:
    - 텍스트로 이미지 검색
    - 이미지로 유사 이미지 검색
    - 복합 쿼리 지원 (positive + negative)
    - 임베딩 캐싱 및 저장/로드
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        cache_dir: Optional[str] = "./clip_cache"
    ):
        """
        Args:
            model_name: 사용할 CLIP 모델 이름
            device: 연산 디바이스 ('cuda', 'cpu', None for auto)
            cache_dir: 임베딩 캐시 저장 디렉토리
        """
        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # 모델 및 프로세서 로드
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 캐시 설정
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 데이터베이스
        self.image_paths: List[str] = []
        self.image_embeddings: Optional[torch.Tensor] = None
        self.metadata: Dict[str, Dict] = {}
        
        logger.info("CLIP search engine initialized successfully")
    
    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        단일 이미지를 임베딩으로 변환
        
        Args:
            image: 이미지 경로 또는 PIL Image 객체
            
        Returns:
            정규화된 이미지 임베딩
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 임베딩 생성
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # L2 정규화
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        return image_features
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        텍스트를 임베딩으로 변환
        
        Args:
            text: 텍스트 또는 텍스트 리스트
            
        Returns:
            정규화된 텍스트 임베딩
        """
        if isinstance(text, str):
            text = [text]
        
        # 텍스트 전처리
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 임베딩 생성
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # L2 정규화
            text_features = F.normalize(text_features, p=2, dim=-1)
        
        return text_features
    
    def index_images(
        self,
        image_dir: str,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
        batch_size: int = 32,
        save_cache: bool = True
    ) -> int:
        """
        디렉토리의 모든 이미지를 인덱싱
        
        Args:
            image_dir: 이미지가 있는 디렉토리 경로
            extensions: 처리할 이미지 확장자
            batch_size: 배치 처리 크기
            save_cache: 캐시 저장 여부
            
        Returns:
            인덱싱된 이미지 수
        """
        image_dir = Path(image_dir)
        
        # 캐시 확인
        cache_file = self.cache_dir / f"{image_dir.name}_embeddings.pkl" if self.cache_dir else None
        if cache_file and cache_file.exists():
            logger.info(f"Loading cached embeddings from {cache_file}")
            return self.load_index(cache_file)
        
        # 이미지 파일 수집
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(image_files)} images to index")
        
        if not image_files:
            return 0
        
        # 배치 처리
        embeddings_list = []
        valid_paths = []
        
        for i in tqdm(range(0, len(image_files), batch_size), desc="Indexing images"):
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            batch_paths = []
            
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    batch_paths.append(str(img_path))
                    
                    # 메타데이터 수집
                    self.metadata[str(img_path)] = {
                        'name': img_path.name,
                        'size': img_path.stat().st_size,
                        'dimensions': img.size,
                        'format': img.format
                    }
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {e}")
                    continue
            
            if batch_images:
                # 배치 임베딩 생성
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    batch_features = self.model.get_image_features(**inputs)
                    batch_features = F.normalize(batch_features, p=2, dim=-1)
                
                embeddings_list.append(batch_features.cpu())
                valid_paths.extend(batch_paths)
        
        # 임베딩 결합
        self.image_embeddings = torch.cat(embeddings_list, dim=0)
        self.image_paths = valid_paths
        
        # 캐시 저장
        if save_cache and cache_file:
            self.save_index(cache_file)
        
        logger.info(f"Successfully indexed {len(self.image_paths)} images")
        return len(self.image_paths)
    
    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        텍스트 쿼리로 이미지 검색
        
        Args:
            query: 검색 쿼리 텍스트
            top_k: 반환할 최대 결과 수
            threshold: 최소 유사도 임계값
            
        Returns:
            검색 결과 리스트
        """
        if self.image_embeddings is None:
            raise ValueError("No images indexed. Call index_images() first.")
        
        # 쿼리 인코딩
        query_features = self.encode_text(query)
        
        # 유사도 계산 (코사인 유사도)
        similarities = (self.image_embeddings @ query_features.cpu().T).squeeze()
        
        # Top-K 선택
        if threshold is not None:
            mask = similarities >= threshold
            indices = torch.where(mask)[0]
            scores = similarities[mask]
            sorted_indices = scores.argsort(descending=True)[:top_k]
            top_indices = indices[sorted_indices]
        else:
            top_indices = similarities.argsort(descending=True)[:top_k]
        
        # 결과 생성
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                path=self.image_paths[idx],
                score=similarities[idx].item(),
                metadata=self.metadata.get(self.image_paths[idx])
            ))
        
        return results
    
    def search_by_image(
        self,
        image: Union[str, Image.Image],
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        이미지로 유사한 이미지 검색
        
        Args:
            image: 쿼리 이미지 (경로 또는 PIL Image)
            top_k: 반환할 최대 결과 수
            exclude_self: 쿼리 이미지 자체를 결과에서 제외
            
        Returns:
            검색 결과 리스트
        """
        if self.image_embeddings is None:
            raise ValueError("No images indexed. Call index_images() first.")
        
        # 쿼리 이미지 인코딩
        query_features = self.encode_image(image)
        
        # 유사도 계산
        similarities = (self.image_embeddings @ query_features.cpu().T).squeeze()
        
        # 자기 자신 제외 처리
        if exclude_self and isinstance(image, str):
            if image in self.image_paths:
                self_idx = self.image_paths.index(image)
                similarities[self_idx] = -1
        
        # Top-K 선택
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        # 결과 생성
        results = []
        for idx in top_indices:
            if similarities[idx] > -1:  # 제외된 항목 스킵
                results.append(SearchResult(
                    path=self.image_paths[idx],
                    score=similarities[idx].item(),
                    metadata=self.metadata.get(self.image_paths[idx])
                ))
        
        return results
    
    def advanced_search(
        self,
        positive_queries: List[str],
        negative_queries: Optional[List[str]] = None,
        top_k: int = 10,
        weight_positive: float = 1.0,
        weight_negative: float = 0.5
    ) -> List[SearchResult]:
        """
        고급 검색: 포함/제외 조건을 사용한 복합 쿼리
        
        Args:
            positive_queries: 포함해야 할 특징들
            negative_queries: 제외해야 할 특징들
            top_k: 반환할 최대 결과 수
            weight_positive: 포지티브 쿼리 가중치
            weight_negative: 네거티브 쿼리 가중치
            
        Returns:
            검색 결과 리스트
        """
        if self.image_embeddings is None:
            raise ValueError("No images indexed. Call index_images() first.")
        
        scores = torch.zeros(len(self.image_embeddings))
        
        # 포지티브 쿼리 처리
        if positive_queries:
            pos_features = self.encode_text(positive_queries)
            for i in range(pos_features.shape[0]):
                similarities = (self.image_embeddings @ pos_features[i].cpu().unsqueeze(1)).squeeze()
                scores += weight_positive * similarities / len(positive_queries)
        
        # 네거티브 쿼리 처리
        if negative_queries:
            neg_features = self.encode_text(negative_queries)
            for i in range(neg_features.shape[0]):
                similarities = (self.image_embeddings @ neg_features[i].cpu().unsqueeze(1)).squeeze()
                scores -= weight_negative * similarities / len(negative_queries)
        
        # Top-K 선택
        top_indices = scores.argsort(descending=True)[:top_k]
        
        # 결과 생성
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                path=self.image_paths[idx],
                score=scores[idx].item(),
                metadata=self.metadata.get(self.image_paths[idx])
            ))
        
        return results
    
    def save_index(self, filepath: str):
        """
        인덱스를 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        data = {
            'image_paths': self.image_paths,
            'image_embeddings': self.image_embeddings.cpu() if self.image_embeddings is not None else None,
            'metadata': self.metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str) -> int:
        """
        저장된 인덱스 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드된 이미지 수
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.image_paths = data['image_paths']
        self.image_embeddings = data['image_embeddings']
        self.metadata = data['metadata']
        
        logger.info(f"Index loaded from {filepath}: {len(self.image_paths)} images")
        return len(self.image_paths)
    
    def visualize_search_results(
        self,
        results: List[SearchResult],
        query: str,
        cols: int = 5
    ):
        """
        검색 결과를 시각화
        
        Args:
            results: 검색 결과 리스트
            query: 검색 쿼리
            cols: 열 수
        """
        n_results = len(results)
        rows = (n_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Search Results for: "{query}"', fontsize=16)
        
        for idx, result in enumerate(results):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            try:
                img = Image.open(result.path)
                ax.imshow(img)
                ax.set_title(f'Score: {result.score:.3f}', fontsize=10)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading image", 
                       ha='center', va='center')
                ax.axis('off')
        
        # 남은 서브플롯 숨기기
        for idx in range(n_results, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def compute_similarity_matrix(self, queries: List[str]) -> np.ndarray:
        """
        여러 쿼리와 모든 이미지 간의 유사도 행렬 계산
        
        Args:
            queries: 쿼리 리스트
            
        Returns:
            유사도 행렬 (queries x images)
        """
        if self.image_embeddings is None:
            raise ValueError("No images indexed. Call index_images() first.")
        
        # 모든 쿼리 인코딩
        query_features = self.encode_text(queries)
        
        # 유사도 행렬 계산
        similarity_matrix = query_features.cpu() @ self.image_embeddings.T
        
        return similarity_matrix.numpy()


def demonstrate_clip_search():
    """
    CLIP 검색 엔진 데모
    """
    print("="*60)
    print("CLIP Image Search Engine Demo")
    print("="*60)
    
    # 엔진 초기화
    engine = CLIPImageSearchEngine()
    
    # 샘플 이미지 생성 (실제로는 실제 이미지 경로 사용)
    print("\n1. Creating sample images...")
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # 샘플 이미지 생성 (예시)
    for i in range(5):
        img = Image.new('RGB', (224, 224), 
                       color=(np.random.randint(0, 255), 
                             np.random.randint(0, 255),
                             np.random.randint(0, 255)))
        img.save(sample_dir / f"sample_{i}.jpg")
    
    # 인덱싱
    print("\n2. Indexing images...")
    n_indexed = engine.index_images(str(sample_dir))
    print(f"   Indexed {n_indexed} images")
    
    # 텍스트 검색
    print("\n3. Text-based search:")
    queries = [
        "a red object",
        "something blue",
        "bright colors",
        "dark image"
    ]
    
    for query in queries:
        results = engine.search_by_text(query, top_k=3)
        print(f"\n   Query: '{query}'")
        for result in results:
            print(f"      {result}")
    
    # 고급 검색
    print("\n4. Advanced search (positive + negative):")
    results = engine.advanced_search(
        positive_queries=["bright", "colorful"],
        negative_queries=["dark", "monochrome"],
        top_k=3
    )
    print(f"   Positive: bright, colorful")
    print(f"   Negative: dark, monochrome")
    for result in results:
        print(f"      {result}")
    
    # 유사도 행렬
    print("\n5. Similarity matrix:")
    sim_matrix = engine.compute_similarity_matrix(["red", "blue", "green"])
    print(f"   Shape: {sim_matrix.shape}")
    print(f"   Mean similarity: {sim_matrix.mean():.3f}")
    print(f"   Max similarity: {sim_matrix.max():.3f}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")


if __name__ == "__main__":
    demonstrate_clip_search()