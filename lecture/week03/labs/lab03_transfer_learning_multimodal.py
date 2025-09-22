"""
Lab 03: Transfer Learning과 멀티모달 API 통합 실습
Week 3: 딥러닝 영상처리

이 실습에서는 전이학습과 멀티모달 API를 통합하여
자연어 기반 사진첩 검색 앱을 구축합니다.
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
import json
import time
from dataclasses import dataclass
import logging

# Google Gemini (선택적)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Gemini API not available. Install google-generativeai to enable.")

# Together AI (선택적)
try:
    import together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    print("Together AI not available. Install together to enable.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhotoMetadata:
    """사진 메타데이터"""
    path: str
    filename: str
    caption: Optional[str] = None
    tags: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None
    timestamp: Optional[float] = None


class SmartPhotoAlbum:
    """
    스마트 사진첩 애플리케이션
    
    기능:
    1. Transfer Learning을 사용한 이미지 분류
    2. CLIP을 사용한 텍스트-이미지 검색
    3. Gemini/Together AI를 사용한 이미지 캡션 생성
    4. 통합 검색 및 관리 인터페이스
    """
    
    def __init__(self):
        """모델 및 API 초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 1. Transfer Learning 모델 초기화 (이미지 분류용)
        self.classifier = self._init_classifier()
        
        # 2. CLIP 모델 초기화 (텍스트-이미지 검색용)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        
        # 3. Gemini API 초기화 (캡션 생성용)
        self.gemini_model = None
        if GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini API initialized")
        
        # 4. Together AI 초기화 (대체 캡션 생성)
        self.together_available = False
        if TOGETHER_AVAILABLE and os.getenv('TOGETHER_API_KEY'):
            together.api_key = os.getenv('TOGETHER_API_KEY')
            self.together_available = True
            logger.info("Together AI initialized")
        
        # 사진 데이터베이스
        self.photos: Dict[str, PhotoMetadata] = {}
        self.embeddings = None
        self.photo_paths = []
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _init_classifier(self) -> nn.Module:
        """Transfer Learning 분류기 초기화"""
        # ResNet50 사용 (ImageNet pretrained)
        model = models.resnet50(pretrained=True)
        
        # Feature Extraction 모드
        for param in model.parameters():
            param.requires_grad = False
        
        # 새로운 분류 레이어 (예: 10개 카테고리)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )
        
        model.to(self.device)
        model.eval()
        return model
    
    def add_photo(self, image_path: str) -> PhotoMetadata:
        """
        사진을 앨범에 추가
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            생성된 PhotoMetadata
        """
        # 메타데이터 생성
        metadata = PhotoMetadata(
            path=image_path,
            filename=Path(image_path).name,
            timestamp=time.time()
        )
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 1. CLIP 임베딩 생성
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            metadata.embedding = image_features.cpu().numpy()
        
        # 2. 자동 캡션 생성
        if self.gemini_model:
            try:
                prompt = "Describe this image in one detailed sentence."
                response = self.gemini_model.generate_content([prompt, image])
                metadata.caption = response.text
            except Exception as e:
                logger.warning(f"Caption generation failed: {e}")
        
        # 3. 자동 태그 생성 (분류기 사용)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(image_tensor)
            probs, indices = outputs.topk(3)
            
            # 카테고리 이름 (예시)
            categories = ['nature', 'people', 'animals', 'food', 'buildings',
                         'vehicles', 'sports', 'art', 'technology', 'other']
            
            tags = []
            for idx, prob in zip(indices[0], probs[0]):
                if prob > 0.1:  # 임계값
                    tags.append(categories[idx])
            
            metadata.tags = tags
        
        # 데이터베이스에 추가
        self.photos[image_path] = metadata
        self.photo_paths.append(image_path)
        
        # 임베딩 업데이트
        if self.embeddings is None:
            self.embeddings = metadata.embedding
        else:
            self.embeddings = np.vstack([self.embeddings, metadata.embedding])
        
        return metadata
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Tuple[str, float, PhotoMetadata]]:
        """
        텍스트로 사진 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            (이미지 경로, 유사도 점수, 메타데이터) 튜플 리스트
        """
        if not self.photos:
            return []
        
        # 쿼리 인코딩
        text_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # 유사도 계산
        text_features_np = text_features.cpu().numpy()
        similarities = np.dot(self.embeddings, text_features_np.T).squeeze()
        
        # Top-K 선택
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            path = self.photo_paths[idx]
            score = similarities[idx]
            metadata = self.photos[path]
            results.append((path, score, metadata))
        
        return results
    
    def search_by_image(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float, PhotoMetadata]]:
        """
        이미지로 유사한 사진 검색
        
        Args:
            query_image_path: 쿼리 이미지 경로
            top_k: 반환할 결과 수
            
        Returns:
            (이미지 경로, 유사도 점수, 메타데이터) 튜플 리스트
        """
        if not self.photos:
            return []
        
        # 쿼리 이미지 인코딩
        image = Image.open(query_image_path).convert('RGB')
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}
        
        with torch.no_grad():
            query_features = self.clip_model.get_image_features(**clip_inputs)
            query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
        
        # 유사도 계산
        query_features_np = query_features.cpu().numpy()
        similarities = np.dot(self.embeddings, query_features_np.T).squeeze()
        
        # 자기 자신 제외
        if query_image_path in self.photo_paths:
            self_idx = self.photo_paths.index(query_image_path)
            similarities[self_idx] = -1
        
        # Top-K 선택
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > -1:
                path = self.photo_paths[idx]
                score = similarities[idx]
                metadata = self.photos[path]
                results.append((path, score, metadata))
        
        return results
    
    def advanced_search(
        self,
        include_terms: List[str],
        exclude_terms: Optional[List[str]] = None,
        tags_filter: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float, PhotoMetadata]]:
        """
        고급 검색: 다중 조건
        
        Args:
            include_terms: 포함해야 할 검색어
            exclude_terms: 제외할 검색어
            tags_filter: 태그 필터
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과
        """
        if not self.photos:
            return []
        
        scores = np.zeros(len(self.photo_paths))
        
        # Include terms 처리
        for term in include_terms:
            text_inputs = self.clip_processor(text=[term], return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            text_features_np = text_features.cpu().numpy()
            similarities = np.dot(self.embeddings, text_features_np.T).squeeze()
            scores += similarities / len(include_terms)
        
        # Exclude terms 처리
        if exclude_terms:
            for term in exclude_terms:
                text_inputs = self.clip_processor(text=[term], return_tensors="pt", padding=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                text_features_np = text_features.cpu().numpy()
                similarities = np.dot(self.embeddings, text_features_np.T).squeeze()
                scores -= similarities * 0.5 / len(exclude_terms)
        
        # Tag 필터링
        if tags_filter:
            for idx, path in enumerate(self.photo_paths):
                metadata = self.photos[path]
                if metadata.tags:
                    if not any(tag in tags_filter for tag in metadata.tags):
                        scores[idx] = -float('inf')
        
        # Top-K 선택
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > -float('inf'):
                path = self.photo_paths[idx]
                score = scores[idx]
                metadata = self.photos[path]
                results.append((path, score, metadata))
        
        return results


def create_gradio_app():
    """Gradio 인터페이스 생성"""
    album = SmartPhotoAlbum()
    
    def upload_photos(files):
        """사진 업로드 처리"""
        if not files:
            return "No files uploaded"
        
        results = []
        for file in files:
            try:
                metadata = album.add_photo(file.name)
                results.append(f"✓ {metadata.filename}")
                if metadata.caption:
                    results.append(f"  Caption: {metadata.caption[:100]}...")
                if metadata.tags:
                    results.append(f"  Tags: {', '.join(metadata.tags)}")
            except Exception as e:
                results.append(f"✗ Error processing {file.name}: {e}")
        
        return "\n".join(results)
    
    def text_search(query, num_results):
        """텍스트 검색"""
        results = album.search_by_text(query, int(num_results))
        
        if not results:
            return None, "No results found"
        
        # 첫 번째 결과 이미지
        top_image = Image.open(results[0][0])
        
        # 결과 정보
        info = []
        for path, score, metadata in results:
            info.append(f"📷 {metadata.filename}")
            info.append(f"   Score: {score:.3f}")
            if metadata.caption:
                info.append(f"   Caption: {metadata.caption[:100]}...")
            if metadata.tags:
                info.append(f"   Tags: {', '.join(metadata.tags)}")
            info.append("")
        
        return top_image, "\n".join(info)
    
    def image_search(query_image, num_results):
        """이미지 검색"""
        if query_image is None:
            return None, "Please upload an image"
        
        # 임시 저장
        temp_path = "temp_query.jpg"
        query_image.save(temp_path)
        
        results = album.search_by_image(temp_path, int(num_results))
        
        if not results:
            return None, "No similar images found"
        
        # 첫 번째 결과 이미지
        top_image = Image.open(results[0][0])
        
        # 결과 정보
        info = []
        for path, score, metadata in results:
            info.append(f"📷 {metadata.filename}")
            info.append(f"   Similarity: {score:.3f}")
            if metadata.caption:
                info.append(f"   Caption: {metadata.caption[:100]}...")
            info.append("")
        
        return top_image, "\n".join(info)
    
    def advanced_search_fn(include_terms, exclude_terms, tag_filter, num_results):
        """고급 검색"""
        include_list = [t.strip() for t in include_terms.split(',') if t.strip()]
        exclude_list = [t.strip() for t in exclude_terms.split(',') if t.strip()] if exclude_terms else None
        tag_list = [t.strip() for t in tag_filter.split(',') if t.strip()] if tag_filter else None
        
        results = album.advanced_search(
            include_list,
            exclude_list,
            tag_list,
            int(num_results)
        )
        
        if not results:
            return None, "No results found"
        
        # 첫 번째 결과 이미지
        top_image = Image.open(results[0][0])
        
        # 결과 정보
        info = []
        for path, score, metadata in results:
            info.append(f"📷 {metadata.filename}")
            info.append(f"   Score: {score:.3f}")
            if metadata.tags:
                info.append(f"   Tags: {', '.join(metadata.tags)}")
            info.append("")
        
        return top_image, "\n".join(info)
    
    # Gradio 인터페이스
    with gr.Blocks(title="Smart Photo Album") as app:
        gr.Markdown("""
        # 🖼️ Smart Photo Album
        ### Transfer Learning + CLIP + Multimodal API를 활용한 지능형 사진첩
        """)
        
        with gr.Tab("📤 Upload Photos"):
            file_upload = gr.File(label="Select photos", file_count="multiple", file_types=["image"])
            upload_btn = gr.Button("Upload and Process", variant="primary")
            upload_output = gr.Textbox(label="Processing Results", lines=10)
            
            upload_btn.click(upload_photos, inputs=[file_upload], outputs=[upload_output])
        
        with gr.Tab("🔍 Text Search"):
            gr.Markdown("자연어로 사진을 검색합니다")
            text_query = gr.Textbox(label="Search Query", placeholder="e.g., 'sunset at beach', 'happy people'")
            num_results_text = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Results")
            search_text_btn = gr.Button("Search", variant="primary")
            
            with gr.Row():
                text_result_image = gr.Image(label="Top Result", type="pil")
                text_result_info = gr.Textbox(label="Search Results", lines=15)
            
            search_text_btn.click(
                text_search,
                inputs=[text_query, num_results_text],
                outputs=[text_result_image, text_result_info]
            )
        
        with gr.Tab("🖼️ Image Search"):
            gr.Markdown("유사한 이미지를 검색합니다")
            query_image = gr.Image(label="Upload Query Image", type="pil")
            num_results_image = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Results")
            search_image_btn = gr.Button("Find Similar", variant="primary")
            
            with gr.Row():
                image_result_image = gr.Image(label="Top Similar", type="pil")
                image_result_info = gr.Textbox(label="Similar Images", lines=15)
            
            search_image_btn.click(
                image_search,
                inputs=[query_image, num_results_image],
                outputs=[image_result_image, image_result_info]
            )
        
        with gr.Tab("⚙️ Advanced Search"):
            gr.Markdown("고급 검색 옵션")
            include_input = gr.Textbox(label="Include Terms (comma-separated)", 
                                      placeholder="sunset, beach, ocean")
            exclude_input = gr.Textbox(label="Exclude Terms (comma-separated)", 
                                      placeholder="people, buildings")
            tag_input = gr.Textbox(label="Tag Filter (comma-separated)", 
                                  placeholder="nature, landscape")
            num_results_adv = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Results")
            search_adv_btn = gr.Button("Advanced Search", variant="primary")
            
            with gr.Row():
                adv_result_image = gr.Image(label="Top Result", type="pil")
                adv_result_info = gr.Textbox(label="Search Results", lines=15)
            
            search_adv_btn.click(
                advanced_search_fn,
                inputs=[include_input, exclude_input, tag_input, num_results_adv],
                outputs=[adv_result_image, adv_result_info]
            )
        
        with gr.Tab("📊 Album Stats"):
            stats_btn = gr.Button("Show Statistics", variant="primary")
            stats_output = gr.Textbox(label="Album Statistics", lines=10)
            
            def show_stats():
                stats = []
                stats.append(f"Total Photos: {len(album.photos)}")
                
                if album.photos:
                    # 태그 통계
                    all_tags = []
                    for metadata in album.photos.values():
                        if metadata.tags:
                            all_tags.extend(metadata.tags)
                    
                    if all_tags:
                        from collections import Counter
                        tag_counts = Counter(all_tags)
                        stats.append("\nTop Tags:")
                        for tag, count in tag_counts.most_common(5):
                            stats.append(f"  • {tag}: {count}")
                    
                    # 캡션 있는 사진 수
                    caption_count = sum(1 for m in album.photos.values() if m.caption)
                    stats.append(f"\nPhotos with Captions: {caption_count}")
                
                return "\n".join(stats)
            
            stats_btn.click(show_stats, outputs=[stats_output])
    
    return app


if __name__ == "__main__":
    # 환경 변수 확인
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY not set. Caption generation will be disabled.")
    
    # 앱 실행
    app = create_gradio_app()
    app.launch(share=True, debug=True)