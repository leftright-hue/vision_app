"""
SAM (Segment Anything Model) API 활용
Meta의 세그멘테이션 모델 활용법
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union
import cv2
import requests
from io import BytesIO
import gradio as gr


class SAMSegmentation:
    """SAM을 활용한 세그멘테이션"""
    
    def __init__(self, model_type: str = "vit_h"):
        """
        Args:
            model_type: SAM 모델 타입 ('vit_h', 'vit_l', 'vit_b')
        """
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 실제 환경에서는 segment-anything 설치 필요
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
            
            # 모델 체크포인트 경로 (실제 환경에서 다운로드 필요)
            checkpoint_paths = {
                'vit_h': 'sam_vit_h_4b8939.pth',
                'vit_l': 'sam_vit_l_0b3195.pth',
                'vit_b': 'sam_vit_b_01ec64.pth'
            }
            
            # SAM 모델 로드
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_paths[model_type])
            self.sam.to(device=self.device)
            
            # 예측기 초기화
            self.predictor = SamPredictor(self.sam)
            
            # 자동 마스크 생성기
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            
            self.initialized = True
        except ImportError:
            print("Note: segment-anything not installed. Using mock implementation for demo.")
            self.initialized = False
    
    def set_image(self, image: Union[np.ndarray, Image.Image]):
        """
        세그멘테이션할 이미지 설정
        
        Args:
            image: 입력 이미지
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        self.current_image = image
        
        if self.initialized:
            self.predictor.set_image(image)
    
    def segment_with_points(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = False
    ) -> Dict:
        """
        포인트 프롬프트로 세그멘테이션
        
        Args:
            point_coords: 포인트 좌표 [[x, y], ...]
            point_labels: 포인트 라벨 (1: 전경, 0: 배경)
            multimask_output: 여러 마스크 출력 여부
        
        Returns:
            세그멘테이션 결과
        """
        if not self.initialized:
            # 데모용 더미 결과
            h, w = self.current_image.shape[:2]
            return {
                'masks': [np.random.rand(h, w) > 0.5],
                'scores': [0.95],
                'logits': [np.random.randn(h, w)]
            }
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        
        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }
    
    def segment_with_box(
        self,
        box: np.ndarray,
        multimask_output: bool = False
    ) -> Dict:
        """
        박스 프롬프트로 세그멘테이션
        
        Args:
            box: 바운딩 박스 [x1, y1, x2, y2]
            multimask_output: 여러 마스크 출력 여부
        
        Returns:
            세그멘테이션 결과
        """
        if not self.initialized:
            # 데모용 더미 결과
            h, w = self.current_image.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = box.astype(int)
            mask[y1:y2, x1:x2] = True
            return {
                'masks': [mask],
                'scores': [0.98],
                'logits': [np.random.randn(h, w)]
            }
        
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=multimask_output
        )
        
        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }
    
    def segment_everything(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        """
        전체 이미지 자동 세그멘테이션
        
        Args:
            image: 입력 이미지
        
        Returns:
            모든 세그먼트 리스트
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if not self.initialized:
            # 데모용 더미 결과
            h, w = image.shape[:2]
            segments = []
            for i in range(5):  # 5개의 더미 세그먼트
                mask = np.zeros((h, w), dtype=bool)
                # 랜덤한 원형 영역 생성
                cx, cy = np.random.randint(0, w), np.random.randint(0, h)
                radius = np.random.randint(20, 50)
                y, x = np.ogrid[:h, :w]
                mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
                
                segments.append({
                    'segmentation': mask,
                    'area': mask.sum(),
                    'bbox': [cx-radius, cy-radius, radius*2, radius*2],
                    'predicted_iou': np.random.rand(),
                    'stability_score': np.random.rand()
                })
            return segments
        
        masks = self.mask_generator.generate(image)
        return masks
    
    def visualize_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        score: float = None,
        point_coords: np.ndarray = None,
        box: np.ndarray = None
    ) -> np.ndarray:
        """
        마스크 시각화
        
        Args:
            image: 원본 이미지
            mask: 세그멘테이션 마스크
            score: 신뢰도 점수
            point_coords: 포인트 좌표
            box: 바운딩 박스
        
        Returns:
            시각화된 이미지
        """
        # 마스크 오버레이
        masked_image = image.copy()
        mask_color = np.array([30, 144, 255])  # 파란색
        masked_image[mask] = masked_image[mask] * 0.5 + mask_color * 0.5
        
        # 포인트 표시
        if point_coords is not None:
            for point in point_coords:
                cv2.circle(masked_image, tuple(point.astype(int)), 5, (255, 0, 0), -1)
        
        # 박스 표시
        if box is not None:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 점수 표시
        if score is not None:
            cv2.putText(masked_image, f"Score: {score:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return masked_image.astype(np.uint8)


class InteractiveSAMDemo:
    """인터랙티브 SAM 데모"""
    
    def __init__(self):
        self.sam = SAMSegmentation()
        self.current_image = None
        self.points = []
        self.labels = []
        
    def reset(self):
        """상태 초기화"""
        self.points = []
        self.labels = []
    
    def add_point(self, image, x, y, is_positive):
        """포인트 추가"""
        if image is None:
            return None
        
        if self.current_image is None or not np.array_equal(self.current_image, image):
            self.current_image = image
            self.sam.set_image(image)
            self.reset()
        
        # 포인트 추가
        self.points.append([x, y])
        self.labels.append(1 if is_positive else 0)
        
        # 세그멘테이션 수행
        if len(self.points) > 0:
            point_coords = np.array(self.points)
            point_labels = np.array(self.labels)
            
            results = self.sam.segment_with_points(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
            
            # 최고 점수 마스크 선택
            best_mask = results['masks'][0]
            best_score = results['scores'][0]
            
            # 시각화
            vis_image = self.sam.visualize_mask(
                image,
                best_mask,
                best_score,
                point_coords
            )
            
            return vis_image
        
        return image
    
    def segment_with_box(self, image, x1, y1, x2, y2):
        """박스로 세그멘테이션"""
        if image is None:
            return None
        
        self.sam.set_image(image)
        
        # 박스 세그멘테이션
        box = np.array([x1, y1, x2, y2])
        results = self.sam.segment_with_box(box, multimask_output=False)
        
        # 시각화
        best_mask = results['masks'][0]
        best_score = results['scores'][0]
        
        vis_image = self.sam.visualize_mask(
            image,
            best_mask,
            best_score,
            box=box
        )
        
        return vis_image
    
    def segment_everything(self, image):
        """전체 세그멘테이션"""
        if image is None:
            return None
        
        # 자동 세그멘테이션
        masks = self.sam.segment_everything(image)
        
        # 모든 마스크 시각화
        vis_image = image.copy()
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            color = np.random.randint(0, 255, size=3)
            vis_image[mask] = vis_image[mask] * 0.5 + color * 0.5
        
        return vis_image.astype(np.uint8)


def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    demo = InteractiveSAMDemo()
    
    def process_point_prompt(image, x, y, is_positive):
        """포인트 프롬프트 처리"""
        return demo.add_point(image, int(x), int(y), is_positive)
    
    def process_box_prompt(image, x1, y1, x2, y2):
        """박스 프롬프트 처리"""
        return demo.segment_with_box(image, x1, y1, x2, y2)
    
    def process_auto_segment(image):
        """자동 세그멘테이션 처리"""
        return demo.segment_everything(image)
    
    def reset_points():
        """포인트 리셋"""
        demo.reset()
        return None
    
    with gr.Blocks(title="SAM Segmentation Demo") as app:
        gr.Markdown("# 🎯 SAM (Segment Anything Model) Demo")
        
        with gr.Tab("Point Prompts"):
            with gr.Row():
                with gr.Column():
                    point_image = gr.Image(label="Input Image", type="numpy")
                    with gr.Row():
                        x_coord = gr.Number(label="X Coordinate", value=100)
                        y_coord = gr.Number(label="Y Coordinate", value=100)
                    is_positive = gr.Checkbox(label="Positive Point (Include)", value=True)
                    with gr.Row():
                        add_point_btn = gr.Button("Add Point")
                        reset_btn = gr.Button("Reset Points")
                
                point_output = gr.Image(label="Segmentation Result")
            
            add_point_btn.click(
                process_point_prompt,
                inputs=[point_image, x_coord, y_coord, is_positive],
                outputs=point_output
            )
            reset_btn.click(reset_points, outputs=point_output)
        
        with gr.Tab("Box Prompt"):
            with gr.Row():
                with gr.Column():
                    box_image = gr.Image(label="Input Image", type="numpy")
                    with gr.Row():
                        x1 = gr.Number(label="X1", value=50)
                        y1 = gr.Number(label="Y1", value=50)
                    with gr.Row():
                        x2 = gr.Number(label="X2", value=200)
                        y2 = gr.Number(label="Y2", value=200)
                    box_segment_btn = gr.Button("Segment with Box")
                
                box_output = gr.Image(label="Segmentation Result")
            
            box_segment_btn.click(
                process_box_prompt,
                inputs=[box_image, x1, y1, x2, y2],
                outputs=box_output
            )
        
        with gr.Tab("Automatic Segmentation"):
            with gr.Row():
                with gr.Column():
                    auto_image = gr.Image(label="Input Image", type="numpy")
                    auto_segment_btn = gr.Button("Segment Everything")
                
                auto_output = gr.Image(label="All Segments")
            
            auto_segment_btn.click(
                process_auto_segment,
                inputs=auto_image,
                outputs=auto_output
            )
        
        gr.Markdown("""
        ## About SAM
        
        SAM (Segment Anything Model)은 Meta에서 개발한 범용 세그멘테이션 모델입니다.
        
        ### Features
        - **Zero-shot Segmentation**: 추가 학습 없이 다양한 객체 세그멘테이션
        - **Prompt Engineering**: 포인트, 박스, 텍스트 등 다양한 프롬프트 지원
        - **High Quality**: 고품질 마스크 생성
        
        ### Usage
        1. **Point Prompts**: 클릭으로 객체 선택
        2. **Box Prompt**: 바운딩 박스로 영역 지정
        3. **Automatic**: 전체 이미지 자동 분할
        
        ### Tips
        - Positive points: 포함할 영역 지정
        - Negative points: 제외할 영역 지정
        - Multiple points: 더 정확한 세그멘테이션
        """)
    
    return app


# 실제 SAM API 사용 예제 (Hugging Face Inference API)
class SAMHuggingFaceAPI:
    """Hugging Face Inference API를 통한 SAM 사용"""
    
    def __init__(self, api_key: str = None):
        self.api_url = "https://api-inference.huggingface.co/models/facebook/sam-vit-huge"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def segment_image(self, image_path: str) -> Dict:
        """
        이미지 세그멘테이션 (HF API)
        
        Args:
            image_path: 이미지 파일 경로
        
        Returns:
            세그멘테이션 결과
        """
        with open(image_path, "rb") as f:
            data = f.read()
        
        response = requests.post(self.api_url, headers=self.headers, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API request failed with status {response.status_code}"}


if __name__ == "__main__":
    print("SAM Segmentation Module")
    print("-" * 50)
    
    # SAM 초기화
    sam = SAMSegmentation()
    
    # 테스트 이미지 생성
    test_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (50, 50), (150, 150), (0, 0, 255), -1)
    cv2.circle(test_image, (180, 180), 30, (0, 255, 0), -1)
    
    # 이미지 설정
    sam.set_image(test_image)
    
    # 포인트 프롬프트 테스트
    points = np.array([[100, 100], [180, 180]])
    labels = np.array([1, 1])  # 모두 전경
    
    results = sam.segment_with_points(points, labels)
    print(f"Point segmentation - Masks shape: {results['masks'][0].shape}")
    print(f"Point segmentation - Score: {results['scores'][0]:.3f}")
    
    # 박스 프롬프트 테스트
    box = np.array([50, 50, 150, 150])
    results = sam.segment_with_box(box)
    print(f"\nBox segmentation - Masks shape: {results['masks'][0].shape}")
    print(f"Box segmentation - Score: {results['scores'][0]:.3f}")
    
    # 자동 세그멘테이션 테스트
    segments = sam.segment_everything(test_image)
    print(f"\nAutomatic segmentation - Found {len(segments)} segments")
    
    # Gradio 앱 실행 (실제 환경에서)
    # app = create_gradio_interface()
    # app.launch()
    
    print("\nSAM segmentation setup complete!")