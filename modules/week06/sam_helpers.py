"""
SAM (Segment Anything Model) 헬퍼 클래스
3-tier fallback 전략: HuggingFace Transformers → Official segment-anything → Simulation
"""

import numpy as np
from PIL import Image
import streamlit as st
from typing import Optional, List, Dict, Tuple, Any
import warnings

class SAMHelper:
    """
    SAM 모델 래퍼 클래스
    - 1순위: HuggingFace Transformers (transformers 패키지)
    - 2순위: Official segment-anything (segment-anything 패키지)
    - 3순위: Simulation mode (기본 이미지 처리)
    """

    def __init__(self, model_type: str = "vit_b"):
        """
        Args:
            model_type: 'vit_b', 'vit_l', 'vit_h' 중 선택
                - vit_b: ~375MB (기본, 빠름)
                - vit_l: ~1.2GB (균형)
                - vit_h: ~2.4GB (최고 성능)
        """
        self.model_type = model_type
        self.mode = None  # 'huggingface', 'official', 'simulation'
        self.model = None
        self.processor = None
        self.predictor = None
        self.device = None

        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화 with 3-tier fallback"""

        # Try 1: HuggingFace Transformers
        if self._try_huggingface():
            self.mode = 'huggingface'
            st.success("✅ SAM 로드 성공 (HuggingFace Transformers)")
            return

        # Try 2: Official segment-anything
        if self._try_official():
            self.mode = 'official'
            st.success("✅ SAM 로드 성공 (Official segment-anything)")
            return

        # Fallback: Simulation mode
        self.mode = 'simulation'
        st.warning("⚠️ SAM 시뮬레이션 모드 (실제 모델 미사용)\n\n"
                  "실제 SAM을 사용하려면:\n"
                  "```bash\n"
                  "pip install transformers torch\n"
                  "```")

    def _try_huggingface(self) -> bool:
        """HuggingFace Transformers로 SAM 로드 시도"""
        try:
            from transformers import SamModel, SamProcessor
            import torch

            # 모델명 매핑
            model_names = {
                'vit_b': 'facebook/sam-vit-base',
                'vit_l': 'facebook/sam-vit-large',
                'vit_h': 'facebook/sam-vit-huge'
            }
            model_name = model_names.get(self.model_type, 'facebook/sam-vit-base')

            # GPU 체크
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                st.info("ℹ️ GPU를 사용할 수 없어 CPU로 실행합니다. (느릴 수 있음)")

            # 모델 로드 with progress
            with st.spinner(f"SAM 모델 로딩 중... ({model_name})"):
                self.processor = SamProcessor.from_pretrained(model_name)
                self.model = SamModel.from_pretrained(model_name).to(self.device)
                self.model.eval()

            return True

        except ImportError:
            return False
        except Exception as e:
            st.error(f"HuggingFace SAM 로드 실패: {e}")
            return False

    def _try_official(self) -> bool:
        """Official segment-anything 패키지로 로드 시도"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import torch

            # 체크포인트 경로 (사용자가 다운로드해야 함)
            checkpoint_paths = {
                'vit_b': 'sam_vit_b_01ec64.pth',
                'vit_l': 'sam_vit_l_0b3195.pth',
                'vit_h': 'sam_vit_h_4b8939.pth'
            }
            checkpoint = checkpoint_paths.get(self.model_type, 'sam_vit_b_01ec64.pth')

            # GPU 체크
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # 모델 로드
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)

            return True

        except ImportError:
            return False
        except FileNotFoundError:
            st.warning("⚠️ SAM 체크포인트 파일을 찾을 수 없습니다.")
            return False
        except Exception as e:
            return False

    def preprocess_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """메모리 최적화를 위한 이미지 전처리"""
        # 이미지 크기 제한
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            st.info(f"ℹ️ 이미지를 {new_size}로 리사이징했습니다.")

        return image

    def segment_with_points(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        labels: List[int]
    ) -> np.ndarray:
        """
        포인트 프롬프트로 세그멘테이션 수행

        Args:
            image: PIL Image
            points: [(x1, y1), (x2, y2), ...] 좌표 리스트
            labels: [1, 0, ...] 1=foreground, 0=background

        Returns:
            mask: (H, W) bool array
        """
        image = self.preprocess_image(image)

        if self.mode == 'huggingface':
            return self._segment_huggingface(image, points, labels)
        elif self.mode == 'official':
            return self._segment_official(image, points, labels)
        else:
            return self._segment_simulation(image, points, labels)

    def segment_with_box(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        박스 프롬프트로 세그멘테이션 수행

        Args:
            image: PIL Image
            box: (x1, y1, x2, y2) 박스 좌표

        Returns:
            mask: (H, W) bool array
        """
        image = self.preprocess_image(image)

        if self.mode == 'huggingface':
            return self._segment_huggingface_box(image, box)
        elif self.mode == 'official':
            return self._segment_official_box(image, box)
        else:
            return self._segment_simulation_box(image, box)

    def generate_auto_masks(
        self,
        image: Image.Image,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        자동 마스크 생성 (전체 이미지 세그멘테이션)

        Returns:
            List of dicts with 'segmentation', 'area', 'bbox', 'predicted_iou'
        """
        image = self.preprocess_image(image)

        if self.mode == 'huggingface':
            return self._auto_masks_huggingface(image, points_per_side)
        elif self.mode == 'official':
            return self._auto_masks_official(image)
        else:
            return self._auto_masks_simulation(image)

    # ==================== HuggingFace Implementation ====================

    def _segment_huggingface(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        labels: List[int]
    ) -> np.ndarray:
        """HuggingFace Transformers를 사용한 세그멘테이션"""
        import torch

        # Prepare inputs
        input_points = [points]  # batch dimension
        input_labels = [labels]

        inputs = self.processor(
            image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        # 가장 높은 score의 마스크 선택
        mask = masks[0, 0].numpy()  # (H, W)
        return mask > 0

    def _segment_huggingface_box(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """HuggingFace: 박스 프롬프트"""
        import torch

        input_boxes = [[box]]  # batch dimension

        inputs = self.processor(
            image,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        mask = masks[0, 0].numpy()
        return mask > 0

    def _auto_masks_huggingface(
        self,
        image: Image.Image,
        points_per_side: int
    ) -> List[Dict[str, Any]]:
        """HuggingFace: 자동 마스크 생성 (그리드 기반 샘플링)"""
        import torch

        w, h = image.size

        # 그리드 포인트 생성
        step = max(w, h) // points_per_side
        grid_x = np.arange(step // 2, w, step)
        grid_y = np.arange(step // 2, h, step)
        points_grid = [(x, y) for y in grid_y for x in grid_x]

        masks_data = []

        # 각 포인트에 대해 세그멘테이션 수행 (배치 처리)
        batch_size = 16
        for i in range(0, len(points_grid), batch_size):
            batch_points = points_grid[i:i+batch_size]
            batch_labels = [1] * len(batch_points)

            inputs = self.processor(
                [image] * len(batch_points),
                input_points=[[p] for p in batch_points],
                input_labels=[[l] for l in batch_labels],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )

            # 결과 수집
            for j, mask_set in enumerate(masks):
                mask = mask_set[0, 0].numpy() > 0
                area = int(mask.sum())

                if area > 100:  # 최소 영역 필터
                    y_indices, x_indices = np.where(mask)
                    bbox = (
                        int(x_indices.min()),
                        int(y_indices.min()),
                        int(x_indices.max()),
                        int(y_indices.max())
                    )

                    masks_data.append({
                        'segmentation': mask,
                        'area': area,
                        'bbox': bbox,
                        'predicted_iou': 0.9  # HF doesn't provide IoU
                    })

        # 영역 크기로 정렬
        masks_data.sort(key=lambda x: x['area'], reverse=True)
        return masks_data

    # ==================== Official Implementation ====================

    def _segment_official(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        labels: List[int]
    ) -> np.ndarray:
        """Official segment-anything을 사용한 세그멘테이션"""
        image_np = np.array(image)

        self.predictor.set_image(image_np)

        point_coords = np.array(points)
        point_labels = np.array(labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # 가장 높은 score의 마스크 선택
        best_mask = masks[scores.argmax()]
        return best_mask

    def _segment_official_box(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Official: 박스 프롬프트"""
        image_np = np.array(image)
        self.predictor.set_image(image_np)

        box_np = np.array(box)
        masks, scores, logits = self.predictor.predict(
            box=box_np,
            multimask_output=True
        )

        best_mask = masks[scores.argmax()]
        return best_mask

    def _auto_masks_official(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Official: 자동 마스크 생성"""
        try:
            from segment_anything import SamAutomaticMaskGenerator

            mask_generator = SamAutomaticMaskGenerator(self.predictor.model)
            image_np = np.array(image)
            masks = mask_generator.generate(image_np)

            return masks
        except Exception as e:
            st.error(f"자동 마스크 생성 실패: {e}")
            return []

    # ==================== Simulation Mode ====================

    def _segment_simulation(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        labels: List[int]
    ) -> np.ndarray:
        """시뮬레이션 모드: 간단한 색상 기반 세그멘테이션"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        # foreground 포인트 주변을 마스킹
        for (x, y), label in zip(points, labels):
            if label == 1:  # foreground
                radius = min(h, w) // 8
                y_min, y_max = max(0, y - radius), min(h, y + radius)
                x_min, x_max = max(0, x - radius), min(w, x + radius)

                # 원형 마스크
                yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                circle = (xx - x)**2 + (yy - y)**2 <= radius**2
                mask[y_min:y_max, x_min:x_max] |= circle

        return mask

    def _segment_simulation_box(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """시뮬레이션: 박스 영역을 마스킹"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = True

        return mask

    def _auto_masks_simulation(self, image: Image.Image) -> List[Dict[str, Any]]:
        """시뮬레이션: 그리드 기반 가짜 마스크"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        masks_data = []

        # 간단한 그리드 분할
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                mask = np.zeros((h, w), dtype=bool)
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                mask[y1:y2, x1:x2] = True

                masks_data.append({
                    'segmentation': mask,
                    'area': int(mask.sum()),
                    'bbox': (x1, y1, x2, y2),
                    'predicted_iou': 0.5
                })

        return masks_data


@st.cache_resource
def get_sam_helper(model_type: str = "vit_b") -> SAMHelper:
    """캐시된 SAM 헬퍼 인스턴스 반환"""
    return SAMHelper(model_type=model_type)
