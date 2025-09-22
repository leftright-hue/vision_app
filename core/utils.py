"""
유틸리티 함수들
공통으로 사용되는 헬퍼 함수들을 제공합니다.
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import Union, List, Tuple, Optional
import cv2

class ImageUtils:
    """이미지 관련 유틸리티"""

    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """PIL 이미지를 base64 문자열로 변환"""
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    def base64_to_image(base64_string: str) -> Image.Image:
        """base64 문자열을 PIL 이미지로 변환"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    @staticmethod
    def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (128, 128)) -> Image.Image:
        """썸네일 생성"""
        img_copy = image.copy()
        img_copy.thumbnail(size, Image.Resampling.LANCZOS)
        return img_copy

    @staticmethod
    def blend_images(image1: Image.Image, image2: Image.Image, alpha: float = 0.5) -> Image.Image:
        """두 이미지 블렌딩"""
        return Image.blend(image1, image2, alpha)

    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """밝기 조정"""
        result = image.astype(np.float32) * factor
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """대비 조정"""
        mean = image.mean()
        result = (image.astype(np.float32) - mean) * factor + mean
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
        """감마 보정"""
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                         for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    @staticmethod
    def rotate_image(image: Image.Image, angle: float, expand: bool = True) -> Image.Image:
        """이미지 회전"""
        return image.rotate(angle, expand=expand)

    @staticmethod
    def flip_image(image: Image.Image, mode: str = 'horizontal') -> Image.Image:
        """이미지 뒤집기"""
        if mode == 'horizontal':
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif mode == 'vertical':
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def crop_center(image: Image.Image, crop_size: Tuple[int, int]) -> Image.Image:
        """중앙 크롭"""
        width, height = image.size
        crop_width, crop_height = crop_size

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        return image.crop((left, top, right, bottom))

    @staticmethod
    def pad_image(image: np.ndarray, padding: int, value: int = 0) -> np.ndarray:
        """이미지 패딩"""
        if len(image.shape) == 2:
            return np.pad(image, padding, constant_values=value)
        else:
            return np.pad(image, ((padding, padding), (padding, padding), (0, 0)),
                        constant_values=value)

    @staticmethod
    def get_image_channels(image: Union[Image.Image, np.ndarray]) -> int:
        """이미지 채널 수 반환"""
        if isinstance(image, Image.Image):
            if image.mode == 'L':
                return 1
            elif image.mode == 'RGB':
                return 3
            elif image.mode == 'RGBA':
                return 4
            else:
                return len(image.mode)
        else:
            return 1 if len(image.shape) == 2 else image.shape[2]

    @staticmethod
    def combine_images_grid(images: List[Image.Image], cols: int = 3,
                          spacing: int = 10) -> Image.Image:
        """여러 이미지를 그리드로 합치기"""
        if not images:
            raise ValueError("No images provided")

        n_images = len(images)
        rows = (n_images + cols - 1) // cols

        # 모든 이미지의 최대 크기 찾기
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        # 캔버스 크기 계산
        canvas_width = cols * max_width + (cols - 1) * spacing
        canvas_height = rows * max_height + (rows - 1) * spacing

        # 캔버스 생성
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        # 이미지 배치
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * (max_width + spacing)
            y = row * (max_height + spacing)

            # 중앙 정렬
            x_offset = (max_width - img.width) // 2
            y_offset = (max_height - img.height) // 2

            canvas.paste(img, (x + x_offset, y + y_offset))

        return canvas

class FilterUtils:
    """필터 관련 유틸리티"""

    @staticmethod
    def get_standard_filters() -> dict:
        """표준 필터 세트 반환"""
        return {
            'identity': np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]),
            'blur': np.ones((3, 3)) / 9,
            'gaussian_3x3': np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]]) / 16,
            'sharpen': np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]]),
            'edge_detection': np.array([[-1, -1, -1],
                                       [-1, 8, -1],
                                       [-1, -1, -1]]),
            'sobel_x': np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]),
            'sobel_y': np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]]),
            'laplacian': np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]]),
            'emboss': np.array([[-2, -1, 0],
                               [-1, 1, 1],
                               [0, 1, 2]])
        }

    @staticmethod
    def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """가우시안 커널 생성"""
        kernel = np.zeros((size, size))
        center = size // 2

        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

        return kernel / kernel.sum()

    @staticmethod
    def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """미디언 필터 적용"""
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def apply_bilateral_filter(image: np.ndarray, d: int = 9,
                             sigma_color: float = 75,
                             sigma_space: float = 75) -> np.ndarray:
        """양방향 필터 적용"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)