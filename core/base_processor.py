"""
기본 이미지 처리 클래스
모든 모듈에서 공통으로 사용하는 기능을 제공합니다.
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from typing import Union, Tuple, List, Dict, Any

class BaseImageProcessor:
    """이미지 처리 기본 클래스"""

    def __init__(self):
        self.setup_matplotlib()

    def setup_matplotlib(self):
        """Matplotlib 설정"""
        if os.name == 'nt':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

    def load_image(self, image_path: str) -> Image.Image:
        """이미지 파일 로드"""
        try:
            return Image.open(image_path)
        except Exception as e:
            raise ValueError(f"이미지 로드 실패: {e}")

    def save_image(self, image: Image.Image, save_path: str):
        """이미지 파일 저장"""
        try:
            image.save(save_path)
            print(f"이미지 저장 완료: {save_path}")
        except Exception as e:
            raise ValueError(f"이미지 저장 실패: {e}")

    def pil_to_numpy(self, image: Image.Image) -> np.ndarray:
        """PIL 이미지를 NumPy 배열로 변환"""
        return np.array(image)

    def numpy_to_pil(self, array: np.ndarray) -> Image.Image:
        """NumPy 배열을 PIL 이미지로 변환"""
        # 정규화된 값이면 0-255로 변환
        if array.dtype == np.float32 or array.dtype == np.float64:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
        return Image.fromarray(array.astype(np.uint8))

    def resize_image(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """이미지 크기 조정"""
        return image.resize(size, Image.Resampling.LANCZOS)

    def convert_to_grayscale(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """그레이스케일 변환"""
        if isinstance(image, Image.Image):
            image = self.pil_to_numpy(image)

        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 정규화 (0-1 범위)"""
        return image.astype(np.float32) / 255.0

    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """정규화된 이미지를 원래 범위로 복원"""
        return (image * 255).astype(np.uint8)

    def apply_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """컨볼루션 연산 적용"""
        return cv2.filter2D(image, -1, kernel)

    def get_image_stats(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """이미지 통계 정보 반환"""
        if isinstance(image, Image.Image):
            image = self.pil_to_numpy(image)

        stats = {
            'shape': str(image.shape),  # 튜플을 문자열로 변환
            'dtype': str(image.dtype),
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std())
        }

        if isinstance(image, Image.Image):
            stats['size'] = image.size
            stats['mode'] = image.mode

        return stats

    def create_figure(self, figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """Figure와 Axes 생성"""
        return plt.subplots(figsize=figsize)

    def display_images(self, images: List[np.ndarray], titles: List[str] = None,
                      cols: int = 3, figsize: Tuple[int, int] = (15, 10)):
        """여러 이미지를 그리드로 표시"""
        n_images = len(images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_images > 1 else [axes]

        for idx, (img, ax) in enumerate(zip(images, axes)):
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)

            if titles and idx < len(titles):
                ax.set_title(titles[idx])
            ax.axis('off')

        # 빈 subplot 숨기기
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def calculate_histogram(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """히스토그램 계산"""
        histograms = {}

        if len(image.shape) == 2:
            # 그레이스케일
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            histograms['gray'] = hist.flatten()
        else:
            # 컬러
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms[color] = hist.flatten()

        return histograms

    def plot_histogram(self, image: np.ndarray, ax: plt.Axes = None):
        """히스토그램 플롯"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        histograms = self.calculate_histogram(image)

        for color, hist in histograms.items():
            ax.plot(hist, color=color if color != 'gray' else 'black',
                   alpha=0.7, label=color.capitalize())

        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax