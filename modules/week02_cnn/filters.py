"""
이미지 필터 모듈
다양한 이미지 필터를 제공합니다.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Union, Optional
from core.utils import FilterUtils

class ImageFilters:
    """이미지 필터 클래스"""

    def __init__(self):
        self.filters = self._initialize_filters()
        self.filter_utils = FilterUtils()

    def _initialize_filters(self) -> dict:
        """필터 초기화"""
        return {
            'None': None,
            'Blur': np.ones((5, 5)) / 25,
            'Gaussian': np.array([
                [1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]
            ]),
            'Sharpen': np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]),
            'Edge Detection': np.array([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]),
            'Emboss': np.array([
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2]
            ]),
            'Sobel X': np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]),
            'Sobel Y': np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]),
            'Laplacian': np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ])
        }

    def apply_filter(self, image: Union[Image.Image, np.ndarray],
                    filter_name: str) -> Union[Image.Image, np.ndarray]:
        """필터 적용"""
        if filter_name == 'None' or filter_name not in self.filters:
            return image

        # PIL to numpy
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            was_pil = True
        else:
            img_array = image
            was_pil = False

        # 그레이스케일 변환 (필요시)
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array

        # 필터 적용
        kernel = self.filters[filter_name]
        if kernel is not None:
            filtered = cv2.filter2D(img_gray, -1, kernel)
        else:
            filtered = img_gray

        # 원래 형식으로 변환
        if was_pil:
            return Image.fromarray(filtered.astype(np.uint8))
        else:
            return filtered

    def apply_custom_filter(self, image: Union[Image.Image, np.ndarray],
                          kernel: np.ndarray) -> Union[Image.Image, np.ndarray]:
        """커스텀 필터 적용"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            was_pil = True
        else:
            img_array = image
            was_pil = False

        filtered = cv2.filter2D(img_array, -1, kernel)

        if was_pil:
            return Image.fromarray(filtered.astype(np.uint8))
        else:
            return filtered

    def apply_gaussian_blur(self, image: Union[Image.Image, np.ndarray],
                          kernel_size: int = 5,
                          sigma: float = 1.0) -> Union[Image.Image, np.ndarray]:
        """가우시안 블러 적용"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            was_pil = True
        else:
            img_array = image
            was_pil = False

        blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)

        if was_pil:
            return Image.fromarray(blurred.astype(np.uint8))
        else:
            return blurred

    def detect_edges_canny(self, image: Union[Image.Image, np.ndarray],
                          low_threshold: int = 50,
                          high_threshold: int = 150) -> np.ndarray:
        """Canny 엣지 검출"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges

    def apply_morphological_operation(self, image: Union[Image.Image, np.ndarray],
                                     operation: str = 'erosion',
                                     kernel_size: int = 3) -> np.ndarray:
        """모폴로지 연산 적용"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation == 'erosion':
            result = cv2.erode(img_array, kernel, iterations=1)
        elif operation == 'dilation':
            result = cv2.dilate(img_array, kernel, iterations=1)
        elif operation == 'opening':
            result = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            result = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        elif operation == 'gradient':
            result = cv2.morphologyEx(img_array, cv2.MORPH_GRADIENT, kernel)
        else:
            result = img_array

        return result

    def get_filter_info(self, filter_name: str) -> dict:
        """필터 정보 반환"""
        descriptions = {
            'Blur': '평균 필터로 노이즈 제거',
            'Gaussian': '가우시안 분포를 사용한 자연스러운 블러',
            'Sharpen': '이미지 선명도 증가',
            'Edge Detection': '모든 방향의 엣지 검출',
            'Emboss': '3D 엠보싱 효과',
            'Sobel X': '수직 엣지 검출',
            'Sobel Y': '수평 엣지 검출',
            'Laplacian': '2차 미분을 이용한 엣지 검출'
        }

        if filter_name in self.filters:
            return {
                'name': filter_name,
                'kernel': self.filters[filter_name],
                'description': descriptions.get(filter_name, ''),
                'kernel_size': self.filters[filter_name].shape if self.filters[filter_name] is not None else None
            }
        return {}