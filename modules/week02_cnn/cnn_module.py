"""
Week 2 CNN 모듈
CNN과 이미지 처리 관련 기능을 제공합니다.
"""

import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt

from core.base_processor import BaseImageProcessor
from core.ai_models import AIModelManager
from .filters import ImageFilters


class InstagramFilterMaker:
    """🎭 나만의 Instagram 필터 제작소 - Streamlit 버전"""
    
    def __init__(self):
        self.current_image = None
        self.original_image = None
    
    def load_image_from_upload(self, uploaded_file):
        """📸 업로드된 사진 불러오기"""
        try:
            # PIL Image로 변환
            pil_image = Image.open(uploaded_file)
            # OpenCV 형식으로 변환 (BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            self.current_image = opencv_image
            self.original_image = opencv_image.copy()
            return True
        except Exception as e:
            st.error(f"이미지 로드 실패: {e}")
            return False
    
    def get_image_info(self):
        """🔍 사진 정보 반환"""
        if self.current_image is None:
            return None
        
        height, width = self.current_image.shape[:2]
        channels = self.current_image.shape[2] if len(self.current_image.shape) == 3 else 1
        
        info = {
            'width': width,
            'height': height,
            'channels': channels,
            'total_pixels': width * height,
            'min_value': self.current_image.min(),
            'max_value': self.current_image.max(),
            'mean_brightness': self.current_image.mean()
        }
        return info
    
    def display_color_spaces(self):
        """🌈 색깔의 다양한 표현 방식 보기"""
        if self.current_image is None:
            return None
        
        # 색상 공간 변환
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
        img_gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # 서브플롯 생성
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('🌈 색깔의 6가지 변신 - 같은 사진, 다른 표현!', fontsize=16)
        
        # 원본 (RGB)
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('🌟 RGB 원본\n(우리가 보는 그대로)')
        axes[0, 0].axis('off')
        
        # HSV
        axes[0, 1].imshow(img_hsv)
        axes[0, 1].set_title('🎨 HSV\n(색깔+진하기+밝기)')
        axes[0, 1].axis('off')
        
        # LAB
        axes[0, 2].imshow(img_lab)
        axes[0, 2].set_title('🔬 LAB\n(과학자 방식)')
        axes[0, 2].axis('off')
        
        # Grayscale
        axes[1, 0].imshow(img_gray, cmap='gray')
        axes[1, 0].set_title('🎬 흑백\n(옛날 영화처럼)')
        axes[1, 0].axis('off')
        
        # RGB 채널 분리
        r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        # 이미지 크기가 다를 수 있으므로 안전하게 처리
        try:
            rgb_combined = np.hstack([r, g, b])
        except ValueError:
            # 크기가 다른 경우 각 채널을 같은 크기로 조정
            min_height = min(r.shape[0], g.shape[0], b.shape[0])
            min_width = min(r.shape[1], g.shape[1], b.shape[1])
            r = r[:min_height, :min_width]
            g = g[:min_height, :min_width]
            b = b[:min_height, :min_width]
            rgb_combined = np.hstack([r, g, b])
        axes[1, 1].imshow(rgb_combined, cmap='gray')
        axes[1, 1].set_title('🔴🟢🔵 RGB 채널\n(빨강|초록|파랑)')
        axes[1, 1].axis('off')
        
        # HSV 채널 분리
        h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        # 이미지 크기가 다를 수 있으므로 안전하게 처리
        try:
            hsv_combined = np.hstack([h, s, v])
        except ValueError:
            # 크기가 다른 경우 각 채널을 같은 크기로 조정
            min_height = min(h.shape[0], s.shape[0], v.shape[0])
            min_width = min(h.shape[1], s.shape[1], v.shape[1])
            h = h[:min_height, :min_width]
            s = s[:min_height, :min_width]
            v = v[:min_height, :min_width]
            hsv_combined = np.hstack([h, s, v])
        axes[1, 2].imshow(hsv_combined, cmap='gray')
        axes[1, 2].set_title('🌈💪☀️ HSV 채널\n(색깔|진하기|밝기)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def apply_convolution_filters(self):
        """🎭 Instagram 필터 6종 세트"""
        if self.current_image is None:
            return None, None
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # 다양한 필터 정의
        filters = {
            '🌟 원본': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            '✨ 선명': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            '💫 몽환': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
            '🔍 경계': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            '📐 세로선': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            '📏 가로선': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        }
        
        # 결과 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('🎭 Instagram 필터 6종 세트 - 어떤 게 가장 멋있나요?', fontsize=16)
        
        axes = axes.flatten()
        results = {}
        
        for i, (name, kernel) in enumerate(filters.items()):
            if '원본' in name:
                result = gray
            else:
                result = cv2.filter2D(gray, -1, kernel)
            
            results[name] = result
            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(f'{name}')
            axes[i].axis('off')
            
            # 각 필터의 특징 설명
            descriptions = {
                '🌟 원본': '원래 모습',
                '✨ 선명': '또렷하게!',
                '💫 몽환': '부드럽게~',
                '🔍 경계': '윤곽선만!',
                '📐 세로선': '세로 강조',
                '📏 가로선': '가로 강조'
            }
            
            if name in descriptions:
                axes[i].text(0.5, -0.1, descriptions[name], 
                           transform=axes[i].transAxes, ha='center', fontsize=10)
        
        plt.tight_layout()
        return fig, results
    
    def edge_detection_comparison(self):
        """🕵️ 탐정게임: 사진 속 경계선 찾기 대결!"""
        if self.current_image is None:
            return None
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거를 위한 블러링
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # 다양한 엣지 검출 방법
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        canny = cv2.Canny(blurred, 50, 150)
        
        # 결과 시각화
        images = [gray, sobel_x, sobel_y, sobel_combined, laplacian, canny]
        titles = ['📷 원본', '📐 세로탐정', '📏 가로탐정', '🎯 합체탐정', '💫 전방향탐정', '🏆 천재탐정']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('🕵️경계선 찾기 탐정 대회 - 누가 가장 잘 찾을까요?', fontsize=16)
        
        axes = axes.flatten()
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig

class CNNModule(BaseImageProcessor):
    """CNN 모듈 클래스"""

    def __init__(self):
        super().__init__()
        self.ai_manager = AIModelManager()
        self.filters = ImageFilters()

    def render(self):
        """Week 2 모듈 렌더링 - 메인 메서드"""
        self.render_ui()

    def render_ui(self):
        """Week 2 모듈 UI 렌더링"""
        st.title("🧠 Week 2: CNN과 디지털 이미지")
        st.markdown("---")

        # 탭 생성
        tabs = st.tabs([
            "📚 이론",
            "🔬 이미지 기초",
            "🎨 필터링",
            "🤖 CNN 시각화",
            "🚀 HuggingFace",
            "📊 통합 분석",
            "🎭 Instagram 필터"
        ])

        with tabs[0]:
            self._render_theory_tab()

        with tabs[1]:
            self._render_image_basics_tab()

        with tabs[2]:
            self._render_filtering_tab()

        with tabs[3]:
            self._render_cnn_visualization_tab()

        with tabs[4]:
            self._render_huggingface_tab()

        with tabs[5]:
            self._render_integrated_analysis_tab()

        with tabs[6]:
            self._render_instagram_filter_tab()

    def _render_theory_tab(self):
        """이론 탭"""
        st.header("📖 CNN 이론")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. 디지털 이미지의 구조")
            st.markdown("""
            - **픽셀**: 이미지의 기본 단위
            - **채널**: RGB, 그레이스케일
            - **해상도**: 가로 × 세로 픽셀 수
            - **비트 깊이**: 색상 표현 능력
            """)

            st.subheader("2. Convolution 연산")
            st.markdown("""
            - **커널/필터**: 특징 추출 도구
            - **스트라이드**: 커널 이동 간격
            - **패딩**: 경계 처리 방법
            - **특징 맵**: Convolution 결과
            """)

        with col2:
            st.subheader("3. CNN 구조")
            st.markdown("""
            - **Convolutional Layer**: 특징 추출
            - **Activation (ReLU)**: 비선형성
            - **Pooling Layer**: 크기 축소
            - **Fully Connected**: 최종 분류
            """)

            st.subheader("4. 주요 아키텍처")
            st.markdown("""
            - **LeNet-5** (1998): 최초의 CNN
            - **AlexNet** (2012): GPU 활용
            - **VGGNet** (2014): 깊은 네트워크
            - **ResNet** (2015): Skip Connection
            """)

    def _render_image_basics_tab(self):
        """이미지 기초 탭"""
        st.header("🔬 디지털 이미지 기초")

        # 탭 생성: 예제 학습과 사용자 이미지 분석
        sub_tabs = st.tabs(["📚 예제로 학습하기", "🔍 내 이미지 분석하기"])

        with sub_tabs[0]:
            st.markdown("### 1. 픽셀과 이미지 배열")

            # 픽셀 배열 시각화
            col1, col2 = st.columns(2)

            with col1:
                # 그레이스케일 이미지 예제
                grayscale_example = np.array([
                    [0,   50,  100, 150, 200],
                    [10,  60,  110, 160, 210],
                    [20,  70,  120, 170, 220],
                    [30,  80,  130, 180, 230],
                    [40,  90,  140, 190, 255]
                ], dtype=np.uint8)

                st.image(grayscale_example, caption="5x5 그레이스케일 이미지", width='stretch')
                st.caption("💡 **설명**: 각 픽셀은 0(검정)~255(흰색) 사이의 값을 가집니다.")
                st.code(f"Shape: {grayscale_example.shape}\nDtype: {grayscale_example.dtype}")

            with col2:
                # RGB 컬러 이미지 예제 - 크기를 키워서 잘 보이도록
                color_example = np.array([
                    [[255, 0, 0], [0, 255, 0]],    # 빨강, 초록
                    [[0, 0, 255], [255, 255, 255]]  # 파랑, 흰색
                ], dtype=np.uint8)

                # 이미지를 크게 확대 (2x2 -> 100x100 픽셀로)
                from PIL import Image as PILImage
                color_pil = PILImage.fromarray(color_example, mode='RGB')
                color_pil_resized = color_pil.resize((100, 100), PILImage.Resampling.NEAREST)

                st.image(color_pil_resized, caption="2x2 RGB 컬러 이미지 (확대)", width='stretch')
                st.caption("💡 **설명**: RGB 이미지는 3개 채널(R, G, B)의 조합입니다.")
                st.code(f"원본 Shape: {color_example.shape}\nDtype: {color_example.dtype}")

                # 색상 설명 추가
                st.markdown("""
                **픽셀 색상**:
                - 좌상단: 🔴 빨강 (255, 0, 0)
                - 우상단: 🟢 초록 (0, 255, 0)
                - 좌하단: 🔵 파랑 (0, 0, 255)
                - 우하단: ⚪ 흰색 (255, 255, 255)
                """)

            st.markdown("---")
            st.markdown("### 2. 색상 공간 변환")

            # 색상 그라데이션 샘플 생성
            size = 100
            gradient_img = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(size):
                for j in range(size):
                    gradient_img[i, j, 0] = int(255 * i / size)  # R
                    gradient_img[i, j, 1] = int(255 * j / size)  # G
                    gradient_img[i, j, 2] = int(255 * (1 - i/size))  # B

            # 색상 공간 변환
            hsv_img = cv2.cvtColor(gradient_img, cv2.COLOR_RGB2HSV)
            gray_img = cv2.cvtColor(gradient_img, cv2.COLOR_RGB2GRAY)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(gradient_img, caption="RGB 원본", width='stretch')
                st.caption("💡 3개 채널 (R, G, B)")

            with col2:
                st.image(hsv_img, caption="HSV 색상공간", width='stretch')
                st.caption("💡 색상, 채도, 명도")

            with col3:
                st.image(gray_img, caption="그레이스케일", width='stretch')
                st.caption("💡 단일 채널 (밝기)")

            st.markdown("---")
            st.markdown("### 3. 이미지 처리 기본 연산")

            # 기본 연산 예제
            original = np.full((50, 50), 128, dtype=np.uint8)

            operations = {
                "원본": original,
                "밝기 +50": np.clip(original + 50, 0, 255).astype(np.uint8),
                "밝기 -50": np.clip(original - 50, 0, 255).astype(np.uint8),
                "대비 x1.5": np.clip(original * 1.5, 0, 255).astype(np.uint8),
                "대비 x0.5": np.clip(original * 0.5, 0, 255).astype(np.uint8),
                "감마 2.2": (np.power(original / 255.0, 2.2) * 255).astype(np.uint8)
            }

            cols = st.columns(3)
            for idx, (name, img) in enumerate(operations.items()):
                with cols[idx % 3]:
                    st.image(img, caption=name, width='stretch')
                    st.caption(f"픽셀 값: {img[0, 0]}")

            st.markdown("---")
            st.markdown("### 4. 이미지 메타데이터와 속성")

            # 샘플 이미지 생성 및 메타데이터 표시
            sample_meta_img = gradient_img

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**📊 기본 속성**")
                st.code(f"""
해상도: {sample_meta_img.shape[1]}x{sample_meta_img.shape[0]}
색상 채널: {sample_meta_img.shape[2]}
비트 깊이: 8비트 (0-255)
픽셀 수: {sample_meta_img.shape[0] * sample_meta_img.shape[1]:,}
데이터 타입: {sample_meta_img.dtype}
                """, language="text")

            with col2:
                st.write("**📷 EXIF 정보 (예시)**")
                st.code("""
카메라: Canon EOS R5
렌즈: 24-70mm f/2.8
ISO: 400
셔터 속도: 1/125s
조리개: f/5.6
촬영 일시: 2024-01-15
                """, language="text")

            with col3:
                st.write("**🌍 GPS 정보 (예시)**")
                st.code("""
위도: 37.5665° N
경도: 126.9780° E
고도: 38m
위치: 서울, 대한민국
                """, language="text")

            st.markdown("---")
            st.markdown("### 5. 이미지 압축과 파일 형식")

            # 파일 형식별 비교
            col1, col2 = st.columns(2)

            with col1:
                st.write("**📁 무손실 압축 형식**")

                # PNG 예시
                st.markdown("**PNG (Portable Network Graphics)**")
                st.success("""
                ✅ 투명도(알파 채널) 지원
                ✅ 무손실 압축
                ✅ 웹 그래픽, 로고, 아이콘에 적합
                ❌ 파일 크기가 큼
                """)

                # BMP 예시
                st.markdown("**BMP (Bitmap)**")
                st.warning("""
                ✅ 압축 없음, 원본 품질 유지
                ✅ 간단한 구조
                ❌ 매우 큰 파일 크기
                ❌ 웹에서 잘 지원 안 됨
                """)

            with col2:
                st.write("**📉 손실 압축 형식**")

                # JPEG 예시
                st.markdown("**JPEG (Joint Photographic Experts Group)**")
                st.info("""
                ✅ 높은 압축률 (10:1 ~ 20:1)
                ✅ 사진에 최적화
                ✅ 작은 파일 크기
                ❌ 투명도 미지원
                ❌ 텍스트/선 이미지에 부적합
                """)

                # WebP 예시
                st.markdown("**WebP (Google)**")
                st.success("""
                ✅ JPEG보다 25-35% 작은 크기
                ✅ 무손실/손실 모두 지원
                ✅ 투명도 지원
                ❌ 구형 브라우저 미지원
                """)

            # 파일 크기 비교 시뮬레이션
            st.markdown("#### 📊 파일 크기 비교 (100x100 RGB 이미지)")

            file_sizes = {
                "BMP (무압축)": 30.0,  # KB
                "PNG (무손실)": 12.5,
                "JPEG (품질 90%)": 4.2,
                "JPEG (품질 70%)": 2.1,
                "WebP (무손실)": 8.3,
                "WebP (손실)": 1.8
            }

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            formats = list(file_sizes.keys())
            sizes = list(file_sizes.values())
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9']

            bars = ax.barh(formats, sizes, color=colors)
            ax.set_xlabel('파일 크기 (KB)')
            ax.set_title('이미지 포맷별 파일 크기 비교')

            # 막대 위에 크기 표시
            for bar, size in zip(bars, sizes):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{size} KB', va='center')

            plt.tight_layout()
            st.pyplot(fig)

            st.info("""
            📖 **학습 포인트**:
            - **픽셀**: 이미지의 최소 단위
            - **채널**: 색상 정보를 담는 레이어 (RGB = 3채널)
            - **색상 공간**: RGB, HSV, LAB 등 색상 표현 방식
            - **메타데이터**: EXIF, GPS 등 추가 정보
            - **압축 방식**: 무손실 vs 손실 압축의 트레이드오프
            - **파일 형식 선택**: 용도에 따른 최적 포맷 선택
            """)

        with sub_tabs[1]:
            st.markdown("### 내 이미지로 실습하기")

            uploaded_file = st.file_uploader(
                "이미지 업로드 (선택사항)",
                type=['png', 'jpg', 'jpeg'],
                key="basics_upload"
            )

            if uploaded_file:
                image = Image.open(uploaded_file)
                img_array = np.array(image)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("원본 이미지")
                    st.image(image, width='stretch')

                    # 이미지 정보
                    st.subheader("📊 이미지 정보")
                    stats = self.get_image_stats(image)
                    for key, value in stats.items():
                        st.metric(key, value)

                with col2:
                    st.subheader("채널 분리")

                    if len(img_array.shape) == 3:
                        # RGB 채널 분리
                        channels = ['Red', 'Green', 'Blue']
                        for i, channel in enumerate(channels):
                            st.write(f"**{channel} Channel**")
                            st.image(img_array[:, :, i], width='stretch')
                    else:
                        st.write("**Grayscale Image**")
                        st.image(img_array, width='stretch')
            else:
                st.info("👆 이미지를 업로드하여 직접 분석해보세요!")

    def _render_filtering_tab(self):
        """필터링 탭"""
        st.header("🎨 이미지 필터링")

        # 탭 생성: 예제 학습과 사용자 이미지 분석
        sub_tabs = st.tabs(["📚 필터 개념 학습", "🔍 내 이미지에 필터 적용"])

        with sub_tabs[0]:
            st.markdown("### Convolution 연산과 필터")

            # 샘플 이미지 생성 (체커보드 패턴)
            size = 100
            checkerboard = np.zeros((size, size), dtype=np.uint8)
            for i in range(0, size, 20):
                for j in range(0, size, 20):
                    if (i//20 + j//20) % 2 == 0:
                        checkerboard[i:i+20, j:j+20] = 255

            # 필터별 결과 보여주기
            st.markdown("#### 주요 필터와 효과")

            filter_examples = {
                'Blur': ('평균 필터로 노이즈 제거', np.ones((5, 5)) / 25),
                'Sharpen': ('이미지 선명도 증가', np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
                'Edge Detection': ('경계선 검출', np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
            }

            for filter_name, (description, kernel) in filter_examples.items():
                st.markdown(f"**{filter_name} - {description}**")

                cols = st.columns([1, 1, 2])

                with cols[0]:
                    st.image(checkerboard, caption="원본", width='stretch')

                with cols[1]:
                    filtered = cv2.filter2D(checkerboard, -1, kernel)
                    st.image(filtered, caption=f"{filter_name} 적용", width='stretch')

                with cols[2]:
                    st.code(f"커널:\n{kernel}", language='python')
                    st.caption(f"💡 {description}")

                st.markdown("---")

            st.markdown("### Convolution 연산 과정")

            # Convolution 과정 설명
            st.info("""
            **Convolution 연산 단계**:
            1. **커널 배치**: 이미지 위에 커널을 놓습니다
            2. **원소별 곱셈**: 겹치는 픽셀과 커널 값을 곱합니다
            3. **합산**: 모든 곱셈 결과를 더합니다
            4. **이동**: 커널을 다음 위치로 이동합니다
            5. **반복**: 전체 이미지에 대해 반복합니다
            """)

            # 시각적 예제
            example_img = np.array([
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]
            ], dtype=np.uint8)

            example_kernel = np.array([
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**입력 이미지**")
                st.code(str(example_img))

            with col2:
                st.write("**Sobel X 커널**")
                st.code(str(example_kernel))

            with col3:
                st.write("**결과**")
                result = cv2.filter2D(example_img, -1, example_kernel)
                st.code(str(result))

            st.success("""
            📖 **학습 포인트**:
            - **커널/필터**: 특정 특징을 추출하는 작은 행렬
            - **Convolution**: 커널을 이미지에 적용하는 연산
            - **용도**: 블러, 샤프닝, 엣지 검출, 노이즈 제거 등
            - **CNN 연결**: CNN의 합성곱층은 이 원리를 학습 가능하게 만든 것!
            """)

        with sub_tabs[1]:
            st.markdown("### 내 이미지에 필터 적용하기")

            uploaded_file = st.file_uploader(
                "이미지 업로드 (선택사항)",
                type=['png', 'jpg', 'jpeg'],
                key="filter_upload"
            )

            if uploaded_file:
                image = Image.open(uploaded_file)

                # 필터 선택
                filter_name = st.selectbox(
                    "필터 선택",
                    list(self.filters.filters.keys())
                )

                # 필터 적용
                filtered_image = self.filters.apply_filter(image, filter_name)

                # 결과 표시
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("원본")
                    st.image(image, width='stretch')

                with col2:
                    st.subheader(f"{filter_name} 적용")
                    st.image(filtered_image, width='stretch')

                # 필터 정보 표시
                if filter_name != 'None':
                    with st.expander("필터 상세 정보"):
                        kernel = self.filters.filters[filter_name]
                        st.code(str(kernel), language='python')

                        filter_info = self.filters.get_filter_info(filter_name)
                        if filter_info.get('description'):
                            st.write(f"**설명**: {filter_info['description']}")
                        if filter_info.get('kernel_size'):
                            st.write(f"**커널 크기**: {filter_info['kernel_size']}")
            else:
                # 기본 예제 이미지 제공
                st.info("👆 이미지를 업로드하거나 아래 예제를 사용해보세요!")

                if st.button("예제 이미지 사용"):
                    # 그라데이션 이미지 생성
                    size = 200
                    example_img = np.zeros((size, size, 3), dtype=np.uint8)
                    for i in range(size):
                        for j in range(size):
                            example_img[i, j] = [i * 255 // size, j * 255 // size, 128]

                    st.session_state['example_image'] = Image.fromarray(example_img)
                    st.rerun()

    def _render_instagram_filter_tab(self):
        """🎭 Instagram 필터 제작소 탭"""
        st.header("🎨 Instagram 필터 제작소")
        st.markdown("### 딥러닝 영상처리 - Week 2 실습")
        
        st.markdown("""
        📱 **오늘 우리는 Instagram, Snapchat 같은 필터를 직접 만들어볼 거예요!**
        
        🎯 **학습 목표:**
        - 📷 디지털 사진이 컴퓨터에서 어떻게 표현되는지 체험하기
        - 🎭 Instagram처럼 다양한 필터 효과 직접 만들어보기
        - 🔍 컴퓨터가 사진에서 물체의 경계를 찾는 방법 이해하기
        """)
        
        # Instagram 필터 제작소 인스턴스 생성
        if 'instagram_filter_maker' not in st.session_state:
            st.session_state.instagram_filter_maker = InstagramFilterMaker()
        
        filter_maker = st.session_state.instagram_filter_maker
        
        # 서브 탭 생성
        sub_tabs = st.tabs([
            "📸 사진 업로드", 
            "🌈 색깔 변신쇼", 
            "🎭 필터 체험", 
            "🕵️ 경계선 탐정", 
            "🎮 연습문제"
        ])
        
        with sub_tabs[0]:
            st.header("📸 사진 업로드")
            
            uploaded_file = st.file_uploader(
                "사진을 업로드하세요",
                type=['png', 'jpg', 'jpeg'],
                help="PNG, JPG, JPEG 형식을 지원합니다",
                key="instagram_upload"
            )
            
            if uploaded_file is not None:
                if filter_maker.load_image_from_upload(uploaded_file):
                    st.success("✅ 사진 업로드 성공!")
                    
                    # 이미지 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🖼️ 업로드된 사진")
                        # BGR을 RGB로 변환하여 표시
                        if filter_maker.current_image is not None:
                            display_image = cv2.cvtColor(filter_maker.current_image, cv2.COLOR_BGR2RGB)
                            st.image(display_image, use_column_width=True)
                    
                    with col2:
                        st.subheader("🔍 사진 정보")
                        info = filter_maker.get_image_info()
                        if info:
                            st.write(f"📐 크기: {info['width']} × {info['height']} 픽셀")
                            st.write(f"🎨 색상 채널: {info['channels']}개")
                            st.write(f"📊 픽셀값 범위: {info['min_value']} ~ {info['max_value']}")
                            st.write(f"✨ 평균 밝기: {info['mean_brightness']:.1f}/255")
                            
                            # 밝기 판정
                            avg_brightness = info['mean_brightness']
                            if avg_brightness > 180:
                                st.write("☀️ 매우 밝은 사진이에요!")
                            elif avg_brightness > 120:
                                st.write("🌤️ 적당히 밝은 사진이에요!")
                            elif avg_brightness > 60:
                                st.write("🌥️ 조금 어두운 사진이에요!")
                            else:
                                st.write("🌙 매우 어두운 사진이에요!")
        
        with sub_tabs[1]:
            st.header("🌈 색깔의 다양한 표현 방식")
            
            if filter_maker.current_image is not None:
                if st.button("🎭 색깔 변신쇼 시작!", key="color_transform"):
                    with st.spinner("🌈 색깔 변신 중..."):
                        fig = filter_maker.display_color_spaces()
                        if fig:
                            st.pyplot(fig)
                            st.markdown("""
                            **🎨 색깔 표현 방식 설명:**
                            - **RGB**: 빨강(Red) + 초록(Green) + 파랑(Blue) 조합
                            - **HSV**: 색조(Hue) + 채도(Saturation) + 명도(Value)
                            - **LAB**: 과학적 색상 표현 방식
                            - **흑백**: 색상 정보 제거, 밝기만 유지
                            """)
            else:
                st.info("먼저 📸 사진 업로드 탭에서 사진을 업로드해주세요!")
        
        with sub_tabs[2]:
            st.header("🎭 Instagram 필터 6종 세트")
            
            if filter_maker.current_image is not None:
                if st.button("🎪 필터 마술쇼 시작!", key="filter_show"):
                    with st.spinner("🎭 필터 적용 중..."):
                        fig, results = filter_maker.apply_convolution_filters()
                        if fig:
                            st.pyplot(fig)
                            
                            st.markdown("""
                            **🎨 필터 설명:**
                            - **✨ 선명**: 이미지의 경계를 더 또렷하게 만듦
                            - **💫 몽환**: 이미지를 부드럽게 블러 처리
                            - **🔍 경계**: 물체의 윤곽선만 추출
                            - **📐 세로선**: 세로 방향 경계 강조
                            - **📏 가로선**: 가로 방향 경계 강조
                            """)
            else:
                st.info("먼저 📸 사진 업로드 탭에서 사진을 업로드해주세요!")
        
        with sub_tabs[3]:
            st.header("🕵️ 경계선 찾기 탐정 대회")
            
            if filter_maker.current_image is not None:
                if st.button("🔍 탐정 대회 시작!", key="edge_detection"):
                    with st.spinner("🕵️ 경계선 찾는 중..."):
                        fig = filter_maker.edge_detection_comparison()
                        if fig:
                            st.pyplot(fig)
                            
                            st.markdown("""
                            **🕵️ 탐정 방법 설명:**
                            - **📐 세로탐정 (Sobel X)**: 세로 방향 경계선 전문
                            - **📏 가로탐정 (Sobel Y)**: 가로 방향 경계선 전문  
                            - **🎯 합체탐정**: 세로+가로 탐정의 합체
                            - **💫 전방향탐정 (Laplacian)**: 모든 방향 동시 탐지
                            - **🏆 천재탐정 (Canny)**: 가장 정확한 경계선 검출
                            """)
            else:
                st.info("먼저 📸 사진 업로드 탭에서 사진을 업로드해주세요!")
        
        with sub_tabs[4]:
            st.header("🎮 연습문제")
            
            # 연습문제 1: 기본 연산
            st.subheader("🎨 연습문제 1: 디지털 아트 창작")
            
            ex1_col1, ex1_col2, ex1_col3 = st.columns(3)
            
            with ex1_col1:
                st.write("**🏁 체스판 만들기**")
                size = st.slider("체스판 크기", 50, 200, 100, key="instagram_checkerboard_size")
                square_size = st.slider("사각형 크기", 5, 20, 10, key="instagram_square_size")
                
                if st.button("체스판 생성", key="instagram_create_checkerboard"):
                    checkerboard = self._create_checkerboard(size, square_size)
                    st.image(checkerboard, caption="생성된 체스판")
            
            with ex1_col2:
                st.write("**☀️ 밝기 조절**")
                if filter_maker.current_image is not None:
                    brightness_factor = st.slider("밝기 조절", 0.1, 3.0, 1.0, 0.1, key="instagram_brightness")
                    
                    if st.button("밝기 적용", key="instagram_apply_brightness"):
                        adjusted = self._adjust_brightness(filter_maker.current_image, brightness_factor)
                        display_adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
                        st.image(display_adjusted, caption=f"밝기 {brightness_factor}x")
                else:
                    st.info("사진을 먼저 업로드하세요")
            
            with ex1_col3:
                st.write("**🌪️ 이미지 회전**")
                if filter_maker.current_image is not None:
                    rotation = st.selectbox("회전 각도", [0, 90, 180, 270], key="instagram_rotation")
                    
                    if st.button("회전 적용", key="instagram_apply_rotation"):
                        rotated = self._rotate_image(filter_maker.current_image, rotation)
                        display_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
                        st.image(display_rotated, caption=f"{rotation}도 회전")
                else:
                    st.info("사진을 먼저 업로드하세요")
            
            st.markdown("---")
            
            # 연습문제 2: 특수 효과
            st.subheader("🎬 연습문제 2: 특수 효과 제작소")
            
            ex2_col1, ex2_col2 = st.columns(2)
            
            with ex2_col1:
                st.write("**🏺 엠보싱 효과**")
                if filter_maker.current_image is not None and st.button("엠보싱 적용", key="instagram_apply_emboss"):
                    embossed = self._emboss_filter(filter_maker.current_image)
                    st.image(embossed, caption="엠보싱 효과", cmap='gray')
                elif filter_maker.current_image is None:
                    st.info("사진을 먼저 업로드하세요")
            
            with ex2_col2:
                st.write("**💫 모션 블러**")
                if filter_maker.current_image is not None:
                    blur_size = st.slider("블러 강도", 5, 25, 15, key="instagram_blur_size")
                    blur_angle = st.selectbox("블러 방향", [0, 45, 90], 
                                            format_func=lambda x: f"{x}도 ({'수평' if x==0 else '대각선' if x==45 else '수직'})",
                                            key="instagram_blur_angle")
                    
                    if st.button("모션 블러 적용", key="instagram_apply_motion_blur"):
                        motion_blurred = self._motion_blur_filter(filter_maker.current_image, blur_size, blur_angle)
                        st.image(motion_blurred, caption="모션 블러 효과", cmap='gray')
                else:
                    st.info("사진을 먼저 업로드하세요")

    def _create_checkerboard(self, size=100, square_size=10):
        """🏁 체스판 만들기"""
        checkerboard = np.zeros((size, size), dtype=np.uint8)
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    checkerboard[i:i+square_size, j:j+square_size] = 255
        return checkerboard

    def _adjust_brightness(self, image, factor):
        """☀️ 사진 밝기 조절"""
        adjusted = image.astype(np.float32) * factor
        adjusted = np.clip(adjusted, 0, 255)
        return adjusted.astype(np.uint8)

    def _rotate_image(self, image, angle_90):
        """🌪️ 이미지 회전 (90도 단위)"""
        if angle_90 == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle_90 == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle_90 == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image

    def _emboss_filter(self, image):
        """🏺 엠보싱 필터"""
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.filter2D(gray, -1, kernel)

    def _motion_blur_filter(self, image, size=15, angle=45):
        """💫 모션 블러 필터"""
        kernel = np.zeros((size, size))
        if angle == 0:  # 수평
            kernel[size//2, :] = 1
        elif angle == 45:  # 대각선
            np.fill_diagonal(kernel, 1)
        elif angle == 90:  # 수직
            kernel[:, size//2] = 1
        
        kernel = kernel / kernel.sum()
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.filter2D(gray, -1, kernel)

    def _render_cnn_visualization_tab(self):
        """CNN 시각화 탭"""
        st.header("🤖 CNN 시각화")

        # 탭 생성
        sub_tabs = st.tabs(["📚 CNN 구조 이해하기", "🔬 특징 맵 시각화"])

        with sub_tabs[0]:
            st.markdown("### CNN의 핵심 구성 요소")

            # CNN 구조 설명
            st.markdown("""
            #### 1️⃣ **Convolutional Layer (합성곱층)**
            """)

            col1, col2 = st.columns(2)

            with col1:
                # 합성곱 연산 예제
                conv_input = np.random.randn(5, 5) * 50 + 128
                conv_input = np.clip(conv_input, 0, 255).astype(np.uint8)
                conv_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
                conv_output = cv2.filter2D(conv_input, -1, conv_kernel)

                st.image(conv_input, caption="입력 (5x5)", width=150)
                st.caption("💡 입력 이미지 또는 이전 층의 특징 맵")

            with col2:
                st.image(conv_output, caption="합성곱 결과 (5x5)", width=150)
                st.caption("💡 엣지, 텍스처 등 특징 추출")

            st.info("""
            **핵심 개념**:
            - **필터/커널**: 학습 가능한 가중치 행렬
            - **특징 추출**: 이미지에서 패턴을 자동으로 학습
            - **파라미터 공유**: 같은 필터를 전체 이미지에 적용
            """)

            st.markdown("""
            #### 2️⃣ **Activation Function (활성화 함수)**
            """)

            # ReLU 시각화
            x = np.linspace(-5, 5, 100)
            relu_y = np.maximum(0, x)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**ReLU (Rectified Linear Unit)**")
                st.line_chart({"x": x, "ReLU(x)": relu_y})
                st.caption("💡 음수를 0으로, 양수는 그대로")

            with col2:
                st.write("**ReLU의 효과**")
                st.markdown("""
                - ✅ 계산 효율적
                - ✅ 비선형성 추가
                - ✅ 기울기 소실 문제 완화
                - ✅ 희소성 유도
                """)

            st.markdown("""
            #### 3️⃣ **Pooling Layer (풀링층)**
            """)

            # Max Pooling 예제
            pool_input = np.array([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ], dtype=np.float32)

            # 2x2 Max Pooling
            pool_output = np.array([
                [np.max(pool_input[0:2, 0:2]), np.max(pool_input[0:2, 2:4])],
                [np.max(pool_input[2:4, 0:2]), np.max(pool_input[2:4, 2:4])]
            ])

            col1, col2 = st.columns(2)

            with col1:
                st.write("**입력 (4x4)**")
                st.code(str(pool_input.astype(int)))

            with col2:
                st.write("**Max Pooling 2x2 결과**")
                st.code(str(pool_output.astype(int)))
                st.caption("💡 크기 축소 + 중요 특징 보존")

            st.markdown("""
            #### 4️⃣ **Fully Connected Layer (완전연결층)**
            """)

            st.info("""
            **역할**:
            - 추출된 특징을 최종 출력으로 변환
            - 분류 작업: 클래스 수만큼 출력 뉴런
            - 회귀 작업: 1개 또는 여러 개 출력 값
            """)

            # CNN 전체 구조
            st.markdown("### 전체 CNN 아키텍처")

            st.code("""
            입력 이미지 (28x28x1)
                ↓
            Conv2d(1, 16, 3) → (26x26x16)
                ↓
            ReLU() → (26x26x16)
                ↓
            MaxPool2d(2) → (13x13x16)
                ↓
            Conv2d(16, 32, 3) → (11x11x32)
                ↓
            ReLU() → (11x11x32)
                ↓
            MaxPool2d(2) → (5x5x32)
                ↓
            Flatten() → (800,)
                ↓
            Linear(800, 10) → (10,)
                ↓
            출력 (10개 클래스)
            """, language="text")

            st.success("""
            📖 **핵심 학습 포인트**:
            - **계층적 특징 학습**: 낮은 층 → 단순 특징, 높은 층 → 복잡한 특징
            - **파라미터 효율성**: 합성곱과 풀링으로 파라미터 수 감소
            - **위치 불변성**: 풀링을 통한 작은 이동에 대한 강건성
            - **자동 특징 추출**: 수동 특징 설계 불필요
            """)

        with sub_tabs[1]:
            st.markdown("### 특징 맵 시각화")

            # 간단한 CNN 모델 생성
            model = self._create_simple_cnn()

            st.info("""
            **특징 맵(Feature Map)**이란?
            - CNN의 각 층이 이미지에서 추출한 특징들
            - 초기 층: 엣지, 색상 등 단순 특징
            - 깊은 층: 텍스처, 패턴 등 복잡한 특징
            """)

            # 특징 맵 생성 옵션
            col1, col2 = st.columns(2)

            with col1:
                input_type = st.radio(
                    "입력 선택",
                    ["랜덤 노이즈", "숫자 패턴", "체커보드"]
                )

            with col2:
                if st.button("특징 맵 생성", key="generate_features"):
                    # 입력 텐서 생성
                    if input_type == "랜덤 노이즈":
                        input_tensor = torch.randn(1, 1, 28, 28)
                    elif input_type == "숫자 패턴":
                        # 간단한 숫자 7 패턴
                        pattern = np.zeros((28, 28), dtype=np.float32)
                        pattern[5:8, 8:20] = 1.0  # 위 가로선
                        pattern[8:20, 17:20] = 1.0  # 대각선
                        input_tensor = torch.tensor(pattern).unsqueeze(0).unsqueeze(0)
                    else:  # 체커보드
                        pattern = np.zeros((28, 28), dtype=np.float32)
                        for i in range(0, 28, 4):
                            for j in range(0, 28, 4):
                                if (i//4 + j//4) % 2 == 0:
                                    pattern[i:i+4, j:j+4] = 1.0
                        input_tensor = torch.tensor(pattern).unsqueeze(0).unsqueeze(0)

                    # 특징 추출
                    features = self._extract_features(model, input_tensor)

                    # 입력 표시
                    st.subheader("입력 이미지")
                    input_img = input_tensor[0, 0].numpy()
                    # 범위를 0-1로 정규화
                    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
                    st.image(input_img, caption="입력 (28x28)", width=200)

                    # 각 레이어의 특징 맵 표시
                    st.subheader("각 레이어의 특징 맵")

                    for name, feature in features.items():
                        st.write(f"**{name}** - Shape: {tuple(feature.shape)}")

                        # 특징 맵 시각화
                        if len(feature.shape) == 4:
                            # 처음 4개 채널만 표시
                            cols = st.columns(min(4, feature.shape[1]))
                            for i in range(min(4, feature.shape[1])):
                                with cols[i]:
                                    img = feature[0, i].detach().numpy()
                                    # 정규화
                                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                                    st.image(img, caption=f"채널 {i+1}", width='stretch')

                            if feature.shape[1] > 4:
                                st.caption(f"... 총 {feature.shape[1]}개 채널 중 4개만 표시")

                        st.markdown("---")

    def _render_huggingface_tab(self):
        """HuggingFace 탭"""
        st.header("🚀 HuggingFace 모델")

        uploaded_file = st.file_uploader(
            "이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="hf_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", width='stretch')

            # 모델 선택
            task = st.selectbox(
                "작업 선택",
                ["이미지 분류", "객체 검출", "이미지 세그멘테이션"]
            )

            if task == "이미지 분류":
                if st.button("분류 실행", key="classify"):
                    with st.spinner("분류 중..."):
                        results = self.ai_manager.classify_image(image)
                        st.success("분류 완료!")

                        for result in results[:5]:
                            st.write(f"**{result['label']}**: {result['score']:.2%}")

            elif task == "객체 검출":
                if st.button("검출 실행", key="detect"):
                    with st.spinner("검출 중..."):
                        results = self.ai_manager.detect_objects(image)
                        st.success(f"{len(results)}개 객체 검출!")

                        # 박스 그리기
                        img_with_boxes = self.ai_manager.draw_detection_boxes(image, results)
                        st.image(img_with_boxes, caption="검출 결과", width='stretch')

                        for obj in results:
                            st.write(f"- **{obj['label']}**: {obj['score']:.2%}")

            elif task == "이미지 세그멘테이션":
                if st.button("세그멘테이션 실행", key="segment"):
                    with st.spinner("세그멘테이션 중..."):
                        results = self.ai_manager.segment_image(image)
                        st.success("세그멘테이션 완료!")

                        # 세그멘테이션 결과 표시
                        if results:
                            for i, result in enumerate(results[:3]):  # 최대 3개 세그먼트만 표시
                                label = result.get('label', f'Segment {i+1}')
                                st.write(f"**{label}**: 마스크 크기 {np.array(result['mask']).shape}")

                                # 마스크 시각화
                                mask = np.array(result['mask'])
                                if len(mask.shape) == 3:
                                    mask = mask[:,:,0]  # 첫 번째 채널만 사용
                                st.image(mask, caption=f"{label} 마스크", width='stretch')
                        else:
                            st.warning("세그멘테이션 결과가 없습니다.")

    def _render_integrated_analysis_tab(self):
        """통합 분석 탭"""
        st.header("📊 통합 이미지 분석")

        uploaded_file = st.file_uploader(
            "이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="integrated_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

            # 분석 옵션
            options = st.multiselect(
                "분석 옵션 선택",
                ["기본 정보", "히스토그램", "필터 적용", "AI 분석", "엣지 검출"]
            )

            if st.button("통합 분석 실행", key="run_integrated"):
                results = {}

                if "기본 정보" in options:
                    results["stats"] = self.get_image_stats(image)

                if "히스토그램" in options:
                    results["histogram"] = self.calculate_histogram(np.array(image))

                if "필터 적용" in options:
                    results["filtered"] = {
                        name: self.filters.apply_filter(image, name)
                        for name in ["Blur", "Edge Detection", "Sharpen"]
                    }

                if "AI 분석" in options:
                    with st.spinner("AI 분석 중..."):
                        results["classification"] = self.ai_manager.classify_image(image)[:3]

                if "엣지 검출" in options:
                    gray = self.convert_to_grayscale(image)
                    results["edges"] = cv2.Canny(gray, 50, 150)

                # 결과 표시
                self._display_integrated_results(image, results, options)

    def _display_integrated_results(self, image, results, options):
        """통합 분석 결과 표시"""
        st.success("분석 완료!")

        if "기본 정보" in options:
            st.subheader("📊 기본 정보")
            cols = st.columns(4)
            for idx, (key, value) in enumerate(results["stats"].items()):
                cols[idx % 4].metric(key, value)

        if "히스토그램" in options:
            st.subheader("📈 히스토그램")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            for channel, hist in results["histogram"].items():
                ax.plot(hist, label=channel, alpha=0.7)
            ax.legend()
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        if "필터 적용" in options:
            st.subheader("🎨 필터 결과")
            cols = st.columns(len(results["filtered"]))
            for idx, (name, img) in enumerate(results["filtered"].items()):
                cols[idx].write(name)
                cols[idx].image(img, width='stretch')

        if "AI 분석" in options:
            st.subheader("🤖 AI 분류 결과")
            for result in results["classification"]:
                st.write(f"- **{result['label']}**: {result['score']:.2%}")

        if "엣지 검출" in options:
            st.subheader("🔍 엣지 검출")
            st.image(results["edges"], width='stretch')

    def _create_simple_cnn(self):
        """간단한 CNN 모델 생성"""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc = nn.Linear(32 * 7 * 7, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 32 * 7 * 7)
                x = self.fc(x)
                return x

        return SimpleCNN()

    def _extract_features(self, model, input_tensor):
        """특징 맵 추출"""
        features = {}

        x = input_tensor
        x = F.relu(model.conv1(x))
        features['Conv1'] = x.clone()

        x = model.pool(x)
        features['Pool1'] = x.clone()

        x = F.relu(model.conv2(x))
        features['Conv2'] = x.clone()

        x = model.pool(x)
        features['Pool2'] = x.clone()

        return features