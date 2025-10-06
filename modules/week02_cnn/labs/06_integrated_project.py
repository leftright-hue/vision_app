"""
06. 통합 프로젝트: 이미지 분석 시스템
Week 2: 디지털 이미지 기초와 CNN

Week 2에서 학습한 모든 내용을 통합한 이미지 분석 시스템
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import base64

class IntegratedImageAnalysisSystem:
    """통합 이미지 분석 시스템"""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.initialize_models()
        self.initialize_filters()

    def initialize_models(self):
        """AI 모델 초기화"""
        @st.cache_resource
        def load_models():
            return {
                'classifier': pipeline(
                    "image-classification",
                    model="google/vit-base-patch16-224",
                    device=self.device
                ),
                'detector': pipeline(
                    "object-detection",
                    model="facebook/detr-resnet-50",
                    device=self.device
                )
            }
        self.models = load_models()

    def initialize_filters(self):
        """이미지 필터 초기화"""
        self.filters = {
            'None': None,
            'Blur': np.ones((5, 5)) / 25,
            'Gaussian': np.array([
                [1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]
            ]),
            'Edge Detection': np.array([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]),
            'Sharpen': np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]),
            'Emboss': np.array([
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2]
            ])
        }

    def apply_filter(self, image, filter_name):
        """이미지에 필터 적용"""
        if filter_name == 'None' or self.filters[filter_name] is None:
            return image

        # PIL to numpy
        img_array = np.array(image)

        # 그레이스케일 변환 (필요시)
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array

        # 필터 적용
        filtered = cv2.filter2D(img_gray, -1, self.filters[filter_name])

        # 다시 PIL로 변환
        return Image.fromarray(filtered.astype(np.uint8))

    def analyze_image_properties(self, image):
        """이미지 속성 분석"""
        img_array = np.array(image)

        properties = {
            "크기": f"{image.size[0]} x {image.size[1]} 픽셀",
            "모드": image.mode,
            "채널 수": len(img_array.shape) if len(img_array.shape) == 3 else 1,
            "데이터 타입": str(img_array.dtype),
            "최소값": int(img_array.min()),
            "최대값": int(img_array.max()),
            "평균값": f"{img_array.mean():.2f}",
            "표준편차": f"{img_array.std():.2f}"
        }

        return properties

    def create_histogram(self, image):
        """히스토그램 생성"""
        img_array = np.array(image)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 이미지 표시
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 히스토그램
        if len(img_array.shape) == 3:
            # 컬러 이미지
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
                axes[1].plot(hist, color=color, alpha=0.7, label=color.upper())
            axes[1].legend()
        else:
            # 그레이스케일
            hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
            axes[1].plot(hist, color='gray')

        axes[1].set_title('Histogram')
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)

        return fig

    def detect_edges_multi(self, image):
        """다양한 엣지 검출 방법 적용"""
        img_array = np.array(image)

        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # 다양한 엣지 검출
        edges = {
            'Sobel X': cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
            'Sobel Y': cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3),
            'Laplacian': cv2.Laplacian(gray, cv2.CV_64F),
            'Canny': cv2.Canny(gray, 50, 150)
        }

        return edges

    def run_streamlit_app(self):
        """Streamlit 웹 앱 실행"""
        st.set_page_config(
            page_title="통합 이미지 분석 시스템",
            page_icon="🖼️",
            layout="wide"
        )

        st.title("🎯 Week 2: 통합 이미지 분석 시스템")
        st.markdown("---")

        # 사이드바
        with st.sidebar:
            st.header("⚙️ 설정")
            analysis_mode = st.selectbox(
                "분석 모드",
                ["기본 분석", "필터링", "AI 분석", "엣지 검출", "통합 분석"]
            )

        # 이미지 업로드
        uploaded_file = st.file_uploader(
            "이미지를 선택하세요",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="지원 형식: PNG, JPG, JPEG, BMP"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            # 모드별 처리
            if analysis_mode == "기본 분석":
                self.basic_analysis(image)

            elif analysis_mode == "필터링":
                self.filtering_mode(image)

            elif analysis_mode == "AI 분석":
                self.ai_analysis_mode(image)

            elif analysis_mode == "엣지 검출":
                self.edge_detection_mode(image)

            elif analysis_mode == "통합 분석":
                self.integrated_analysis(image)

    def basic_analysis(self, image):
        """기본 분석 모드"""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 원본 이미지")
            st.image(image, width='stretch')

        with col2:
            st.subheader("📊 이미지 속성")
            properties = self.analyze_image_properties(image)
            for key, value in properties.items():
                st.metric(label=key, value=value)

        # 히스토그램
        st.subheader("📈 히스토그램 분석")
        fig = self.create_histogram(image)
        st.pyplot(fig)

    def filtering_mode(self, image):
        """필터링 모드"""
        st.subheader("🎨 이미지 필터링")

        col1, col2 = st.columns(2)

        with col1:
            st.write("원본 이미지")
            st.image(image, width='stretch')

        with col2:
            filter_name = st.selectbox("필터 선택", list(self.filters.keys()))
            filtered_image = self.apply_filter(image, filter_name)
            st.write(f"{filter_name} 필터 적용")
            st.image(filtered_image, width='stretch')

        # 필터 설명
        if filter_name != 'None':
            st.info(self.get_filter_description(filter_name))

    def ai_analysis_mode(self, image):
        """AI 분석 모드"""
        st.subheader("🤖 AI 기반 이미지 분석")

        # 분석 옵션
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🏷️ 이미지 분류", width='stretch'):
                with st.spinner("분류 중..."):
                    results = self.models['classifier'](image)
                    st.success("분류 완료!")

                    for i, result in enumerate(results[:5]):
                        st.write(f"{i+1}. **{result['label']}**: {result['score']:.2%}")

        with col2:
            if st.button("🎯 객체 검출", width='stretch'):
                with st.spinner("검출 중..."):
                    results = self.models['detector'](image)
                    st.success(f"{len(results)}개 객체 검출!")

                    # 결과 시각화
                    img_with_boxes = self.draw_detection_boxes(image, results)
                    st.image(img_with_boxes, width='stretch')

                    # 검출 결과 목록
                    for obj in results:
                        st.write(f"- **{obj['label']}**: {obj['score']:.2%}")

    def edge_detection_mode(self, image):
        """엣지 검출 모드"""
        st.subheader("🔍 엣지 검출")

        edges = self.detect_edges_multi(image)

        # 2x2 그리드로 표시
        col1, col2 = st.columns(2)

        for i, (name, edge_img) in enumerate(edges.items()):
            if i % 2 == 0:
                with col1:
                    st.write(name)
                    st.image(edge_img, width='stretch', clamp=True)
            else:
                with col2:
                    st.write(name)
                    st.image(edge_img, width='stretch', clamp=True)

    def integrated_analysis(self, image):
        """통합 분석 모드"""
        st.subheader("🔬 통합 이미지 분석")

        # 탭 생성
        tabs = st.tabs(["속성", "필터", "AI 분석", "엣지 검출"])

        with tabs[0]:
            self.basic_analysis(image)

        with tabs[1]:
            self.filtering_mode(image)

        with tabs[2]:
            self.ai_analysis_mode(image)

        with tabs[3]:
            self.edge_detection_mode(image)

    def draw_detection_boxes(self, image, results):
        """객체 검출 박스 그리기"""
        img_array = np.array(image)

        for obj in results:
            box = obj['box']
            xmin, ymin = int(box['xmin']), int(box['ymin'])
            xmax, ymax = int(box['xmax']), int(box['ymax'])

            # 박스 그리기
            cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

            # 레이블 표시
            label = f"{obj['label']}: {obj['score']:.2f}"
            cv2.putText(img_array, label, (xmin, ymin - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return Image.fromarray(img_array)

    def get_filter_description(self, filter_name):
        """필터 설명 반환"""
        descriptions = {
            'Blur': "평균 필터로 이미지를 부드럽게 만들고 노이즈를 제거합니다.",
            'Gaussian': "가우시안 분포를 사용한 자연스러운 블러 효과입니다.",
            'Edge Detection': "이미지의 엣지(경계)를 강조합니다.",
            'Sharpen': "이미지의 선명도를 증가시킵니다.",
            'Emboss': "3D 엠보싱 효과를 생성합니다."
        }
        return descriptions.get(filter_name, "")

def main():
    """메인 실행 함수"""
    system = IntegratedImageAnalysisSystem()
    system.run_streamlit_app()

if __name__ == "__main__":
    main()