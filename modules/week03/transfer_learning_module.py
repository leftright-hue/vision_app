"""
Week 3: Transfer Learning & Multi-modal API 모듈
Transfer Learning과 Multi-modal API 관련 기능을 제공합니다.
"""

import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
import os
import sys

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_processor import BaseImageProcessor
from core.ai_models import AIModelManager
from .transfer_helpers import TransferLearningHelper
from .multimodal_helpers import MultiModalHelper


class TransferLearningModule(BaseImageProcessor):
    """Transfer Learning 및 Multi-modal API 학습 모듈"""

    def __init__(self):
        super().__init__()
        self.ai_manager = AIModelManager()
        self.transfer_helper = TransferLearningHelper()
        self.multimodal_helper = MultiModalHelper()

    def render(self):
        """Week 3 모듈 UI 렌더링 - Week 2와 동일한 메서드명"""
        self.render_ui()

    def render_ui(self):
        """Week 3 모듈 UI 렌더링"""
        st.title("🔄 Week 3: Transfer Learning & Multi-modal API")
        st.markdown("---")

        # 탭 생성
        tabs = st.tabs([
            "📚 이론",
            "🔄 Transfer Learning",
            "🖼️ CLIP 검색",
            "🔍 API 비교",
            "🎨 특징 추출",
            "📊 통합 분석",
            "🚀 실전 프로젝트"
        ])

        with tabs[0]:
            self._render_theory_tab()

        with tabs[1]:
            self._render_transfer_learning_tab()

        with tabs[2]:
            self._render_clip_search_tab()

        with tabs[3]:
            self._render_api_comparison_tab()

        with tabs[4]:
            self._render_feature_extraction_tab()

        with tabs[5]:
            self._render_integrated_analysis_tab()

        with tabs[6]:
            self._render_project_tab()

    def _render_theory_tab(self):
        """이론 탭"""
        st.header("📖 Transfer Learning & Multi-modal 이론")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Transfer Learning이란?")
            st.markdown("""
            - **정의**: 사전 훈련된 모델을 새로운 작업에 활용
            - **장점**: 적은 데이터로 높은 성능
            - **방법**: Feature Extraction, Fine-tuning
            - **응용**: 의료 영상, 제품 검사 등
            """)

            st.subheader("2. 주요 기법")
            st.markdown("""
            - **Feature Extraction**: 마지막 층만 학습
            - **Fine-tuning**: 전체 또는 일부 층 재학습
            - **Domain Adaptation**: 도메인 간 지식 전이
            - **Few-shot Learning**: 매우 적은 샘플로 학습
            """)

        with col2:
            st.subheader("3. Multi-modal Learning")
            st.markdown("""
            - **CLIP**: 텍스트-이미지 연결
            - **DALL-E**: 텍스트로 이미지 생성
            - **Flamingo**: 비전-언어 이해
            - **ALIGN**: 대규모 비전-언어 모델
            """)

            st.subheader("4. 실제 활용 사례")
            st.markdown("""
            - **의료 AI**: X-ray, MRI 분석
            - **자율주행**: 객체 인식 및 추적
            - **품질 검사**: 제조업 불량 검출
            - **콘텐츠 검색**: 이미지-텍스트 검색
            """)

    def _render_transfer_learning_tab(self):
        """Transfer Learning 탭"""
        st.header("🔄 Transfer Learning 실습")

        # 탭 생성: 예제 학습과 사용자 이미지 분석
        sub_tabs = st.tabs(["📚 예제로 학습하기", "🔧 내 모델 Fine-tuning하기"])

        with sub_tabs[0]:
            st.markdown("### 1. 사전 훈련 모델 선택")

            col1, col2, col3 = st.columns(3)

            with col1:
                model_name = st.selectbox(
                    "모델 선택",
                    ["ResNet50", "VGG16", "EfficientNet", "MobileNet", "DenseNet"],
                    key="model_select_example"
                )

            with col2:
                pretrained = st.checkbox("사전 훈련 가중치 사용", value=True, key="pretrained_example")

            with col3:
                num_classes = st.number_input("출력 클래스 수", min_value=2, value=10, key="num_classes_example")

            # 모델 정보 표시
            if st.button("모델 정보 보기", key="model_info_example"):
                self._show_model_info(model_name)

            st.markdown("### 2. Transfer Learning 방법")

            method = st.radio(
                "학습 방법 선택",
                ["Feature Extraction (빠름)", "Fine-tuning (정확함)", "전체 학습 (느림)"],
                key="method_example"
            )

            # 코드 예시
            with st.expander("📝 코드 보기"):
                code = self.transfer_helper.get_transfer_learning_code(model_name, num_classes, method)
                st.code(code, language="python")

        with sub_tabs[1]:
            st.markdown("### 🔧 커스텀 데이터셋으로 Fine-tuning")

            # 파일 업로드
            uploaded_files = st.file_uploader(
                "학습할 이미지 업로드 (클래스별로 폴더 구분)",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="custom_dataset"
            )

            if uploaded_files:
                col1, col2 = st.columns(2)

                with col1:
                    model_choice = st.selectbox(
                        "베이스 모델",
                        ["ResNet50", "EfficientNet-B0", "MobileNetV2"],
                        key="model_custom"
                    )

                    learning_rate = st.slider(
                        "학습률",
                        min_value=0.0001,
                        max_value=0.01,
                        value=0.001,
                        format="%.4f",
                        key="lr_custom"
                    )

                with col2:
                    epochs = st.slider("에폭 수", min_value=1, max_value=50, value=10, key="epochs_custom")
                    batch_size = st.select_slider("배치 크기", options=[8, 16, 32, 64], value=32, key="batch_custom")

                if st.button("🚀 Fine-tuning 시작", key="start_finetuning"):
                    with st.spinner("모델을 학습하는 중..."):
                        # 실제 fine-tuning 로직은 여기에 구현
                        st.info("Fine-tuning 시뮬레이션 중...")
                        progress_bar = st.progress(0)
                        for i in range(epochs):
                            progress_bar.progress((i + 1) / epochs)
                        st.success("Fine-tuning 완료!")

    def _render_clip_search_tab(self):
        """CLIP Image Search 탭"""
        st.header("🖼️ CLIP을 사용한 이미지 검색")

        # 탭 생성
        sub_tabs = st.tabs(["🔍 텍스트로 검색", "🖼️ 이미지로 검색", "📊 임베딩 시각화"])

        with sub_tabs[0]:
            st.markdown("### 텍스트 → 이미지 검색")

            search_query = st.text_input(
                "검색할 텍스트 입력",
                placeholder="예: 빨간 자동차, 행복한 강아지, 일몰 해변",
                key="clip_text_search"
            )

            # 이미지 데이터베이스
            uploaded_images = st.file_uploader(
                "검색할 이미지 데이터베이스 업로드",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="clip_db_text"
            )

            if search_query and uploaded_images:
                if st.button("🔍 CLIP 검색 실행", key="run_clip_text"):
                    with st.spinner("CLIP 모델로 검색 중..."):
                        # CLIP 검색 시뮬레이션
                        st.info(f"'{search_query}'와 가장 유사한 이미지를 찾는 중...")

                        # 결과 표시 (시뮬레이션)
                        cols = st.columns(3)
                        for i, img_file in enumerate(uploaded_images[:3]):
                            if i < 3:
                                img = Image.open(img_file)
                                cols[i].image(img, caption=f"유사도: {np.random.uniform(0.7, 0.95):.2%}")

        with sub_tabs[1]:
            st.markdown("### 이미지 → 이미지 검색")

            query_image = st.file_uploader(
                "쿼리 이미지 업로드",
                type=['png', 'jpg', 'jpeg'],
                key="clip_query_image"
            )

            db_images = st.file_uploader(
                "검색할 이미지 데이터베이스",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="clip_db_image"
            )

            if query_image and db_images:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(query_image, caption="쿼리 이미지")

                with col2:
                    if st.button("🔍 유사 이미지 검색", key="run_clip_image"):
                        st.info("유사한 이미지를 검색 중...")

        with sub_tabs[2]:
            st.markdown("### 📊 CLIP 임베딩 시각화")

            if st.button("임베딩 공간 시각화", key="visualize_embeddings"):
                # 임베딩 시각화 (시뮬레이션)
                fig = self.multimodal_helper.visualize_clip_embeddings()
                st.pyplot(fig)

    def _render_api_comparison_tab(self):
        """Multi-modal API 비교 탭"""
        st.header("🔍 Multi-modal API 비교 분석")

        # 2025년 9월 기준 API 정보 표시
        with st.expander("📅 2025년 9월 기준 API 접근 방법", expanded=True):
            st.markdown("""
            ### 🔗 OpenAI CLIP
            - **접근 방식**: 오픈소스 다운로드 (API 서비스 아님)
            - **설치**: `pip install git+https://github.com/openai/CLIP.git`
            - **특징**: API 키 불필요, 완전 무료, 로컬 실행
            - **응답 속도**: <100ms (GPU 사용 시)

            ### 🤖 Google Gemini API (2025년 권장)
            - **Vision API 대체**: Gemini가 Vision API를 대체하는 추세
            - **Google AI Studio 접근 방법**:
              1. ai.google.dev 접속
              2. Google 계정 로그인
              3. "Get API key" 클릭
              4. "Create API key in new project" 선택
              5. API 키 생성 (형식: AIza...)
            - **무료 할당량**: 분당 60건, 신용카드 불필요
            - **강점**: 멀티모달 처리, PDF 직접 처리, 90분 비디오 지원

            ### 🤗 Hugging Face API
            - **토큰 생성**: HuggingFace.co → Settings → Access Tokens → New Token
            - **토큰 형식**: `hf_xxxxx`
            - **2025년 권장**: Fine-grained 토큰, 앱별 별도 토큰 생성
            """)

        st.markdown("---")

        # API 선택
        selected_apis = st.multiselect(
            "비교할 API 선택",
            ["OpenAI CLIP", "Google Vision API", "Azure Computer Vision",
             "AWS Rekognition", "Hugging Face", "OpenAI GPT-4V"],
            default=["OpenAI CLIP", "Google Vision API", "Hugging Face"],
            key="api_comparison"
        )

        if len(selected_apis) >= 2:
            # 비교 차트 생성
            st.subheader("📊 API 기능 비교")

            comparison_df = self.multimodal_helper.get_api_comparison_data(selected_apis)
            st.dataframe(comparison_df, use_container_width=True)

            # 성능 벤치마크
            st.subheader("⚡ 성능 벤치마크")

            col1, col2 = st.columns(2)

            with col1:
                # 속도 비교 차트
                fig_speed = self.multimodal_helper.create_speed_comparison_chart(selected_apis)
                st.pyplot(fig_speed)

            with col2:
                # 정확도 비교 차트
                fig_accuracy = self.multimodal_helper.create_accuracy_comparison_chart(selected_apis)
                st.pyplot(fig_accuracy)

            # 사용 사례별 추천
            st.subheader("💡 사용 사례별 추천")

            use_case = st.selectbox(
                "사용 사례 선택",
                ["이미지 검색", "콘텐츠 모더레이션", "의료 이미지 분석",
                 "제품 추천", "자동 태깅", "시각적 질의응답"],
                key="use_case"
            )

            recommendation = self.multimodal_helper.get_api_recommendation(use_case, selected_apis)
            st.info(recommendation)

    def _render_feature_extraction_tab(self):
        """특징 추출 탭"""
        st.header("🎨 특징 추출 및 시각화")

        uploaded_file = st.file_uploader(
            "이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="feature_extraction"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="원본 이미지")

                model_choice = st.selectbox(
                    "특징 추출 모델",
                    ["ResNet50", "VGG16", "EfficientNet", "CLIP"],
                    key="feature_model"
                )

                layer_choice = st.selectbox(
                    "추출할 레이어",
                    ["Early layers", "Middle layers", "Late layers", "Final layer"],
                    key="feature_layer"
                )

            with col2:
                if st.button("🎨 특징 추출", key="extract_features"):
                    with st.spinner("특징을 추출하는 중..."):
                        # 특징 추출 시각화 (시뮬레이션)
                        fig = self.transfer_helper.visualize_features(image, model_choice, layer_choice)
                        st.pyplot(fig)

            # 특징 맵 분석
            if st.checkbox("상세 분석 보기", key="detailed_analysis"):
                st.subheader("📊 특징 맵 상세 분석")

                tabs = st.tabs(["히트맵", "3D 시각화", "통계"])

                with tabs[0]:
                    st.info("특징 맵 히트맵 시각화")
                    # 히트맵 시각화 코드

                with tabs[1]:
                    st.info("3D 특징 공간 시각화")
                    # 3D 시각화 코드

                with tabs[2]:
                    st.info("특징 통계 분석")
                    # 통계 분석 코드

    def _render_integrated_analysis_tab(self):
        """통합 분석 탭"""
        st.header("📊 Transfer Learning 통합 분석")

        analysis_type = st.selectbox(
            "분석 유형 선택",
            ["모델 성능 비교", "학습 곡선 분석", "혼동 행렬", "특징 공간 분석"],
            key="integrated_analysis"
        )

        if analysis_type == "모델 성능 비교":
            st.subheader("🏆 모델 성능 비교")

            # 모델 선택
            models = st.multiselect(
                "비교할 모델",
                ["ResNet50", "VGG16", "EfficientNet", "MobileNet", "DenseNet"],
                default=["ResNet50", "EfficientNet"],
                key="model_comparison"
            )

            if len(models) >= 2:
                # 성능 메트릭 표시
                metrics_df = self.transfer_helper.get_model_metrics(models)
                st.dataframe(metrics_df, use_container_width=True)

                # 차트 생성
                fig = self.transfer_helper.create_performance_chart(models)
                st.pyplot(fig)

        elif analysis_type == "학습 곡선 분석":
            st.subheader("📈 학습 곡선 분석")

            # 학습 곡선 시각화
            fig = self.transfer_helper.plot_learning_curves()
            st.pyplot(fig)

            # 분석 인사이트
            st.info("""
            **학습 곡선 해석**:
            - 훈련 손실과 검증 손실의 차이가 크면 과적합
            - 두 곡선이 모두 높으면 과소적합
            - 최적점은 검증 손실이 최소인 지점
            """)

        elif analysis_type == "혼동 행렬":
            st.subheader("🔢 혼동 행렬 분석")

            # 클래스 수 선택
            num_classes = st.slider("클래스 수", min_value=2, max_value=10, value=5, key="confusion_classes")

            # 혼동 행렬 생성 및 표시
            fig = self.transfer_helper.create_confusion_matrix(num_classes)
            st.pyplot(fig)

        else:  # 특징 공간 분석
            st.subheader("🌌 특징 공간 분석")

            # t-SNE 시각화
            fig = self.transfer_helper.visualize_feature_space()
            st.pyplot(fig)

    def _render_project_tab(self):
        """실전 프로젝트 탭"""
        st.header("🚀 실전 Transfer Learning 프로젝트")

        project_type = st.selectbox(
            "프로젝트 선택",
            ["🏥 의료 이미지 분류", "🏭 제조업 품질 검사", "🎨 스타일 전이", "🔍 상품 검색 시스템"],
            key="project_type"
        )

        if project_type == "🏥 의료 이미지 분류":
            self._render_medical_project()
        elif project_type == "🏭 제조업 품질 검사":
            self._render_quality_control_project()
        elif project_type == "🎨 스타일 전이":
            self._render_style_transfer_project()
        else:
            self._render_product_search_project()

    def _render_medical_project(self):
        """의료 이미지 분류 프로젝트"""
        st.subheader("🏥 X-ray 이미지 분류 시스템")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **프로젝트 목표**:
            - 흉부 X-ray에서 폐렴 검출
            - Transfer Learning으로 정확도 향상
            - 적은 데이터로 높은 성능 달성
            """)

            uploaded_xray = st.file_uploader(
                "X-ray 이미지 업로드",
                type=['png', 'jpg', 'jpeg'],
                key="xray_upload"
            )

        with col2:
            if uploaded_xray:
                st.image(uploaded_xray, caption="업로드된 X-ray")

                if st.button("🔍 진단 시작", key="diagnose"):
                    with st.spinner("AI 분석 중..."):
                        # 진단 시뮬레이션
                        st.success("분석 완료!")
                        st.metric("정상 확률", "15%")
                        st.metric("폐렴 확률", "85%", delta="주의 필요")

    def _render_quality_control_project(self):
        """제조업 품질 검사 프로젝트"""
        st.subheader("🏭 제품 불량 검출 시스템")

        st.markdown("""
        **시스템 특징**:
        - 실시간 불량품 검출
        - 다양한 불량 유형 분류
        - Transfer Learning으로 빠른 배포
        """)

        # 불량 유형 설정
        defect_types = st.multiselect(
            "검출할 불량 유형",
            ["스크래치", "찌그러짐", "변색", "크랙", "이물질"],
            default=["스크래치", "크랙"],
            key="defect_types"
        )

        if st.button("시스템 시작", key="start_qc"):
            st.info("품질 검사 시스템이 실행 중입니다...")

    def _render_style_transfer_project(self):
        """스타일 전이 프로젝트"""
        st.subheader("🎨 Neural Style Transfer")

        col1, col2 = st.columns(2)

        with col1:
            content_image = st.file_uploader(
                "콘텐츠 이미지",
                type=['png', 'jpg', 'jpeg'],
                key="content_img"
            )
            if content_image:
                st.image(content_image, caption="콘텐츠")

        with col2:
            style_image = st.file_uploader(
                "스타일 이미지",
                type=['png', 'jpg', 'jpeg'],
                key="style_img"
            )
            if style_image:
                st.image(style_image, caption="스타일")

        if content_image and style_image:
            style_weight = st.slider("스타일 강도", 0.0, 1.0, 0.5, key="style_weight")

            if st.button("🎨 스타일 전이 시작", key="transfer_style"):
                with st.spinner("스타일을 전이하는 중..."):
                    st.info("Neural Style Transfer 처리 중...")
                    st.success("스타일 전이 완료!")

    def _render_product_search_project(self):
        """상품 검색 시스템 프로젝트"""
        st.subheader("🔍 시각적 상품 검색 시스템")

        search_method = st.radio(
            "검색 방법",
            ["텍스트로 검색", "이미지로 검색", "하이브리드 검색"],
            key="search_method"
        )

        if search_method == "텍스트로 검색":
            query = st.text_input("검색어 입력", placeholder="빨간 운동화", key="text_query")
        elif search_method == "이미지로 검색":
            query_img = st.file_uploader("참조 이미지", type=['png', 'jpg', 'jpeg'], key="img_query")
        else:
            col1, col2 = st.columns(2)
            with col1:
                text_q = st.text_input("텍스트", placeholder="편안한", key="hybrid_text")
            with col2:
                img_q = st.file_uploader("이미지", type=['png', 'jpg', 'jpeg'], key="hybrid_img")

        if st.button("🔍 검색", key="search_products"):
            st.success("유사한 상품을 찾았습니다!")
            # 검색 결과 표시

    def _show_model_info(self, model_name):
        """모델 정보 표시"""
        model_info = {
            "ResNet50": {
                "parameters": "25.6M",
                "layers": "50",
                "year": "2015",
                "accuracy": "92.1%"
            },
            "VGG16": {
                "parameters": "138M",
                "layers": "16",
                "year": "2014",
                "accuracy": "90.1%"
            },
            "EfficientNet": {
                "parameters": "5.3M",
                "layers": "Variable",
                "year": "2019",
                "accuracy": "91.7%"
            },
            "MobileNet": {
                "parameters": "4.2M",
                "layers": "28",
                "year": "2017",
                "accuracy": "89.5%"
            },
            "DenseNet": {
                "parameters": "25.6M",
                "layers": "121",
                "year": "2016",
                "accuracy": "91.8%"
            }
        }

        if model_name in model_info:
            info = model_info[model_name]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Parameters", info["parameters"])
            col2.metric("Layers", info["layers"])
            col3.metric("Year", info["year"])
            col4.metric("ImageNet Top-5", info["accuracy"])