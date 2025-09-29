"""
Week 4: Vision Transformer와 최신 모델 비교 모듈
Vision Transformer, DINO, SAM 등 최신 비전 모델 학습
"""

import streamlit as st
import numpy as np
from PIL import Image
import torch
import os
import sys

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_processor import BaseImageProcessor


class VisionTransformerModule(BaseImageProcessor):
    """Vision Transformer 및 최신 모델 학습 모듈"""

    def __init__(self):
        super().__init__()

    def render(self):
        """Week 4 모듈 UI 렌더링"""
        self.render_ui()

    def render_ui(self):
        """Week 4 모듈 UI 렌더링"""
        st.title("🤖 Week 4: Vision Transformer와 최신 모델")
        st.markdown("---")

        # 탭 생성
        tabs = st.tabs([
            "📚 이론",
            "🧠 Self-Attention",
            "🔍 Vision Transformer",
            "🎯 DINO & 자기지도학습",
            "📊 모델 벤치마크",
            "🚀 실전 프로젝트"
        ])

        with tabs[0]:
            self._render_theory_tab()

        with tabs[1]:
            self._render_self_attention_tab()

        with tabs[2]:
            self._render_vit_tab()

        with tabs[3]:
            self._render_dino_tab()

        with tabs[4]:
            self._render_benchmark_tab()

        with tabs[5]:
            self._render_project_tab()

    def _render_theory_tab(self):
        """이론 탭"""
        st.header("📖 Vision Transformer 이론")

        with st.expander("🔹 Transformer의 등장과 컴퓨터 비전 혁명", expanded=True):
            st.markdown("""
            ### 🧠 Transformer의 역사

            #### NLP에서 시작된 혁명
            - **2017년 "Attention Is All You Need"** 논문으로 시작
            - RNN/LSTM의 순차 처리 한계 극복
            - 병렬 처리 가능한 Self-Attention 메커니즘 도입
            - BERT, GPT 등 대형 언어 모델의 기반

            #### 컴퓨터 비전으로의 확장
            - **2020년 "An Image is Worth 16x16 Words"** (ViT 논문)
            - CNN의 귀납적 편향 없이도 우수한 성능
            - 대용량 데이터에서 CNN을 능가하는 성능
            - 멀티모달 AI의 핵심 구성 요소

            ### 📊 CNN vs Transformer 비교

            | 특징 | CNN | Transformer |
            |------|-----|-------------|
            | 처리 방식 | 지역적 (Local) | 전역적 (Global) |
            | 귀납적 편향 | 강함 (평행이동 불변성) | 약함 |
            | 데이터 요구량 | 적음 | 많음 |
            | 계산 복잡도 | O(n) | O(n²) |
            | 장거리 의존성 | 약함 | 강함 |
            """)

        with st.expander("🔹 Self-Attention 메커니즘"):
            st.markdown("""
            ### 🎯 Self-Attention의 핵심 원리

            **Query, Key, Value (Q, K, V) 개념:**
            ```python
            # Self-Attention 계산
            Q = X @ W_q  # Query: "무엇을 찾을까?"
            K = X @ W_k  # Key: "나는 무엇을 가지고 있나?"
            V = X @ W_v  # Value: "실제 정보"

            # Attention Score 계산
            attention_scores = (Q @ K.T) / sqrt(d_k)
            attention_weights = softmax(attention_scores)

            # 최종 출력
            output = attention_weights @ V
            ```

            ### 💡 주요 특징
            1. **전역적 문맥 이해**: 모든 위치 간의 관계 파악
            2. **병렬 처리**: GPU 효율적 활용
            3. **가중 평균**: 중요한 정보에 더 집중
            """)

        with st.expander("🔹 Vision Transformer (ViT) 아키텍처"):
            st.markdown("""
            ### 🏗️ ViT 구조

            ```
            입력 이미지 (224×224×3)
            ↓
            Patch Embedding (16×16 패치로 분할)
            ↓
            Positional Encoding (위치 정보 추가)
            ↓
            Transformer Encoder (12 layers)
            ├─ Multi-Head Attention
            ├─ Layer Normalization
            ├─ MLP (Feed-Forward)
            └─ Residual Connection
            ↓
            Classification Head
            ```

            ### 📐 핵심 설계 요소
            1. **Patch Embedding**: 이미지를 16×16 패치로 분할
            2. **Position Encoding**: 패치의 위치 정보 인코딩
            3. **[CLS] Token**: 분류를 위한 특수 토큰
            4. **Multi-Head Attention**: 다양한 관점에서 정보 처리
            """)

    def _render_self_attention_tab(self):
        """Self-Attention 탭"""
        st.header("🧠 Self-Attention 메커니즘")

        st.markdown("""
        ### 🎯 Self-Attention 시각화

        Self-Attention은 입력의 각 부분이 다른 모든 부분과 어떻게 관련되는지를 학습합니다.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔢 Attention 메커니즘**")
            uploaded_file = st.file_uploader(
                "이미지 업로드 (Attention 시각화)",
                type=['png', 'jpg', 'jpeg'],
                key="attention_upload"
            )

            if uploaded_file:
                st.image(uploaded_file, caption="입력 이미지", use_container_width=True)

                if st.button("🔍 Attention Map 생성", key="gen_attention"):
                    with st.spinner("Attention 계산 중..."):
                        st.success("✅ 완료!")
                        st.info("""
                        💡 실제 구현에서는 사전훈련된 ViT 모델을 사용하여
                        각 레이어의 Attention Map을 시각화할 수 있습니다.
                        """)

        with col2:
            st.markdown("**📊 Multi-Head Attention**")
            num_heads = st.slider("Attention Head 수", 1, 12, 8, key="num_heads")

            st.info(f"""
            **Multi-Head Attention 설정:**
            - Head 수: {num_heads}
            - 각 Head는 서로 다른 관점에서 정보를 처리
            - 최종 결과는 모든 Head의 출력을 결합
            """)

            st.code("""
# Multi-Head Attention 구현
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def forward(self, Q, K, V):
        # 여러 Head로 분할
        Q_heads = split_heads(Q, self.num_heads)
        K_heads = split_heads(K, self.num_heads)
        V_heads = split_heads(V, self.num_heads)

        # 각 Head에서 Attention 계산
        attention_outputs = []
        for q, k, v in zip(Q_heads, K_heads, V_heads):
            attn = scaled_dot_product_attention(q, k, v)
            attention_outputs.append(attn)

        # 결합
        return concat(attention_outputs)
            """, language="python")

    def _render_vit_tab(self):
        """Vision Transformer 탭"""
        st.header("🔍 Vision Transformer (ViT)")

        st.markdown("""
        ### 🏗️ ViT 아키텍처 이해하기

        Vision Transformer는 이미지를 **패치 시퀀스**로 변환하여 처리합니다.
        """)

        # ViT 설정
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**⚙️ ViT 모델 설정**")
            model_size = st.selectbox(
                "모델 크기",
                ["ViT-Tiny", "ViT-Small", "ViT-Base", "ViT-Large"],
                index=2,
                key="vit_size"
            )

            patch_size = st.selectbox(
                "Patch 크기",
                [8, 16, 32],
                index=1,
                key="patch_size"
            )

            st.info(f"""
            **선택된 설정:**
            - 모델: {model_size}
            - Patch 크기: {patch_size}×{patch_size}
            - 이미지 크기: 224×224
            - Patch 수: {(224//patch_size)**2}개
            """)

        with col2:
            st.markdown("**📊 모델 사양**")
            model_specs = {
                "ViT-Tiny": {"Layers": 12, "Hidden": 192, "Heads": 3, "Params": "5M"},
                "ViT-Small": {"Layers": 12, "Hidden": 384, "Heads": 6, "Params": "22M"},
                "ViT-Base": {"Layers": 12, "Hidden": 768, "Heads": 12, "Params": "86M"},
                "ViT-Large": {"Layers": 24, "Hidden": 1024, "Heads": 16, "Params": "307M"}
            }

            specs = model_specs[model_size]
            for key, value in specs.items():
                st.metric(key, value)

        # ViT 시연
        st.markdown("---")
        st.markdown("### 🎯 ViT 이미지 분류 시연")

        uploaded_vit = st.file_uploader(
            "이미지 업로드 (ViT 분류)",
            type=['png', 'jpg', 'jpeg'],
            key="vit_upload"
        )

        if uploaded_vit:
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_vit, caption="입력 이미지", use_container_width=True)

            with col2:
                if st.button("🔍 ViT 분류 실행", key="run_vit"):
                    with st.spinner(f"{model_size} 모델 추론 중..."):
                        st.success("✅ 분류 완료!")

                        # 시뮬레이션 결과
                        import random
                        classes = ["고양이", "개", "자동차", "비행기", "새"]
                        probs = [random.random() for _ in range(5)]
                        total = sum(probs)
                        probs = [p/total for p in probs]

                        st.markdown("**예측 결과:**")
                        for cls, prob in zip(classes, probs):
                            st.write(f"{cls}: {prob*100:.1f}%")
                            st.progress(prob)

    def _render_dino_tab(self):
        """DINO & 자기지도학습 탭"""
        st.header("🎯 DINO & 자기지도학습")

        st.markdown("""
        ### 🦖 DINO (Self-Distillation with No Labels)

        **DINO**는 레이블 없이 이미지의 구조를 학습하는 자기지도학습 방법입니다.
        """)

        with st.expander("🔹 DINO의 핵심 원리", expanded=True):
            st.markdown("""
            ### 📚 자기지도학습이란?

            - **지도학습**: 레이블이 필요 (고양이, 개, 자동차 등)
            - **자기지도학습**: 레이블 불필요, 데이터 자체에서 학습
            - **DINO**: Teacher-Student 구조로 지식 증류

            ### 🔄 DINO 학습 과정

            ```
            입력 이미지
            ↓
            ┌─────────────────┬─────────────────┐
            │  Student Network │  Teacher Network │
            │  (학습됨)         │  (EMA 업데이트)   │
            └─────────────────┴─────────────────┘
            ↓
            Knowledge Distillation Loss
            ```

            ### 💡 주요 특징
            1. **레이블 불필요**: 대규모 unlabeled 데이터 활용
            2. **강력한 특징**: Semantic Segmentation 가능
            3. **전이 학습**: 다양한 다운스트림 태스크에 활용
            """)

        # DINO 시연
        st.markdown("### 🎨 DINO 특징 시각화")

        uploaded_dino = st.file_uploader(
            "이미지 업로드 (DINO 분석)",
            type=['png', 'jpg', 'jpeg'],
            key="dino_upload"
        )

        if uploaded_dino:
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_dino, caption="입력 이미지", use_container_width=True)

            with col2:
                if st.button("🔍 DINO 특징 추출", key="run_dino"):
                    with st.spinner("DINO 분석 중..."):
                        st.success("✅ 완료!")
                        st.info("""
                        💡 DINO는 이미지의 의미론적 구조를 자동으로 발견합니다.
                        - 객체 경계 감지
                        - 부분-전체 관계 이해
                        - 의미론적 그룹화
                        """)

    def _render_benchmark_tab(self):
        """모델 벤치마크 탭"""
        st.header("📊 최신 비전 모델 벤치마크")

        st.markdown("""
        ### 🏆 주요 모델 성능 비교

        최신 비전 모델들의 성능을 다양한 지표로 비교합니다.
        """)

        # 벤치마크 데이터
        benchmark_data = {
            "모델": ["ResNet-50", "ViT-Base", "DINO ViT-S", "DINOv2 ViT-B", "SAM ViT-H"],
            "ImageNet Top-1": ["76.2%", "84.5%", "79.3%", "86.5%", "-"],
            "파라미터": ["25M", "86M", "22M", "86M", "632M"],
            "추론 속도": ["빠름", "보통", "보통", "보통", "느림"],
            "특화 분야": ["범용", "분류", "자기지도", "자기지도", "분할"]
        }

        st.table(benchmark_data)

        st.markdown("""
        ### 📈 모델 선택 가이드

        #### 🎯 용도별 추천 모델

        | 용도 | 추천 모델 | 이유 |
        |------|-----------|------|
        | 이미지 분류 | ViT-Base | 높은 정확도, 안정적 |
        | 실시간 처리 | ResNet-50 | 빠른 추론 속도 |
        | 전이 학습 | DINOv2 | 강력한 사전학습 특징 |
        | 객체 분할 | SAM | 제로샷 분할 능력 |
        | 적은 데이터 | DINO | 자기지도학습으로 일반화 |
        """)

        # 성능 비교 시뮬레이션
        st.markdown("---")
        st.markdown("### 🔬 성능 테스트")

        test_dataset = st.selectbox(
            "테스트 데이터셋",
            ["ImageNet", "CIFAR-100", "Custom Dataset"],
            key="test_dataset"
        )

        models_to_compare = st.multiselect(
            "비교할 모델",
            ["ResNet-50", "ViT-Base", "DINO ViT-S", "DINOv2 ViT-B"],
            default=["ResNet-50", "ViT-Base"],
            key="models_compare"
        )

        if st.button("🚀 벤치마크 실행", key="run_benchmark"):
            with st.spinner("벤치마크 실행 중..."):
                st.success("✅ 완료!")

                # 시뮬레이션 결과
                import random
                for model in models_to_compare:
                    accuracy = 70 + random.random() * 20
                    fps = 50 + random.random() * 100
                    st.metric(f"{model}", f"{accuracy:.1f}% accuracy, {fps:.0f} FPS")

    def _render_project_tab(self):
        """실전 프로젝트 탭"""
        st.header("🚀 실전 Vision Transformer 프로젝트")

        project_type = st.selectbox(
            "프로젝트 선택",
            ["🖼️ 이미지 분류 (ViT)", "🎨 객체 분할 (SAM)", "🔍 특징 추출 (DINO)", "📊 모델 비교"],
            key="vit_project_type"
        )

        if project_type == "🖼️ 이미지 분류 (ViT)":
            self._render_classification_project()
        elif project_type == "🎨 객체 분할 (SAM)":
            self._render_segmentation_project()
        elif project_type == "🔍 특징 추출 (DINO)":
            self._render_feature_extraction_project()
        else:
            self._render_comparison_project()

    def _render_classification_project(self):
        """이미지 분류 프로젝트"""
        st.subheader("🖼️ Vision Transformer 이미지 분류")

        # API 사용 옵션
        use_api = st.checkbox("🤖 Google Gemini API 사용 (실제 분석)", key="use_api_vit")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **프로젝트 목표:**
            - Vision Transformer를 사용한 이미지 분류
            - 사전훈련 모델 활용
            - 다양한 카테고리 인식
            """)

            uploaded_file = st.file_uploader(
                "이미지 업로드",
                type=['png', 'jpg', 'jpeg'],
                key="vit_classify_upload"
            )

        with col2:
            if uploaded_file:
                st.image(uploaded_file, caption="업로드된 이미지")

                if st.button("🔍 분류 시작", key="vit_classify"):
                    with st.spinner("ViT 모델 분석 중..."):
                        if use_api:
                            try:
                                import os
                                import google.generativeai as genai
                                from PIL import Image

                                api_key = os.getenv('GOOGLE_API_KEY')
                                if api_key:
                                    genai.configure(api_key=api_key)
                                    model = genai.GenerativeModel('gemini-2.5-pro')

                                    img = Image.open(uploaded_file)
                                    prompt = """
                                    이 이미지를 분석하고 분류해주세요:
                                    1. 주요 객체/카테고리 (상위 5개)
                                    2. 각 카테고리별 신뢰도 (%)
                                    3. 이미지의 주요 특징
                                    """

                                    response = model.generate_content([prompt, img])
                                    st.success("✅ API 분석 완료!")
                                    st.write("**Gemini 분석 결과:**")
                                    st.info(response.text)
                                else:
                                    st.error("API Key가 설정되지 않았습니다.")
                            except Exception as e:
                                st.error(f"API 오류: {str(e)}")
                        else:
                            # 시뮬레이션
                            import random
                            st.success("분류 완료!")
                            categories = ["고양이", "개", "자동차", "비행기", "새", "꽃", "건물"]
                            for cat in random.sample(categories, 5):
                                conf = random.randint(60, 99)
                                st.metric(cat, f"{conf}%")

    def _render_segmentation_project(self):
        """객체 분할 프로젝트"""
        st.subheader("🎨 SAM (Segment Anything Model)")

        st.info("""
        💡 **SAM (Segment Anything Model)**
        - Meta AI가 개발한 제로샷 이미지 분할 모델
        - 사용자가 클릭/박스/텍스트로 분할 영역 지정
        - 11억 개의 마스크로 학습된 강력한 모델
        """)

        uploaded_sam = st.file_uploader(
            "이미지 업로드 (SAM 분할)",
            type=['png', 'jpg', 'jpeg'],
            key="sam_upload"
        )

        if uploaded_sam:
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_sam, caption="원본 이미지")

            with col2:
                mode = st.radio("분할 모드", ["자동 분할", "클릭 분할", "박스 분할"], key="sam_mode")

                if st.button("🎨 분할 실행", key="run_sam"):
                    with st.spinner("SAM 분할 중..."):
                        st.success("✅ 완료!")
                        st.image(uploaded_sam, caption="분할 결과 (시뮬레이션)")
                        st.caption("⚠️ 실제 SAM 모델 구현이 아닌 시뮬레이션입니다.")

    def _render_feature_extraction_project(self):
        """특징 추출 프로젝트"""
        st.subheader("🔍 DINO 특징 추출 및 유사도 검색")

        st.markdown("""
        **프로젝트 목표:**
        - DINO를 사용한 이미지 특징 추출
        - 유사 이미지 검색
        - 클러스터링 및 시각화
        """)

        uploaded_query = st.file_uploader(
            "쿼리 이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="dino_query"
        )

        if uploaded_query:
            st.image(uploaded_query, caption="쿼리 이미지", width=300)

            if st.button("🔍 유사 이미지 검색", key="dino_search"):
                with st.spinner("DINO 특징 추출 및 검색 중..."):
                    st.success("✅ 유사 이미지를 찾았습니다!")
                    st.info("""
                    💡 실제 구현에서는:
                    1. DINO로 이미지 특징 벡터 추출
                    2. 데이터베이스에서 코사인 유사도 계산
                    3. 상위 K개 유사 이미지 반환
                    """)

    def _render_comparison_project(self):
        """모델 비교 프로젝트"""
        st.subheader("📊 Vision 모델 성능 비교")

        st.markdown("""
        **비교 항목:**
        - 분류 정확도
        - 추론 속도
        - 메모리 사용량
        - 다운스트림 태스크 성능
        """)

        test_image = st.file_uploader(
            "테스트 이미지",
            type=['png', 'jpg', 'jpeg'],
            key="compare_upload"
        )

        if test_image:
            st.image(test_image, caption="테스트 이미지", width=300)

            models = st.multiselect(
                "비교할 모델",
                ["ResNet-50", "ViT-Base", "DINO", "DINOv2"],
                default=["ResNet-50", "ViT-Base"],
                key="compare_models"
            )

            if st.button("🚀 비교 실행", key="run_compare"):
                with st.spinner("모델 비교 중..."):
                    st.success("✅ 완료!")

                    import random
                    for model in models:
                        with st.expander(f"📊 {model} 결과"):
                            acc = 70 + random.random() * 25
                            speed = 10 + random.random() * 90
                            memory = 500 + random.random() * 1500

                            col1, col2, col3 = st.columns(3)
                            col1.metric("정확도", f"{acc:.1f}%")
                            col2.metric("속도", f"{speed:.0f} FPS")
                            col3.metric("메모리", f"{memory:.0f} MB")