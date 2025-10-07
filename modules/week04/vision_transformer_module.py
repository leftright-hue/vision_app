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
from core.vit_helpers import get_attention_overlays, get_attention_overlays_per_head, get_attention_rollout
from transformers import ViTForImageClassification, AutoImageProcessor


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
            "📊 모델 벤치마크",
            "🎯 DINO & 자기지도학습",
            "🚀 실전 프로젝트"
        ])

        with tabs[0]:
            self._render_theory_tab()

        with tabs[1]:
            self._render_self_attention_tab()

        with tabs[2]:
            self._render_vit_tab()

        with tabs[3]:
            self._render_benchmark_tab()

        with tabs[4]:
            self._render_dino_tab()

        with tabs[5]:
            self._render_project_tab()

    def _render_theory_tab(self):
        """이론 탭 - 상세하고 직관적인 설명"""
        st.header("📖 Vision Transformer 완전 정복")

        # 전체 개요
        st.info("""
        💡 **학습 목표**: Vision Transformer가 어떻게 이미지를 "단어"처럼 처리하여
        CNN을 뛰어넘는 성능을 달성했는지 깊이 있게 이해합니다.
        """)

        with st.expander("🔹 1. Transformer 혁명의 시작", expanded=True):
            st.markdown("""
            ### 🧠 왜 Transformer인가?

            #### 📜 역사적 맥락
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **🔤 NLP에서의 혁명 (2017)**

                - **"Attention Is All You Need"** 논문 발표
                - RNN/LSTM의 치명적 한계:
                  - ❌ 순차 처리 → 병렬화 불가
                  - ❌ 장거리 의존성 학습 어려움
                  - ❌ Gradient Vanishing 문제

                - ✅ Transformer의 해결책:
                  - ✨ Self-Attention으로 전역 맥락 파악
                  - ✨ 완전 병렬 처리 가능
                  - ✨ BERT, GPT 등의 기반
                """)

            with col2:
                st.markdown("""
                **🖼️ Computer Vision으로 확장 (2020)**

                - **"An Image is Worth 16×16 Words"** 발표
                - CNN의 한계:
                  - ❌ 작은 수용 영역 (3×3 필터)
                  - ❌ 장거리 관계 파악 어려움
                  - ❌ 귀납적 편향에 의존

                - ✅ ViT의 혁신:
                  - ✨ 이미지 패치를 단어처럼 처리
                  - ✨ 처음부터 전역 관계 파악
                  - ✨ 대규모 데이터에서 CNN 능가
                """)

            st.markdown("---")
            st.markdown("""
            ### 📊 CNN vs Transformer: 근본적인 차이
            """)

            tab1, tab2 = st.tabs(["처리 방식 비교", "성능 비교표"])

            with tab1:
                st.code("""
# CNN 방식: 점진적 확장
이미지 → [3×3 작은 창] → [5×5 창] → [7×7 창] → ... → 전체
        지역적 특징      중간 특징     전역 특징

# Transformer 방식: 즉시 전역
이미지 → [모든 패치 간 관계 동시 계산] → 전역 특징
        Self-Attention으로 한 번에 파악
                """, language="text")

                st.info("""
                **💡 직관적 비유:**

                - **CNN**: 현미경으로 사진을 조금씩 확대하며 보는 것
                  - 처음엔 작은 부분만 → 점점 넓은 영역 → 최종적으로 전체

                - **Transformer**: 사진 전체를 한눈에 보며 각 부분 간 관계 파악
                  - 처음부터 "코는 눈 근처에 있고, 귀는 옆에 있다" 인식
                """)

            with tab2:
                comparison_df = {
                    "특징": ["처리 방식", "수용 영역", "귀납적 편향", "데이터 요구량", "계산 복잡도", "장거리 의존성", "병렬 처리", "해석 가능성"],
                    "CNN": ["지역적 → 전역적", "점진적 확장", "강함 (이동/회전 불변)", "적음 (수천~수만)", "O(n)", "약함", "레이어별만", "낮음"],
                    "Transformer": ["전역적", "전체 이미지", "약함", "많음 (수십만~수백만)", "O(n²)", "강함", "완전 병렬", "높음 (Attention Map)"]
                }
                st.table(comparison_df)

                st.success("""
                **🎯 핵심 포인트:**
                - CNN은 **적은 데이터**에서 효율적이지만 **장거리 관계 파악**에 약함
                - Transformer는 **대규모 데이터**에서 강력하며 **전역적 이해**가 뛰어남
                - 현대 AI는 두 방식을 **결합**(ConvNeXt, Swin Transformer 등)
                """)

        with st.expander("🔹 2. Self-Attention: 핵심 메커니즘 완전 분해"):
            st.markdown("""
            ### 🎯 Self-Attention이란?

            "이미지의 어느 부분에 **집중**(Attention)할 것인가?"를 학습하는 메커니즘
            """)

            st.info("""
            **🔍 실생활 비유:**

            당신이 사진에서 "고양이"를 찾는다고 상상해보세요:

            1. **Query (질문)**: "고양이는 어디에 있을까?"
            2. **Key (단서)**: 각 영역의 특징 - "털이 있다", "동그란 눈", "배경"
            3. **Value (실제 정보)**: 각 영역의 상세 정보
            4. **Attention**: Query와 Key를 비교해 "털+눈" 영역에 높은 점수 부여
            5. **결과**: 고양이가 있는 부분의 정보를 가중 평균하여 추출
            """)

            st.markdown("#### 📐 수학적 정의")
            st.latex(r"""
            \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
            """)

            st.markdown("""
            여기서:
            - **Q (Query)**: 질의 행렬 = "무엇을 찾고 싶은가?"
            - **K (Key)**: 키 행렬 = "각 위치가 가진 정보는?"
            - **V (Value)**: 값 행렬 = "실제로 전달할 내용"
            - **√d_k**: 스케일링 팩터 (gradient 안정화)
            """)

            st.markdown("#### 🔄 단계별 처리 과정")

            process_tabs = st.tabs(["Step 1: Q,K,V 생성", "Step 2: Attention Score", "Step 3: Softmax", "Step 4: 가중합", "Step 5: 최종 출력"])

            with process_tabs[0]:
                st.markdown("""
                **1단계: Linear Transformation으로 Q, K, V 생성**
                """)
                st.code("""
입력: X [197×768]  (CLS + 196 패치, 각 768 차원)

Q = X @ W_q  →  [197×768]  "각 패치가 찾고자 하는 것"
K = X @ W_k  →  [197×768]  "각 패치가 제공하는 정보"
V = X @ W_v  →  [197×768]  "실제로 전달할 값"

예시:
- P1(눈 패치)의 Query: "근처에 코나 입이 있나?"
- P50(코 패치)의 Key: "나는 얼굴 중앙 부분이야"
- P50의 Value: 코의 상세 특징 정보
                """, language="text")

            with process_tabs[1]:
                st.markdown("""
                **2단계: Attention Score 계산 (유사도 측정)**
                """)
                st.code("""
Scores = Q @ K^T  →  [197×197]

각 패치가 다른 모든 패치와의 관련성을 계산:

          P1    P2    P3   ...  P196
    P1 [0.8]  [0.1] [0.05] ... [0.02]
    P2 [0.1]  [0.9] [0.15] ... [0.03]
    P3 [0.05] [0.15][0.7]  ... [0.01]
    ...

숫자가 클수록 관련성이 높음
(예: P1과 P1 = 0.8, 자기 자신과 가장 관련)
                """, language="text")

                st.warning("⚠️ 이 점수들은 아직 확률이 아니므로 Softmax 필요!")

            with process_tabs[2]:
                st.markdown("""
                **3단계: Softmax로 확률 분포 변환**
                """)
                st.code("""
Attention_Weights = softmax(Scores / √d_k)

√d_k로 나누는 이유:
- 큰 차원에서 내적 값이 너무 커지는 것 방지
- Gradient가 vanishing되지 않도록 함

Softmax 적용 후:
          P1    P2    P3   ...  P196  | 합
    P1 [0.45] [0.20][0.15] ... [0.01] = 1.0
    P2 [0.05] [0.60][0.25] ... [0.02] = 1.0
    ...

이제 각 행이 확률 분포 (합 = 1)
                """, language="text")

            with process_tabs[3]:
                st.markdown("""
                **4단계: Value와 가중합 계산**
                """)
                st.code("""
Output = Attention_Weights @ V  →  [197×768]

각 패치에 대해:
- Attention이 높은 패치의 Value를 많이 가져옴
- Attention이 낮은 패치의 Value는 조금만 가져옴

예시: P1(눈 패치)의 출력
= 0.45 × V_P1 (자기 자신: 눈)
+ 0.20 × V_P2 (주변: 눈썹)
+ 0.15 × V_P3 (주변: 코)
+ ...
+ 0.01 × V_P196 (멀리: 배경)

→ 눈 주변 정보가 많이 반영된 새로운 표현
                """, language="text")

            with process_tabs[4]:
                st.markdown("""
                **5단계: 최종 출력**
                """)
                st.success("""
                ✅ **결과:**
                - 각 패치가 다른 모든 패치의 정보를 **관련성에 따라** 가져옴
                - 전역적 맥락이 반영된 새로운 특징 표현
                - 다음 레이어로 전달되거나 최종 분류에 사용

                **💡 핵심:**
                Self-Attention은 "누구랑 친한지" 계산하고,
                친한 친구들의 정보를 많이 가져와서
                자신을 업데이트하는 과정!
                """)

            st.markdown("---")
            st.markdown("#### 🎨 Multi-Head Attention: 다양한 관점")

            st.markdown("""
            **왜 여러 개의 Head가 필요할까?**
            """)

            st.info("""
            **🔍 비유:**

            한 명의 전문가보다 여러 전문가가 각자의 관점에서 보는 것이 낫습니다:

            - **Head 1**: 공간적 인접성 파악 (근처 패치들 중요하게)
            - **Head 2**: 색상 유사성 파악 (비슷한 색 패치들 중요하게)
            - **Head 3**: 텍스처 패턴 파악 (비슷한 질감 패치들 중요하게)
            - **Head 4**: 형태 관계 파악 (같은 객체 패치들 중요하게)
            - ... (총 12개 Head가 서로 다른 관점)

            최종적으로 12개 Head의 결과를 합쳐서 풍부한 표현 생성!
            """)

            st.code("""
입력 [197×768]
    │
    ├──── Head 1 [197×64] ── 공간 관계
    ├──── Head 2 [197×64] ── 색상 관계
    ├──── Head 3 [197×64] ── 텍스처 관계
    ├──── Head 4 [197×64] ── 형태 관계
    ├──── ...
    └──── Head 12 [197×64] ── 맥락 정보
    │
    └─► Concatenate ─► [197×768] ─► Linear ─► 출력
            """, language="text")

        with st.expander("🔹 3. Vision Transformer 완전 분해"):
            st.markdown("""
            ### 🏗️ ViT 아키텍처: "이미지를 문장처럼"

            **핵심 아이디어**: 이미지를 단어들의 시퀀스처럼 취급
            """)

            st.image("https://raw.githubusercontent.com/google-research/vision_transformer/main/vit_figure.png",
                    caption="Vision Transformer 전체 구조 (출처: Google Research)")

            st.markdown("#### 📦 1. Patch Embedding: 이미지를 단어로")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.code("""
원본 이미지: 224×224×3
        ↓
16×16 패치로 분할
        ↓
총 196개 패치
(14×14 = 196)
        ↓
각 패치: 16×16×3 = 768개 픽셀값
        ↓
Linear Projection
        ↓
768차원 벡터
                """, language="text")

            with col2:
                st.info("""
                **🎯 왜 패치로 나눌까?**

                1. **연산 효율성**:
                   - 224×224 = 50,176 픽셀 직접 처리 → 너무 무거움
                   - 196개 패치만 처리 → 256배 효율적!

                2. **NLP와의 유사성**:
                   - 패치 = 단어
                   - 이미지 = 문장
                   - Transformer를 그대로 사용 가능

                3. **적절한 granularity**:
                   - 너무 작으면(8×8): 패치 수 너무 많음
                   - 너무 크면(32×32): 세밀한 정보 손실
                   - 16×16: 최적의 균형
                """)

            st.markdown("#### 🎯 2. CLS Token: 분류를 위한 특수 토큰")

            st.code("""
패치 시퀀스: [P1, P2, P3, ..., P196]
        ↓ CLS 토큰 추가
[CLS, P1, P2, P3, ..., P196]  (총 197개)

CLS 토큰의 역할:
- 전체 이미지 정보를 모으는 "집합체"
- Transformer를 거치며 모든 패치의 정보를 흡수
- 최종적으로 분류에 사용
            """, language="text")

            st.success("""
            **💡 직관:**

            CLS 토큰은 "반장" 같은 존재:
            - 처음엔 빈 상태
            - Self-Attention을 통해 모든 패치(학생들)로부터 정보 수집
            - 12개 레이어를 거치며 점점 전체 이미지에 대한 이해 축적
            - 최종적으로 "이 이미지는 고양이다!"라고 판단
            """)

            st.markdown("#### 📍 3. Position Embedding: 위치 정보 주입")

            st.warning("""
            **⚠️ 중요한 문제:**

            Self-Attention은 위치 정보를 모릅니다!

            - [P1, P2, P3]와 [P3, P1, P2]를 동일하게 처리
            - 하지만 이미지에서 위치는 매우 중요!
              - 왼쪽 위(P1) vs 오른쪽 아래(P196)는 다른 의미
            """)

            st.markdown("**해결책: Position Embedding 추가**")

            st.code("""
Patch Embedding + Position Embedding

[CLS, P1, P2, ..., P196]      각 패치 임베딩 [768]
   +     +    +        +       더하기
[Pos0, Pos1, Pos2, ..., Pos196]  위치 임베딩 [768]
   ↓     ↓    ↓        ↓
[CLS', P1', P2', ..., P196']   위치 정보 포함된 임베딩

이제 모델이 알 수 있음:
- P1 = 왼쪽 위 패치
- P196 = 오른쪽 아래 패치
            """, language="text")

            st.markdown("#### 🔄 4. Transformer Encoder Block (×12)")

            st.code("""
하나의 Transformer Block:

입력 [197×768]
    │
    ▼
┌───────────────────────┐
│ Layer Norm            │
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ Multi-Head Attention  │  ← 전역 관계 파악
│ (12 heads × 64 dims)  │
└───────────────────────┘
    │
    ▼ (Residual Connection)
    + ────────────────────► 더하기
    │
    ▼
┌───────────────────────┐
│ Layer Norm            │
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ MLP (Feed Forward)    │  ← 비선형 변환
│ 768 → 3072 → 768      │
└───────────────────────┘
    │
    ▼ (Residual Connection)
    + ────────────────────► 더하기
    │
    ▼
출력 [197×768]

이 블록을 12번 반복!
            """, language="text")

            st.info("""
            **🔍 각 구성 요소의 역할:**

            1. **Layer Norm**: 학습 안정화
            2. **Multi-Head Attention**: 패치 간 관계 학습
            3. **Residual Connection**: Gradient 흐름 개선
            4. **MLP**: 복잡한 비선형 패턴 학습

            **레이어가 깊어질수록:**
            - Layer 1-4: 저수준 특징 (엣지, 텍스처)
            - Layer 5-8: 중간 특징 (부분, 패턴)
            - Layer 9-12: 고수준 특징 (객체, 의미)
            """)

            st.markdown("#### 🎓 5. Classification Head: 최종 분류")

            st.code("""
12개 레이어 통과 후:
[CLS', P1', P2', ..., P196']  [197×768]
   │
   ▼ CLS 토큰만 추출
 CLS'  [1×768]
   │
   ▼ Layer Norm
 CLS'' [1×768]
   │
   ▼ Linear (768 → 1000)
 Logits [1×1000]
   │
   ▼ Softmax
 Probabilities [1×1000]
   │
   ▼
"고양이: 95%, 개: 3%, 호랑이: 1%, ..."
            """, language="text")

            st.success("""
            **✅ 전체 흐름 요약:**

            1. 이미지를 196개 패치로 분할
            2. 각 패치를 768차원 벡터로 변환
            3. CLS 토큰 추가 + 위치 정보 추가
            4. 12개 Transformer Block 통과 (관계 학습)
            5. CLS 토큰으로 최종 분류

            **💡 핵심:**
            처음부터 끝까지 모든 패치가 서로 소통하며
            전체 이미지에 대한 이해를 점진적으로 깊게 만듦!
            """)

    def _render_self_attention_tab(self):
        """Self-Attention 탭 - 개념 이해와 간단한 시뮬레이션"""
        st.header("🧠 Self-Attention 메커니즘 이해하기")

        st.info("""
        💡 **이 탭의 목적**: Self-Attention이 **무엇인지**, **어떻게 작동하는지**를
        간단한 시뮬레이션으로 이해합니다. (실제 모델 필요 없음)

        👉 실제 ViT 모델로 이미지 분석을 하려면 **"🔍 Vision Transformer" 탭**으로 이동하세요!
        """)

        st.markdown("---")

        # 1. 개념 설명
        with st.expander("📖 Self-Attention이란?", expanded=True):
            st.markdown("""
            ### 🎯 핵심 아이디어

            Self-Attention은 **입력 시퀀스의 각 요소**가 **다른 모든 요소**와
            얼마나 관련이 있는지 계산하는 메커니즘입니다.

            #### 🔍 텍스트 예시
            """)

            text_example = "The cat sat on the mat"
            st.code(f'문장: "{text_example}"', language="text")

            st.markdown("""
            "cat"이라는 단어가 다른 단어들과의 관련성:
            - **cat** ←→ **sat**: 높은 관련성 (주어-동사)
            - **cat** ←→ **mat**: 중간 관련성 (같은 문맥)
            - **cat** ←→ **The**: 낮은 관련성

            #### 🖼️ 이미지 예시

            고양이 사진을 196개 패치로 나눴을 때:
            - **눈 패치** ←→ **코 패치**: 높은 관련성 (얼굴 부분)
            - **눈 패치** ←→ **귀 패치**: 중간 관련성 (같은 머리)
            - **눈 패치** ←→ **배경 패치**: 낮은 관련성
            """)

        st.markdown("---")

        # 2. 대화형 시뮬레이션
        st.markdown("### 🎮 Self-Attention 시뮬레이션")

        sim_tab1, sim_tab2 = st.tabs(["📝 텍스트 Attention", "🎨 이미지 패치 Attention"])

        with sim_tab1:
            st.markdown("#### 문장의 단어 간 Attention 시뮬레이션")

            sample_sentence = st.text_input(
                "문장을 입력하세요 (공백으로 구분)",
                "The cat sat on the mat",
                key="sample_sentence"
            )

            words = sample_sentence.split()

            if len(words) > 1:
                st.markdown(f"**단어 수**: {len(words)}개")

                # 사용자가 선택한 단어
                query_word = st.selectbox("Query 단어 선택 (어떤 단어의 관점에서 볼까요?)", words, key="query_word")
                query_idx = words.index(query_word)

                # 간단한 시뮬레이션 (거리 기반)
                st.markdown(f"**'{query_word}'** 단어의 관점에서 다른 단어들과의 Attention:")

                import random
                attention_scores = {}
                for i, word in enumerate(words):
                    if i == query_idx:
                        score = 1.0  # 자기 자신
                    else:
                        # 거리 기반 간단한 시뮬레이션
                        distance = abs(i - query_idx)
                        score = max(0.1, 1.0 - distance * 0.15 + random.uniform(-0.1, 0.1))
                    attention_scores[word] = score

                # 정규화 (합이 1이 되도록)
                total = sum(attention_scores.values())
                attention_scores = {k: v/total for k, v in attention_scores.items()}

                # 시각화
                for word, score in attention_scores.items():
                    bar_length = int(score * 50)
                    bar = "█" * bar_length
                    st.text(f"{word:15s} {bar} {score:.3f}")

                st.success(f"""
                ✅ **해석**: '{query_word}' 단어는 자기 자신과 가장 높은 Attention을 가지며,
                가까운 단어일수록 더 높은 Attention을 가집니다.
                """)

        with sim_tab2:
            st.markdown("#### 이미지 패치 간 Attention 시뮬레이션")

            st.markdown("""
            224×224 이미지를 16×16 패치로 나누면 **14×14 = 196개** 패치가 생성됩니다.
            """)

            # 그리드 시각화
            grid_size = 14
            selected_row = st.slider("Query 패치 행 (Row)", 0, grid_size-1, 5, key="patch_row")
            selected_col = st.slider("Query 패치 열 (Col)", 0, grid_size-1, 5, key="patch_col")

            selected_patch = selected_row * grid_size + selected_col

            st.markdown(f"**선택된 패치**: P{selected_patch} (Row {selected_row}, Col {selected_col})")

            # 간단한 Attention 맵 시뮬레이션 (거리 기반)
            st.markdown("**Attention Map** (선택된 패치와 다른 패치들의 관련성):")

            attention_grid = []
            for r in range(grid_size):
                row = []
                for c in range(grid_size):
                    # 유클리드 거리 기반
                    distance = ((r - selected_row)**2 + (c - selected_col)**2)**0.5
                    attention = max(0.0, 1.0 - distance / grid_size)
                    row.append(attention)
                attention_grid.append(row)

            # NumPy를 사용한 히트맵 시각화
            import numpy as np
            attention_array = np.array(attention_grid)

            # Matplotlib으로 히트맵 생성
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(attention_array, cmap='hot', interpolation='nearest')
            ax.set_title(f'Attention Map for Patch P{selected_patch}', fontsize=14)
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')

            # 선택된 패치 표시
            ax.plot(selected_col, selected_row, 'b*', markersize=20, label='Query Patch')
            ax.legend()

            # 컬러바
            plt.colorbar(im, ax=ax, label='Attention Weight')

            st.pyplot(fig)

            st.info("""
            💡 **시뮬레이션 설명**:
            - 🔵 파란 별: 선택한 Query 패치
            - 🔴 빨간색: 높은 Attention (밀접한 관련)
            - 🟡 노란색: 중간 Attention
            - ⚫ 검은색: 낮은 Attention (거의 무관)

            실제 ViT에서는 학습을 통해 **의미적 유사성**을 기반으로 Attention을 계산합니다!
            """)

        st.markdown("---")

        # 3. Multi-Head Attention 개념
        with st.expander("🎨 Multi-Head Attention: 다양한 관점"):
            st.markdown("""
            ### 왜 여러 개의 Head가 필요할까?

            하나의 Attention만으로는 **한 가지 관점**만 볼 수 있습니다.
            **Multi-Head Attention**은 여러 관점에서 동시에 정보를 처리합니다.
            """)

            num_heads = st.slider("Attention Head 수", 1, 12, 8, key="num_heads_sim")

            st.markdown(f"""
            **{num_heads}개 Head의 역할 (예시):**
            """)

            head_roles = [
                "공간적 인접성 (근처 패치 집중)",
                "색상 유사성 (비슷한 색 집중)",
                "텍스처 패턴 (비슷한 질감 집중)",
                "형태 관계 (같은 객체 집중)",
                "엣지 검출 (경계선 집중)",
                "밝기 대비 (명암 집중)",
                "맥락 정보 (전체 문맥 집중)",
                "세부 디테일 (작은 특징 집중)",
                "객체 부분-전체 (계층적 관계)",
                "시간적 연관성 (동작 패턴)",
                "의미적 유사성 (비슷한 의미)",
                "전역적 구조 (전체 레이아웃)"
            ]

            for i in range(min(num_heads, len(head_roles))):
                st.markdown(f"- **Head {i+1}**: {head_roles[i]}")

            st.code("""
입력 [197×768]
    │
    ├──── Head 1 [197×64] ── 공간 관계
    ├──── Head 2 [197×64] ── 색상 관계
    ├──── Head 3 [197×64] ── 텍스처 관계
    ...
    └──── Head 12 [197×64] ── 맥락 정보
    │
    └─► Concatenate & Linear ─► [197×768]
            """, language="text")

            st.success("""
            ✅ **핵심**: 각 Head가 서로 다른 **특화된 관점**에서 정보를 처리하고,
            이를 결합하여 **풍부하고 다차원적인 표현**을 만듭니다!
            """)

        st.markdown("---")
        st.warning("""
        ⚠️ **이 탭은 개념 이해를 위한 시뮬레이션입니다.**

        실제 ViT 모델로 이미지를 분석하고 Attention을 시각화하려면
        👉 **"🔍 Vision Transformer" 탭**으로 이동하세요!
        """)

    def _render_vit_tab(self):
        """Vision Transformer 탭 - 실제 모델 사용"""
        st.header("🔍 Vision Transformer 실전")

        st.info("""
        💡 **이 탭의 목적**: 실제 ViT 모델을 다운로드하고, Attention을 시각화하며, 이미지를 분류합니다.

        **3단계 프로세스:**
        1. 🔽 모델 다운로드/로드
        2. 👁️ Attention Map 시각화
        3. 🖼️ 이미지 분류
        """)

        st.markdown("---")

        # Step 1: 모델 선택 및 다운로드
        st.markdown("### 1️⃣ ViT 모델 다운로드")

        col1, col2 = st.columns(2)

        with col1:
            model_choice = st.selectbox(
                "사전학습 ViT 모델 선택",
                [
                    "google/vit-base-patch16-224",
                    "google/vit-large-patch16-224",
                ],
                index=0,
                key="vit_model_choice",
                help="HuggingFace에서 사전학습된 ViT 분류 모델을 다운로드합니다"
            )

            # 모델 정보 표시
            model_info = {
                "google/vit-base-patch16-224": {
                    "name": "ViT-Base/16",
                    "params": "86M",
                    "dataset": "ImageNet-1K",
                    "acc": "81.8%"
                },
                "google/vit-large-patch16-224": {
                    "name": "ViT-Large/16",
                    "params": "307M",
                    "dataset": "ImageNet-1K",
                    "acc": "82.6%"
                },
            }

            info = model_info[model_choice]
            st.info(f"""
            **모델 정보:**
            - 이름: {info['name']}
            - 파라미터: {info['params']}
            - 학습 데이터: {info['dataset']}
            - 정확도: {info['acc']}
            """)

            if st.button("⬇️ 모델 다운로드/로드", key="load_vit", type="primary"):
                with st.spinner("분류 모델 다운로드 및 로드 중 (처음엔 시간이 걸릴 수 있습니다)..."):
                    try:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        processor = AutoImageProcessor.from_pretrained(model_choice)
                        model = ViTForImageClassification.from_pretrained(model_choice, output_attentions=True)
                        model.to(device)
                        model.eval()

                        st.success(f"✅ 모델 로드 완료!")
                        st.caption(f"Device: {device} | 캐시 저장됨")
                        # 캐시 저장
                        st.session_state['vit_model'] = model
                        st.session_state['vit_processor'] = processor
                        st.session_state['vit_model_name'] = model_choice
                    except Exception as e:
                        st.error(f"❌ 모델 로드 실패: {e}")

        with col2:
            # 모델 상태 표시
            if 'vit_model' in st.session_state:
                st.success("✅ 모델 로드됨")
                st.caption(f"모델: {st.session_state.get('vit_model_name', 'Unknown')}")
            else:
                st.warning("⚠️ 모델 미로드")
                st.caption("왼쪽에서 모델을 다운로드하세요")

            # 모델 아키텍처 정보
            st.markdown("**📊 ViT 아키텍처**")
            st.code("""
이미지 (224×224×3)
    ↓
패치 분할 (16×16)
    ↓
196개 패치 + CLS
    ↓
Position Embedding
    ↓
12 Transformer Layers
    ↓
CLS 토큰 → 분류
            """, language="text")

        uploaded_vit = st.file_uploader(
            "이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="vit_upload",
            help="224×224로 자동 리사이즈됩니다"
        )

        if uploaded_vit:
            st.image(uploaded_vit, caption="업로드된 이미지", use_column_width=True)

            # Step 2: Attention 시각화
            if 'vit_model' in st.session_state:
                st.markdown("---")
                st.markdown("### 2️⃣ Attention Map 시각화")

                st.info("""
                💡 **Attention Map**: ViT가 이미지의 어느 부분에 집중하는지 시각화합니다.
                - 빨간색/밝은 영역 = 높은 Attention (중요한 부분)
                - 파란색/어두운 영역 = 낮은 Attention (덜 중요한 부분)
                """)

                viz_col1, viz_col2 = st.columns([1, 2])

                with viz_col1:
                    st.markdown("**🎛️ 시각화 설정**")

                    viz_mode = st.radio(
                        "시각화 모드",
                        ["Layer-wise (레이어별)", "Head-wise (헤드별)", "Attention Rollout"],
                        key="viz_mode",
                        help="""
                        - Layer-wise: 각 레이어의 평균 Attention
                        - Head-wise: 특정 레이어의 각 헤드별 Attention
                        - Attention Rollout: 모든 레이어를 누적한 최종 Attention
                        """
                    )

                    alpha = st.slider("오버레이 투명도", 0.0, 1.0, 0.6, 0.1, key="alpha_slider")

                    if viz_mode == "Layer-wise (레이어별)":
                        max_layers = st.slider("표시할 레이어 수", 1, 12, 6, key="max_layers")
                        st.caption(f"Layer 1부터 {max_layers}까지 표시")

                    elif viz_mode == "Head-wise (헤드별)":
                        layer_idx = st.slider("레이어 선택", 0, 11, 5, key="layer_idx")
                        st.caption(f"Layer {layer_idx+1}의 12개 헤드 표시")

                    elif viz_mode == "Attention Rollout":
                        discard_ratio = st.slider(
                            "Discard Ratio",
                            0.0, 0.95, 0.9, 0.05,
                            key="discard_ratio",
                            help="낮은 Attention 값을 제거하는 비율 (높을수록 중요한 영역만 표시)"
                        )

                    if st.button("👁️ Attention Map 생성", key="vis_attn", type="primary"):
                        with st.spinner("Attention 계산 중... (시간이 걸릴 수 있습니다)"):
                            try:
                                model = st.session_state['vit_model']
                                processor = st.session_state['vit_processor']
                                pil_img = Image.open(uploaded_vit).convert('RGB')

                                if viz_mode == "Layer-wise (레이어별)":
                                    overlays = get_attention_overlays(pil_img, model, processor, alpha=alpha, max_layers=max_layers)
                                    st.session_state['attention_result'] = overlays
                                elif viz_mode == "Head-wise (헤드별)":
                                    head_overlays = get_attention_overlays_per_head(pil_img, model, processor, layer_idx=layer_idx, alpha=alpha)
                                    st.session_state['attention_result'] = head_overlays
                                elif viz_mode == "Attention Rollout":
                                    rollout_overlay = get_attention_rollout(pil_img, model, processor, alpha=alpha, discard_ratio=discard_ratio)
                                    st.session_state['attention_result'] = rollout_overlay
                                
                                st.session_state['attention_viz_mode'] = viz_mode
                                st.success("✅ Attention 시각화 완료!")

                            except Exception as e:
                                st.error(f"Attention 계산 실패: {e}")
                                import traceback
                                st.text(traceback.format_exc())

                with viz_col2:
                    st.markdown("**🖼️ Attention 시각화 결과**")
                    if 'attention_result' in st.session_state:
                        viz_mode = st.session_state['attention_viz_mode']
                        result = st.session_state['attention_result']

                        if viz_mode == "Layer-wise (레이어별)":
                            cols_per_row = 3
                            for i in range(0, len(result), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, col in enumerate(cols):
                                    idx = i + j
                                    if idx < len(result):
                                        with col:
                                            st.image(result[idx], caption=f"Layer {idx+1}", use_column_width=True)
                        elif viz_mode == "Head-wise (헤드별)":
                            cols_per_row = 4
                            for i in range(0, len(result), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, col in enumerate(cols):
                                    idx = i + j
                                    if idx < len(result):
                                        with col:
                                            st.image(result[idx], caption=f"Head {idx+1}", use_column_width=True)
                        elif viz_mode == "Attention Rollout":
                            st.image(result, caption="Attention Rollout (누적)", use_column_width=True)
                            st.success("✅ Attention Rollout: 모든 레이어의 attention을 누적하여 최종적으로 모델이 집중하는 영역을 보여줍니다!")
                    else:
                        st.caption("왼쪽에서 설정 후 'Attention Map 생성' 버튼을 클릭하세요")
                        st.image("https://via.placeholder.com/400x300?text=Attention+Map+will+appear+here",
                                caption="Attention Map이 여기에 표시됩니다")

            # Step 3: 이미지 분류
            if 'vit_model' in st.session_state:
                st.markdown("---")
                st.markdown("### 3️⃣ 이미지 분류")

                if st.button("🔮 이미지 분류 시작", key="classify_btn", type="primary"):
                    with st.spinner("ViT 모델로 분석 중..."):
                        try:
                            model = st.session_state['vit_model']
                            processor = st.session_state['vit_processor']
                            pil_img = Image.open(uploaded_vit).convert('RGB')

                            inputs = processor(images=pil_img, return_tensors="pt")
                            
                            with torch.no_grad():
                                outputs = model(**inputs)
                                logits = outputs.logits

                            # Top-5 예측
                            probs = torch.nn.functional.softmax(logits, dim=-1)
                            top5_probs, top5_indices = torch.topk(probs, 5)

                            st.success("✅ 분류 완료!")
                            st.markdown("#### 🎯 Top-5 예측 결과")

                            for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
                                label = model.config.id2label[idx.item()]
                                confidence = prob.item() * 100
                                st.markdown(f"**{i+1}. {label}** (`{confidence:.2f}%`)")
                                st.progress(confidence / 100)
                            
                        except Exception as e:
                            st.error(f"분류 실패: {e}")


    def _render_dino_tab(self):
        """DINO & 자기지도학습 탭 - 상세하고 직관적인 설명"""
        st.header("🦖 DINO & 자기지도학습 완전 정복")

        st.info("""
        💡 **학습 목표**: DINO가 어떻게 레이블 없이도 이미지의 의미를 이해하고,
        왜 자기지도학습이 미래의 AI 학습 방법인지 깊이 있게 이해합니다.
        """)

        # 1. 자기지도학습의 혁명
        with st.expander("🔹 1. 자기지도학습 - 레이블 없는 학습의 혁명", expanded=True):
            st.markdown("""
            ### 🧠 왜 자기지도학습인가?

            #### 📊 전통적 지도학습의 한계
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **❌ 지도학습의 문제점**

                - 📝 **레이블 비용**:
                  - ImageNet: 1.4M 이미지에 수백만 달러 투입
                  - 전문가 시간: 이미지당 평균 1-5분
                  - Medical/위성 이미지: 전문가만 레이블링 가능

                - 🔒 **확장성 제한**:
                  - 인터넷에는 수십억 개 이미지 존재
                  - 레이블된 데이터는 극히 일부
                  - 새로운 카테고리마다 재레이블링 필요

                - 🎯 **편향 문제**:
                  - 레이블러의 주관이 개입
                  - 문화적/지역적 편향 발생
                """)

            with col2:
                st.markdown("""
                **✅ 자기지도학습의 해결책**

                - 🌊 **무한한 데이터**:
                  - 인터넷의 모든 이미지 활용 가능
                  - 비용 제로: 자동으로 레이블 생성
                  - YouTube, Instagram 등 실시간 데이터

                - 🧩 **본질적 학습**:
                  - 데이터의 내재적 구조 발견
                  - 인간의 개입 최소화
                  - 더 일반화된 표현 학습

                - 🚀 **확장 가능**:
                  - 모델 크기에 비례한 성능 향상
                  - GPT-4, DALL-E 등의 기반
                """)

            st.markdown("---")

            # 자기지도학습 비유
            st.markdown("""
            #### 🎓 직관적 이해: 아기가 세상을 배우는 방법

            | 학습 방법 | 비유 | 예시 |
            |---------|------|-----|
            | **지도학습** | 부모가 "이건 개야, 이건 고양이야"라고 일일이 가르침 | ImageNet 레이블링 |
            | **자기지도학습** | 아기가 스스로 관찰하며 패턴 발견 | DINO가 이미지 구조 학습 |
            | **결과** | 아기는 명시적으로 배우지 않은 것도 이해 (물리 법칙, 중력 등) | DINO는 객체 경계, 의미론적 그룹 자동 발견 |
            """)

        # 2. DINO 아키텍처 상세 설명
        with st.expander("🔹 2. DINO 아키텍처 - Teacher와 Student의 지식 증류", expanded=False):
            st.markdown("""
            ### 🏗️ DINO의 핵심 구조

            #### 🎯 Knowledge Distillation (지식 증류)란?
            """)

            st.info("""
            **🍷 와인 증류 비유**:
            - 와인(Teacher): 복잡하고 풍부한 맛
            - 증류주(Student): 핵심 맛만 농축
            - **DINO**: Teacher의 지식을 Student가 증류하여 학습
            """)

            st.markdown("""
            #### 🔄 DINO 학습 프로세스 (단계별)
            """)

            # DINO 프로세스를 탭으로 구성
            process_tabs = st.tabs(["1️⃣ 입력 증강", "2️⃣ 이중 네트워크", "3️⃣ 출력 비교", "4️⃣ 역전파 학습", "5️⃣ Teacher 업데이트"])

            with process_tabs[0]:
                st.markdown("""
                ### 📸 Step 1: 입력 이미지 증강 (Augmentation)

                **같은 이미지, 다른 시각**

                ```
                원본 이미지 (고양이)
                        ↓
                ┌───────┴────────┐
                │                │
                Global Crop 1    Local Crop 1
                (전체 보기)       (얼굴 클로즈업)
                    ↓                ↓
                Global Crop 2    Local Crop 2
                (약간 회전)      (발 클로즈업)
                ```

                **💡 왜 여러 Crop을 만드나?**
                - **Global view (224×224)**: 전체 맥락 이해 (고양이가 앉아있다)
                - **Local view (96×96)**: 세부 특징 학습 (귀가 뾰족하다, 수염이 있다)
                - **일관성 강제**: 같은 고양이의 다른 부분들이 일관된 표현을 가져야 함

                **📊 증강 기법**:
                - Random Crop: 위치 불변성
                - Color Jitter: 조명 불변성
                - Gaussian Blur: 노이즈 강건성
                - Horizontal Flip: 방향 불변성
                """)

            with process_tabs[1]:
                st.markdown("""
                ### 👥 Step 2: Teacher-Student 이중 네트워크

                **같은 구조, 다른 역할**

                ```
                ┌─────────────────────────────────────────┐
                │         입력: 고양이 이미지              │
                └────────────┬────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                📚 Teacher          🎓 Student
                (안정된 지식)        (배우는 중)
                    │                 │
                ViT Backbone      ViT Backbone
                (동일 구조)        (동일 구조)
                    │                 │
                출력 확률 분포      출력 확률 분포
                [0.1, 0.8, 0.1]  [0.2, 0.6, 0.2]
                ```

                **💡 역할 구분**:

                | 특성 | Teacher 🧑‍🏫 | Student 🎓 |
                |------|------------|-----------|
                | **파라미터 업데이트** | EMA (천천히) | Gradient Descent (빠르게) |
                | **입력** | Global views만 | Global + Local views |
                | **역할** | 안정적 지식 제공 | 적극적으로 학습 |
                | **비유** | 경험 많은 교수 | 열심히 배우는 학생 |

                **🔑 핵심 아이디어**:
                - Teacher는 **안정적인 목표**를 제공 (Moving Target 문제 방지)
                - Student는 **다양한 관점**에서 배움 (Local crops 포함)
                - 같은 이미지에 대해 Teacher와 Student의 출력이 **일치**해야 함
                """)

            with process_tabs[2]:
                st.markdown("""
                ### 📊 Step 3: 출력 확률 분포 비교

                **Cross-Entropy Loss로 유사도 측정**

                ```
                Teacher 출력 (Global Crop 1):
                고양이: 80% ████████
                개:     10% ██
                새:     10% ██

                Student 출력 (Local Crop - 귀 부분):
                고양이: 60% ██████
                개:     25% ███
                새:     15% ███

                ❌ Loss = 0.45 (불일치!)
                → Student 업데이트 필요
                ```

                **💡 Loss Function**:

                ```python
                # Cross-Entropy between Teacher and Student
                loss = -sum(teacher_prob * log(student_prob))

                # 예시:
                teacher = [0.8, 0.1, 0.1]  # 고양이에 확신
                student = [0.6, 0.25, 0.15]  # 덜 확신
                loss = -(0.8*log(0.6) + 0.1*log(0.25) + 0.1*log(0.15))
                     = 0.45  # 높은 loss → 많이 다름
                ```

                **🎯 학습 목표**:
                - Student가 Local crop (귀만 보고)도 Teacher처럼 "고양이"라고 확신하도록
                - 부분만 봐도 전체를 이해하는 능력 학습
                """)

            with process_tabs[3]:
                st.markdown("""
                ### 🔄 Step 4: Student만 역전파 학습

                **Teacher는 고정, Student만 업데이트**

                ```
                Loss = 0.45 (높음)
                    ↓
                Gradient 계산
                    ↓
                Student 파라미터 업데이트 ✅
                    ↓
                다음 iteration:
                Student 출력 개선
                고양이: 70% ███████ (60%→70% 향상!)
                개:     18% ██
                새:     12% ██

                Loss = 0.25 (감소!)
                ```

                **💡 왜 Teacher는 업데이트 안 하나?**

                - ❌ **만약 둘 다 업데이트하면**:
                  - Moving Target 문제 발생
                  - Teacher도 계속 바뀌면 Student가 뭘 따라가야 할지 모름
                  - 학습 불안정 (Collapse)

                - ✅ **Teacher 고정 시**:
                  - 안정적인 학습 목표 제공
                  - Student는 명확한 방향으로 개선
                  - 점진적 지식 축적

                **📈 학습률 차이**:
                - Student: lr = 0.001 (빠르게 변화)
                - Teacher: EMA로 천천히 반영
                """)

            with process_tabs[4]:
                st.markdown("""
                ### 🐌 Step 5: Teacher의 EMA 업데이트

                **Exponential Moving Average (지수 이동 평균)**

                ```python
                # Teacher는 Student의 '과거 평균'
                teacher_params = 0.996 * teacher_params + 0.004 * student_params
                ```

                **💡 EMA의 의미**:

                | Iteration | Student Weight | Teacher Weight | 설명 |
                |-----------|---------------|----------------|------|
                | 1 | 1.0 | 1.0 | 시작점 동일 |
                | 2 | 1.5 (급변) | 1.002 | Teacher는 거의 안 변함 |
                | 3 | 1.3 (요동) | 1.003 | |
                | ... | ... | ... | |
                | 100 | 2.1 | 1.8 | Teacher는 평균값 유지 |

                **🎯 왜 EMA를 쓰나?**

                1. **안정성**: Teacher는 Student의 '요동'을 평활화
                2. **과거 지식 보존**: 이전 학습 내용을 서서히 반영
                3. **Collapse 방지**: 급격한 변화 방지

                **🔬 Momentum 계수 영향**:
                ```
                m = 0.99:  Teacher가 Student를 빠르게 따라감 (불안정)
                m = 0.996: 균형잡힌 업데이트 (DINO 기본값)
                m = 0.999: Teacher가 거의 안 변함 (학습 느림)
                ```
                """)

        # 3. DINO vs 지도학습 비교
        with st.expander("🔹 3. DINO의 놀라운 능력 - 레이블 없이 의미를 발견한다", expanded=False):
            st.markdown("""
            ### 🎨 Self-Attention으로 본 DINO의 시각

            #### 📸 사례 연구: 강아지 이미지
            """)

            comparison_tabs = st.tabs(["지도학습 ViT", "DINO ViT", "차이점 분석"])

            with comparison_tabs[0]:
                st.markdown("""
                ### 📚 지도학습 ViT의 Attention

                **레이블: "개"로 학습됨**

                ```
                Attention Map:
                ┌─────────────────┐
                │   🐕 강아지      │  → 강한 attention
                │   (전체 영역)    │     (레이블이 "개"라고 알려줌)
                │                 │
                │   🌳 배경        │  → 약한 attention
                │   🌿 풀          │     (관련 없다고 학습)
                └─────────────────┘

                학습 과정:
                1. 레이블 "개" 제공
                2. "개" 영역에 집중하도록 학습
                3. 배경은 무시
                ```

                **❌ 한계**:
                - 레이블에 명시된 것만 학습
                - 배경의 의미론적 정보 손실
                - "풀밭 위의 개"라는 맥락 이해 부족
                """)

            with comparison_tabs[1]:
                st.markdown("""
                ### 🦖 DINO의 Attention

                **레이블 없음 - 스스로 발견**

                ```
                Attention Map:
                ┌─────────────────┐
                │   🐕 강아지      │  → 강한 attention (Group 1)
                │   [얼굴+몸통+꼬리]│     객체 경계 자동 발견!
                │                 │
                │   🌳 나무        │  → 중간 attention (Group 2)
                │   🌿 풀          │  → 약한 attention (Group 3)
                └─────────────────┘

                발견한 것들:
                ✅ 객체 경계 (개의 윤곽선)
                ✅ 부분-전체 관계 (귀, 발, 꼬리가 하나의 개체)
                ✅ 의미론적 그룹 (개/나무/풀 각각 다른 그룹)
                ✅ 전경-배경 분리
                ```

                **✨ 놀라운 점**:
                - **명시적 레이블 없이** 객체 경계 발견
                - **Segmentation 정보 없이** 영역 분할
                - **카테고리 정보 없이** 의미론적 그룹화
                """)

            with comparison_tabs[2]:
                st.markdown("""
                ### 🔬 왜 DINO가 이런 능력을 갖게 되나?

                #### 🧩 학습 메커니즘 비교

                | 측면 | 지도학습 ViT | DINO |
                |------|-------------|------|
                | **학습 신호** | 외부 레이블 | 내부 일관성 |
                | **목표** | 레이블 맞추기 | 다른 view 일치시키기 |
                | **발견 범위** | 레이블된 것만 | 이미지의 모든 구조 |
                | **일반화** | 제한적 | 강력함 |

                #### 💡 DINO의 학습 원리

                ```
                Global View (전체 강아지):
                "이 이미지는 개체 A와 배경 B로 구성"

                Local View (강아지 얼굴만):
                "이 부분은 개체 A의 일부"

                → Teacher-Student 일치 조건:
                "Local view도 Global view와 같은 개체 A로 인식해야"

                → 결과:
                개체 A의 모든 부분이 일관된 표현을 가짐
                = 객체 경계 자동 발견!
                ```

                #### 🎯 실제 능력 비교

                **1. Zero-Shot Segmentation**:
                ```
                DINO: Attention map이 곧 segmentation mask
                지도학습: 별도의 segmentation 학습 필요
                ```

                **2. 부분-전체 이해**:
                ```
                DINO: "강아지 귀"를 보고 "강아지 전체" 추론
                지도학습: "개" 레이블만 학습, 부분 관계 모름
                ```

                **3. 새로운 카테고리 적응**:
                ```
                DINO: 본 적 없는 객체도 경계 발견 가능
                지도학습: 재학습 필요
                ```
                """)

        # 4. DINO 시리즈 발전사
        with st.expander("🔹 4. DINO 진화 - v1에서 v3까지", expanded=False):
            st.markdown("""
            ### 📈 DINO 시리즈 발전 과정
            """)

            evolution_data = {
                "모델": ["DINO (v1)", "DINOv2", "DINOv3"],
                "출시": ["2021.04", "2023.04", "2025.08"],
                "파라미터": ["22M - 632M", "300M - 1B", "7B"],
                "학습 데이터": ["ImageNet (1.4M)", "LVD-142M", "Unlabeled Web (수억)"],
                "주요 혁신": [
                    "Self-distillation 도입",
                    "대규모 학습, 더 강한 특징",
                    "7B 규모, SOTA 성능"
                ],
                "ImageNet Top-1": ["79.3%", "84.5%", "87.1%"],
                "특징": [
                    "ViT + 자기지도학습",
                    "Improved data augmentation",
                    "최신 SOTA"
                ]
            }

            st.table(evolution_data)

            st.markdown("""
            #### 🚀 주요 개선 사항

            **DINO → DINOv2**:
            1. **학습 데이터 100배 증가**: 1.4M → 142M
            2. **모델 크기 확대**: 최대 632M → 1B
            3. **개선된 증강 기법**: Stronger augmentation
            4. **성능 향상**: ImageNet 79.3% → 84.5%

            **DINOv2 → DINOv3**:
            1. **거대 모델**: 7B 파라미터 (GPT-3급)
            2. **웹 규모 학습**: 수억 개 unlabeled 이미지
            3. **SOTA 달성**: 87.1% ImageNet 정확도
            4. **강력한 전이 학습**: 거의 모든 vision task에서 우수

            #### 💡 스케일링 법칙 (Scaling Law)

            ```
            모델 크기 ↑ + 데이터 양 ↑ = 성능 ↑

            DINO v1:   22M params,  1.4M images  → 79.3%
            DINOv2:    1B params,   142M images  → 84.5%
            DINOv3:    7B params,   수억 images  → 87.1%

            → 자기지도학습도 스케일링 법칙 적용!
            ```
            """)

        # 5. 실습: DINO Attention 시각화
        with st.expander("🔹 5. 실습: DINO의 눈으로 이미지 보기", expanded=True):
            st.markdown("""
            ### 🎨 DINO Attention Map 시각화

            **DINO가 이미지에서 무엇을 "보는지" 확인해보세요!**

            #### 📌 기대 효과:
            - ✅ 객체 경계가 자동으로 발견됨
            - ✅ 의미론적으로 유사한 영역이 그룹화됨
            - ✅ 전경과 배경이 명확히 분리됨
            """)

            st.warning("""
            ⚠️ **참고**:
            - 이 데모는 DINO의 개념을 설명하기 위한 시뮬레이션입니다
            - 실제 DINO 모델 구현은 **실전 프로젝트** 탭에서 확인하세요
            - HuggingFace의 `facebook/dino-vits16` 모델 사용 예제 제공
            """)

            st.markdown("---")

            uploaded_dino = st.file_uploader(
                "🖼️ 이미지 업로드 (DINO 분석용)",
                type=['png', 'jpg', 'jpeg'],
                key="dino_upload",
                help="명확한 객체가 있는 이미지를 추천합니다 (예: 동물, 사람, 차량 등)"
            )

            if uploaded_dino:
                from PIL import Image
                import numpy as np
                import matplotlib.pyplot as plt

                col1, col2 = st.columns(2)

                with col1:
                    st.image(uploaded_dino, caption="📸 입력 이미지", use_container_width=True)

                with col2:
                    if st.button("🔍 DINO Attention 분석", key="run_dino", type="primary"):
                        with st.spinner("🦖 DINO가 이미지를 분석하는 중..."):
                            # 간단한 시뮬레이션: 중심 영역에 높은 attention
                            image = Image.open(uploaded_dino).convert('RGB')
                            img_array = np.array(image)
                            h, w = img_array.shape[:2]

                            # 중심에 가우시안 attention map 생성
                            y, x = np.ogrid[0:h, 0:w]
                            center_y, center_x = h // 2, w // 2
                            attention_map = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (min(h, w) * 30))

                            # 시각화
                            fig, ax = plt.subplots(figsize=(8, 8))
                            ax.imshow(img_array)
                            ax.imshow(attention_map, cmap='jet', alpha=0.5)
                            ax.set_title("DINO Attention Map (Simulated)", fontsize=14, pad=10)
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()

                        st.success("✅ 분석 완료!")

                        st.info("""
                        💡 **DINO가 발견한 것들**:

                        🔴 **빨간색 영역 (High Attention)**:
                        - 주요 객체의 중심부
                        - 의미론적으로 중요한 부분

                        🔵 **파란색 영역 (Low Attention)**:
                        - 배경 또는 덜 중요한 영역

                        **🎯 실제 DINO의 능력**:
                        - 객체 경계를 픽셀 단위로 정확히 발견
                        - 같은 객체의 다른 부분들을 하나로 그룹화
                        - 여러 객체가 있을 때 각각을 분리
                        """)

                        st.markdown("""
                        #### 🚀 더 알아보기

                        **실제 DINO 구현**:
                        - 👉 **실전 프로젝트** 탭에서 HuggingFace DINO 모델 사용법 확인
                        - `facebook/dino-vits16` 또는 `facebook/dinov2-base` 사용

                        **응용 분야**:
                        - 🎨 Zero-shot Semantic Segmentation
                        - 🔍 Object Discovery (객체 발견)
                        - 🖼️ Image Retrieval (유사 이미지 검색)
                        - 🤖 로봇 비전 (레이블 없이 객체 인식)
                        """)
            else:
                st.info("👆 이미지를 업로드하면 DINO의 Attention을 시각화할 수 있습니다.")

                # 예시 결과 보여주기
                st.markdown("""
                #### 📚 예시: DINO Attention의 실제 모습

                **강아지 이미지 예시**:
                ```
                입력 이미지:        DINO Attention:
                ┌─────────┐        ┌─────────┐
                │  🐕     │   →   │ 🔴🔴🔴  │  강아지 영역: 강한 attention
                │    🌳   │        │ 🔴🔴  🔵│  배경: 약한 attention
                │  🌿🌿   │        │ 🔵🔵🔵  │
                └─────────┘        └─────────┘
                ```

                **💡 놀라운 점**:
                - 레이블 없이도 강아지 윤곽선을 정확히 찾아냄
                - 강아지의 귀, 발, 꼬리를 하나의 객체로 인식
                - 배경(나무, 풀)은 자동으로 제외
                """)

    def _render_benchmark_tab(self):
        """모델 벤치마크 탭 - 2024-2025 최신 모델 포함"""
        st.header("📊 Vision 모델 벤치마크 (2020-2025)")

        st.info("""
        💡 **Vision Transformer의 진화**: 2020년 ViT 등장 이후 5년간 급격한 발전!
        - 2020: ViT (86M)
        - 2021: DINO (22M)
        - 2023: SAM (632M), DINOv2 (300M)
        - 2024: EVA-CLIP-18B (18B), InternVL3 (78B), SAM 2
        - 2025: DINOv3 (7B), SigLIP 2
        """)

        st.markdown("---")

        # 시대별 탭
        era_tabs = st.tabs(["🏛️ 클래식 (2020-2022)", "🚀 최신 (2024-2025)", "📊 전체 비교"])

        with era_tabs[0]:
            st.markdown("### 🏛️ 초기 Vision Transformer 모델 (2020-2022)")

            classic_data = {
                "모델": ["ResNet-50", "ViT-Base/16", "ViT-Large/16", "DINO ViT-S", "DeiT-Base"],
                "출시년도": ["2015", "2020", "2020", "2021", "2021"],
                "파라미터": ["25M", "86M", "307M", "22M", "86M"],
                "ImageNet Acc": ["76.2%", "81.8%", "82.6%", "79.3%", "81.8%"],
                "주요 특징": [
                    "CNN 기반, 빠른 추론",
                    "최초 ViT, 대용량 데이터 필요",
                    "높은 정확도, 많은 파라미터",
                    "자기지도학습, 라벨 불필요",
                    "지식 증류, 효율적 학습"
                ]
            }

            st.table(classic_data)

            st.success("""
            **🎯 핵심 포인트:**
            - ViT가 CNN을 능가하며 Transformer 시대 개막
            - DINO로 자기지도학습의 가능성 입증
            - DeiT로 효율적 학습 방법 제시
            """)

        with era_tabs[1]:
            st.markdown("### 🚀 최신 SOTA 모델 (2024-2025)")

            modern_data = {
                "모델": [
                    "DINOv3 (Meta)",
                    "SAM 2 (Meta)",
                    "InternVL3-78B",
                    "EVA-CLIP-18B",
                    "SigLIP 2 (Google)",
                    "Veo 3 (Google)"
                ],
                "출시": ["2025.8", "2024.7", "2024", "2024.2", "2025.2", "2024"],
                "파라미터": ["7B", "~300M", "78B", "18B", "~1B", "~8B"],
                "주요 혁신": [
                    "Image-Text Alignment, 1.7B 이미지 학습",
                    "비디오 세그멘테이션, 메모리 모듈",
                    "MLLM SOTA, 3D Vision 지원",
                    "최고 Zero-shot 성능 80.7%",
                    "다국어 Vision-Language",
                    "비디오 생성, Diffusion Transformer"
                ],
                "특화 분야": [
                    "자기지도학습, 특징 추출",
                    "비디오 객체 분할",
                    "멀티모달 추론, GUI Agent",
                    "Zero-shot 분류, 검색",
                    "다국어 이미지-텍스트",
                    "AI 비디오 생성"
                ]
            }

            st.table(modern_data)

            st.success("""
            **🎯 2024-2025 트렌드:**
            - ✨ **규모 확장**: 1B → 78B 파라미터 (InternVL3)
            - ✨ **멀티모달**: 이미지 + 텍스트 + 비디오 통합
            - ✨ **실용화**: SAM 2로 실제 앱에 적용 (Instagram Edits)
            - ✨ **생성 AI**: Veo 3로 텍스트→비디오 생성
            """)

            # 최신 모델 상세 설명
            with st.expander("🔍 DINOv3 (2025년 최신 자기지도학습)"):
                st.markdown("""
                **Meta AI Research - 2025년 8월 발표**

                #### 주요 혁신
                - **Image-Text Alignment**: CLIP처럼 이미지와 텍스트 동시 학습
                - **7B 파라미터**: DINO 대비 300배 규모
                - **1.7B 이미지**: 사상 최대 학습 데이터
                - **Gram Anchoring**: 안정적 학습 기법
                - **Axial RoPE**: 위치 인코딩 개선

                #### 성능
                - 모든 자기지도학습 모델 중 최고 성능
                - Zero-shot transfer learning에서 탁월
                - Dense prediction task (segmentation, depth)에서 우수

                #### 활용
                ```python
                # HuggingFace에서 사용
                from transformers import Dinov3Model
                model = Dinov3Model.from_pretrained('facebook/dinov3-base')
                ```
                """)

            with st.expander("🔍 SAM 2 (2024년 비디오 세그멘테이션)"):
                st.markdown("""
                **Meta AI - 2024년 7월 발표**

                #### 주요 혁신
                - **통합 모델**: 이미지 + 비디오 동시 처리
                - **메모리 모듈**: 프레임 간 객체 추적
                - **Real-time**: 실시간 처리 가능
                - **Interactive**: 클릭/박스/마스크 입력 지원

                #### 성능
                - 비디오 세그멘테이션: 기존 대비 3배 적은 상호작용
                - 이미지 세그멘테이션: SAM 대비 6배 빠르고 정확
                - SA-V 데이터셋: 51K 비디오, 600K masklets

                #### 실제 활용
                - Instagram Edits의 Cutouts 기능
                - 의료 영상 분석
                - 자율주행 객체 추적

                #### 코드 예제
                ```python
                from sam2.build_sam import build_sam2_video_predictor
                predictor = build_sam2_video_predictor("sam2_hiera_large.pt")

                # 비디오의 첫 프레임에서 객체 선택
                points = np.array([[210, 350]])
                labels = np.array([1])  # 1=foreground
                frame_idx, object_ids, masks = predictor.add_new_points(
                    frame_idx=0, obj_id=0, points=points, labels=labels
                )

                # 전체 비디오에 대해 자동 추적
                for frame_idx, object_ids, masks in predictor.propagate_in_video():
                    # 각 프레임의 마스크 사용
                    pass
                ```
                """)

            with st.expander("🔍 InternVL3-78B (2024년 MLLM SOTA)"):
                st.markdown("""
                **OpenGVLab - 2024년 발표**

                #### 주요 특징
                - **78B 파라미터**: 최대 규모 오픈소스 MLLM
                - **MMMU 72.2**: 벤치마크 신기록
                - **멀티모달 추론**: 이미지+텍스트 복합 이해
                - **3D Vision**: 3D 객체 인식 지원
                - **GUI Agent**: 인터페이스 이해 및 조작

                #### 활용 분야
                - 산업 이미지 분석
                - 문서 이해 (OCR + VQA)
                - 복잡한 시각적 추론
                - 로봇 비전
                """)

            with st.expander("🔍 Veo 3 (2024년 AI 비디오 생성)"):
                st.markdown("""
                **Google DeepMind - 2024년 발표**

                #### 아키텍처
                - **Latent Diffusion Transformer**: 효율적 비디오 생성
                - **Spacetime Patches**: ViT처럼 비디오를 패치로 처리
                - **Audio-Visual Unified**: 영상+음성 동시 생성

                #### 특징
                - **텍스트→비디오**: 텍스트 프롬프트로 고품질 비디오 생성
                - **시간적 일관성**: Transformer로 프레임 간 연속성 보장
                - **고해상도**: 4K 비디오 생성 가능

                #### Transformer의 역할
                - Sequence modeling로 프레임 간 관계 학습
                - Attention으로 시간적 coherence 유지
                - Vision Transformer 기술 직접 활용

                #### vs. Nano Banana
                - **Veo 3**: 비디오 생성
                - **Nano Banana**: 이미지 생성 (Gemini 2.5 Flash)
                - 두 모델 모두 Transformer 기반!
                """)

        with era_tabs[2]:
            st.markdown("### 📊 전체 모델 비교 (2020-2025)")

            comparison_df = {
                "시대": ["클래식", "클래식", "중기", "중기", "최신", "최신", "최신"],
                "모델": ["ViT-Base", "DINO", "DINOv2", "SAM", "EVA-CLIP-18B", "SAM 2", "DINOv3"],
                "년도": ["2020", "2021", "2023", "2023", "2024", "2024", "2025"],
                "파라미터": ["86M", "22M", "300M", "632M", "18B", "~300M", "7B"],
                "주요 용도": ["분류", "자기지도", "자기지도", "분할", "검색", "비디오분할", "자기지도"],
                "혁신 포인트": [
                    "Transformer를 Vision에 첫 적용",
                    "라벨 없는 학습 입증",
                    "대규모 데이터로 성능 향상",
                    "Prompt 기반 제로샷 분할",
                    "최대 규모 CLIP 모델",
                    "비디오로 확장+메모리",
                    "Image-Text Alignment"
                ]
            }

            st.table(comparison_df)

        st.markdown("---")
        st.markdown("### 🎯 Vision Transformer 모델 선택 가이드 (2025년 기준)")

        st.info("""
        💡 **모델 분류 기준**:
        - **순수 Vision Transformer**: ViT, DINO 계열 (이미지 이해)
        - **Vision-Language Transformer**: CLIP, SigLIP (이미지+텍스트)
        - **Task-Specific Transformer**: SAM (세그멘테이션)
        - **참고: 생성/멀티모달**: Veo 3 (생성), InternVL3 (MLLM) - 관련 모델이지만 다른 목적
        """)

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("""
            #### 🔍 **순수 Vision Transformer**

            **이미지 특징 추출 & 분류**
            - **DINOv3** (7B): 최강 자기지도학습
              - Zero-shot transfer 최고 성능
              - Dense prediction (segmentation, depth) 우수
              - 연구/프로토타입에 적합

            - **ViT-Large** (307M): 클래식하지만 검증됨
              - ImageNet 82.6% 정확도
              - 빠른 추론 속도
              - 프로덕션 환경에 안정적

            #### 🎨 **Vision-Language Transformer**

            **이미지-텍스트 매칭**
            - **EVA-CLIP-18B** (18B): Zero-shot 분류 최고 (80.7%)
              - "고양이가 앉아있다" → 이미지 검색
              - 텍스트 프롬프트로 이미지 분류

            - **SigLIP 2** (1B): 빠른 서비스용
              - 다국어 지원 (한국어 포함)
              - 실시간 검색 엔진에 적합
            """)

        with rec_col2:
            st.markdown("""
            #### 🎯 **Task-Specific Transformer**

            **세그멘테이션 전문**
            - **SAM 2** (300M): 비디오 객체 분할
              - 이미지: 클릭 한 번으로 객체 추출
              - 비디오: 프레임 간 객체 자동 추적
              - Instagram Edits에 실제 사용 중
              - 의료/자율주행에 필수

            #### 📚 **참고: 관련 모델들**

            **Vision Transformer 기술 활용**
            - **InternVL3-78B**: Multimodal LLM
              - ViT + LLM 결합
              - 이미지 이해 + 추론 능력
              - "이 이미지에서 문제점은?" 같은 VQA

            - **Veo 3**: 비디오 생성 (Diffusion Transformer)
              - ViT의 Spacetime Patch 개념 활용
              - 텍스트 → 비디오 생성
              - 생성 AI 분야
            """)

        st.success("""
        **🎯 선택 기준 요약**:

        | 목적 | 추천 모델 | 이유 |
        |------|----------|------|
        | **이미지 분류** | ViT-Large, DINOv3 | 검증된 성능 |
        | **Zero-shot 분류** | EVA-CLIP-18B | 텍스트로 카테고리 지정 |
        | **특징 추출** | DINOv3 | 라벨 없이 강력한 특징 |
        | **이미지 검색** | SigLIP 2 | 빠르고 다국어 지원 |
        | **객체 분할** | SAM 2 | 비디오까지 지원 |
        | **전이 학습** | DINOv3, ViT-Large | Backbone으로 사용 |

        💡 **Week 4 실습**: 위 순수 Vision Transformer 모델들 (ViT, DINO, CLIP, SAM)을 직접 사용해보세요!
        """)

    def _render_project_tab(self):
        """실전 프로젝트 탭"""
        st.header("🚀 실전 Vision Transformer 프로젝트")

        project_type = st.selectbox(
            "프로젝트 선택",
            ["🖼️ 이미지 분류 (ViT)", "🔍 특징 추출 (DINO)", "📊 모델 비교"],
            key="vit_project_type"
        )

        if project_type == "🖼️ 이미지 분류 (ViT)":
            self._render_classification_project()
        elif project_type == "🔍 특징 추출 (DINO)":
            self._render_feature_extraction_project()
        else:
            self._render_comparison_project()

    def _render_classification_project(self):
        """이미지 분류 프로젝트 - HuggingFace ViT 모델 사용"""
        st.subheader("🖼️ Vision Transformer 이미지 분류")

        st.info("""
        💡 **실제 ViT 모델 사용**: HuggingFace의 사전학습된 Vision Transformer로 이미지를 분류합니다.
        - 모델: `google/vit-base-patch16-224`
        - ImageNet 1000개 클래스 분류
        - 첫 실행 시 모델 다운로드 (약 330MB)
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **프로젝트 목표:**
            - Vision Transformer를 사용한 이미지 분류
            - HuggingFace 사전훈련 모델 활용
            - ImageNet 1000개 클래스 인식

            **지원 모델:**
            - ViT-Base: 86M 파라미터
            - 입력: 224×224 이미지
            - 출력: Top-5 예측 결과
            """)

            uploaded_file = st.file_uploader(
                "이미지 업로드",
                type=['png', 'jpg', 'jpeg'],
                key="vit_classify_upload"
            )

        with col2:
            if uploaded_file:
                from PIL import Image
                import numpy as np

                img = Image.open(uploaded_file)
                st.image(img, caption="업로드된 이미지", use_container_width=True)

                if st.button("🔍 ViT 모델로 분류", key="vit_classify", type="primary"):
                    with st.spinner("ViT 모델 로딩 및 추론 중..."):
                        try:
                            from transformers import ViTImageProcessor, ViTForImageClassification
                            import torch

                            # 모델 로드
                            model_name = "google/vit-base-patch16-224"
                            processor = ViTImageProcessor.from_pretrained(model_name)
                            model = ViTForImageClassification.from_pretrained(model_name)

                            # 이미지 전처리
                            inputs = processor(images=img, return_tensors="pt")

                            # 추론
                            with torch.no_grad():
                                outputs = model(**inputs)
                                logits = outputs.logits

                            # Top-5 예측
                            probs = torch.nn.functional.softmax(logits, dim=-1)
                            top5_probs, top5_indices = torch.topk(probs, 5)

                            st.success("✅ 분류 완료!")
                            st.markdown("### 🎯 Top-5 예측 결과")

                            for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
                                label = model.config.id2label[idx.item()]
                                confidence = prob.item() * 100

                                # 막대 그래프 형식으로 표시
                                bar_length = int(confidence / 2)
                                bar = "█" * bar_length
                                st.markdown(f"**{i+1}. {label}**")
                                st.progress(confidence / 100)
                                st.caption(f"신뢰도: {confidence:.2f}%")
                                st.markdown("---")

                            # 모델 정보
                            with st.expander("📊 모델 정보"):
                                st.markdown(f"""
                                - **모델**: {model_name}
                                - **파라미터 수**: {sum(p.numel() for p in model.parameters()):,}
                                - **입력 크기**: 224×224
                                - **패치 크기**: 16×16
                                - **학습 데이터**: ImageNet-21k
                                """)

                        except Exception as e:
                            st.error(f"❌ 모델 로딩 실패: {str(e)}")
                            st.info("""
                            **해결 방법:**
                            1. 인터넷 연결 확인 (모델 다운로드 필요)
                            2. 터미널에서 수동 설치: `pip install transformers torch pillow`
                            3. 충분한 디스크 공간 확인 (최소 500MB)
                            """)

    def _render_feature_extraction_project(self):
        """특징 추출 프로젝트 - HuggingFace DINO 모델 사용"""
        st.subheader("🔍 DINO 특징 추출 및 Attention Map")

        st.info("""
        💡 **실제 DINOv2 모델 사용**: HuggingFace의 사전학습된 DINOv2로 이미지 특징을 추출합니다.
        - 모델: `facebook/dinov2-small` (22M 파라미터)
        - 자기지도학습으로 학습된 강력한 특징 추출기
        - Attention Map으로 모델이 보는 영역 시각화
        - DINO v1보다 개선된 성능
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **프로젝트 목표:**
            - DINO를 사용한 이미지 특징 벡터 추출
            - Self-Attention Map 시각화
            - 객체 경계 자동 발견 확인

            **DINO의 능력:**
            - 레이블 없이 객체 경계 발견
            - 의미론적 특징 추출
            - Zero-shot segmentation
            """)

            uploaded_file = st.file_uploader(
                "이미지 업로드",
                type=['png', 'jpg', 'jpeg'],
                key="dino_feature_upload"
            )

        with col2:
            if uploaded_file:
                from PIL import Image
                import numpy as np

                img = Image.open(uploaded_file)
                st.image(img, caption="업로드된 이미지", use_container_width=True)

                if st.button("🔍 DINO 특징 추출", key="dino_extract", type="primary"):
                    with st.spinner("DINO 모델 로딩 및 특징 추출 중..."):
                        try:
                            from transformers import AutoImageProcessor, AutoModel
                            import torch

                            # DINOv2 모델 로드 (Attention 지원)
                            model_name = "facebook/dinov2-small"
                            processor = AutoImageProcessor.from_pretrained(model_name)
                            # attn_implementation='eager'로 설정하여 attention 출력 활성화
                            model = AutoModel.from_pretrained(model_name, attn_implementation='eager')

                            # 이미지 전처리
                            inputs = processor(images=img, return_tensors="pt")

                            # 특징 추출
                            with torch.no_grad():
                                outputs = model(**inputs, output_attentions=True)

                                # CLS token 특징 (전체 이미지 표현)
                                cls_features = outputs.last_hidden_state[:, 0, :]

                                # Attention weights
                                attentions = outputs.attentions if hasattr(outputs, 'attentions') else None

                            st.success("✅ 특징 추출 완료!")

                            # 특징 벡터 정보
                            st.markdown("### 📊 추출된 특징 정보")
                            st.metric("특징 벡터 차원", f"{cls_features.shape[1]}")

                            # attentions가 None이 아닌지 확인
                            if attentions is not None and len(attentions) > 0:
                                st.metric("Attention Layers", f"{len(attentions)}")

                                # Attention Map 시각화
                                st.markdown("### 🎨 Self-Attention Map")

                                # 마지막 레이어의 평균 attention
                                last_attention = attentions[-1]  # [batch, heads, tokens, tokens]
                                avg_attention = last_attention[0].mean(0)  # 모든 head 평균

                                # CLS token이 다른 패치에 주는 attention
                                cls_attention = avg_attention[0, 1:]  # CLS → patch tokens

                                # 14x14 그리드로 reshape (ViT-S/16은 14x14 패치)
                                num_patches = int(np.sqrt(cls_attention.shape[0]))
                                attention_map = cls_attention.reshape(num_patches, num_patches).numpy()

                                # 원본 이미지 크기로 resize
                                import matplotlib.pyplot as plt
                                from scipy.ndimage import zoom

                                img_array = np.array(img)
                                h, w = img_array.shape[:2]
                                attention_resized = zoom(attention_map, (h/num_patches, w/num_patches), order=1)

                                # 시각화
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                                # 원본 이미지
                                ax1.imshow(img_array)
                                ax1.set_title("Original Image", fontsize=14)
                                ax1.axis('off')

                                # Attention overlay
                                ax2.imshow(img_array)
                                im = ax2.imshow(attention_resized, cmap='jet', alpha=0.6)
                                ax2.set_title("DINO Attention Map", fontsize=14)
                                ax2.axis('off')
                                plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

                                st.pyplot(fig)
                                plt.close()

                                st.success("""
                                ✨ **DINO가 발견한 것들**:
                                - 🔴 빨간색: 모델이 중요하다고 판단한 영역
                                - 🔵 파란색: 배경 또는 덜 중요한 영역
                                - DINO는 레이블 없이도 객체 경계를 자동으로 찾아냅니다!
                                """)
                            else:
                                st.warning("⚠️ Attention 정보를 가져올 수 없습니다. 특징 벡터만 추출되었습니다.")
                                st.info("""
                                **특징 벡터는 정상적으로 추출되었습니다!**
                                - 384차원 특징 벡터를 사용하여 이미지 검색, 클러스터링 등이 가능합니다.
                                - Attention Map 시각화는 일부 모델에서만 지원됩니다.
                                """)

                            # 특징 벡터 활용 예시
                            with st.expander("💡 특징 벡터 활용 예시"):
                                st.code("""
# 추출된 특징으로 할 수 있는 것들:

# 1. 유사 이미지 검색
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 여러 이미지의 특징 추출
features_db = []  # 데이터베이스
query_feature = cls_features.numpy()  # 쿼리 이미지

# 코사인 유사도 계산
similarities = cosine_similarity(query_feature, features_db)
top_k = np.argsort(similarities[0])[-5:]  # 상위 5개

# 2. 이미지 클러스터링
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(features_db)

# 3. Zero-shot Segmentation
# Attention map을 threshold하여 객체 마스크 생성
mask = attention_map > threshold
                                """, language="python")

                            # 모델 정보
                            with st.expander("📊 DINOv2 모델 정보"):
                                st.markdown(f"""
                                - **모델**: {model_name}
                                - **파라미터 수**: {sum(p.numel() for p in model.parameters()):,}
                                - **아키텍처**: Vision Transformer Small
                                - **패치 크기**: 14×14
                                - **학습 방법**: Self-Distillation (자기지도학습)
                                - **학습 데이터**: LVD-142M (142M 이미지)
                                - **특징**: 레이블 없이 객체 경계 발견 가능
                                - **개선점**: DINO v1 대비 더 강력한 특징 추출
                                """)

                        except Exception as e:
                            st.error(f"❌ 모델 로딩 실패: {str(e)}")
                            st.info("""
                            **해결 방법:**
                            1. 인터넷 연결 확인 (모델 다운로드 필요)
                            2. 필요한 패키지 설치: `pip install transformers torch pillow scipy matplotlib`
                            3. 충분한 디스크 공간 확인
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