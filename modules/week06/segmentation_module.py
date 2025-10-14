"""
Week 6: 이미지 세그멘테이션과 SAM (Segment Anything Model)
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import io

from core.base_processor import BaseImageProcessor
from .sam_helpers import get_sam_helper


class SegmentationModule(BaseImageProcessor):
    """Week 6: 이미지 세그멘테이션 모듈"""

    def __init__(self):
        super().__init__()
        self.name = "Week 6: Segmentation & SAM"

    def render(self):
        """메인 렌더링 함수"""
        st.title("🎨 Week 6: 이미지 세그멘테이션과 SAM")

        st.markdown("""
        ## 학습 목표
        - **이론**: U-Net, Instance/Panoptic Segmentation, SAM 원리 이해
        - **실습**: SAM을 활용한 interactive segmentation 구현
        - **응용**: 배경 제거, 자동 라벨링 도구 제작
        """)

        # 환경 체크
        self._check_environment()

        # 5개 탭 구성
        tabs = st.tabs([
            "📚 개념 소개",
            "🎯 SAM 기초",
            "🖱️ Interactive 세그멘테이션",
            "🤖 Auto Mask 생성",
            "💼 실전 응용"
        ])

        with tabs[0]:
            self.render_theory()

        with tabs[1]:
            self.render_sam_basics()

        with tabs[2]:
            self.render_interactive()

        with tabs[3]:
            self.render_auto_mask()

        with tabs[4]:
            self.render_practical()

    def _check_environment(self):
        """환경 체크 및 설정"""
        with st.expander("🔧 환경 설정 확인", expanded=False):
            st.markdown("""
            ### 필요한 패키지
            - `transformers`: SAM 모델 로딩 (HuggingFace)
            - `torch`: PyTorch 백엔드
            - `Pillow`, `numpy`, `matplotlib`: 이미지 처리

            ### 3-Tier Fallback 전략
            1. **HuggingFace Transformers** (권장): `pip install transformers torch`
            2. **Official segment-anything**: `pip install segment-anything`
            3. **Simulation Mode**: 패키지 없이 기본 기능 사용
            """)

            issues = []

            # Check transformers
            try:
                import transformers
                st.success(f"✅ transformers {transformers.__version__}")
            except ImportError:
                issues.append("transformers")
                st.warning("⚠️ transformers 미설치 (시뮬레이션 모드)")

            # Check torch
            try:
                import torch
                device = "GPU" if torch.cuda.is_available() else "CPU"
                st.success(f"✅ torch {torch.__version__} ({device})")
            except ImportError:
                issues.append("torch")
                st.warning("⚠️ torch 미설치 (시뮬레이션 모드)")

            if issues:
                st.info(f"""
                ### 🔧 설치 방법
                ```bash
                pip install transformers torch
                ```

                시뮬레이션 모드에서도 기본 기능은 사용 가능합니다.
                """)

    # ==================== Tab 1: 개념 소개 ====================

    def render_theory(self):
        """세그멘테이션 이론 설명"""
        st.header("📚 세그멘테이션 개념 소개")

        st.markdown("""
        ## 1. 세그멘테이션이란?

        **세그멘테이션(Segmentation)**은 이미지의 각 픽셀을 특정 클래스나 객체에 할당하는 작업입니다.

        ### 1.1 Classification vs Detection vs Segmentation

        | 태스크 | 목표 | 출력 |
        |--------|------|------|
        | **Classification** | 이미지 전체 분류 | "고양이" |
        | **Object Detection** | 객체 위치 탐지 | 바운딩 박스 + 클래스 |
        | **Segmentation** | 픽셀 단위 분류 | 픽셀별 마스크 |

        """)

        # 세그멘테이션 타입 비교 이미지
        import os
        seg_img_path = os.path.join(os.path.dirname(__file__), "assets", "3.png")
        if os.path.exists(seg_img_path):
            st.image(seg_img_path,
                    caption="분류 vs 객체 탐지 vs 세그멘테이션 비교",
                    use_container_width=True)

        st.markdown("""

        ---

        ## 2. 세그멘테이션 종류

        ### 2.1 Semantic Segmentation
        - **정의**: 같은 클래스의 모든 픽셀을 같은 레이블로 지정
        - **특징**: 개별 객체 구분 안 함 (모든 사람 → "person")
        - **예**: 도로, 하늘, 건물 등 영역 분할

        ### 2.2 Instance Segmentation
        - **정의**: 같은 클래스라도 개별 객체를 구분
        - **특징**: 각 객체마다 다른 마스크
        - **예**: person1, person2, person3 구분

        ### 2.3 Panoptic Segmentation
        - **정의**: Semantic + Instance 결합
        - **특징**: 배경(stuff)은 semantic, 객체(thing)는 instance
        - **예**: 도로(semantic) + 자동차들(instance)
        """)

        # 세그멘테이션 종류 비교 이미지
        import os
        seg_img_path = os.path.join(os.path.dirname(__file__), "assets", "segmentation.png")
        if os.path.exists(seg_img_path):
            st.image(seg_img_path,
                    caption="Semantic vs Instance vs Panoptic Segmentation 비교",
                    use_container_width=True)

        st.markdown("""

        ---

        ## 3. U-Net 아키텍처

        U-Net은 의료 이미지 세그멘테이션을 위해 개발된 대표적인 아키텍처입니다.
        """)

        # U-Net 구조 다이어그램
        import os
        unet_img_path = os.path.join(os.path.dirname(__file__), "assets", "unet_architecture.png")
        if os.path.exists(unet_img_path):
            st.image(unet_img_path,
                    caption="U-Net Architecture (Ronneberger et al., 2015)",
                    use_container_width=True)
        else:
            st.image("https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png",
                    caption="U-Net Architecture (Ronneberger et al., 2015)",
                    use_container_width=True)

        st.markdown("""
        ### U-Net 핵심 구조

        U-Net은 **"U"자 형태**의 아키텍처를 가지고 있어 이와 같이 명명되었습니다. 인코더-디코더 구조의 대칭적 형태를 나타냅니다.

        ---

        #### 🔽 1단계: Contracting Path (축소 경로) - "정보 압축"

        **비유**: 망원경으로 사진을 찍듯이, 큰 그림을 점점 작게 압축합니다.

        - **동작 원리**: 이미지를 점진적으로 다운샘플링하면서 고수준 특징을 추출
        - **목적**: 객체와 배경을 구분하는 의미론적 정보를 학습하기 위함
        - **예시**: 512×512 → 256×256 → 128×128 → 64×64로 축소

        💡 *공간적 해상도는 감소하지만 추상화 수준은 증가합니다*

        ---

        #### 🎯 2단계: Bottleneck (병목) - "핵심 이해"

        **비유**: 정보를 최대한 압축하여 핵심만 추출하는 과정입니다.

        - **동작 원리**: 이미지의 핵심 정보만을 포함하는 최소 표현을 생성
        - **목적**: 전체 이미지의 맥락적 정보를 통합적으로 이해하기 위함

        💡 *최소 해상도이지만 최대 의미 정보를 포함합니다*

        ---

        #### 🔼 3단계: Expansive Path (확장 경로) - "디테일 복원"

        **비유**: 압축 파일을 다시 풀듯이, 작은 이미지를 원래 크기로 키웁니다.

        - **동작 원리**: 압축된 특징 맵을 점진적으로 업샘플링하여 원래 해상도로 복원
        - **목적**: 각 픽셀 단위의 정확한 세그멘테이션 마스크를 생성하기 위함
        - **예시**: 64×64 → 128×128 → 256×256 → 512×512로 확대

        💡 *해상도를 복원하면서 픽셀 단위의 세밀한 분류 정보를 재구성합니다*

        ---

        #### ⭐ 4단계: Skip Connections (지름길 연결) - "디테일 보존"

        **비유**: 정보 손실을 방지하기 위한 직접적인 연결 통로입니다.

        - **동작 원리**: 인코더의 고해상도 특징을 디코더의 해당 레벨에 직접 연결
        - **목적**: 다운샘플링 과정에서 손실된 세부 정보를 복원하기 위함
        - **효과**: 경계선이 더 선명하고 정확한 세그멘테이션 결과 생성

        💡 *이는 U-Net의 핵심 설계로, 공간적 세부 정보의 보존을 가능하게 합니다*

        ---

        ### 🎓 U-Net의 장점

        1. **세부 정보 보존**: Skip Connection을 통한 정밀한 경계 분할 성능
        2. **데이터 효율성**: 제한된 훈련 데이터에서도 효과적인 학습 가능
        3. **계산 효율성**: 최적화된 구조를 통한 빠른 추론 속도
        4. **범용성**: 의료 영상, 위성 이미지, 자율주행 등 다양한 도메인 적용

        ---

        ## 4. Segment Anything Model (SAM)

        **SAM(Segment Anything Model)**은 Meta AI가 2023년 발표한 "무엇이든 분할하는" AI입니다.

        ---

        ### 🎯 SAM이 뭐가 특별한가요?

        **비유**: SAM은 **"만능 가위"** 같아요!

        기존 세그멘테이션 모델이 특정 도메인에 특화되어 있는 반면, SAM은 다양한 객체에 대해 범용적인 분할 성능을 제공합니다.

        ---

        #### ✨ SAM의 3가지 슈퍼파워

        **1. 🌟 Zero-Shot 능력 - "처음 봐도 알아요!"**

        **비유**: 외국인이 처음 본 과일도 "이게 과일이구나"라고 아는 것처럼

        - **무엇을 하나요?** 한 번도 본 적 없는 물건도 정확하게 분할
        - **예시**:
          - 고양이만 학습했는데 강아지도 분할 가능
          - 자동차 데이터로 학습했으나 자전거 분할도 가능
        - **왜 대단한가요?** 기존 모델은 학습한 것만 분할 가능했어요

        💡 *1,100만 개 이미지 데이터셋으로 훈련되어 광범위한 객체 인식이 가능합니다*

        ---

        **2. 🖱️ Prompt 기반 - "원하는 대로 조작 가능!"**

        **비유**: 리모컨으로 TV 채널을 바꾸듯이, 클릭과 박스로 AI를 조종

        3가지 조작 방법:
        - **Point (클릭)**: 특정 좌표의 객체 분할 요청 👆
        - **Box (박스)**: 지정된 영역 내 객체 분할 요청 ⬜
        - **Mask (마스크)**: "이거 더 정확하게 다듬어줘!" ✂️

        💡 *마치 포토샵의 마술 지팡이 도구처럼 쉬워요!*

        ---

        **3. ⚡ 실시간 처리 - "빠르고 정확해요!"**

        **특징**: 초고속 처리 성능

        - **속도**: 클릭하면 즉시 결과 (< 1초)
        - **대화형**: 결과를 보면서 계속 수정 가능
        - **실용성**: 실제 작업에 바로 사용 가능

        💡 *느린 기존 모델과 달리 바로바로 반응해요!*

        ---

        ### 🔧 SAM의 기술적 원리

        SAM은 **Transformer 기반 아키텍처**로 구성되며, 3개의 핵심 컴포넌트가 상호작용합니다.

        ---

        #### 1단계: 이미지 인코더 (Image Encoder) 🔍

        **아키텍처**: Vision Transformer (ViT) 기반의 계층적 인코더

        **동작 원리**:
        - **패치 임베딩**: 이미지를 16×16 패치로 분할하여 토큰화
        - **위치 인코딩**: 각 패치의 공간적 위치 정보를 임베딩에 추가
        - **Self-Attention**: 패치 간 전역적 관계를 학습하여 컨텍스트 파악
        - **다중 스케일 특징**: 다양한 해상도에서 특징 맵 생성 (1/4, 1/8, 1/16, 1/32)

        **핵심 기술**:
        ```
        Input Image (1024×1024) 
        → Patch Embedding (64×64×768)
        → Multi-Head Attention Layers
        → Hierarchical Feature Maps
        ```

        ---

        #### 2단계: 프롬프트 인코더 (Prompt Encoder) 🎯

        **아키텍처**: 다중 모달 입력 처리를 위한 특화된 인코더

        **Point 프롬프트 처리**:
        - **위치 임베딩**: 2D 좌표를 고차원 벡터로 변환
        - **타입 임베딩**: Positive/Negative 클릭을 구분하는 학습된 임베딩
        - **결합**: `point_embedding = positional_embedding + type_embedding`

        **Box 프롬프트 처리**:
        - **모서리 인코딩**: 4개 모서리 좌표를 각각 임베딩
        - **형태 정보**: 박스의 가로/세로 비율 및 크기 정보 포함

        **Mask 프롬프트 처리**:
        - **CNN 기반**: 2D 마스크를 합성곱으로 인코딩
        - **다운샘플링**: 이미지 해상도와 맞춤

        ---

        #### 3단계: 마스크 디코더 (Mask Decoder) ✂️

        **아키텍처**: Transformer 디코더 + CNN 업샘플링

        **Cross-Attention 메커니즘**:
        - **Query**: 프롬프트 임베딩 (사용자 의도)
        - **Key/Value**: 이미지 특징 맵 (시각적 정보)
        - **출력**: 프롬프트에 맞는 관련 이미지 영역 식별

        **다중 스케일 융합**:
        ```
        Low-level features (세부 경계) + High-level features (의미 정보)
        → Skip connections로 결합
        → 정밀한 마스크 생성
        ```

        **Ambiguity-Aware 출력**:
        - **3개 마스크 후보**: 서로 다른 세분화 수준
          - **세밀**: 특정 부분만 (예: 자동차 바퀴)
          - **중간**: 단일 객체 (예: 자동차 전체)  
          - **광범위**: 관련 영역 전체 (예: 자동차 + 그림자)
        - **IoU 점수**: 각 마스크의 품질 예측값 제공

        **업샘플링 과정**:
        ```
        64×64 feature map 
        → Bilinear interpolation (×4)
        → 256×256 mask
        → Final upsampling (×4)
        → 1024×1024 final mask
        ```

        💡 *모호한 상황에 대응하는 다중 해석 능력이 SAM의 핵심 혁신입니다*

        ---

        ### ⚙️ SAM의 학습 전략

        **데이터 엔진 (Data Engine)**:
        1. **Manual Stage**: 전문가가 120K 이미지에 수동 라벨링
        2. **Semi-automatic**: SAM 보조로 180만 마스크 생성  
        3. **Fully automatic**: SAM만으로 1100만 마스크 자동 생성

        **손실 함수 (Loss Function)**:
        ```
        Total Loss = Focal Loss (분류) + Dice Loss (마스크) + IoU Loss (품질)
        ```

        **Zero-shot 일반화의 비밀**:
        - **대규모 데이터**: 1100만 고품질 마스크
        - **다양성**: 다양한 도메인, 객체, 스타일
        - **프롬프트 학습**: 다양한 사용자 입력 패턴 학습

        ---

        ### 📊 기술적 비교: 기존 모델 vs SAM

        | 구분 | 기존 세그멘테이션 모델 | SAM |
        |------|---------------------|-----|
        | **아키텍처** | CNN 기반 (FCN, U-Net, DeepLab) | Vision Transformer + Cross-Attention |
        | **학습 방식** | 고정된 클래스 레이블 학습 | 프롬프트 기반 학습 |
        | **일반화** | 특정 도메인/클래스에 한정 | Zero-shot 범용 분할 ⭐ |
        | **입력 방식** | 이미지만 처리 | 이미지 + 다중 프롬프트 ⭐ |
        | **출력** | 단일 마스크 | 다중 후보 마스크 + 신뢰도 ⭐ |
        | **데이터** | 수천~수만 라벨링 이미지 | 1100만 자동 생성 마스크 |
        | **추론 속도** | 도메인별 최적화 필요 | 실시간 처리 (50ms) ⭐ |

        ---

        ### 🔬 SAM vs U-Net 원리 비교

        | 측면 | U-Net | SAM |
        |------|-------|-----|
        | **인코더** | CNN 기반 다운샘플링 | Vision Transformer |
        | **특징 학습** | 지역적 Convolution | 전역적 Self-Attention |
        | **디코더** | 대칭적 업샘플링 | Cross-Attention + 업샘플링 |
        | **Skip Connection** | 동일 레벨 특징 결합 | 프롬프트 가이드 특징 융합 |
        | **학습 목표** | 픽셀별 클래스 분류 | 프롬프트 조건부 분할 |
        | **모호성 처리** | 단일 해석 | 다중 해석 + 불확실성 추정 |

        ---

        ### 💼 실제 응용 분야

        #### 1. 📸 **이미지 편집**
        - **응용**: 정밀한 배경 분리 및 교체
        - **활용처**: 사진 스튜디오, 디지털 콘텐츠 제작

        #### 2. 🏷️ **데이터 어노테이션**
        - **응용**: 대량 이미지 데이터셋의 자동 라벨링
        - **활용처**: 머신러닝 모델 개발, 컴퓨터 비전 연구

        #### 3. 🩺 **의료 영상 분석**
        - **예시**: X-ray, MRI에서 병변 부위 자동 감지
        - **사용자**: 의사, 병원

        #### 4. 🎬 **영상 편집**
        - **예시**: 유튜브 영상에서 배경만 교체, 특정 물건 제거
        - **사용자**: 크리에이터, 영상 편집자

        #### 5. 🚗 **자율주행 차량**
        - **예시**: 도로, 차량, 보행자 실시간 구분
        - **사용자**: 자동차 회사, 자율주행 개발사

        #### 6. 🛍️ **쇼핑몰 상품 사진**
        - **예시**: 옷, 신발 등 상품만 깔끔하게 추출
        - **사용자**: 이커머스, 온라인 쇼핑몰

        ---

        ### 🎉 정리하면?

        ### 💡 핵심 원리 요약

        **U-Net**: 
        ```
        CNN 인코더-디코더 + Skip Connections
        → 의료 이미지 등 특정 도메인에서 정밀한 분할
        ```

        **SAM**: 
        ```
        Vision Transformer + Cross-Attention + 프롬프트 학습
        → 범용 도메인에서 대화형 실시간 분할
        ```

        **결론**: U-Net은 **특정 태스크의 정확성**, SAM은 **범용성과 유연성**에 특화

        더 이상 복잡한 설정 없이, 클릭 몇 번으로 원하는 객체를 정확하게 분할할 수 있어요! 🎯

        ---

        ## 참고 자료
        - [U-Net 논문](https://arxiv.org/abs/1505.04597)
        - [SAM 논문](https://arxiv.org/abs/2304.02643)
        - [SAM Demo](https://segment-anything.com/)
        """)

    # ==================== Tab 2: SAM 기초 ====================

    def render_sam_basics(self):
        """SAM 기본 사용법"""
        st.header("🎯 SAM 기초 사용법")

        st.markdown("""
        이 탭에서는 SAM의 3가지 프롬프트 방식을 실습합니다:
        1. **Point Prompt**: 클릭으로 객체 지정
        2. **Box Prompt**: 박스로 영역 지정
        3. **Mask Prompt**: 기존 마스크 개선
        """)

        # 모델 선택
        col1, col2 = st.columns([1, 2])
        with col1:
            model_type = st.selectbox(
                "SAM 모델 선택",
                ["vit_b", "vit_l", "vit_h"],
                index=0,
                help="vit_b: 빠름(375MB), vit_l: 균형(1.2GB), vit_h: 고품질(2.4GB)"
            )

        with col2:
            st.info(f"""
            **선택된 모델**: {model_type}
            - vit_b: ~375MB, 빠른 추론
            - vit_l: ~1.2GB, 균형잡힌 성능
            - vit_h: ~2.4GB, 최고 품질
            """)

        # SAM 헬퍼 로드
        sam = get_sam_helper(model_type)

        st.markdown(f"**현재 모드**: `{sam.mode}` ({sam.device if sam.device else 'simulation'})")

        # 이미지 업로드
        uploaded = st.file_uploader(
            "이미지 선택",
            type=['png', 'jpg', 'jpeg'],
            key="sam_basics_upload"
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="원본 이미지", use_container_width=True)

            # 프롬프트 방식 선택
            prompt_type = st.radio(
                "프롬프트 방식",
                ["Point Prompt", "Box Prompt"],
                horizontal=True
            )

            if prompt_type == "Point Prompt":
                self._demo_point_prompt(image, sam)
            else:
                self._demo_box_prompt(image, sam)

    def _demo_point_prompt(self, image: Image.Image, sam):
        """포인트 프롬프트 데모"""
        st.subheader("🖱️ Point Prompt")

        st.markdown("""
        **사용법**:
        1. 분할하고 싶은 객체 위에 포인트 지정
        2. Label: 1=foreground (포함), 0=background (제외)
        3. 여러 포인트로 정확도 향상
        """)

        # 포인트 입력
        num_points = st.number_input("포인트 개수", 1, 10, 1)

        points = []
        labels = []

        cols = st.columns(3)
        for i in range(num_points):
            with cols[i % 3]:
                st.markdown(f"**Point {i+1}**")
                x = st.number_input(f"X{i+1}", 0, image.width, image.width//2, key=f"x{i}")
                y = st.number_input(f"Y{i+1}", 0, image.height, image.height//2, key=f"y{i}")
                label = st.selectbox(f"Label{i+1}", [1, 0], key=f"label{i}",
                                    help="1=foreground, 0=background")
                points.append((x, y))
                labels.append(label)

        if st.button("🎨 세그멘테이션 실행", key="point_segment"):
            with st.spinner("처리 중..."):
                # 세그멘테이션 수행
                mask = sam.segment_with_points(image, points, labels)

                # 시각화
                self._visualize_segmentation(image, mask, points, labels)

    def _demo_box_prompt(self, image: Image.Image, sam):
        """박스 프롬프트 데모"""
        st.subheader("📦 Box Prompt")

        st.markdown("""
        **사용법**:
        1. 객체를 포함하는 박스 좌표 입력 (x1, y1, x2, y2)
        2. 박스 내부의 주요 객체를 자동으로 분할
        """)

        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("X1 (좌상단)", 0, image.width, 0)
            y1 = st.number_input("Y1 (좌상단)", 0, image.height, 0)

        with col2:
            x2 = st.number_input("X2 (우하단)", 0, image.width, image.width)
            y2 = st.number_input("Y2 (우하단)", 0, image.height, image.height)

        box = (x1, y1, x2, y2)

        if st.button("🎨 세그멘테이션 실행", key="box_segment"):
            with st.spinner("처리 중..."):
                # 세그멘테이션 수행
                mask = sam.segment_with_box(image, box)

                # 시각화
                self._visualize_segmentation(image, mask, box=box)

    def _visualize_segmentation(
        self,
        image: Image.Image,
        mask: np.ndarray,
        points: Optional[List[Tuple[int, int]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None
    ):
        """세그멘테이션 결과 시각화"""

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. 원본 + 프롬프트
        axes[0].imshow(image)
        if points:
            for (x, y), label in zip(points, labels):
                color = 'green' if label == 1 else 'red'
                axes[0].plot(x, y, marker='o', markersize=10, color=color)
        if box:
            from matplotlib.patches import Rectangle
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                           edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
        axes[0].set_title("원본 + 프롬프트")
        axes[0].axis('off')

        # 2. 마스크
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("생성된 마스크")
        axes[1].axis('off')

        # 3. 오버레이
        axes[2].imshow(image)
        axes[2].imshow(mask, alpha=0.5, cmap='jet')
        axes[2].set_title("오버레이")
        axes[2].axis('off')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 통계
        total_pixels = mask.size
        selected_pixels = mask.sum()
        percentage = (selected_pixels / total_pixels) * 100

        st.success(f"""
        **세그멘테이션 완료**
        - 선택된 픽셀: {selected_pixels:,} / {total_pixels:,}
        - 비율: {percentage:.2f}%
        """)

    # ==================== Tab 3: Interactive 세그멘테이션 ====================

    def render_interactive(self):
        """Interactive 세그멘테이션 실습"""
        st.header("🖱️ Interactive 세그멘테이션")

        st.markdown("""
        ## Interactive Annotation

        실전에서는 반복적으로 포인트를 추가하며 마스크를 개선합니다.

        ### 워크플로우
        1. 초기 포인트로 대략적인 마스크 생성
        2. 누락된 영역에 foreground point 추가
        3. 잘못 포함된 영역에 background point 추가
        4. 만족할 때까지 반복
        """)

        model_type = st.selectbox(
            "모델 선택",
            ["vit_b", "vit_l", "vit_h"],
            key="interactive_model"
        )
        sam = get_sam_helper(model_type)

        uploaded = st.file_uploader(
            "이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="interactive_upload"
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")

            # 세션 상태로 포인트 관리
            if 'interactive_points' not in st.session_state:
                st.session_state.interactive_points = []
                st.session_state.interactive_labels = []

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(image, caption="작업 이미지", use_container_width=True)

            with col2:
                st.markdown("### 포인트 추가")

                x = st.number_input("X 좌표", 0, image.width, image.width//2, key="int_x")
                y = st.number_input("Y 좌표", 0, image.height, image.height//2, key="int_y")
                label = st.radio("타입", [1, 0], format_func=lambda x: "Foreground" if x == 1 else "Background", key="int_label")

                if st.button("➕ 포인트 추가"):
                    st.session_state.interactive_points.append((x, y))
                    st.session_state.interactive_labels.append(label)
                    st.success(f"포인트 추가됨 (총 {len(st.session_state.interactive_points)}개)")

                if st.button("🗑️ 모두 지우기"):
                    st.session_state.interactive_points = []
                    st.session_state.interactive_labels = []
                    st.rerun()

                st.markdown(f"**현재 포인트**: {len(st.session_state.interactive_points)}개")

            # 세그멘테이션 실행
            if st.session_state.interactive_points:
                if st.button("🎨 세그멘테이션 업데이트", type="primary"):
                    with st.spinner("처리 중..."):
                        mask = sam.segment_with_points(
                            image,
                            st.session_state.interactive_points,
                            st.session_state.interactive_labels
                        )

                        self._visualize_segmentation(
                            image,
                            mask,
                            st.session_state.interactive_points,
                            st.session_state.interactive_labels
                        )

                        # 다운로드
                        self._offer_mask_download(mask, "interactive_mask.png")

    def _offer_mask_download(self, mask: np.ndarray, filename: str):
        """마스크 다운로드 제공"""
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        buf = io.BytesIO()
        mask_img.save(buf, format='PNG')
        buf.seek(0)

        st.download_button(
            label="💾 마스크 다운로드",
            data=buf,
            file_name=filename,
            mime="image/png"
        )

    # ==================== Tab 4: Auto Mask 생성 ====================

    def render_auto_mask(self):
        """자동 마스크 생성"""
        st.header("🤖 Auto Mask 생성")

        st.markdown("""
        ## Automatic Mask Generation

        프롬프트 없이 이미지 전체를 자동으로 세그멘테이션합니다.

        ### 동작 원리
        1. 이미지에 그리드 포인트 샘플링
        2. 각 포인트에서 세그멘테이션 수행
        3. 중복 마스크 제거 (NMS)
        4. 품질 필터링 (IoU, stability score)

        ### 활용 사례
        - 데이터셋 자동 라벨링
        - 객체 개수 카운팅
        - 전체 장면 분석
        """)

        # ⚠️ 메모리 경고
        st.warning("""
        ⚠️ **메모리 사용량 주의**

        자동 마스크 생성은 **매우 많은 메모리(8~12GB)**를 사용합니다.

        **권장 설정**:
        - 이미지 크기: 512px 이하로 리사이즈
        - Points per side: 8~16 (낮을수록 빠름)
        - 메모리 부족 시 브라우저가 멈출 수 있습니다
        """)

        model_type = st.selectbox("모델", ["vit_b", "vit_l", "vit_h"], key="auto_model")
        sam = get_sam_helper(model_type)

        uploaded = st.file_uploader("이미지", type=['png', 'jpg', 'jpeg'], key="auto_upload")

        if uploaded:
            image = Image.open(uploaded).convert("RGB")

            # 이미지 크기 체크
            col1, col2 = st.columns(2)
            with col1:
                st.metric("원본 크기", f"{image.width}×{image.height}")
            with col2:
                resize_enabled = st.checkbox("이미지 리사이즈 (권장)", value=True)

            if resize_enabled:
                max_size = st.slider("최대 크기", 256, 1024, 512, step=128,
                                    help="이미지를 이 크기로 리사이즈 (메모리 절약)")

                # 리사이즈
                if max(image.width, image.height) > max_size:
                    ratio = max_size / max(image.width, image.height)
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    st.info(f"✅ 리사이즈됨: {new_size[0]}×{new_size[1]}")

            st.image(image, caption="처리할 이미지", use_container_width=True)

            # 파라미터 설정
            with st.expander("⚙️ 고급 설정"):
                points_per_side = st.slider("Points per side", 8, 64, 16,
                                           help="그리드 밀도 (높을수록 정밀하지만 매우 느리고 메모리 많이 사용)")
                pred_iou_thresh = st.slider("IoU threshold", 0.5, 1.0, 0.88,
                                            help="마스크 품질 임계값")
                stability_score_thresh = st.slider("Stability threshold", 0.5, 1.0, 0.95,
                                                  help="마스크 안정성 임계값")

            if st.button("🤖 자동 마스크 생성", type="primary"):
                try:
                    with st.spinner("자동 마스크 생성 중... (시간이 걸릴 수 있습니다)"):
                        masks = sam.generate_auto_masks(
                            image,
                            points_per_side=points_per_side,
                            pred_iou_thresh=pred_iou_thresh,
                            stability_score_thresh=stability_score_thresh
                        )

                        st.success(f"✅ {len(masks)}개 마스크 생성 완료")

                        # 시각화
                        self._visualize_auto_masks(image, masks)

                except RuntimeError as e:
                    if "not enough memory" in str(e):
                        st.error("""
                        ❌ **메모리 부족 에러**

                        자동 마스크 생성에 필요한 메모리가 부족합니다.

                        **해결 방법**:
                        1. ✅ 이미지 리사이즈 활성화 (512px 이하 권장)
                        2. Points per side를 8~12로 낮추기
                        3. 다른 프로그램 종료
                        4. 더 작은 이미지 사용

                        또는 **Interactive 세그멘테이션** 탭을 사용하시기 바랍니다 (메모리 효율적).
                        """)
                    else:
                        st.error(f"에러 발생: {e}")
                except Exception as e:
                    st.error(f"예상치 못한 에러: {e}")

    def _visualize_auto_masks(self, image: Image.Image, masks: List[Dict[str, Any]]):
        """자동 마스크 시각화"""

        if not masks:
            st.warning("생성된 마스크가 없습니다.")
            return

        # 통계
        total_area = sum(m['area'] for m in masks)
        avg_area = total_area / len(masks)

        col1, col2, col3 = st.columns(3)
        col1.metric("총 마스크 수", len(masks))
        col2.metric("평균 영역", f"{avg_area:.0f}px")
        col3.metric("총 커버리지", f"{(total_area / (image.width * image.height) * 100):.1f}%")

        # 컬러맵으로 모든 마스크 표시
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(image)
        axes[0].set_title("원본 이미지")
        axes[0].axis('off')

        # 모든 마스크 합성
        combined_mask = np.zeros((*masks[0]['segmentation'].shape, 3), dtype=np.uint8)
        for i, mask_data in enumerate(masks[:50]):  # 상위 50개만
            mask = mask_data['segmentation']
            color = np.random.randint(0, 255, 3)
            combined_mask[mask] = color

        axes[1].imshow(image)
        axes[1].imshow(combined_mask, alpha=0.5)
        axes[1].set_title(f"자동 마스크 ({len(masks)}개)")
        axes[1].axis('off')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 개별 마스크 보기
        if st.checkbox("개별 마스크 보기"):
            mask_idx = st.slider("마스크 선택", 0, len(masks)-1, 0)
            selected_mask = masks[mask_idx]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image)
            ax.imshow(selected_mask['segmentation'], alpha=0.6, cmap='jet')
            ax.set_title(f"Mask {mask_idx} (Area: {selected_mask['area']}px)")
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

            # bbox 정보
            x1, y1, x2, y2 = selected_mask['bbox']
            st.info(f"BBox: ({x1}, {y1}) → ({x2}, {y2})")

    # ==================== Tab 5: 실전 응용 ====================

    def render_practical(self):
        """실전 응용 예제"""
        st.header("💼 실전 응용")

        st.markdown("""
        ## SAM 활용 사례

        세그멘테이션 기술의 실전 응용 예제를 실습합니다.
        """)

        app_type = st.selectbox(
            "응용 예제 선택",
            [
                "배경 제거 (증명사진)",
                "자동 라벨링 도구",
                "객체 카운팅"
            ]
        )

        if app_type == "배경 제거 (증명사진)":
            self._app_background_removal()
        elif app_type == "자동 라벨링 도구":
            self._app_auto_labeling()
        else:
            self._app_object_counting()

    def _app_background_removal(self):
        """배경 제거 앱"""
        st.subheader("📸 배경 제거 (증명사진 편집기)")

        st.markdown("""
        **사용법**:
        1. 인물 사진 업로드
        2. 인물 위에 포인트 지정
        3. 배경 제거 및 새 배경 선택
        """)

        sam = get_sam_helper("vit_b")
        uploaded = st.file_uploader("인물 사진", type=['png', 'jpg', 'jpeg'], key="bg_remove")

        if uploaded:
            image = Image.open(uploaded).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="원본", use_container_width=True)

            with col2:
                # 포인트 입력
                x = st.number_input("X (인물)", 0, image.width, image.width//2)
                y = st.number_input("Y (인물)", 0, image.height, image.height//2)

                # 새 배경 색상
                bg_color = st.color_picker("새 배경 색상", "#FFFFFF")

            if st.button("🎨 배경 제거", type="primary"):
                with st.spinner("처리 중..."):
                    try:
                        # 세그멘테이션
                        mask = sam.segment_with_points(image, [(x, y)], [1])
                        
                        # 마스크 유효성 검사
                        if mask is None:
                            st.error("세그멘테이션에 실패했습니다. 다른 점을 선택해보세요.")
                            return
                        
                        # 마스크 형태 검증
                        if len(mask.shape) != 2:
                            st.error(f"마스크 형태가 올바르지 않습니다: {mask.shape}")
                            return
                        
                        st.success(f"마스크 생성 완료: {mask.shape}")

                        # 배경 교체
                        result_image = self._replace_background(image, mask, bg_color)
                        
                    except Exception as e:
                        st.error(f"세그멘테이션 처리 중 오류 발생: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
                        return

                    st.image(result_image, caption="결과", use_container_width=True)

                    # 다운로드
                    buf = io.BytesIO()
                    result_image.save(buf, format='PNG')
                    buf.seek(0)

                    st.download_button(
                        "💾 다운로드",
                        data=buf,
                        file_name="background_removed.png",
                        mime="image/png"
                    )

    def _replace_background(
        self,
        image: Image.Image,
        mask: np.ndarray,
        bg_color: str
    ) -> Image.Image:
        """배경 교체"""
        # RGB 변환
        r = int(bg_color[1:3], 16)
        g = int(bg_color[3:5], 16)
        b = int(bg_color[5:7], 16)

        # 새 배경 생성
        bg = Image.new("RGB", image.size, (r, g, b))

        # 마스크를 PIL 이미지로 변환하면서 크기 확인
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        
        # 마스크와 이미지 크기가 다른 경우 마스크를 이미지 크기에 맞게 조정
        if mask_img.size != image.size:
            mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)
        
        # 마스크가 L 모드(그레이스케일)인지 확인하고, 필요시 변환
        if mask_img.mode != 'L':
            mask_img = mask_img.convert('L')

        # 합성
        try:
            result = Image.composite(image, bg, mask_img)
        except ValueError as e:
            # 크기 불일치 문제가 여전히 있는 경우, 대체 방법 사용
            st.error(f"이미지 합성 중 오류 발생: {e}")
            # numpy를 사용한 대체 방법
            image_np = np.array(image)
            bg_np = np.array(bg)
            mask_np = np.array(mask_img) / 255.0
            
            # 마스크를 3차원으로 확장
            if len(mask_np.shape) == 2:
                mask_np = np.stack([mask_np] * 3, axis=-1)
            
            # 배경 교체
            result_np = image_np * mask_np + bg_np * (1 - mask_np)
            result = Image.fromarray(result_np.astype(np.uint8))
        
        return result

    def _app_auto_labeling(self):
        """자동 라벨링 도구"""
        st.subheader("🏷️ 자동 라벨링 도구")

        st.markdown("""
        **목적**: 객체 탐지 학습 데이터 생성 자동화

        **워크플로우**:
        1. 이미지 업로드
        2. 자동 마스크 생성
        3. 각 마스크에 클래스 레이블 할당
        4. COCO/YOLO 포맷으로 내보내기
        """)

        st.warning("⚠️ 메모리 사용량이 많습니다. 작은 이미지 사용 또는 리사이징을 권장합니다.")

        sam = get_sam_helper("vit_b")
        uploaded = st.file_uploader("이미지", type=['png', 'jpg', 'jpeg'], key="label_img")

        if uploaded:
            image = Image.open(uploaded).convert("RGB")

            # 이미지 리사이즈
            resize = st.checkbox("이미지 리사이즈", value=True, key="label_resize")
            if resize:
                max_size = st.slider("최대 크기", 256, 1024, 512, step=128, key="label_max")
                if max(image.width, image.height) > max_size:
                    ratio = max_size / max(image.width, image.height)
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    st.info(f"✅ 리사이즈: {new_size[0]}×{new_size[1]}")

            st.image(image, use_container_width=True)

            if st.button("🤖 자동 마스크 생성"):
                try:
                    with st.spinner("처리 중..."):
                        masks = sam.generate_auto_masks(image, points_per_side=16)
                        st.session_state.labeling_masks = masks
                        st.success(f"✅ {len(masks)}개 후보 생성")
                except RuntimeError as e:
                    if "not enough memory" in str(e):
                        st.error("❌ 메모리 부족! 이미지 크기를 축소하여 다시 시도하시기 바랍니다.")
                    else:
                        st.error(f"에러: {e}")
                except Exception as e:
                    st.error(f"예상치 못한 에러: {e}")

            if 'labeling_masks' in st.session_state:
                masks = st.session_state.labeling_masks

                st.markdown("### 레이블 할당")

                # 클래스 정의
                class_names = st.text_input("클래스 (쉼표 구분)", "person,car,dog,cat")
                classes = [c.strip() for c in class_names.split(",")]

                # 마스크별 레이블
                for i, mask_data in enumerate(masks[:10]):  # 상위 10개
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        # 미리보기
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.imshow(image)
                        ax.imshow(mask_data['segmentation'], alpha=0.5)
                        ax.set_title(f"Mask {i}")
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()

                    with col2:
                        label = st.selectbox(f"Class", ["(skip)"] + classes, key=f"class_{i}")

                    with col3:
                        st.metric("Area", f"{mask_data['area']}px")

                st.info("💡 실제 서비스에서는 여기서 JSON/XML 파일로 내보내기 구현")

    def _app_object_counting(self):
        """객체 카운팅"""
        st.subheader("🔢 객체 카운팅")

        st.markdown("""
        **응용 분야**:
        - 군중 계수 (crowd counting)
        - 세포 카운팅 (medical imaging)
        - 재고 관리 (inventory counting)
        """)

        st.warning("⚠️ 메모리 사용량이 많습니다. 이미지 리사이즈를 권장합니다.")

        sam = get_sam_helper("vit_b")
        uploaded = st.file_uploader("이미지", type=['png', 'jpg', 'jpeg'], key="count_img")

        if uploaded:
            image = Image.open(uploaded).convert("RGB")

            # 이미지 리사이즈
            resize = st.checkbox("이미지 리사이즈", value=True, key="count_resize")
            if resize:
                max_size = st.slider("최대 크기", 256, 1024, 512, step=128, key="count_max")
                if max(image.width, image.height) > max_size:
                    ratio = max_size / max(image.width, image.height)
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    st.info(f"✅ 리사이즈: {new_size[0]}×{new_size[1]}")

            st.image(image, use_container_width=True)

            # 필터링 파라미터
            min_area = st.slider("최소 객체 크기 (px²)", 100, 5000, 500)
            max_area = st.slider("최대 객체 크기 (px²)", 1000, 50000, 10000)

            if st.button("🔢 객체 카운팅", type="primary"):
                try:
                    with st.spinner("처리 중..."):
                        masks = sam.generate_auto_masks(image, points_per_side=16)

                        # 필터링
                        filtered = [
                            m for m in masks
                            if min_area <= m['area'] <= max_area
                        ]

                        st.success(f"✅ 검출된 객체: **{len(filtered)}개**")

                        # 시각화
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.imshow(image)

                        for i, mask_data in enumerate(filtered):
                            mask = mask_data['segmentation']
                            color = np.random.rand(3)

                            # 마스크 오버레이
                            colored_mask = np.zeros((*mask.shape, 3))
                            colored_mask[mask] = color
                            ax.imshow(colored_mask, alpha=0.4)

                            # 번호 표시
                            x1, y1, x2, y2 = mask_data['bbox']
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            ax.text(cx, cy, str(i+1), color='white',
                                   fontsize=12, weight='bold',
                                   ha='center', va='center',
                                   bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8))

                        ax.set_title(f"총 {len(filtered)}개 객체 검출")
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()

                        # 통계
                        if filtered:
                            areas = [m['area'] for m in filtered]
                            st.markdown(f"""
                            ### 통계
                            - 평균 크기: {np.mean(areas):.0f}px²
                            - 최소 크기: {np.min(areas)}px²
                            - 최대 크기: {np.max(areas)}px²
                            """)

                except RuntimeError as e:
                    if "not enough memory" in str(e):
                        st.error("❌ 메모리 부족! 이미지 크기 축소 또는 Interactive 탭 사용을 권장합니다.")
                    else:
                        st.error(f"에러: {e}")
                except Exception as e:
                    st.error(f"예상치 못한 에러: {e}")
