"""
Lab 2: SAM Interactive Annotation
- SAM을 활용한 interactive segmentation 실습
- Point/Box 프롬프트 체험
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

# 상위 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.week06.sam_helpers import get_sam_helper


def run():
    st.title("🖱️ Lab 2: SAM Interactive Annotation")

    st.markdown("""
    ## 학습 목표
    - SAM의 Point/Box 프롬프트 이해
    - Interactive annotation 워크플로우 체험
    - 효율적인 annotation 전략 학습
    """)

    tabs = st.tabs([
        "1️⃣ Point Prompt",
        "2️⃣ Box Prompt",
        "3️⃣ Multi-step Annotation"
    ])

    with tabs[0]:
        demo_point_prompt()

    with tabs[1]:
        demo_box_prompt()

    with tabs[2]:
        demo_multistep()


def demo_point_prompt():
    """Point 프롬프트 데모"""
    st.header("1️⃣ Point Prompt 실습")

    st.markdown("""
    ### Point Prompt 사용법

    1. **Foreground Point** (label=1): 분할하고 싶은 객체 위의 점
    2. **Background Point** (label=0): 제외하고 싶은 영역의 점
    3. 여러 포인트를 조합하여 정밀도 향상

    ### 실습 시나리오
    - 초기에는 객체 중앙에 1개의 fg point
    - 누락된 영역에 추가 fg point
    - 잘못 포함된 영역에 bg point 추가
    """)

    # 모델 선택
    model_type = st.selectbox("SAM 모델", ["vit_b", "vit_l", "vit_h"], key="point_model")
    sam = get_sam_helper(model_type)

    st.info(f"**현재 모드**: {sam.mode}")

    # 이미지 업로드
    uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], key="point_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="원본 이미지", use_container_width=True)

        # 포인트 입력
        st.markdown("### 포인트 입력")

        num_points = st.number_input("포인트 개수", 1, 10, 1)

        points = []
        labels = []

        cols = st.columns(min(num_points, 4))
        for i in range(num_points):
            with cols[i % 4]:
                st.markdown(f"**Point {i+1}**")
                x = st.number_input(f"X", 0, image.width, image.width//2, key=f"px{i}")
                y = st.number_input(f"Y", 0, image.height, image.height//2, key=f"py{i}")
                label = st.selectbox(
                    f"Type",
                    options=[1, 0],
                    format_func=lambda x: "Foreground" if x == 1 else "Background",
                    key=f"pl{i}"
                )
                points.append((x, y))
                labels.append(label)

        if st.button("🎨 세그멘테이션 실행", type="primary"):
            with st.spinner("처리 중..."):
                mask = sam.segment_with_points(image, points, labels)

                # 시각화
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # 원본 + 포인트
                axes[0].imshow(image)
                for (x, y), label in zip(points, labels):
                    color = 'lime' if label == 1 else 'red'
                    marker = 'o' if label == 1 else 'x'
                    axes[0].plot(x, y, marker=marker, markersize=15, color=color,
                               markeredgewidth=3, markeredgecolor='white')
                axes[0].set_title("원본 + 프롬프트")
                axes[0].axis('off')

                # 마스크
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title("생성된 마스크")
                axes[1].axis('off')

                # 오버레이
                axes[2].imshow(image)
                axes[2].imshow(mask, alpha=0.5, cmap='jet')
                axes[2].set_title("오버레이")
                axes[2].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # 통계
                selected = mask.sum()
                total = mask.size
                st.success(f"✅ 선택된 픽셀: {selected:,} / {total:,} ({selected/total*100:.2f}%)")


def demo_box_prompt():
    """Box 프롬프트 데모"""
    st.header("2️⃣ Box Prompt 실습")

    st.markdown("""
    ### Box Prompt 장점
    - 빠른 annotation (마우스로 박스만 그리면 됨)
    - 대략적인 위치만으로도 정확한 마스크 생성
    - Object Detection 결과와 연계 가능

    ### 실습
    박스 좌표를 입력하여 객체를 분할해봅시다.
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b", "vit_l", "vit_h"], key="box_model")
    sam = get_sam_helper(model_type)

    uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], key="box_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="원본 이미지", use_container_width=True)

        # Box 입력
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("X1 (좌상단)", 0, image.width, 50)
            y1 = st.number_input("Y1 (좌상단)", 0, image.height, 50)

        with col2:
            x2 = st.number_input("X2 (우하단)", 0, image.width, image.width - 50)
            y2 = st.number_input("Y2 (우하단)", 0, image.height, image.height - 50)

        box = (x1, y1, x2, y2)

        if st.button("🎨 세그멘테이션 실행", type="primary", key="box_segment"):
            with st.spinner("처리 중..."):
                mask = sam.segment_with_box(image, box)

                # 시각화
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # 원본 + 박스
                axes[0].imshow(image)
                from matplotlib.patches import Rectangle
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=3, edgecolor='red', facecolor='none')
                axes[0].add_patch(rect)
                axes[0].set_title("원본 + 박스")
                axes[0].axis('off')

                # 마스크
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title("생성된 마스크")
                axes[1].axis('off')

                # 오버레이
                axes[2].imshow(image)
                axes[2].imshow(mask, alpha=0.5, cmap='jet')
                axes[2].set_title("오버레이")
                axes[2].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # 통계
                selected = mask.sum()
                total = mask.size
                st.success(f"✅ 선택된 픽셀: {selected:,} / {total:,} ({selected/total*100:.2f}%)")


def demo_multistep():
    """Multi-step annotation 데모"""
    st.header("3️⃣ Multi-step Annotation 실습")

    st.markdown("""
    ### Interactive Annotation Workflow

    실무에서는 여러 단계를 거쳐 마스크를 점진적으로 개선합니다.

    **단계**:
    1. 초기 포인트로 대략적인 마스크 생성
    2. 결과 확인 후 누락/과잉 영역 식별
    3. 추가 포인트로 마스크 개선
    4. 만족할 때까지 반복

    ### 실습
    아래에서 점진적으로 포인트를 추가하며 마스크를 개선해보세요.
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b"], key="multi_model")
    sam = get_sam_helper(model_type)

    uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], key="multi_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        # 세션 상태로 포인트 관리
        if 'multi_points' not in st.session_state:
            st.session_state.multi_points = []
            st.session_state.multi_labels = []
            st.session_state.multi_history = []

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption="작업 이미지", use_container_width=True)

        with col2:
            st.markdown("### 포인트 추가")

            x = st.number_input("X", 0, image.width, image.width//2, key="mx")
            y = st.number_input("Y", 0, image.height, image.height//2, key="my")
            label = st.radio("Type", [1, 0],
                           format_func=lambda x: "Foreground" if x == 1 else "Background",
                           key="ml")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("➕ 추가"):
                    st.session_state.multi_points.append((x, y))
                    st.session_state.multi_labels.append(label)
                    st.success("추가됨")

            with col_b:
                if st.button("🗑️ 초기화"):
                    st.session_state.multi_points = []
                    st.session_state.multi_labels = []
                    st.session_state.multi_history = []
                    st.rerun()

            st.metric("현재 포인트", len(st.session_state.multi_points))

        # 세그멘테이션 실행
        if st.session_state.multi_points:
            if st.button("🎨 업데이트", type="primary", key="multi_segment"):
                with st.spinner("처리 중..."):
                    mask = sam.segment_with_points(
                        image,
                        st.session_state.multi_points,
                        st.session_state.multi_labels
                    )

                    # 히스토리 저장
                    st.session_state.multi_history.append({
                        'mask': mask.copy(),
                        'n_points': len(st.session_state.multi_points)
                    })

                    # 시각화
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                    axes[0].imshow(image)
                    for (px, py), pl in zip(st.session_state.multi_points,
                                           st.session_state.multi_labels):
                        color = 'lime' if pl == 1 else 'red'
                        marker = 'o' if pl == 1 else 'x'
                        axes[0].plot(px, py, marker=marker, markersize=12,
                                   color=color, markeredgewidth=2,
                                   markeredgecolor='white')
                    axes[0].set_title(f"포인트 ({len(st.session_state.multi_points)}개)")
                    axes[0].axis('off')

                    axes[1].imshow(image)
                    axes[1].imshow(mask, alpha=0.5, cmap='jet')
                    axes[1].set_title("현재 마스크")
                    axes[1].axis('off')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        # 히스토리 보기
        if st.session_state.multi_history:
            st.markdown("### Annotation 히스토리")

            history_idx = st.slider("버전 선택", 0,
                                   len(st.session_state.multi_history)-1,
                                   len(st.session_state.multi_history)-1)

            hist = st.session_state.multi_history[history_idx]
            st.info(f"버전 {history_idx+1}: {hist['n_points']}개 포인트")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image)
            ax.imshow(hist['mask'], alpha=0.5, cmap='jet')
            ax.set_title(f"버전 {history_idx+1}")
            ax.axis('off')
            st.pyplot(fig)
            plt.close()


if __name__ == "__main__":
    run()
