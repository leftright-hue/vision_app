"""
Lab 3: Auto Mask Generation
- 자동 마스크 생성 실습
- 객체 카운팅 응용
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from modules.week06.sam_helpers import get_sam_helper


def run():
    st.title("🤖 Lab 3: Auto Mask Generation")

    st.markdown("""
    ## 학습 목표
    - Automatic segmentation 원리 이해
    - 파라미터 조정 및 영향 분석
    - 객체 카운팅 응용
    """)

    tabs = st.tabs([
        "1️⃣ Auto Mask 기초",
        "2️⃣ 파라미터 조정",
        "3️⃣ 객체 카운팅"
    ])

    with tabs[0]:
        demo_auto_mask_basic()

    with tabs[1]:
        demo_parameter_tuning()

    with tabs[2]:
        demo_object_counting()


def demo_auto_mask_basic():
    """자동 마스크 생성 기초"""
    st.header("1️⃣ Auto Mask 기초")

    st.markdown("""
    ### Automatic Mask Generation 원리

    1. **Grid Sampling**: 이미지에 규칙적인 그리드 포인트 생성
    2. **Per-point Segmentation**: 각 포인트에서 세그멘테이션 수행
    3. **NMS (Non-Maximum Suppression)**: 중복 마스크 제거
    4. **Quality Filtering**: IoU와 stability score로 필터링

    ### 장점
    - 프롬프트 불필요
    - 전체 장면 분석
    - 데이터 라벨링 자동화

    ### 단점
    - 계산 비용 높음
    - 처리 시간 길음
    - 품질 필터링 필요
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b"], key="auto_model")
    sam = get_sam_helper(model_type)

    st.info(f"**현재 모드**: {sam.mode}")

    uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], key="auto_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="원본 이미지", use_container_width=True)

        if st.button("🤖 자동 마스크 생성", type="primary"):
            with st.spinner("처리 중... (시간이 걸릴 수 있습니다)"):
                masks = sam.generate_auto_masks(image, points_per_side=16)

                st.success(f"✅ {len(masks)}개 마스크 생성 완료")

                # 시각화
                visualize_auto_masks(image, masks)

                # 세션 상태에 저장
                st.session_state.auto_masks = masks


def demo_parameter_tuning():
    """파라미터 조정"""
    st.header("2️⃣ 파라미터 조정")

    st.markdown("""
    ### 주요 파라미터

    | 파라미터 | 역할 | 영향 |
    |---------|------|------|
    | `points_per_side` | 그리드 밀도 | 높을수록 정밀, 느림 |
    | `pred_iou_thresh` | IoU 임계값 | 높을수록 고품질 마스크만 |
    | `stability_score_thresh` | 안정성 임계값 | 높을수록 안정적인 마스크만 |

    ### 실습
    파라미터를 조정하며 결과 변화를 관찰하세요.
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b"], key="param_model")
    sam = get_sam_helper(model_type)

    uploaded = st.file_uploader("이미지", type=['png', 'jpg', 'jpeg'], key="param_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### 파라미터 설정")

            points_per_side = st.slider(
                "Points per side",
                8, 64, 16,
                help="그리드 밀도 (8=빠름/거칠, 64=느림/정밀)"
            )

            pred_iou_thresh = st.slider(
                "IoU threshold",
                0.5, 1.0, 0.88, 0.01,
                help="마스크 품질 임계값"
            )

            stability_score_thresh = st.slider(
                "Stability threshold",
                0.5, 1.0, 0.95, 0.01,
                help="마스크 안정성 임계값"
            )

        if st.button("🎨 생성", type="primary", key="param_generate"):
            with st.spinner("처리 중..."):
                masks = sam.generate_auto_masks(
                    image,
                    points_per_side=points_per_side,
                    pred_iou_thresh=pred_iou_thresh,
                    stability_score_thresh=stability_score_thresh
                )

                st.metric("생성된 마스크 수", len(masks))

                # 파라미터별 비교
                st.markdown("### 결과 분석")

                col1, col2, col3 = st.columns(3)
                if masks:
                    areas = [m['area'] for m in masks]
                    col1.metric("평균 크기", f"{np.mean(areas):.0f}px²")
                    col2.metric("최소 크기", f"{np.min(areas)}px²")
                    col3.metric("최대 크기", f"{np.max(areas)}px²")

                # 시각화
                visualize_auto_masks(image, masks)


def demo_object_counting():
    """객체 카운팅"""
    st.header("3️⃣ 객체 카운팅 응용")

    st.markdown("""
    ### 객체 카운팅 응용 사례

    - **군중 계수** (Crowd Counting): 공공장소 관리
    - **세포 카운팅** (Cell Counting): 의료 이미지 분석
    - **재고 관리** (Inventory): 물류 자동화
    - **동물 개체 수**: 생태 조사

    ### 실습
    자동 마스크 생성 + 크기 필터링으로 객체를 카운팅해봅시다.
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b"], key="count_model")
    sam = get_sam_helper(model_type)

    uploaded = st.file_uploader("이미지", type=['png', 'jpg', 'jpeg'], key="count_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="원본 이미지", use_container_width=True)

        # 필터링 파라미터
        col1, col2 = st.columns(2)
        with col1:
            min_area = st.number_input("최소 크기 (px²)", 100, 10000, 500)

        with col2:
            max_area = st.number_input("최대 크기 (px²)", 1000, 100000, 10000)

        if st.button("🔢 객체 카운팅", type="primary"):
            with st.spinner("처리 중..."):
                # 자동 마스크 생성
                masks = sam.generate_auto_masks(image, points_per_side=24)

                # 크기 필터링
                filtered = [
                    m for m in masks
                    if min_area <= m['area'] <= max_area
                ]

                st.success(f"### 검출된 객체: **{len(filtered)}개**")

                # 통계
                if filtered:
                    areas = [m['area'] for m in filtered]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("평균 크기", f"{np.mean(areas):.0f}px²")
                    col2.metric("표준편차", f"{np.std(areas):.0f}px²")
                    col3.metric("중앙값", f"{np.median(areas):.0f}px²")

                # 시각화
                visualize_counting(image, filtered)


def visualize_auto_masks(image: Image.Image, masks: list):
    """자동 마스크 시각화"""
    if not masks:
        st.warning("생성된 마스크가 없습니다.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 원본
    axes[0].imshow(image)
    axes[0].set_title("원본")
    axes[0].axis('off')

    # 마스크 오버레이
    combined_mask = np.zeros((*masks[0]['segmentation'].shape, 3), dtype=np.uint8)
    for i, mask_data in enumerate(masks[:50]):  # 상위 50개
        mask = mask_data['segmentation']
        color = np.random.randint(50, 255, 3)
        combined_mask[mask] = color

    axes[1].imshow(image)
    axes[1].imshow(combined_mask, alpha=0.6)
    axes[1].set_title(f"자동 마스크 ({len(masks)}개)")
    axes[1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def visualize_counting(image: Image.Image, masks: list):
    """객체 카운팅 시각화"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)

    for i, mask_data in enumerate(masks):
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
               fontsize=10, weight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8))

    ax.set_title(f"총 {len(masks)}개 객체 검출")
    ax.axis('off')
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    run()
