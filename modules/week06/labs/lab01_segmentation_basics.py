"""
Lab 1: 세그멘테이션 기초
- Semantic vs Instance vs Panoptic Segmentation 비교
- 기본적인 이미지 세그멘테이션 체험
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import Tuple, List


def run():
    st.title("🎨 Lab 1: 세그멘테이션 기초")

    st.markdown("""
    ## 학습 목표
    - 세그멘테이션의 종류 이해하기
    - Semantic, Instance, Panoptic의 차이 체험
    - 간단한 색상 기반 세그멘테이션 구현
    """)

    tabs = st.tabs([
        "1️⃣ 세그멘테이션 종류",
        "2️⃣ 색상 기반 세그멘테이션",
        "3️⃣ 실습 과제"
    ])

    with tabs[0]:
        demo_segmentation_types()

    with tabs[1]:
        demo_color_based_segmentation()

    with tabs[2]:
        show_exercises()


def demo_segmentation_types():
    """세그멘테이션 종류 시각화"""
    st.header("1️⃣ 세그멘테이션 종류 비교")

    st.markdown("""
    ### 시각화 데모

    아래 버튼을 클릭하면 각 세그멘테이션 방식의 차이를 볼 수 있습니다.
    """)

    # 샘플 이미지 생성
    if st.button("샘플 이미지 생성"):
        sample_img = create_sample_scene()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 원본
        axes[0, 0].imshow(sample_img)
        axes[0, 0].set_title("원본 이미지", fontsize=14)
        axes[0, 0].axis('off')

        # Semantic
        semantic_mask = create_semantic_mask(sample_img)
        axes[0, 1].imshow(semantic_mask, cmap='tab10')
        axes[0, 1].set_title("Semantic Segmentation\n(같은 클래스 → 같은 색)", fontsize=14)
        axes[0, 1].axis('off')

        # Instance
        instance_mask = create_instance_mask(sample_img)
        axes[1, 0].imshow(instance_mask, cmap='tab20')
        axes[1, 0].set_title("Instance Segmentation\n(개별 객체 → 다른 색)", fontsize=14)
        axes[1, 0].axis('off')

        # Panoptic
        panoptic_mask = create_panoptic_mask(sample_img)
        axes[1, 1].imshow(panoptic_mask, cmap='tab20')
        axes[1, 1].set_title("Panoptic Segmentation\n(Semantic + Instance)", fontsize=14)
        axes[1, 1].axis('off')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 설명
        st.markdown("""
        ### 비교 설명

        | 방식 | 특징 | 예시 |
        |------|------|------|
        | **Semantic** | 같은 클래스 → 같은 레이블 | 모든 사람 → "person" |
        | **Instance** | 개별 객체 구분 | person1, person2, person3 |
        | **Panoptic** | Stuff(semantic) + Thing(instance) | 하늘(semantic) + 차들(instance) |

        ---

        ### 활용 분야

        - **Semantic**: 자율주행 (도로, 인도, 차선 구분)
        - **Instance**: 로봇 비전 (개별 물체 추적)
        - **Panoptic**: 완전한 장면 이해 (배경 + 객체)
        """)


def create_sample_scene() -> np.ndarray:
    """샘플 장면 생성 (간단한 도형들)"""
    img = Image.new('RGB', (400, 300), color='skyblue')  # 배경 (하늘)
    draw = ImageDraw.Draw(img)

    # 땅
    draw.rectangle([0, 200, 400, 300], fill='green')

    # 사람 1 (왼쪽)
    draw.ellipse([50, 120, 90, 180], fill='red')  # 몸
    draw.ellipse([60, 100, 80, 120], fill='peachpuff')  # 머리

    # 사람 2 (오른쪽)
    draw.ellipse([310, 130, 350, 190], fill='blue')  # 몸
    draw.ellipse([320, 110, 340, 130], fill='peachpuff')  # 머리

    # 집
    draw.rectangle([150, 150, 250, 250], fill='brown')
    draw.polygon([(150, 150), (200, 100), (250, 150)], fill='darkred')

    return np.array(img)


def create_semantic_mask(img: np.ndarray) -> np.ndarray:
    """Semantic 마스크 생성"""
    # 색상 기반으로 클래스 할당
    mask = np.zeros(img.shape[:2], dtype=np.int32)

    # Sky (파란색 계열)
    sky_mask = (img[:, :, 0] < 200) & (img[:, :, 2] > 200)
    mask[sky_mask] = 1

    # Ground (녹색 계열)
    ground_mask = (img[:, :, 1] > 200) & (img[:, :, 0] < 100)
    mask[ground_mask] = 2

    # Person (빨강/파랑/살색 계열)
    person_mask = (
        ((img[:, :, 0] > 200) & (img[:, :, 1] < 100)) |  # 빨강
        ((img[:, :, 2] > 200) & (img[:, :, 0] < 100)) |  # 파랑
        ((img[:, :, 0] > 200) & (img[:, :, 1] > 150))    # 살색
    )
    mask[person_mask] = 3

    # Building (갈색/진빨강 계열)
    building_mask = (
        ((img[:, :, 0] > 100) & (img[:, :, 0] < 170)) |  # 갈색
        ((img[:, :, 0] > 100) & (img[:, :, 1] < 50))     # 진빨강
    )
    mask[building_mask] = 4

    return mask


def create_instance_mask(img: np.ndarray) -> np.ndarray:
    """Instance 마스크 생성"""
    mask = np.zeros(img.shape[:2], dtype=np.int32)

    # 왼쪽 사람 (빨강)
    person1_mask = (img[:, :, 0] > 200) & (img[:, :, 1] < 100) & (img[:, :, 2] < 100)
    person1_head = (img[:, :, 0] > 200) & (img[:, :, 1] > 150) & (img[:, :, 0] < 260)
    mask[person1_mask | (person1_head & (np.arange(img.shape[0])[:, None] < 130))] = 1

    # 오른쪽 사람 (파랑)
    person2_mask = (img[:, :, 2] > 200) & (img[:, :, 0] < 100) & (img[:, :, 1] < 100)
    person2_head = (img[:, :, 0] > 200) & (img[:, :, 1] > 150) & (img[:, :, 0] > 260)
    mask[person2_mask | (person2_head & (np.arange(img.shape[0])[:, None] < 140))] = 2

    # 집
    building_mask = (
        ((img[:, :, 0] > 100) & (img[:, :, 0] < 170) & (img[:, :, 1] > 30)) |
        ((img[:, :, 0] > 100) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50))
    )
    mask[building_mask] = 3

    return mask


def create_panoptic_mask(img: np.ndarray) -> np.ndarray:
    """Panoptic 마스크 생성 (Stuff + Thing)"""
    mask = np.zeros(img.shape[:2], dtype=np.int32)

    # Stuff (배경, semantic)
    # Sky
    sky_mask = (img[:, :, 0] < 200) & (img[:, :, 2] > 200)
    mask[sky_mask] = 1

    # Ground
    ground_mask = (img[:, :, 1] > 200) & (img[:, :, 0] < 100)
    mask[ground_mask] = 2

    # Things (객체, instance)
    # Person 1
    person1_mask = (img[:, :, 0] > 200) & (img[:, :, 1] < 100) & (img[:, :, 2] < 100)
    person1_head = (img[:, :, 0] > 200) & (img[:, :, 1] > 150) & (img[:, :, 0] < 260)
    mask[person1_mask | (person1_head & (np.arange(img.shape[0])[:, None] < 130))] = 10

    # Person 2
    person2_mask = (img[:, :, 2] > 200) & (img[:, :, 0] < 100) & (img[:, :, 1] < 100)
    person2_head = (img[:, :, 0] > 200) & (img[:, :, 1] > 150) & (img[:, :, 0] > 260)
    mask[person2_mask | (person2_head & (np.arange(img.shape[0])[:, None] < 140))] = 11

    # Building
    building_mask = (
        ((img[:, :, 0] > 100) & (img[:, :, 0] < 170) & (img[:, :, 1] > 30)) |
        ((img[:, :, 0] > 100) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50))
    )
    mask[building_mask] = 12

    return mask


def demo_color_based_segmentation():
    """색상 기반 세그멘테이션 데모"""
    st.header("2️⃣ 색상 기반 세그멘테이션")

    st.markdown("""
    간단한 색상 기반 세그멘테이션을 직접 구현해봅시다.

    **원리**: 특정 색상 범위의 픽셀을 찾아 마스킹
    """)

    uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="원본 이미지", use_container_width=True)

        with col2:
            st.markdown("### 색상 범위 설정")

            # RGB 범위 슬라이더
            r_min, r_max = st.slider("R 범위", 0, 255, (100, 255))
            g_min, g_max = st.slider("G 범위", 0, 255, (0, 100))
            b_min, b_max = st.slider("B 범위", 0, 255, (0, 100))

            if st.button("세그멘테이션 실행"):
                # 색상 범위 내 픽셀 찾기
                mask = (
                    (img_array[:, :, 0] >= r_min) & (img_array[:, :, 0] <= r_max) &
                    (img_array[:, :, 1] >= g_min) & (img_array[:, :, 1] <= g_max) &
                    (img_array[:, :, 2] >= b_min) & (img_array[:, :, 2] <= b_max)
                )

                # 시각화
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(image)
                axes[0].set_title("원본")
                axes[0].axis('off')

                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title("마스크")
                axes[1].axis('off')

                axes[2].imshow(image)
                axes[2].imshow(mask, alpha=0.5, cmap='Reds')
                axes[2].set_title("오버레이")
                axes[2].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # 통계
                total = mask.size
                selected = mask.sum()
                st.metric("선택된 픽셀", f"{selected:,} / {total:,} ({selected/total*100:.2f}%)")


def show_exercises():
    """실습 과제"""
    st.header("3️⃣ 실습 과제")

    st.markdown("""
    ## 과제 1: 다중 클래스 세그멘테이션

    아래 이미지에서 **3가지 이상의 클래스**를 색상 기반으로 분할하세요.

    ### 요구사항
    1. 각 클래스마다 다른 RGB 범위 설정
    2. 3개 이상의 클래스 마스크 생성
    3. 결과를 다른 색상으로 시각화

    ### 힌트
    ```python
    # 클래스 1: 빨간색 계열
    mask1 = (img[:,:,0] > 200) & (img[:,:,1] < 100)

    # 클래스 2: 녹색 계열
    mask2 = (img[:,:,1] > 200) & (img[:,:,0] < 100)

    # 클래스 3: 파란색 계열
    mask3 = (img[:,:,2] > 200) & (img[:,:,0] < 100)

    # 결합
    combined = mask1.astype(int) + mask2.astype(int)*2 + mask3.astype(int)*3
    ```

    ---

    ## 과제 2: 실전 응용

    자신의 이미지를 업로드하고, 특정 객체(예: 하늘, 나무, 건물)를 색상 기반으로 분할해보세요.

    ### 단계
    1. 분할하고 싶은 객체가 포함된 이미지 준비
    2. 해당 객체의 대표 색상 범위 찾기
    3. 세그멘테이션 수행
    4. 결과 분석

    ---

    ## 제출

    - 코드 스크린샷
    - 결과 이미지
    - 간단한 설명 (어떤 클래스를 분할했는지, 어려웠던 점)

    ---

    ## 심화 과제 (선택)

    ### HSV 색공간 활용
    RGB 대신 HSV(Hue, Saturation, Value) 색공간을 사용하면 더 쉽게 색상 기반 세그멘테이션이 가능합니다.

    ```python
    from PIL import Image
    import numpy as np

    # RGB → HSV 변환
    img = Image.open('image.jpg').convert('RGB')
    img_hsv = img.convert('HSV')
    hsv_array = np.array(img_hsv)

    # Hue(색상) 기반 마스킹
    # 빨강: 0-10 or 170-180
    # 녹색: 40-80
    # 파랑: 100-140
    ```

    HSV를 사용하면 조명 변화에 더 강건한 세그멘테이션이 가능합니다!
    """)


if __name__ == "__main__":
    run()
