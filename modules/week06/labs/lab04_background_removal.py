"""
Lab 4: Background Removal Application
- 배경 제거 앱 구현
- 증명사진 편집기
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from modules.week06.sam_helpers import get_sam_helper


def run():
    st.title("📸 Lab 4: Background Removal")

    st.markdown("""
    ## 학습 목표
    - 배경 제거 알고리즘 이해
    - 증명사진 편집기 구현
    - 실전 응용 기법 학습
    """)

    tabs = st.tabs([
        "1️⃣ 기본 배경 제거",
        "2️⃣ 증명사진 편집기",
        "3️⃣ 고급 기능"
    ])

    with tabs[0]:
        demo_basic_removal()

    with tabs[1]:
        demo_id_photo_editor()

    with tabs[2]:
        demo_advanced_features()


def demo_basic_removal():
    """기본 배경 제거"""
    st.header("1️⃣ 기본 배경 제거")

    st.markdown("""
    ### 배경 제거 프로세스

    1. **객체 세그멘테이션**: SAM으로 전경 객체 분할
    2. **마스크 생성**: 이진 마스크 (객체=1, 배경=0)
    3. **배경 교체**: 새로운 배경과 합성

    ### 구현 단계
    ```python
    # 1. 세그멘테이션
    mask = sam.segment_with_points(image, points, labels)

    # 2. 배경 생성
    new_bg = create_background(bg_color, image.size)

    # 3. 합성
    result = composite(foreground, new_bg, mask)
    ```
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b"], key="basic_model")
    sam = get_sam_helper(model_type)

    uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], key="basic_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="원본", use_container_width=True)

            # 포인트 입력
            st.markdown("### 객체 지정")
            x = st.number_input("X (객체 위)", 0, image.width, image.width//2)
            y = st.number_input("Y (객체 위)", 0, image.height, image.height//2)

        with col2:
            st.markdown("### 새 배경 설정")

            bg_mode = st.radio("배경 타입", ["단색", "그라데이션", "투명"])

            if bg_mode == "단색":
                bg_color = st.color_picker("배경 색상", "#FFFFFF")
            elif bg_mode == "그라데이션":
                color1 = st.color_picker("색상 1", "#FFFFFF")
                color2 = st.color_picker("색상 2", "#E0E0E0")

        if st.button("🎨 배경 제거", type="primary"):
            with st.spinner("처리 중..."):
                # 세그멘테이션
                mask = sam.segment_with_points(image, [(x, y)], [1])

                # 배경 생성
                if bg_mode == "단색":
                    result = replace_solid_background(image, mask, bg_color)
                elif bg_mode == "그라데이션":
                    result = replace_gradient_background(image, mask, color1, color2)
                else:
                    result = make_transparent(image, mask)

                # 결과 표시
                st.image(result, caption="결과", use_container_width=True)

                # 다운로드
                offer_download(result, "background_removed.png")


def demo_id_photo_editor():
    """증명사진 편집기"""
    st.header("2️⃣ 증명사진 편집기")

    st.markdown("""
    ### 증명사진 표준 규격

    | 용도 | 크기 (px) | 배경색 |
    |------|----------|--------|
    | 여권 | 413 x 531 | 흰색 |
    | 비자 | 354 x 472 | 흰색 |
    | 이력서 | 295 x 413 | 파란색/흰색 |
    | 운전면허증 | 260 x 354 | 회색 |

    ### 기능
    - 자동 크기 조정
    - 배경 색상 변경
    - 표준 규격 맞추기
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b"], key="id_model")
    sam = get_sam_helper(model_type)

    uploaded = st.file_uploader("인물 사진", type=['png', 'jpg', 'jpeg'], key="id_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### 설정")

            # 용도 선택
            photo_type = st.selectbox(
                "용도",
                ["여권", "비자", "이력서", "운전면허증", "사용자 정의"]
            )

            # 규격 설정
            if photo_type == "여권":
                width, height = 413, 531
                bg_color = "#FFFFFF"
            elif photo_type == "비자":
                width, height = 354, 472
                bg_color = "#FFFFFF"
            elif photo_type == "이력서":
                width, height = 295, 413
                bg_color = st.color_picker("배경", "#E3F2FD")
            elif photo_type == "운전면허증":
                width, height = 260, 354
                bg_color = "#F5F5F5"
            else:
                width = st.number_input("너비 (px)", 100, 1000, 413)
                height = st.number_input("높이 (px)", 100, 1000, 531)
                bg_color = st.color_picker("배경", "#FFFFFF")

            st.info(f"**크기**: {width} x {height} px")

            # 포인트
            x = st.number_input("X (얼굴)", 0, image.width, image.width//2, key="id_x")
            y = st.number_input("Y (얼굴)", 0, image.height, image.height//3, key="id_y")

        if st.button("🎨 증명사진 생성", type="primary"):
            with st.spinner("처리 중..."):
                # 세그멘테이션
                mask = sam.segment_with_points(image, [(x, y)], [1])

                # 증명사진 생성
                result = create_id_photo(image, mask, width, height, bg_color)

                # 결과
                st.image(result, caption=f"{photo_type} 증명사진", width=width)

                # 다운로드
                offer_download(result, f"id_photo_{photo_type}.png")

                st.success(f"✅ {photo_type} 규격 증명사진 생성 완료")


def demo_advanced_features():
    """고급 기능"""
    st.header("3️⃣ 고급 기능")

    st.markdown("""
    ### 추가 기능들

    1. **Edge Refinement**: 마스크 경계 다듬기
    2. **Hair Matting**: 머리카락 디테일 보존
    3. **Shadow Removal**: 배경 그림자 제거
    4. **Batch Processing**: 여러 이미지 일괄 처리

    ### 실습: 마스크 후처리
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b"], key="adv_model")
    sam = get_sam_helper(model_type)

    uploaded = st.file_uploader("이미지", type=['png', 'jpg', 'jpeg'], key="adv_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

        # 후처리 옵션
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 마스크 후처리")

            enable_erosion = st.checkbox("Erosion (경계 축소)", value=False)
            erosion_size = st.slider("Erosion 크기", 1, 10, 3) if enable_erosion else 0

            enable_dilation = st.checkbox("Dilation (경계 확장)", value=False)
            dilation_size = st.slider("Dilation 크기", 1, 10, 3) if enable_dilation else 0

            enable_blur = st.checkbox("Gaussian Blur (부드럽게)", value=True)
            blur_size = st.slider("Blur 크기", 1, 21, 5, step=2) if enable_blur else 0

        with col2:
            st.markdown("### 배경 설정")

            x = st.number_input("X", 0, image.width, image.width//2, key="adv_x")
            y = st.number_input("Y", 0, image.height, image.height//2, key="adv_y")
            bg_color = st.color_picker("배경 색상", "#FFFFFF")

        if st.button("🎨 처리", type="primary", key="adv_process"):
            with st.spinner("처리 중..."):
                # 세그멘테이션
                mask = sam.segment_with_points(image, [(x, y)], [1])

                # 후처리
                mask_refined = postprocess_mask(
                    mask,
                    erosion_size=erosion_size,
                    dilation_size=dilation_size,
                    blur_size=blur_size
                )

                # 배경 교체
                result_original = replace_solid_background(image, mask, bg_color)
                result_refined = replace_solid_background(image, mask_refined, bg_color)

                # 비교
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(image)
                axes[0].set_title("원본")
                axes[0].axis('off')

                axes[1].imshow(result_original)
                axes[1].set_title("기본 마스크")
                axes[1].axis('off')

                axes[2].imshow(result_refined)
                axes[2].set_title("후처리 적용")
                axes[2].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # 다운로드
                offer_download(result_refined, "refined_result.png")


# ==================== 헬퍼 함수 ====================

def replace_solid_background(image: Image.Image, mask: np.ndarray, bg_color: str) -> Image.Image:
    """단색 배경 교체"""
    r = int(bg_color[1:3], 16)
    g = int(bg_color[3:5], 16)
    b = int(bg_color[5:7], 16)

    bg = Image.new("RGB", image.size, (r, g, b))
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    return Image.composite(image, bg, mask_img)


def replace_gradient_background(
    image: Image.Image,
    mask: np.ndarray,
    color1: str,
    color2: str
) -> Image.Image:
    """그라데이션 배경 교체"""
    # 그라데이션 생성
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

    gradient = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    for i in range(image.height):
        ratio = i / image.height
        gradient[i, :, 0] = int(r1 * (1 - ratio) + r2 * ratio)
        gradient[i, :, 1] = int(g1 * (1 - ratio) + g2 * ratio)
        gradient[i, :, 2] = int(b1 * (1 - ratio) + b2 * ratio)

    bg = Image.fromarray(gradient)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    return Image.composite(image, bg, mask_img)


def make_transparent(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """투명 배경 생성"""
    image_rgba = image.convert("RGBA")
    alpha = (mask * 255).astype(np.uint8)

    image_array = np.array(image_rgba)
    image_array[:, :, 3] = alpha

    return Image.fromarray(image_array)


def create_id_photo(
    image: Image.Image,
    mask: np.ndarray,
    width: int,
    height: int,
    bg_color: str
) -> Image.Image:
    """증명사진 생성"""
    # 배경 교체
    result = replace_solid_background(image, mask, bg_color)

    # 크기 조정 (비율 유지)
    ratio = min(width / result.width, height / result.height) * 0.8
    new_size = (int(result.width * ratio), int(result.height * ratio))
    result_resized = result.resize(new_size, Image.Resampling.LANCZOS)

    # 캔버스 중앙 배치
    r = int(bg_color[1:3], 16)
    g = int(bg_color[3:5], 16)
    b = int(bg_color[5:7], 16)

    canvas = Image.new("RGB", (width, height), (r, g, b))
    x_offset = (width - result_resized.width) // 2
    y_offset = (height - result_resized.height) // 4  # 상단으로 약간 치우침

    canvas.paste(result_resized, (x_offset, y_offset))

    return canvas


def postprocess_mask(
    mask: np.ndarray,
    erosion_size: int = 0,
    dilation_size: int = 0,
    blur_size: int = 0
) -> np.ndarray:
    """마스크 후처리"""
    from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter

    mask_processed = mask.copy()

    # Erosion
    if erosion_size > 0:
        mask_processed = binary_erosion(mask_processed, iterations=erosion_size)

    # Dilation
    if dilation_size > 0:
        mask_processed = binary_dilation(mask_processed, iterations=dilation_size)

    # Gaussian blur
    if blur_size > 0:
        mask_float = mask_processed.astype(float)
        mask_float = gaussian_filter(mask_float, sigma=blur_size/3)
        mask_processed = mask_float > 0.5

    return mask_processed


def offer_download(image: Image.Image, filename: str):
    """이미지 다운로드 제공"""
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)

    st.download_button(
        label="💾 다운로드",
        data=buf,
        file_name=filename,
        mime="image/png"
    )


if __name__ == "__main__":
    run()
