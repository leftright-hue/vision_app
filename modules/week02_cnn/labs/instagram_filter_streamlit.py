"""
🎨 Instagram 필터 제작소 - Streamlit 버전
딥러닝 영상처리 강의 - Week 2 실습용 웹 인터페이스

📖 학습 목표:
- 📷 디지털 사진이 컴퓨터에서 어떻게 표현되는지 체험하기
- 🎭 Instagram처럼 다양한 필터 효과 직접 만들어보기  
- 🔍 컴퓨터가 사진에서 물체의 경계를 찾는 방법 이해하기
- 🎪 재미있는 실습으로 컴퓨터 비전의 기초 원리 체득하기
"""

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

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
        rgb_combined = np.hstack([r, g, b])
        axes[1, 1].imshow(rgb_combined, cmap='gray')
        axes[1, 1].set_title('🔴🟢🔵 RGB 채널\n(빨강|초록|파랑)')
        axes[1, 1].axis('off')
        
        # HSV 채널 분리
        h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        hsv_combined = np.hstack([h, s, v])
        axes[1, 2].imshow(hsv_combined, cmap='gray')
        axes[1, 2].set_title('🌈💪☀️ HSV 채널\n(색깔|진하기|밝기)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def apply_convolution_filters(self):
        """🎭 Instagram 필터 6종 세트"""
        if self.current_image is None:
            return None
        
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

def create_checkerboard(size=100, square_size=10):
    """🏁 체스판 만들기"""
    checkerboard = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, square_size):
        for j in range(0, size, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size] = 255
    return checkerboard

def adjust_brightness(image, factor):
    """☀️ 사진 밝기 조절"""
    adjusted = image.astype(np.float32) * factor
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

def rotate_image(image, angle_90):
    """🌪️ 이미지 회전 (90도 단위)"""
    if angle_90 == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle_90 == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle_90 == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image

def emboss_filter(image):
    """🏺 엠보싱 필터"""
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.filter2D(gray, -1, kernel)

def motion_blur_filter(image, size=15, angle=45):
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

def render_instagram_filter_lab():
    """🎨 Instagram 필터 제작소 메인 UI"""
    
    st.title("🎨 Instagram 필터 제작소")
    st.markdown("### 딥러닝 영상처리 - Week 2 실습")
    
    st.markdown("""
    📱 **오늘 우리는 Instagram, Snapchat 같은 필터를 직접 만들어볼 거예요!**
    
    🎯 **학습 목표:**
    - 📷 디지털 사진이 컴퓨터에서 어떻게 표현되는지 체험하기
    - 🎭 Instagram처럼 다양한 필터 효과 직접 만들어보기
    - 🔍 컴퓨터가 사진에서 물체의 경계를 찾는 방법 이해하기
    """)
    
    # Instagram 필터 제작소 인스턴스 생성
    if 'filter_maker' not in st.session_state:
        st.session_state.filter_maker = InstagramFilterMaker()
    
    filter_maker = st.session_state.filter_maker
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📸 사진 업로드", 
        "🌈 색깔 변신쇼", 
        "🎭 필터 체험", 
        "🕵️ 경계선 탐정", 
        "🎮 연습문제"
    ])
    
    with tab1:
        st.header("📸 사진 업로드")
        
        uploaded_file = st.file_uploader(
            "사진을 업로드하세요",
            type=['png', 'jpg', 'jpeg'],
            help="PNG, JPG, JPEG 형식을 지원합니다"
        )
        
        if uploaded_file is not None:
            if filter_maker.load_image_from_upload(uploaded_file):
                st.success("✅ 사진 업로드 성공!")
                
                # 이미지 표시
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🖼️ 업로드된 사진")
                    # BGR을 RGB로 변환하여 표시
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
    
    with tab2:
        st.header("🌈 색깔의 다양한 표현 방식")
        
        if filter_maker.current_image is not None:
            if st.button("🎭 색깔 변신쇼 시작!"):
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
    
    with tab3:
        st.header("🎭 Instagram 필터 6종 세트")
        
        if filter_maker.current_image is not None:
            if st.button("🎪 필터 마술쇼 시작!"):
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
    
    with tab4:
        st.header("🕵️ 경계선 찾기 탐정 대회")
        
        if filter_maker.current_image is not None:
            if st.button("🔍 탐정 대회 시작!"):
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
    
    with tab5:
        st.header("🎮 연습문제")
        
        # 연습문제 1: 기본 연산
        st.subheader("🎨 연습문제 1: 디지털 아트 창작")
        
        ex1_col1, ex1_col2, ex1_col3 = st.columns(3)
        
        with ex1_col1:
            st.write("**🏁 체스판 만들기**")
            size = st.slider("체스판 크기", 50, 200, 100, key="checkerboard_size")
            square_size = st.slider("사각형 크기", 5, 20, 10, key="square_size")
            
            if st.button("체스판 생성", key="create_checkerboard"):
                checkerboard = create_checkerboard(size, square_size)
                st.image(checkerboard, caption="생성된 체스판")
        
        with ex1_col2:
            st.write("**☀️ 밝기 조절**")
            if filter_maker.current_image is not None:
                brightness_factor = st.slider("밝기 조절", 0.1, 3.0, 1.0, 0.1, key="brightness")
                
                if st.button("밝기 적용", key="apply_brightness"):
                    adjusted = adjust_brightness(filter_maker.current_image, brightness_factor)
                    display_adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
                    st.image(display_adjusted, caption=f"밝기 {brightness_factor}x")
            else:
                st.info("사진을 먼저 업로드하세요")
        
        with ex1_col3:
            st.write("**🌪️ 이미지 회전**")
            if filter_maker.current_image is not None:
                rotation = st.selectbox("회전 각도", [0, 90, 180, 270], key="rotation")
                
                if st.button("회전 적용", key="apply_rotation"):
                    rotated = rotate_image(filter_maker.current_image, rotation)
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
            if filter_maker.current_image is not None and st.button("엠보싱 적용", key="apply_emboss"):
                embossed = emboss_filter(filter_maker.current_image)
                st.image(embossed, caption="엠보싱 효과", cmap='gray')
            elif filter_maker.current_image is None:
                st.info("사진을 먼저 업로드하세요")
        
        with ex2_col2:
            st.write("**💫 모션 블러**")
            if filter_maker.current_image is not None:
                blur_size = st.slider("블러 강도", 5, 25, 15, key="blur_size")
                blur_angle = st.selectbox("블러 방향", [0, 45, 90], 
                                        format_func=lambda x: f"{x}도 ({'수평' if x==0 else '대각선' if x==45 else '수직'})",
                                        key="blur_angle")
                
                if st.button("모션 블러 적용", key="apply_motion_blur"):
                    motion_blurred = motion_blur_filter(filter_maker.current_image, blur_size, blur_angle)
                    st.image(motion_blurred, caption="모션 블러 효과", cmap='gray')
            else:
                st.info("사진을 먼저 업로드하세요")

if __name__ == "__main__":
    render_instagram_filter_lab()