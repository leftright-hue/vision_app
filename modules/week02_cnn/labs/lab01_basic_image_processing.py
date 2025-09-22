#!/usr/bin/env python3
"""
🎨 Week 1 Lab: 나만의 Instagram 필터 만들기 실습
딥러닝 영상처리 강의 - 1주차 실습용 코드

📖 학습 목표:
- 📷 디지털 사진이 컴퓨터에서 어떻게 표현되는지 체험하기
- 🎭 Instagram처럼 다양한 필터 효과 직접 만들어보기  
- 🔍 컴퓨터가 사진에서 물체의 경계를 찾는 방법 이해하기
- 🎪 재미있는 실습으로 컴퓨터 비전의 기초 원리 체득하기

🎬 실습 시나리오:
여러분은 새로운 SNS 앱의 필터 개발자가 되었습니다!
사용자들이 좋아할 만한 다양한 필터를 만들어보세요.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

class InstagramFilterMaker:
    """🎭 나만의 Instagram 필터 제작소
    
    실생활 비유: 📱 스마트폰의 사진 앱처럼!
    - 사진을 불러오고 → 다양한 필터 적용 → 결과 확인
    - 마치 포토샵이나 인스타그램 필터를 직접 만드는 것!
    """
    
    def __init__(self):
        self.current_image = None      # 현재 작업 중인 사진
        self.original_image = None     # 원본 사진 (백업용)
        print("🎨 Instagram 필터 제작소가 열렸습니다!")
        print("   📸 사진을 불러와서 멋진 필터를 만들어보세요!")
    
    def load_image(self, image_path):
        """📸 사진 불러오기 - 스마트폰에서 사진을 선택하는 것처럼!"""
        print(f"📂 사진 불러오는 중: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"😅 앗! 사진을 찾을 수 없어요: {image_path}")
            print("💡 팁: 파일 경로를 다시 확인해보세요!")
            return False
        
        try:
            self.current_image = cv2.imread(image_path)
            self.original_image = self.current_image.copy()
            print(f"🎉 사진 불러오기 성공!")
            print(f"📏 사진 크기: {self.current_image.shape[1]}×{self.current_image.shape[0]} 픽셀")
            print(f"🎨 색상 채널: {self.current_image.shape[2]}개 (Red, Green, Blue)")
            return True
        except Exception as e:
            print(f"😰 사진 불러오기 실패: {e}")
            print("💡 팁: jpg, png 파일인지 확인해보세요!")
            return False
    
    def show_image_info(self):
        """🔍 사진 정보 살펴보기 - 의사선생님이 환자를 진찰하는 것처럼!"""
        if self.current_image is None:
            print("😅 앗! 먼저 사진을 불러와주세요!")
            return
        
        height, width = self.current_image.shape[:2]
        channels = self.current_image.shape[2] if len(self.current_image.shape) == 3 else 1
        
        print("🏥 사진 건강검진 결과:")
        print(f"📐 크기: {width} × {height} 픽셀")
        
        if width * height > 1000000:
            print("   📱 고해상도 사진이네요! (100만 픽셀 이상)")
        elif width * height > 300000:
            print("   📷 중간 해상도 사진이에요!")
        else:
            print("   🖼️ 작은 크기의 사진이에요!")
            
        print(f"🎨 색상 채널: {channels}개 ({'컬러' if channels == 3 else '흑백'})")
        print(f"📊 픽셀값 범위: {self.current_image.min()} ~ {self.current_image.max()}")
        print(f"✨ 평균 밝기: {self.current_image.mean():.1f}/255")
        
        # 밝기 판정
        avg_brightness = self.current_image.mean()
        if avg_brightness > 180:
            print("   ☀️ 매우 밝은 사진이에요!")
        elif avg_brightness > 120:
            print("   🌤️ 적당히 밝은 사진이에요!")
        elif avg_brightness > 60:
            print("   🌥️ 조금 어두운 사진이에요!")
        else:
            print("   🌙 매우 어두운 사진이에요!")
    
    def display_color_spaces(self):
        """🌈 색깔의 다양한 표현 방식 보기 - 같은 그림을 다른 언어로 번역하는 것처럼!
        
        실생활 비유: 🎨 미술 시간에 같은 그림을 다양한 방법으로 표현하기
        - RGB: 빨강+초록+파랑 물감으로 섞기 
        - HSV: 색깔+진하기+밝기로 표현하기
        - LAB: 과학자들이 쓰는 특별한 방법
        """
        if self.current_image is None:
            print("😅 앗! 먼저 사진을 불러와주세요!")
            return
        
        # 색상 공간 변환
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
        img_gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        print("🎭 색깔 변신 쇼가 시작됩니다!")
        
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
        plt.show()
    
    def apply_convolution_filters(self):
        """🎭 Instagram 필터 6종 세트 - 마법의 돋보기로 사진 변신시키기!
        
        실생활 비유: 🔍 특수 안경이나 렌즈 착용하기
        - 선명하게 보는 안경 (Sharpen)
        - 몽환적인 렌즈 (Blur) 
        - 경계선 찾기 탐정경 (Edge Detection)
        - 각종 특수효과 렌즈들!
        """
        if self.current_image is None:
            print("😅 앗! 먼저 사진을 불러와주세요!")
            return
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        print("🎪 필터 마술쇼가 시작됩니다!")
        
        # 다양한 필터 정의 - 각각의 마법 수치들!
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
        
        for i, (name, kernel) in enumerate(filters.items()):
            if '원본' in name:
                result = gray
            else:
                result = cv2.filter2D(gray, -1, kernel)
            
            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(f'{name}')
            axes[i].axis('off')
            
            # 각 필터의 특징 설명을 위한 작은 텍스트 추가
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
        plt.show()
    
    def edge_detection_comparison(self):
        """🕵️ 탐정게임: 사진 속 경계선 찾기 대결!
        
        실생활 비유: 🖍️ 색칠공부 윤곽선 그리기
        - Sobel: 세로 방향과 가로 방향 따로 찾기
        - Laplacian: 모든 방향 한 번에 찾기  
        - Canny: 가장 똑똑한 탐정 (최고 추천!)
        """
        if self.current_image is None:
            print("😅 앗! 먼저 사진을 불러와주세요!")
            return
        
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
        
        print("🔍 경계선 탐정 대회가 시작됩니다!")
        
        # 결과 시각화
        images = [gray, sobel_x, sobel_y, sobel_combined, laplacian, canny]
        titles = ['📷 원본', '📐 세로탐정', '📏 가로탐정', '🎯 합체탐정', '💫 전방향탐정', '🏆 천재탐정']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('🕵️ 경계선 찾기 탐정 대회 - 누가 가장 잘 찾을까요?', fontsize=16)
        
        axes = axes.flatten()
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def exercise_1_basic_operations():
    """🎮 연습문제 1: 디지털 아트 창작 게임
    
    🎯 미션: 여러분은 디지털 아티스트가 되어 3가지 작품을 만들어야 합니다!
    """
    print("🎨 미션 1: 디지털 아트 창작 게임")
    print("=" * 50)
    print("🏆 목표: SNS에 올릴 멋진 디지털 아트 3작품 완성하기!")
    
    print("\\n🎯 미션 리스트:")
    print("1. 🏁 체스판 패턴 그리기 (100×100 크기)")
    print("2. ☀️ 어두운 사진을 밝게 만들기 (+50% 밝기)")  
    print("3. 🔄 사진을 멋지게 회전시키기 (90도 시계방향)")
    
    print("\\n🎁 도움말 상자:")
    print("📦 체스판 만들기: np.zeros() + 슬라이싱 마술")
    print("💡 밝기 조절: 사진 × 1.5 (하지만 255 넘지 않게!)")
    print("🌪️ 회전 마법: cv2.rotate() 또는 np.rot90() 사용")
    
    print("\\n🎪 실생활 비유:")
    print("- 체스판: 🧩 바둑판이나 타일 바닥 패턴 만들기")
    print("- 밝기: ☀️ 방의 전등을 더 밝게 키는 것")  
    print("- 회전: 🌪️ 액자를 벽에서 돌려서 걸기")
    
    print("\\n" + "="*50)
    print("✏️ 코딩 작업대 - 여기서 마법을 부려보세요!")
    print("="*50)
    
    # 1. 체크보드 패턴 생성
    def create_checkerboard(size=100, square_size=10):
        """🏁 체스판 만들기 - 흑백 네모들의 패턴 파티!"""
        print("🎨 체스판 그리는 중...")
        # TODO: 학생이 구현
        # 힌트: 바둑판처럼 검은색(0)과 하얀색(255)이 번갈아 나오게!
        pass
    
    # 2. 밝기 조절  
    def adjust_brightness(image, factor):
        """☀️ 사진 밝기 마법사 - 어두운 사진도 화사하게!"""
        print(f"✨ 밝기를 {factor}배 조절하는 중...")
        # TODO: 학생이 구현
        # 힌트: 모든 픽셀값에 factor를 곱하되, 255를 넘지 않게 조심!
        pass
    
    # 3. 이미지 회전
    def rotate_image(image, angle):
        """🌪️ 회전 마술사 - 사진을 빙글빙글 돌려보자!"""
        print(f"🔄 {angle}도 회전하는 중...")
        # TODO: 학생이 구현  
        # 힌트: cv2.rotate() 사용하거나 np.rot90() 활용
        pass
    
    print("\\n🎉 함수 3개를 완성했다면 호출해서 결과를 확인해보세요!")
    print("💡 예시: create_checkerboard(), adjust_brightness(my_image, 1.5)")
    
    # 완성 확인을 위한 체크리스트
    print("\\n✅ 완성 체크리스트:")
    print("□ 체스판이 제대로 나타나나요?")
    print("□ 밝기가 자연스럽게 조절되나요?")
    print("□ 회전이 정확히 90도인가요?")


def exercise_2_color_analysis():
    """🌈 연습문제 2: 컬러 탐정이 되어보자!
    
    🎯 미션: 여러분은 색깔 전문 탐정이 되어 사진 속 색깔의 비밀을 파헤쳐야 합니다!
    """
    print("🕵️ 미션 2: 컬러 탐정 수사대")
    print("=" * 50)
    print("🔍 목표: 사진 속 색깔들의 숨겨진 비밀 3가지 밝혀내기!")
    
    print("\\n🎯 수사 미션:")
    print("1. 🔴 특정 색깔 범죄자 잡기 (빨간색 영역만 추출)")
    print("2. 📊 색깔 인구조사 하기 (히스토그램 그래프)")
    print("3. 🏆 색깔 인기순위 Top 5 발표")
    
    print("\\n🔧 탐정 도구함:")
    print("🎨 HSV 색상공간 + cv2.inRange() = 색깔 체포기")
    print("📈 cv2.calcHist() = 색깔 계수기")  
    print("🔢 np.unique() + np.bincount() = 인기도 측정기")
    
    print("\\n🌟 실생활 예시:")
    print("- 색깔 찾기: 🍎 빨간 사과만 골라내기")
    print("- 히스토그램: 📊 반 학생들 키 분포 그래프") 
    print("- Top 5: 🎵 음악 차트 순위 매기기")
    
    # 학생 구현 공간
    # ========== 여기에 코드 작성 ==========
    
    def extract_color_range(image, lower_hsv, upper_hsv):
        """특정 색상 범위 추출"""
        # TODO: 학생이 구현
        pass
    
    def plot_color_histogram(image):
        """색상 히스토그램 생성"""
        # TODO: 학생이 구현
        pass
    
    def find_dominant_colors(image, k=5):
        """주요 색상 추출"""
        # TODO: 학생이 구현
        pass
    
    # =====================================


def exercise_3_custom_filter():
    """🎨 연습문제 3: 나만의 특수효과 제작소
    
    🎯 미션: 여러분은 Hollywood 특수효과 전문가가 되어 새로운 필터를 개발해야 합니다!
    """
    print("🎬 미션 3: Hollywood 특수효과 제작소")
    print("=" * 50)
    print("🌟 목표: 영화에서나 볼 법한 멋진 특수효과 3가지 만들기!")
    
    print("\\n🎯 특수효과 미션:")
    print("1. 🏺 고급스러운 엠보싱 효과 (동전 새기기 느낌)")
    print("2. 💫 속도감 있는 모션 블러 (빠르게 움직이는 느낌)")  
    print("3. 🎭 나만의 창의적 필터 (자유 창작!)")
    
    print("\\n🛠️ 특수효과 비법:")
    print("💎 엠보싱: [[-2,-1,0],[-1,1,1],[0,1,2]] - 동전처럼 도드라지게!")
    print("🌪️ 모션블러: 대각선 방향으로 흐르듯 번지게!")
    print("🎨 창의필터: 여러 효과 조합하거나 색상 마법 부리기!")
    
    print("\\n🎪 실생활 비유:")
    print("- 엠보싱: 🪙 동전 표면의 양각 무늬")
    print("- 모션블러: 🏎️ 빠른 차가 지나갈 때의 잔상")
    print("- 창의필터: 🎭 연극 무대의 조명 효과")
    
    # 학생 구현 공간
    # ========== 여기에 코드 작성 ==========
    
    def emboss_filter(image):
        """엠보싱 필터"""
        # TODO: 학생이 구현
        pass
    
    def motion_blur_filter(image, size=15, angle=45):
        """모션 블러 필터"""
        # TODO: 학생이 구현
        pass
    
    def creative_filter(image):
        """창의적 필터 (자유 구현)"""
        # TODO: 학생이 자유롭게 구현
        pass
    
    # =====================================


def demonstration_code():
    """🎪 시연: Instagram 필터 제작소 완전판!"""
    print("🎬 대형 시연쇼: Instagram 필터 제작소 완전판!")
    print("=" * 60)
    print("📱 실제 앱처럼 동작하는 필터 시스템을 구경하세요!")
    
    # 데모 이미지 생성 (실제 수업에서는 샘플 이미지 사용)
    demo_image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # 예쁜 데모 작품 그리기 🎨
    cv2.rectangle(demo_image, (50, 50), (150, 150), (255, 100, 100), -1)  # 따뜻한 빨강 사각형
    cv2.circle(demo_image, (100, 100), 30, (100, 255, 100), -1)  # 상큼한 녹색 원
    cv2.putText(demo_image, "DEMO", (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Instagram 필터 제작소 가동! 🏭
    processor = InstagramFilterMaker()
    processor.current_image = demo_image
    processor.original_image = demo_image.copy()
    
    print("\\n🏥 1단계: 사진 건강진단")
    processor.show_image_info()
    
    print("\\n🌈 2단계: 색깔 변신쇼")
    processor.display_color_spaces()
    
    print("\\n🎭 3단계: 필터 6종 세트 체험")
    processor.apply_convolution_filters()
    
    print("\\n🕵️ 4단계: 경계선 탐정 대회")
    processor.edge_detection_comparison()
    
    print("\\n🎉 시연 완료! 이제 여러분도 직접 해보세요!")


if __name__ == "__main__":
    """🎮 Instagram 필터 제작소 입구"""
    
    print("🎨 Instagram 필터 제작소에 오신 것을 환영합니다!")
    print("=" * 70)
    print("🎓 Week 1 Lab: 나만의 Instagram 필터 만들기 실습")
    print("딥러닝 영상처리 강의 - 1주차 실습")
    print("=" * 70)
    print("📱 오늘 우리는 Instagram, Snapchat 같은 필터를 직접 만들어볼 거예요!")
    
    # 메뉴 선택
    while True:
        print("\\n🎪 필터 제작소 메뉴:")
        print("1. 🎬 시연쇼 관람하기 (필터 작동 원리 구경)")
        print("2. 🎨 미션 1: 디지털 아트 창작 게임")  
        print("3. 🕵️ 미션 2: 컬러 탐정 수사대")
        print("4. 🎭 미션 3: Hollywood 특수효과 제작소")
        print("5. 🛠️ 자유 실습 모드 (창작의 자유!)")
        print("0. 🚪 나가기")
        
        try:
            choice = input("\\n선택하세요 (0-5): ")
            
            if choice == '0':
                print("🎉 수고하셨습니다! Instagram 필터 제작소를 이용해주셔서 감사해요!")
                print("📱 오늘 배운 것들로 멋진 필터를 만들어보세요!")
                break
            elif choice == '1':
                demonstration_code()
            elif choice == '2':
                exercise_1_basic_operations()
            elif choice == '3':
                exercise_2_color_analysis()
            elif choice == '4':
                exercise_3_custom_filter()
            elif choice == '5':
                print("🛠️ 자유 실습 모드 - 창작의 자유!")
                print("=" * 50)
                print("🎨 여기서는 뭐든 자유롭게 실험해보세요!")
                print("💡 아이디어: 나만의 필터 조합, 새로운 효과 실험...")
                
                # 이미지 프로세서 인스턴스 생성
                processor = InstagramFilterMaker()
                print("\\n✨ 필터 제작소가 준비되었습니다!")
                print("📁 사용법: processor.load_image('your_image.jpg')로 사진을 불러오세요.")
                print("🎭 그 다음: processor.apply_convolution_filters() 같은 함수들을 써보세요!")
                
            else:
                print("😅 잘못된 번호예요! 0-5 중에서 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\\n🎉 Instagram 필터 제작소를 나가시는군요!")
            print("📱 오늘 배운 필터 기술로 멋진 작품을 만들어보세요!")
            break
        except Exception as e:
            print(f"😅 앗! 예상치 못한 일이 일어났어요: {e}")
            print("💡 다시 시도해보시거나 선생님께 문의해보세요!")
    
    print("\\n🌟 Instagram 필터 제작소 방문을 마칩니다!")
    print("📸 오늘 배운 것:")
    print("   - 디지털 사진이 어떻게 구성되는지")
    print("   - 다양한 필터 효과들의 원리") 
    print("   - 컴퓨터가 경계선을 찾는 방법")
    print("🚀 다음 시간엔 더 고급 기술을 배워보겠습니다!")