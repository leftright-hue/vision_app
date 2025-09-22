"""
01. 디지털 이미지의 구조와 표현
Week 2: 디지털 이미지 기초와 CNN

이 파일은 디지털 이미지의 기본 구조, 색상 공간,
메타데이터 등을 실습하는 코드입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import cv2
import os

class DigitalImageBasics:
    """디지털 이미지 기초 실습 클래스"""

    def __init__(self):
        self.setup_korean_font()

    def setup_korean_font(self):
        """한글 폰트 설정"""
        if os.name == 'nt':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

    def demonstrate_pixel_array(self):
        """1.1 픽셀과 이미지 배열 실습"""
        print("\n=== 1.1 픽셀과 이미지 배열 ===")

        # 그레이스케일 이미지 생성
        grayscale_image = np.array([
            [0,   50,  100, 150, 200],
            [10,  60,  110, 160, 210],
            [20,  70,  120, 170, 220],
            [30,  80,  130, 180, 230],
            [40,  90,  140, 190, 255]
        ], dtype=np.uint8)

        # 컬러 이미지 생성
        color_image = np.array([
            [[255, 0, 0], [0, 255, 0]],    # 빨강, 초록
            [[0, 0, 255], [255, 255, 255]]  # 파랑, 흰색
        ], dtype=np.uint8)

        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 그레이스케일 이미지 표시
        im1 = axes[0].imshow(grayscale_image, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('그레이스케일 이미지 (5x5)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0], label='픽셀 값')

        # 픽셀 값 표시
        for i in range(5):
            for j in range(5):
                axes[0].text(j, i, f'{grayscale_image[i,j]}',
                           ha='center', va='center', color='red', fontsize=10)

        # 컬러 이미지 표시
        axes[1].imshow(color_image)
        axes[1].set_title('RGB 컬러 이미지 (2x2)')
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])

        # RGB 값 표시
        colors = ['빨강\n(255,0,0)', '초록\n(0,255,0)',
                 '파랑\n(0,0,255)', '흰색\n(255,255,255)']
        positions = [(0,0), (1,0), (0,1), (1,1)]
        for pos, color_text in zip(positions, colors):
            axes[1].text(pos[0], pos[1], color_text,
                       ha='center', va='center', fontsize=9)

        # 3D 배열 구조 시각화
        axes[2].axis('off')
        axes[2].set_title('RGB 이미지의 3D 구조')

        # 3D 구조 설명 텍스트
        structure_text = """
        RGB 이미지 = 3차원 배열

        Shape: (높이, 너비, 채널)
        예: (2, 2, 3)

        채널:
        - R (Red): 빨강 채널
        - G (Green): 초록 채널
        - B (Blue): 파랑 채널

        각 채널 값: 0-255
        """
        axes[2].text(0.5, 0.5, structure_text, ha='center', va='center',
                    fontsize=11, bbox=dict(boxstyle="round", facecolor='lightblue'))

        plt.tight_layout()
        plt.savefig('01_pixel_array_demo.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 배열 정보 출력
        print(f"그레이스케일 이미지 shape: {grayscale_image.shape}")
        print(f"그레이스케일 이미지 dtype: {grayscale_image.dtype}")
        print(f"컬러 이미지 shape: {color_image.shape}")
        print(f"컬러 이미지 dtype: {color_image.dtype}")

    def demonstrate_color_spaces(self):
        """1.2 색상 공간 실습"""
        print("\n=== 1.2 색상 공간 (Color Spaces) ===")

        # 샘플 이미지 생성
        sample_img = self.create_sample_color_image()

        # 색상 공간 변환
        hsv_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2HSV)
        lab_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2LAB)
        gray_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY)

        # 시각화
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # RGB
        axes[0, 0].imshow(sample_img)
        axes[0, 0].set_title('RGB 원본')

        # RGB 채널별 표시
        channel_names = ['R (빨강)', 'G (초록)', 'B (파랑)']
        for i, name in enumerate(channel_names):
            axes[0, i+1].imshow(sample_img[:,:,i], cmap='gray')
            axes[0, i+1].set_title(f'{name} 채널')

        # HSV
        axes[1, 0].imshow(hsv_img)
        axes[1, 0].set_title('HSV 색상공간')

        # HSV 채널별 표시
        hsv_names = ['H (색상)', 'S (채도)', 'V (명도)']
        for i, name in enumerate(hsv_names):
            axes[1, i+1].imshow(hsv_img[:,:,i], cmap='gray')
            axes[1, i+1].set_title(f'{name}')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('01_color_spaces_demo.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("색상 공간 변환 완료:")
        print(f"- RGB shape: {sample_img.shape}")
        print(f"- HSV shape: {hsv_img.shape}")
        print(f"- Grayscale shape: {gray_img.shape}")

    def create_sample_color_image(self, size=100):
        """색상 그라데이션 샘플 이미지 생성"""
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # 색상 그라데이션 생성
        for i in range(size):
            for j in range(size):
                img[i, j, 0] = int(255 * i / size)  # R
                img[i, j, 1] = int(255 * j / size)  # G
                img[i, j, 2] = int(255 * (1 - i/size))  # B

        return img

    def demonstrate_image_metadata(self):
        """1.3 이미지 메타데이터 실습"""
        print("\n=== 1.3 이미지 메타데이터 ===")

        # 샘플 이미지 생성 및 저장
        sample_img = self.create_sample_color_image(200)
        img_pil = Image.fromarray(sample_img)

        # 기본 속성 출력
        print("기본 이미지 속성:")
        print(f"- 크기: {img_pil.size}")
        print(f"- 모드: {img_pil.mode}")
        print(f"- 포맷: {img_pil.format}")

        # 다양한 형식으로 저장 및 크기 비교
        formats = {
            'PNG': '01_sample.png',
            'JPEG': '01_sample.jpg',
            'BMP': '01_sample.bmp'
        }

        print("\n파일 형식별 크기 비교:")
        for format_name, filename in formats.items():
            img_pil.save(filename)
            file_size = os.path.getsize(filename) / 1024  # KB
            print(f"- {format_name}: {file_size:.2f} KB")

            # 파일 삭제 (정리)
            if os.path.exists(filename):
                os.remove(filename)

    def demonstrate_basic_operations(self):
        """1.5 이미지 처리 기본 연산"""
        print("\n=== 1.5 이미지 처리 기본 연산 ===")

        # 원본 이미지 생성
        original = np.array([
            [100, 150, 200],
            [120, 180, 220],
            [140, 160, 180]
        ], dtype=np.uint8)

        # 밝기 조정
        brighter = np.clip(original + 50, 0, 255).astype(np.uint8)
        darker = np.clip(original - 50, 0, 255).astype(np.uint8)

        # 대비 조정
        higher_contrast = np.clip(original * 1.5, 0, 255).astype(np.uint8)
        lower_contrast = np.clip(original * 0.5, 0, 255).astype(np.uint8)

        # 감마 보정
        gamma = 2.2
        gamma_corrected = np.power(original / 255.0, gamma) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)

        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        images = [original, brighter, darker,
                 higher_contrast, lower_contrast, gamma_corrected]
        titles = ['원본', '밝기 +50', '밝기 -50',
                 '대비 x1.5', '대비 x0.5', f'감마 {gamma}']

        for ax, img, title in zip(axes.flat, images, titles):
            im = ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(title)

            # 픽셀 값 표시
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{img[i,j]}', ha='center', va='center',
                          color='red', fontsize=10)

            ax.set_xticks(range(3))
            ax.set_yticks(range(3))

        plt.tight_layout()
        plt.savefig('01_basic_operations.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("원본 픽셀 값:")
        print(original)
        print("\n밝기 증가 (+50):")
        print(brighter)
        print("\n대비 증가 (x1.5):")
        print(higher_contrast)

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("📚 01. 디지털 이미지의 구조와 표현")
    print("=" * 60)

    basics = DigitalImageBasics()

    # 1.1 픽셀과 이미지 배열
    basics.demonstrate_pixel_array()

    # 1.2 색상 공간
    basics.demonstrate_color_spaces()

    # 1.3 이미지 메타데이터
    basics.demonstrate_image_metadata()

    # 1.5 기본 연산
    basics.demonstrate_basic_operations()

    print("\n" + "=" * 60)
    print("✅ 01. 디지털 이미지 기초 실습 완료!")
    print("생성된 파일:")
    print("  - 01_pixel_array_demo.png")
    print("  - 01_color_spaces_demo.png")
    print("  - 01_basic_operations.png")
    print("=" * 60)

if __name__ == "__main__":
    main()