"""
02. 이미지 필터링과 Convolution 연산
Week 2: 디지털 이미지 기초와 CNN

이 파일은 이미지 필터링의 원리와 Convolution 연산을
구현하고 시각화하는 코드입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
import os

class ImageFilteringConvolution:
    """이미지 필터링과 Convolution 실습 클래스"""

    def __init__(self):
        self.setup_korean_font()
        self.filters = self.create_filters()

    def setup_korean_font(self):
        """한글 폰트 설정"""
        if os.name == 'nt':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

    def create_filters(self):
        """다양한 필터 생성"""
        filters = {
            '박스 필터': np.ones((3, 3)) / 9,

            '가우시안': np.array([
                [1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]
            ]),

            'Sobel 수직': np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]),

            'Sobel 수평': np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]),

            '라플라시안': np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ]),

            '샤프닝': np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]),

            '엣지 검출': np.array([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ])
        }
        return filters

    def manual_convolution_2d(self, image, kernel, padding=0, stride=1):
        """
        2.2 수동 Convolution 구현

        Args:
            image: 입력 이미지
            kernel: 필터/커널
            padding: 패딩 크기
            stride: 스트라이드

        Returns:
            convolved: 합성곱 결과
        """
        # 패딩 적용
        if padding > 0:
            image = np.pad(image, padding, mode='constant')

        img_height, img_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # 출력 크기 계산
        output_height = (img_height - kernel_height) // stride + 1
        output_width = (img_width - kernel_width) // stride + 1

        # 출력 배열 초기화
        output = np.zeros((output_height, output_width))

        # Convolution 연산
        for i in range(output_height):
            for j in range(output_width):
                # 현재 위치
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + kernel_height
                w_end = w_start + kernel_width

                # 윈도우 추출
                window = image[h_start:h_end, w_start:w_end]

                # 요소별 곱셈 후 합계
                output[i, j] = np.sum(window * kernel)

        return output

    def demonstrate_convolution_process(self):
        """2.2 Convolution 연산 과정 시각화"""
        print("\n=== 2.2 Convolution 연산 과정 ===")

        # 작은 테스트 이미지
        test_image = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9]
        ], dtype=float)

        # 엣지 검출 커널
        kernel = self.filters['엣지 검출']

        # 단계별 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 입력 이미지
        im1 = axes[0, 0].imshow(test_image, cmap='gray')
        axes[0, 0].set_title('입력 이미지 (5x5)')
        for i in range(5):
            for j in range(5):
                axes[0, 0].text(j, i, f'{test_image[i,j]:.0f}',
                              ha='center', va='center', color='red')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. 커널
        im2 = axes[0, 1].imshow(kernel, cmap='RdBu', vmin=-8, vmax=8)
        axes[0, 1].set_title('커널 (3x3)')
        for i in range(3):
            for j in range(3):
                axes[0, 1].text(j, i, f'{kernel[i,j]:.0f}',
                              ha='center', va='center',
                              color='white' if abs(kernel[i,j]) > 2 else 'black')
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. Convolution 과정 (첫 번째 위치)
        axes[0, 2].imshow(test_image, cmap='gray', alpha=0.3)
        rect = plt.Rectangle((0, 0), 3, 3, linewidth=3,
                            edgecolor='red', facecolor='none')
        axes[0, 2].add_patch(rect)
        axes[0, 2].set_title('슬라이딩 윈도우')

        # 윈도우 내 계산 표시
        window = test_image[0:3, 0:3]
        result = np.sum(window * kernel)
        axes[0, 2].text(1, 1, f'결과:\n{result:.1f}',
                      ha='center', va='center',
                      bbox=dict(boxstyle="round", facecolor='yellow'))

        # 4. 전체 Convolution 결과
        conv_result = self.manual_convolution_2d(test_image, kernel)
        im4 = axes[1, 0].imshow(conv_result, cmap='coolwarm')
        axes[1, 0].set_title('Convolution 결과 (3x3)')
        for i in range(conv_result.shape[0]):
            for j in range(conv_result.shape[1]):
                axes[1, 0].text(j, i, f'{conv_result[i,j]:.1f}',
                              ha='center', va='center')
        plt.colorbar(im4, ax=axes[1, 0])

        # 5. 패딩 적용 예시
        padded_result = self.manual_convolution_2d(test_image, kernel, padding=1)
        im5 = axes[1, 1].imshow(padded_result, cmap='coolwarm')
        axes[1, 1].set_title('패딩 적용 결과 (5x5)')
        plt.colorbar(im5, ax=axes[1, 1])

        # 6. 스트라이드 2 예시
        stride2_result = self.manual_convolution_2d(test_image, kernel, stride=2)
        im6 = axes[1, 2].imshow(stride2_result, cmap='coolwarm')
        axes[1, 2].set_title('스트라이드=2 결과 (2x2)')
        for i in range(stride2_result.shape[0]):
            for j in range(stride2_result.shape[1]):
                axes[1, 2].text(j, i, f'{stride2_result[i,j]:.1f}',
                              ha='center', va='center')
        plt.colorbar(im6, ax=axes[1, 2])

        plt.tight_layout()
        plt.savefig('02_convolution_process.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 출력 크기 계산 공식
        print("\n📐 출력 크기 계산 공식:")
        print("Output = (Input - Kernel + 2×Padding) / Stride + 1")
        print(f"\n예시:")
        print(f"- 입력: 5x5, 커널: 3x3, 패딩: 0, 스트라이드: 1")
        print(f"  출력: (5 - 3 + 0) / 1 + 1 = 3x3")
        print(f"- 입력: 5x5, 커널: 3x3, 패딩: 1, 스트라이드: 1")
        print(f"  출력: (5 - 3 + 2) / 1 + 1 = 5x5")

    def demonstrate_filter_effects(self):
        """2.3 주요 필터 효과 시각화"""
        print("\n=== 2.3 주요 필터 유형과 응용 ===")

        # 샘플 이미지 생성
        sample_img = self.create_sample_image()

        # 필터 적용
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flat

        # 원본
        axes[0].imshow(sample_img, cmap='gray')
        axes[0].set_title('원본 이미지', fontsize=12, fontweight='bold')

        # 각 필터 적용
        filter_names = list(self.filters.keys())
        for idx, filter_name in enumerate(filter_names[:8]):
            filtered = cv2.filter2D(sample_img, -1, self.filters[filter_name])
            axes[idx+1].imshow(filtered, cmap='gray')
            axes[idx+1].set_title(filter_name, fontsize=12, fontweight='bold')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('02_filter_effects.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 필터별 설명
        print("\n📋 필터별 용도:")
        print("- 박스 필터: 평균값으로 노이즈 제거")
        print("- 가우시안: 자연스러운 블러 효과")
        print("- Sobel 수직/수평: 방향성 있는 엣지 검출")
        print("- 라플라시안: 2차 미분으로 엣지 검출")
        print("- 샤프닝: 이미지 선명도 증가")
        print("- 엣지 검출: 모든 방향의 엣지 강조")

    def demonstrate_edge_detection_comparison(self):
        """엣지 검출 필터 비교"""
        print("\n=== 엣지 검출 필터 비교 ===")

        # 샘플 이미지
        img = self.create_sample_image()

        # 엣지 검출 필터들
        sobel_x = cv2.filter2D(img, -1, self.filters['Sobel 수직'])
        sobel_y = cv2.filter2D(img, -1, self.filters['Sobel 수평'])
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        laplacian = cv2.filter2D(img, -1, self.filters['라플라시안'])

        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        images = [img, sobel_x, sobel_y, sobel_combined, laplacian]
        titles = ['원본', 'Sobel X (수직 엣지)', 'Sobel Y (수평 엣지)',
                 'Sobel 합성', '라플라시안']

        for ax, image, title in zip(axes.flat[:5], images, titles):
            ax.imshow(image, cmap='gray')
            ax.set_title(title, fontsize=12)
            ax.axis('off')

        # 엣지 방향 시각화
        axes[1, 2].axis('off')
        axes[1, 2].set_title('엣지 방향과 강도', fontsize=12)

        # 그라디언트 방향 계산
        angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        magnitude = sobel_combined

        # 퀴버 플롯
        y, x = np.mgrid[0:img.shape[0]:10, 0:img.shape[1]:10]
        u = sobel_x[::10, ::10]
        v = sobel_y[::10, ::10]
        axes[1, 2].quiver(x, y, u, v, magnitude[::10, ::10], cmap='hot')
        axes[1, 2].set_xlim(0, img.shape[1])
        axes[1, 2].set_ylim(img.shape[0], 0)

        plt.tight_layout()
        plt.savefig('02_edge_detection_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def create_sample_image(self, size=100):
        """샘플 이미지 생성"""
        img = np.zeros((size, size))

        # 다양한 패턴 추가
        # 원
        center = size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        img[mask] = 200

        # 사각형
        img[20:40, 20:40] = 150
        img[60:80, 60:80] = 100

        # 대각선
        for i in range(min(size, 80)):
            img[i, i] = 255
            if size-1-i >= 0:
                img[i, size-1-i] = 180

        # 노이즈 추가
        noise = np.random.normal(0, 10, (size, size))
        img = np.clip(img + noise, 0, 255)

        return img.astype(np.uint8)

    def demonstrate_frequency_domain(self):
        """주파수 도메인에서의 필터링"""
        print("\n=== 주파수 도메인 필터링 ===")

        # 샘플 이미지
        img = self.create_sample_image()

        # FFT 변환
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

        # 저주파 통과 필터 (Low-pass filter)
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2

        # 마스크 생성
        mask_low = np.zeros((rows, cols), np.uint8)
        r = 30  # 반경
        center = (crow, ccol)
        cv2.circle(mask_low, center, r, 1, -1)

        # 고주파 통과 필터 (High-pass filter)
        mask_high = 1 - mask_low

        # 필터 적용
        f_shift_low = f_shift * mask_low
        f_shift_high = f_shift * mask_high

        # 역변환
        img_low = np.fft.ifft2(np.fft.ifftshift(f_shift_low))
        img_low = np.abs(img_low)
        img_high = np.fft.ifft2(np.fft.ifftshift(f_shift_high))
        img_high = np.abs(img_high)

        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('원본 이미지')

        axes[0, 1].imshow(magnitude_spectrum, cmap='gray')
        axes[0, 1].set_title('주파수 스펙트럼')

        axes[0, 2].imshow(mask_low, cmap='gray')
        axes[0, 2].set_title('저주파 통과 마스크')

        axes[1, 0].imshow(img_low, cmap='gray')
        axes[1, 0].set_title('저주파 통과 결과 (블러)')

        axes[1, 1].imshow(mask_high, cmap='gray')
        axes[1, 1].set_title('고주파 통과 마스크')

        axes[1, 2].imshow(img_high, cmap='gray')
        axes[1, 2].set_title('고주파 통과 결과 (엣지)')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('02_frequency_domain.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🔍 02. 이미지 필터링과 Convolution 연산")
    print("=" * 60)

    filtering = ImageFilteringConvolution()

    # 2.2 Convolution 연산 과정
    filtering.demonstrate_convolution_process()

    # 2.3 주요 필터 효과
    filtering.demonstrate_filter_effects()

    # 엣지 검출 비교
    filtering.demonstrate_edge_detection_comparison()

    # 주파수 도메인 필터링
    filtering.demonstrate_frequency_domain()

    print("\n" + "=" * 60)
    print("✅ 02. 이미지 필터링 실습 완료!")
    print("생성된 파일:")
    print("  - 02_convolution_process.png")
    print("  - 02_filter_effects.png")
    print("  - 02_edge_detection_comparison.png")
    print("  - 02_frequency_domain.png")
    print("=" * 60)

if __name__ == "__main__":
    main()