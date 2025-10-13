"""
Lab 02: 시간적 특징 추출 (Temporal Feature Extraction)

이 실습에서는 비디오의 시간적 특징을 추출하는 방법을 배웁니다:
- Optical Flow 계산 (Farneback 알고리즘)
- Optical Flow 시각화 (HSV, Arrow)
- 모션 크기 및 방향 분석
- 움직임 히트맵 생성

사용법:
    python lab02_temporal_features.py --input sample.mp4
    python lab02_temporal_features.py --webcam  # 실시간 Optical Flow
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Tuple, Optional


def compute_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    method: str = 'farneback'
) -> np.ndarray:
    """
    두 프레임 사이의 Optical Flow를 계산합니다.

    Args:
        frame1: 첫 번째 프레임 (grayscale 또는 BGR)
        frame2: 두 번째 프레임 (grayscale 또는 BGR)
        method: 'farneback' 또는 'lucas_kanade'

    Returns:
        flow: (H, W, 2) 형태의 optical flow (x, y 방향)
    """
    # Grayscale 변환
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1

    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2

    if method == 'farneback':
        # Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,      # 피라미드 스케일
            levels=3,           # 피라미드 레벨
            winsize=15,         # 윈도우 크기
            iterations=3,       # 반복 횟수
            poly_n=5,           # 다항식 확장 크기
            poly_sigma=1.2,     # 가우시안 표준편차
            flags=0
        )
    else:
        raise ValueError(f"지원하지 않는 메서드: {method}")

    return flow


def visualize_flow_hsv(flow: np.ndarray) -> np.ndarray:
    """
    Optical Flow를 HSV 색상 공간으로 시각화합니다.
    - Hue: 방향
    - Saturation: 255 (고정)
    - Value: 크기 (magnitude)

    Args:
        flow: (H, W, 2) Optical flow

    Returns:
        hsv_image: (H, W, 3) BGR 이미지
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # 크기와 각도 계산
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # HSV 값 설정
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: 방향 (0-179)
    hsv[..., 1] = 255                     # Saturation: 최대
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: 크기

    # BGR로 변환
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def visualize_flow_arrows(
    image: np.ndarray,
    flow: np.ndarray,
    step: int = 16,
    scale: float = 3.0,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Optical Flow를 화살표로 시각화합니다.

    Args:
        image: 원본 이미지
        flow: (H, W, 2) Optical flow
        step: 화살표 간격
        scale: 화살표 길이 스케일
        color: 화살표 색상 (BGR)

    Returns:
        annotated_image: 화살표가 그려진 이미지
    """
    h, w = flow.shape[:2]
    result = image.copy()

    # 그리드 포인트에서 화살표 그리기
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)

    for yi, xi in zip(y, x):
        fx, fy = flow[yi, xi]
        mag = np.sqrt(fx**2 + fy**2)

        # 움직임이 작으면 스킵
        if mag < 1.0:
            continue

        # 화살표 그리기
        cv2.arrowedLine(
            result,
            (xi, yi),
            (int(xi + fx * scale), int(yi + fy * scale)),
            color,
            1,
            tipLength=0.3
        )

    return result


def compute_motion_statistics(flow: np.ndarray) -> dict:
    """
    Optical Flow의 통계량을 계산합니다.

    Args:
        flow: (H, W, 2) Optical flow

    Returns:
        통계 딕셔너리
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    stats = {
        'mean_magnitude': float(np.mean(mag)),
        'max_magnitude': float(np.max(mag)),
        'median_magnitude': float(np.median(mag)),
        'std_magnitude': float(np.std(mag)),
        'dominant_direction': float(np.mean(ang) * 180 / np.pi)  # 도 단위
    }

    return stats


def create_motion_heatmap(flow: np.ndarray, history_length: int = 30) -> np.ndarray:
    """
    움직임 히트맵을 생성합니다 (누적 magnitude).

    Args:
        flow: (H, W, 2) Optical flow
        history_length: 히스토리 길이 (프레임 수)

    Returns:
        heatmap: (H, W, 3) BGR 히트맵 이미지
    """
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 정규화
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 컬러맵 적용
    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

    return heatmap


def process_video(
    video_path: str,
    output_dir: Optional[str] = None,
    visualize_mode: str = 'hsv'
):
    """
    비디오 파일에서 Optical Flow를 계산하고 시각화합니다.

    Args:
        video_path: 비디오 파일 경로
        output_dir: 시각화 결과를 저장할 디렉토리 (None이면 실시간 표시만)
        visualize_mode: 'hsv', 'arrows', 'heatmap'
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

    # 출력 디렉토리 생성
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    ret, prev_frame = cap.read()

    if not ret:
        raise ValueError("첫 번째 프레임을 읽을 수 없습니다")

    frame_idx = 0

    print(f"비디오 처리 중... (시각화 모드: {visualize_mode})")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Optical Flow 계산
        flow = compute_optical_flow(prev_frame, frame)

        # 통계 계산
        stats = compute_motion_statistics(flow)

        # 시각화
        if visualize_mode == 'hsv':
            vis = visualize_flow_hsv(flow)
        elif visualize_mode == 'arrows':
            vis = visualize_flow_arrows(frame, flow, step=16)
        elif visualize_mode == 'heatmap':
            vis = create_motion_heatmap(flow)
        else:
            vis = frame

        # 통계 정보 표시
        info_text = f"Frame {frame_idx} | Mag: {stats['mean_magnitude']:.2f}"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        # 화면 표시
        cv2.imshow('Optical Flow', vis)

        # 저장
        if output_dir:
            output_path = os.path.join(output_dir, f"flow_{frame_idx:04d}.jpg")
            cv2.imwrite(output_path, vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  {frame_idx} 프레임 처리됨...")

    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ 총 {frame_idx} 프레임 처리 완료")


def process_webcam(visualize_mode: str = 'hsv'):
    """
    웹캠에서 실시간 Optical Flow를 계산하고 시각화합니다.

    Args:
        visualize_mode: 'hsv', 'arrows', 'heatmap'
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise ValueError("웹캠을 열 수 없습니다")

    print("웹캠 시작됨. 종료하려면 'q' 키를 누르세요.")

    ret, prev_frame = cap.read()

    if not ret:
        raise ValueError("첫 번째 프레임을 읽을 수 없습니다")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Optical Flow 계산
        flow = compute_optical_flow(prev_frame, frame)

        # 통계 계산
        stats = compute_motion_statistics(flow)

        # 시각화
        if visualize_mode == 'hsv':
            vis = visualize_flow_hsv(flow)
        elif visualize_mode == 'arrows':
            vis = visualize_flow_arrows(frame, flow, step=16)
        elif visualize_mode == 'heatmap':
            vis = create_motion_heatmap(flow)
        else:
            vis = frame

        # 통계 정보 표시
        info = [
            f"Magnitude: {stats['mean_magnitude']:.2f}",
            f"Max: {stats['max_magnitude']:.2f}",
            f"Direction: {stats['dominant_direction']:.1f} deg"
        ]

        for i, text in enumerate(info):
            cv2.putText(vis, text, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Optical Flow - Webcam', vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ 총 {frame_count} 프레임 처리됨")


def main():
    parser = argparse.ArgumentParser(description="Lab 02: 시간적 특징 추출")
    parser.add_argument("--input", type=str, help="입력 비디오 파일 경로")
    parser.add_argument("--webcam", action="store_true", help="웹캠 사용")
    parser.add_argument("--output-dir", type=str, help="결과 저장 디렉토리")
    parser.add_argument("--mode", type=str, default='hsv',
                       choices=['hsv', 'arrows', 'heatmap'],
                       help="시각화 모드 (hsv, arrows, heatmap)")

    args = parser.parse_args()

    if args.webcam:
        # 웹캠 실시간 처리
        process_webcam(visualize_mode=args.mode)

    elif args.input:
        # 비디오 파일 처리
        print(f"📹 비디오 파일: {args.input}")
        print(f"🎨 시각화 모드: {args.mode}")

        try:
            process_video(
                args.input,
                output_dir=args.output_dir,
                visualize_mode=args.mode
            )
        except Exception as e:
            print(f"❌ 오류: {e}")

    else:
        parser.print_help()
        print("\n예제:")
        print("  python lab02_temporal_features.py --input sample.mp4 --mode hsv")
        print("  python lab02_temporal_features.py --webcam --mode arrows")


if __name__ == "__main__":
    main()
