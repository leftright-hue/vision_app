"""
Lab 01: 비디오 처리 기초 (Video Processing Basics)

이 실습에서는 OpenCV를 사용한 비디오 입출력 기초를 배웁니다:
- 비디오 파일 읽기 및 프레임 추출
- 비디오 정보 확인 (FPS, 해상도, 총 프레임 수)
- 프레임 저장 및 비디오 생성
- 실시간 웹캠 입력 처리

사용법:
    python lab01_video_basics.py --input sample.mp4
    python lab01_video_basics.py --webcam  # 웹캠 사용
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple


def get_video_info(video_path: str) -> dict:
    """
    비디오 파일의 메타데이터 정보를 추출합니다.

    Args:
        video_path: 비디오 파일 경로

    Returns:
        비디오 정보 딕셔너리 (fps, width, height, frame_count, duration)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': duration
    }


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_rate: int = 30,
    max_frames: Optional[int] = None
) -> int:
    """
    비디오에서 프레임을 추출하여 이미지로 저장합니다.

    Args:
        video_path: 비디오 파일 경로
        output_dir: 프레임을 저장할 디렉토리
        sample_rate: N 프레임마다 1개씩 추출 (기본값: 30)
        max_frames: 최대 추출 프레임 수 (None이면 무제한)

    Returns:
        추출된 프레임 수
    """
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

    frame_idx = 0
    saved_count = 0

    print(f"비디오 프레임 추출 중... (sample_rate={sample_rate})")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # sample_rate에 따라 프레임 샘플링
        if frame_idx % sample_rate == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

            if saved_count % 10 == 0:
                print(f"  {saved_count} 프레임 저장됨...")

            if max_frames and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    print(f"✅ 총 {saved_count}개 프레임 추출 완료")
    return saved_count


def create_video_from_frames(
    frames_dir: str,
    output_path: str,
    fps: float = 30.0,
    frame_pattern: str = "frame_%04d.jpg"
) -> bool:
    """
    이미지 프레임들을 모아 비디오 파일을 생성합니다.

    Args:
        frames_dir: 프레임 이미지가 있는 디렉토리
        output_path: 생성할 비디오 파일 경로
        fps: 비디오 FPS
        frame_pattern: 프레임 파일명 패턴

    Returns:
        성공 여부
    """
    # 첫 번째 프레임으로 크기 확인
    first_frame_path = os.path.join(frames_dir, frame_pattern % 0)

    if not os.path.exists(first_frame_path):
        print(f"❌ 첫 번째 프레임을 찾을 수 없습니다: {first_frame_path}")
        return False

    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]

    # VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    print(f"비디오 생성 중... (fps={fps})")

    while True:
        frame_path = os.path.join(frames_dir, frame_pattern % frame_idx)

        if not os.path.exists(frame_path):
            break

        frame = cv2.imread(frame_path)
        out.write(frame)

        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"  {frame_idx} 프레임 처리됨...")

    out.release()

    print(f"✅ 비디오 생성 완료: {output_path} ({frame_idx} 프레임)")
    return True


def process_webcam(display_info: bool = True):
    """
    웹캠에서 실시간 비디오를 읽어 화면에 표시합니다.

    Args:
        display_info: 화면에 프레임 정보를 표시할지 여부
    """
    cap = cv2.VideoCapture(0)  # 0 = 기본 웹캠

    if not cap.isOpened():
        raise ValueError("웹캠을 열 수 없습니다")

    print("웹캠 시작됨. 종료하려면 'q' 키를 누르세요.")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다")
            break

        # 프레임 정보 표시
        if display_info:
            info_text = f"Frame: {frame_count} | Size: {frame.shape[1]}x{frame.shape[0]}"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow('Webcam', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ 총 {frame_count} 프레임 처리됨")


def main():
    parser = argparse.ArgumentParser(description="Lab 01: 비디오 처리 기초")
    parser.add_argument("--input", type=str, help="입력 비디오 파일 경로")
    parser.add_argument("--webcam", action="store_true", help="웹캠 사용")
    parser.add_argument("--output-dir", type=str, default="output_frames",
                       help="프레임 출력 디렉토리")
    parser.add_argument("--sample-rate", type=int, default=30,
                       help="프레임 샘플링 비율 (N 프레임마다 1개)")
    parser.add_argument("--max-frames", type=int, help="최대 추출 프레임 수")
    parser.add_argument("--create-video", action="store_true",
                       help="추출된 프레임으로 비디오 생성")

    args = parser.parse_args()

    if args.webcam:
        # 웹캠 처리
        process_webcam(display_info=True)

    elif args.input:
        # 비디오 파일 처리
        print(f"📹 비디오 파일: {args.input}")

        # 1. 비디오 정보 확인
        try:
            info = get_video_info(args.input)
            print("\n비디오 정보:")
            print(f"  - FPS: {info['fps']:.2f}")
            print(f"  - 해상도: {info['width']}x{info['height']}")
            print(f"  - 총 프레임 수: {info['frame_count']}")
            print(f"  - 재생 시간: {info['duration']:.2f}초")
            print()
        except Exception as e:
            print(f"❌ 오류: {e}")
            return

        # 2. 프레임 추출
        try:
            saved_count = extract_frames(
                args.input,
                args.output_dir,
                sample_rate=args.sample_rate,
                max_frames=args.max_frames
            )
            print()
        except Exception as e:
            print(f"❌ 프레임 추출 실패: {e}")
            return

        # 3. 비디오 재생성 (옵션)
        if args.create_video and saved_count > 0:
            output_video = "output_reconstructed.mp4"
            try:
                create_video_from_frames(
                    args.output_dir,
                    output_video,
                    fps=info['fps'] / args.sample_rate
                )
            except Exception as e:
                print(f"❌ 비디오 생성 실패: {e}")

    else:
        parser.print_help()
        print("\n예제:")
        print("  python lab01_video_basics.py --input sample.mp4")
        print("  python lab01_video_basics.py --webcam")


if __name__ == "__main__":
    main()
