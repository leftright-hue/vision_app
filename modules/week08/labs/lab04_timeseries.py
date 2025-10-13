"""
Lab 04: 시계열 감정 분석 (Time Series Emotion Analysis)

이 실습에서는 여러 이미지 또는 비디오에서 시계열 감정 변화를 분석합니다:
- 여러 이미지 순차 분석
- 비디오 프레임 추출 및 분석
- 시계열 그래프 시각화
- 감정 변화점 탐지
- CSV 파일 저장

사용법:
    python lab04_timeseries.py --images image1.jpg image2.jpg image3.jpg
    python lab04_timeseries.py --input-dir frames/
    python lab04_timeseries.py --video video.mp4 --sample-rate 30
    python lab04_timeseries.py --images *.jpg --output timeline.png --csv results.csv
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional
from PIL import Image
import matplotlib.pyplot as plt

# OpenCV는 선택적 의존성
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from modules.week08.emotion_helpers import EmotionHelper, EmotionTimeSeries


def extract_video_frames(
    video_path: str,
    sample_rate: int = 30,
    max_frames: Optional[int] = None
) -> List[Image.Image]:
    """
    비디오에서 프레임을 추출합니다.

    Args:
        video_path: 비디오 파일 경로
        sample_rate: N 프레임마다 1개씩 추출
        max_frames: 최대 추출 프레임 수

    Returns:
        PIL Image 리스트
    """
    if not HAS_OPENCV:
        raise ImportError("비디오 처리를 위해 OpenCV가 필요합니다: pip install opencv-python")

    print(f"📹 비디오 프레임 추출: {video_path}")
    print(f"   - 샘플링 비율: {sample_rate} 프레임마다 1개")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

    # 비디오 정보
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"   - 총 프레임 수: {total_frames}")
    print(f"   - FPS: {fps:.2f}")

    frames = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 샘플링
        if frame_idx % sample_rate == 0:
            # OpenCV BGR -> RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)
            saved_count += 1

            if saved_count % 10 == 0:
                print(f"   ... {saved_count}개 프레임 추출됨")

            if max_frames and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    print(f"✅ 총 {len(frames)}개 프레임 추출 완료")

    return frames


def load_images_from_paths(image_paths: List[str]) -> List[Image.Image]:
    """
    여러 이미지 파일을 로드합니다.

    Args:
        image_paths: 이미지 파일 경로 리스트

    Returns:
        PIL Image 리스트
    """
    print(f"📁 이미지 로드 중... ({len(image_paths)}개)")

    images = []

    for i, path in enumerate(image_paths, 1):
        try:
            img = Image.open(path)
            images.append(img)

            if i % 10 == 0:
                print(f"   ... {i}개 로드됨")

        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {path} - {e}")

    print(f"✅ 총 {len(images)}개 이미지 로드 완료")

    return images


def analyze_timeseries(
    helper: EmotionHelper,
    images: List[Image.Image],
    verbose: bool = True
) -> EmotionTimeSeries:
    """
    이미지 리스트를 순차적으로 분석하여 시계열 데이터를 생성합니다.

    Args:
        helper: EmotionHelper 인스턴스
        images: PIL Image 리스트
        verbose: 상세 출력 여부

    Returns:
        EmotionTimeSeries 인스턴스
    """
    timeseries = EmotionTimeSeries(window_size=len(images))

    print(f"\n🔍 감정 분석 시작 ({len(images)}개 이미지)")
    print(f"   - API 모드: {helper.mode}")

    start_time = time.time()

    for i, image in enumerate(images):
        if verbose and (i % 5 == 0 or i == len(images) - 1):
            print(f"   [{i+1}/{len(images)}] 분석 중...")

        # 감정 분석
        emotions = helper.analyze_basic_emotion(image)

        # 시계열에 추가
        timeseries.add_frame(emotions, timestamp=i)

    elapsed = time.time() - start_time

    print(f"✅ 분석 완료 (소요 시간: {elapsed:.2f}초)")
    print(f"   - 프레임당 평균: {elapsed/len(images):.3f}초")

    return timeseries


def print_timeseries_summary(timeseries: EmotionTimeSeries):
    """
    시계열 분석 결과 요약을 출력합니다.

    Args:
        timeseries: EmotionTimeSeries 인스턴스
    """
    summary = timeseries.get_summary()

    print("\n" + "=" * 60)
    print("📊 시계열 분석 요약")
    print("=" * 60)

    print(f"\n총 프레임 수: {summary['total_frames']}")
    print(f"지배적 감정: {summary['dominant_emotion'].upper()}")
    print(f"평균 신뢰도: {summary['avg_confidence']:.2%}")

    # 변화점
    change_points = summary['change_points']
    print(f"\n감정 변화점: {len(change_points)}개")

    if change_points:
        print("  변화 발생 프레임:")
        for idx in change_points[:5]:
            print(f"    - 프레임 {idx+1}")
        if len(change_points) > 5:
            print(f"    ... 외 {len(change_points)-5}개")

    # 트렌드 분석
    print("\n📈 감정 트렌드:")
    emotions_to_check = ['happy', 'sad', 'angry', 'fear']

    for emotion in emotions_to_check:
        trend = timeseries.get_trend(emotion)
        trend_symbols = {
            'increasing': '↑ 상승',
            'decreasing': '↓ 하락',
            'stable': '→ 안정'
        }
        print(f"  - {emotion.capitalize():<10} : {trend_symbols.get(trend, trend)}")


def save_timeline_plot(
    timeseries: EmotionTimeSeries,
    output_path: str
):
    """
    시계열 그래프를 저장합니다.

    Args:
        timeseries: EmotionTimeSeries 인스턴스
        output_path: 저장할 파일 경로
    """
    print(f"\n📈 시계열 그래프 생성 중...")

    try:
        fig = timeseries.visualize_timeline()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"💾 그래프 저장: {output_path}")

    except Exception as e:
        print(f"❌ 그래프 저장 실패: {e}")


def save_csv_export(
    timeseries: EmotionTimeSeries,
    output_path: str
):
    """
    시계열 데이터를 CSV로 저장합니다.

    Args:
        timeseries: EmotionTimeSeries 인스턴스
        output_path: 저장할 파일 경로
    """
    print(f"\n💾 CSV 내보내기...")

    try:
        timeseries.export_to_csv(output_path)
        print(f"✅ CSV 저장: {output_path}")

    except Exception as e:
        print(f"❌ CSV 저장 실패: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Lab 04: 시계열 감정 분석"
    )

    # 입력 옵션
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--images",
        type=str,
        nargs='+',
        help="분석할 이미지 파일 경로 (여러 개)"
    )

    input_group.add_argument(
        "--input-dir",
        type=str,
        help="이미지 디렉토리 경로"
    )

    input_group.add_argument(
        "--video",
        type=str,
        help="비디오 파일 경로"
    )

    # 비디오 옵션
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=30,
        help="비디오 샘플링 비율 (N 프레임마다 1개, 기본값: 30)"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help="최대 추출 프레임 수"
    )

    # 출력 옵션
    parser.add_argument(
        "--output",
        type=str,
        help="시계열 그래프 저장 경로 (예: timeline.png)"
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="CSV 파일 저장 경로 (예: results.csv)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="최소한의 출력만 표시"
    )

    args = parser.parse_args()

    # EmotionHelper 초기화
    if not args.quiet:
        print("🤖 감정 인식 시스템 초기화 중...")

    helper = EmotionHelper()

    if not args.quiet:
        print(f"✅ 초기화 완료: {helper.mode} 모드\n")

    # 이미지 수집
    images = None

    if args.images:
        # 명시적 이미지 파일 목록
        images = load_images_from_paths(args.images)

    elif args.input_dir:
        # 디렉토리에서 이미지 찾기
        input_dir = Path(args.input_dir)

        if not input_dir.exists():
            print(f"❌ 디렉토리를 찾을 수 없습니다: {args.input_dir}")
            return

        # 이미지 파일 찾기
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files = []

        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))

        if not image_files:
            print(f"❌ 디렉토리에서 이미지를 찾을 수 없습니다: {args.input_dir}")
            return

        images = load_images_from_paths([str(f) for f in image_files])

    elif args.video:
        # 비디오 프레임 추출
        video_path = Path(args.video)

        if not video_path.exists():
            print(f"❌ 비디오 파일을 찾을 수 없습니다: {args.video}")
            return

        try:
            images = extract_video_frames(
                str(video_path),
                sample_rate=args.sample_rate,
                max_frames=args.max_frames
            )
        except Exception as e:
            print(f"❌ 비디오 처리 실패: {e}")
            return

    if not images:
        print("❌ 분석할 이미지가 없습니다")
        return

    # 시계열 분석
    timeseries = analyze_timeseries(helper, images, verbose=not args.quiet)

    # 결과 요약
    if not args.quiet:
        print_timeseries_summary(timeseries)

    # 시계열 그래프 저장
    if args.output:
        save_timeline_plot(timeseries, args.output)

    # CSV 저장
    if args.csv:
        save_csv_export(timeseries, args.csv)

    if not args.quiet:
        print("\n✅ 모든 작업 완료")


if __name__ == "__main__":
    main()
