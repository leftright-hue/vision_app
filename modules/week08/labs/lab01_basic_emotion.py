"""
Lab 01: 기본 감정 인식 (Basic Emotion Recognition)

이 실습에서는 Google Gemini API를 사용한 기본 감정 인식을 학습합니다:
- 7가지 기본 감정 인식 (happy, sad, angry, fear, surprise, disgust, neutral)
- 단일 이미지 분석
- 배치 이미지 처리
- JSON 형식 결과 출력

사용법:
    python lab01_basic_emotion.py --input image.jpg
    python lab01_basic_emotion.py --input images/ --batch
    python lab01_basic_emotion.py --input image.jpg --output result.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from modules.week08.emotion_helpers import EmotionHelper


def analyze_single_image(
    helper: EmotionHelper,
    image_path: str,
    verbose: bool = True
) -> Dict[str, float]:
    """
    단일 이미지의 감정을 분석합니다.

    Args:
        helper: EmotionHelper 인스턴스
        image_path: 이미지 파일 경로
        verbose: 상세 출력 여부

    Returns:
        감정 분석 결과 딕셔너리
    """
    if verbose:
        print(f"📷 이미지 분석: {image_path}")

    try:
        # 이미지 로드
        image = Image.open(image_path)

        if verbose:
            print(f"  - 크기: {image.width}x{image.height}")
            print(f"  - 모드: {image.mode}")

        # 감정 분석
        result = helper.analyze_basic_emotion(image)

        if verbose:
            print(f"  - API 모드: {helper.mode}")
            print()

        return result

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return {}


def analyze_batch_images(
    helper: EmotionHelper,
    input_dir: str,
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp'],
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    디렉토리 내 모든 이미지를 배치 분석합니다.

    Args:
        helper: EmotionHelper 인스턴스
        input_dir: 이미지 디렉토리 경로
        extensions: 처리할 파일 확장자 목록
        verbose: 상세 출력 여부

    Returns:
        파일명을 키로 하는 감정 분석 결과 딕셔너리
    """
    results = {}

    # 이미지 파일 찾기
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))

    if not image_files:
        print(f"⚠️ 디렉토리에서 이미지를 찾을 수 없습니다: {input_dir}")
        return results

    print(f"📁 배치 처리: {len(image_files)}개 이미지")
    print(f"  - API 모드: {helper.mode}")
    print()

    # 각 이미지 처리
    for i, image_path in enumerate(image_files, 1):
        if verbose:
            print(f"[{i}/{len(image_files)}] {image_path.name}")

        result = analyze_single_image(helper, str(image_path), verbose=False)

        if result:
            results[image_path.name] = result

            # 지배적 감정 표시
            if verbose:
                dominant = max(result.items(), key=lambda x: x[1])
                print(f"  → {dominant[0].upper()} ({dominant[1]:.2%})")

    print()
    print(f"✅ 배치 분석 완료: {len(results)}개 이미지")

    return results


def print_emotion_results(
    results: Dict[str, float],
    title: str = "감정 분석 결과",
    top_n: int = 3
):
    """
    감정 분석 결과를 보기 좋게 출력합니다.

    Args:
        results: 감정 분석 결과 딕셔너리
        title: 출력 제목
        top_n: 상위 N개 감정만 표시
    """
    print(f"\n{title}")
    print("=" * 50)

    # 신뢰도 순으로 정렬
    sorted_emotions = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Top N 표시
    print(f"\n🏆 상위 {top_n}개 감정:")
    for i, (emotion, score) in enumerate(sorted_emotions[:top_n], 1):
        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {i}. {emotion.capitalize():<10} {bar} {score:.2%}")

    # 전체 목록
    print(f"\n📊 전체 감정 신뢰도:")
    for emotion, score in sorted_emotions:
        print(f"  - {emotion.capitalize():<10} : {score:.4f}")


def save_results_to_json(
    results: Dict,
    output_path: str,
    pretty: bool = True
):
    """
    분석 결과를 JSON 파일로 저장합니다.

    Args:
        results: 저장할 결과 딕셔너리
        output_path: 출력 파일 경로
        pretty: 보기 좋게 포맷팅 여부
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                json.dump(results, f, ensure_ascii=False)

        print(f"💾 결과 저장: {output_path}")

    except Exception as e:
        print(f"❌ 저장 실패: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Lab 01: Google Gemini API를 사용한 기본 감정 인식"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 이미지 파일 또는 디렉토리 경로"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="배치 모드 (디렉토리 내 모든 이미지 처리)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="결과를 저장할 JSON 파일 경로"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="상위 N개 감정 표시 (기본값: 3)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="최소한의 출력만 표시"
    )

    args = parser.parse_args()

    # EmotionHelper 초기화
    print("🤖 감정 인식 시스템 초기화 중...")
    helper = EmotionHelper()
    print(f"✅ 초기화 완료: {helper.mode} 모드")
    print()

    # 입력 경로 검증
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ 입력 경로를 찾을 수 없습니다: {args.input}")
        return

    # 처리 모드 결정
    results = None

    if args.batch or input_path.is_dir():
        # 배치 모드
        results = analyze_batch_images(
            helper,
            str(input_path),
            verbose=not args.quiet
        )

        # 배치 결과 요약
        if results and not args.quiet:
            print("\n📈 배치 분석 요약:")

            # 전체 이미지에서 가장 많이 나타난 감정
            emotion_counts = {}
            for img_result in results.values():
                dominant = max(img_result.items(), key=lambda x: x[1])[0]
                emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1

            print("\n감정 분포:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(results) * 100
                print(f"  - {emotion.capitalize():<10} : {count}개 ({percentage:.1f}%)")

    else:
        # 단일 이미지 모드
        result = analyze_single_image(
            helper,
            str(input_path),
            verbose=not args.quiet
        )

        if result:
            results = {input_path.name: result}

            # 결과 출력
            if not args.quiet:
                print_emotion_results(result, top_n=args.top)

    # JSON 저장
    if args.output and results:
        save_results_to_json(results, args.output)

    # JSON 출력 (quiet 모드)
    if args.quiet and results:
        print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
