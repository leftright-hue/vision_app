"""
Lab 05: API 성능 비교 (API Performance Comparison)

이 실습에서는 여러 감정 인식 API의 성능을 비교합니다:
- Google Gemini vs OpenAI GPT-4o vs Simulation
- 정확도 (일관성) 비교
- 속도 (응답 시간) 비교
- 비용 추정 비교
- 벤치마크 결과 테이블 출력

사용법:
    python lab05_comparison.py --input image.jpg
    python lab05_comparison.py --input image.jpg --runs 5
    python lab05_comparison.py --batch images/ --output benchmark.txt
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import statistics

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from modules.week08.emotion_helpers import EmotionHelper


# API 비용 추정 (USD)
API_COSTS = {
    'gemini': {
        'per_1k_images': 0.0025,  # Gemini Pro Vision (가상 예시)
        'name': 'Google Gemini 2.5 Pro'
    },
    'openai': {
        'per_1k_images': 0.01,    # GPT-4o Vision
        'name': 'OpenAI GPT-4o'
    },
    'simulation': {
        'per_1k_images': 0.0,     # 무료
        'name': 'Simulation Mode'
    }
}


def benchmark_api(
    api_mode: str,
    image: Image.Image,
    runs: int = 3
) -> Dict:
    """
    특정 API 모드의 성능을 벤치마크합니다.

    Args:
        api_mode: API 모드 ('gemini', 'openai', 'simulation')
        image: 테스트 이미지
        runs: 반복 실행 횟수

    Returns:
        벤치마크 결과 딕셔너리
    """
    print(f"  🔄 {API_COSTS[api_mode]['name']} 테스트 중... ({runs}회 실행)")

    # EmotionHelper 초기화
    helper = EmotionHelper()

    # 강제로 특정 모드 설정
    if api_mode == 'gemini' and helper.gemini_model is None:
        print(f"    ⚠️ Gemini API를 사용할 수 없습니다. 건너뜁니다.")
        return None

    if api_mode == 'openai' and helper.openai_client is None:
        print(f"    ⚠️ OpenAI API를 사용할 수 없습니다. 건너뜁니다.")
        return None

    # 모드 강제 설정
    helper.mode = api_mode

    times = []
    results = []

    for i in range(runs):
        start = time.time()

        try:
            result = helper.analyze_basic_emotion(image)
            elapsed = time.time() - start

            times.append(elapsed)
            results.append(result)

        except Exception as e:
            print(f"    ❌ 실행 {i+1} 실패: {e}")
            continue

    if not times:
        return None

    # 통계 계산
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0

    # 일관성 계산 (결과 간 표준편차)
    consistency = calculate_consistency(results)

    # 비용 계산 (1000회 기준)
    cost_per_1k = API_COSTS[api_mode]['per_1k_images']

    return {
        'mode': api_mode,
        'name': API_COSTS[api_mode]['name'],
        'runs': len(times),
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'consistency': consistency,
        'cost_per_1k': cost_per_1k,
        'results': results
    }


def calculate_consistency(results: List[Dict[str, float]]) -> float:
    """
    여러 분석 결과의 일관성을 계산합니다.

    Args:
        results: 감정 분석 결과 리스트

    Returns:
        일관성 점수 (0.0 ~ 1.0, 높을수록 일관성 높음)
    """
    if len(results) < 2:
        return 1.0

    # 각 감정별 표준편차 계산
    emotions = list(results[0].keys())
    variances = []

    for emotion in emotions:
        values = [r[emotion] for r in results]
        if len(set(values)) > 1:
            variance = statistics.variance(values)
            variances.append(variance)

    if not variances:
        return 1.0

    # 평균 분산 → 일관성 점수로 변환
    avg_variance = statistics.mean(variances)
    consistency = max(0.0, 1.0 - (avg_variance * 10))  # 스케일 조정

    return consistency


def print_comparison_table(benchmarks: List[Dict]):
    """
    벤치마크 결과를 테이블 형식으로 출력합니다.

    Args:
        benchmarks: 벤치마크 결과 리스트
    """
    print("\n" + "=" * 80)
    print("📊 API 성능 비교 결과")
    print("=" * 80)

    # 헤더
    print(f"\n{'API':<25} {'평균 시간':<12} {'일관성':<10} {'비용(1K)':<12} {'실행 횟수':<8}")
    print("-" * 80)

    # 각 API 결과
    for bench in benchmarks:
        if bench is None:
            continue

        print(f"{bench['name']:<25} "
              f"{bench['avg_time']:<12.3f}초 "
              f"{bench['consistency']:<10.2%} "
              f"${bench['cost_per_1k']:<11.4f} "
              f"{bench['runs']:<8}회")

    # 상세 통계
    print("\n" + "=" * 80)
    print("📈 상세 통계")
    print("=" * 80)

    for bench in benchmarks:
        if bench is None:
            continue

        print(f"\n{bench['name']}:")
        print(f"  - 평균 응답 시간: {bench['avg_time']:.3f}초")
        print(f"  - 최소 응답 시간: {bench['min_time']:.3f}초")
        print(f"  - 최대 응답 시간: {bench['max_time']:.3f}초")
        print(f"  - 표준 편차: {bench['std_time']:.3f}초")
        print(f"  - 일관성 점수: {bench['consistency']:.2%}")
        print(f"  - 1,000회 비용: ${bench['cost_per_1k']:.4f}")

        # 지배적 감정 표시
        if bench['results']:
            dominant_emotions = []
            for result in bench['results']:
                dominant = max(result.items(), key=lambda x: x[1])[0]
                dominant_emotions.append(dominant)

            # 가장 많이 나타난 감정
            from collections import Counter
            emotion_counts = Counter(dominant_emotions)
            most_common = emotion_counts.most_common(1)[0]

            print(f"  - 주요 감정: {most_common[0].upper()} ({most_common[1]}/{bench['runs']}회)")


def print_winner_summary(benchmarks: List[Dict]):
    """
    각 카테고리별 최고 성능 API를 요약합니다.

    Args:
        benchmarks: 벤치마크 결과 리스트
    """
    valid_benchmarks = [b for b in benchmarks if b is not None]

    if not valid_benchmarks:
        return

    print("\n" + "=" * 80)
    print("🏆 카테고리별 우승자")
    print("=" * 80)

    # 속도 우승자
    fastest = min(valid_benchmarks, key=lambda x: x['avg_time'])
    print(f"\n⚡ 최고 속도: {fastest['name']}")
    print(f"   - 평균 응답 시간: {fastest['avg_time']:.3f}초")

    # 일관성 우승자
    most_consistent = max(valid_benchmarks, key=lambda x: x['consistency'])
    print(f"\n✅ 최고 일관성: {most_consistent['name']}")
    print(f"   - 일관성 점수: {most_consistent['consistency']:.2%}")

    # 비용 효율성 우승자
    cheapest = min(valid_benchmarks, key=lambda x: x['cost_per_1k'])
    print(f"\n💰 최고 비용 효율: {cheapest['name']}")
    print(f"   - 1,000회 비용: ${cheapest['cost_per_1k']:.4f}")

    # 종합 점수 (정규화된 속도 + 일관성 - 비용)
    for bench in valid_benchmarks:
        speed_score = 1.0 - (bench['avg_time'] / max(b['avg_time'] for b in valid_benchmarks))
        consistency_score = bench['consistency']
        cost_score = 1.0 - (bench['cost_per_1k'] / max(b['cost_per_1k'] for b in valid_benchmarks if b['cost_per_1k'] > 0))

        bench['overall_score'] = (speed_score + consistency_score + cost_score) / 3

    best_overall = max(valid_benchmarks, key=lambda x: x['overall_score'])
    print(f"\n🎯 종합 최고: {best_overall['name']}")
    print(f"   - 종합 점수: {best_overall['overall_score']:.2%}")


def save_benchmark_report(benchmarks: List[Dict], output_path: str):
    """
    벤치마크 결과를 텍스트 파일로 저장합니다.

    Args:
        benchmarks: 벤치마크 결과 리스트
        output_path: 저장할 파일 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("API 성능 비교 벤치마크 보고서\n")
        f.write("=" * 80 + "\n\n")

        for bench in benchmarks:
            if bench is None:
                continue

            f.write(f"{bench['name']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"평균 응답 시간: {bench['avg_time']:.3f}초\n")
            f.write(f"일관성 점수: {bench['consistency']:.2%}\n")
            f.write(f"1,000회 비용: ${bench['cost_per_1k']:.4f}\n")
            f.write(f"실행 횟수: {bench['runs']}\n")
            f.write("\n")

    print(f"\n💾 벤치마크 보고서 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Lab 05: API 성능 비교 벤치마크"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="테스트 이미지 파일 경로"
    )

    parser.add_argument(
        "--batch",
        type=str,
        help="배치 테스트용 이미지 디렉토리"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="각 API당 반복 실행 횟수 (기본값: 3)"
    )

    parser.add_argument(
        "--modes",
        type=str,
        nargs='+',
        default=['gemini', 'openai', 'simulation'],
        choices=['gemini', 'openai', 'simulation'],
        help="테스트할 API 모드 (기본값: 모두)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="벤치마크 보고서 저장 경로 (예: benchmark.txt)"
    )

    args = parser.parse_args()

    if not args.input and not args.batch:
        parser.print_help()
        print("\n예제:")
        print("  python lab05_comparison.py --input image.jpg")
        print("  python lab05_comparison.py --input image.jpg --runs 5")
        return

    # 이미지 로드
    if args.input:
        input_path = Path(args.input)

        if not input_path.exists():
            print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
            return

        image = Image.open(input_path)
        images = [image]

    elif args.batch:
        batch_dir = Path(args.batch)

        if not batch_dir.exists():
            print(f"❌ 디렉토리를 찾을 수 없습니다: {args.batch}")
            return

        # 이미지 찾기
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files = []

        for ext in extensions:
            image_files.extend(batch_dir.glob(f"*{ext}"))

        if not image_files:
            print(f"❌ 디렉토리에서 이미지를 찾을 수 없습니다: {args.batch}")
            return

        images = [Image.open(f) for f in sorted(image_files)]

    print(f"🚀 API 성능 벤치마크 시작")
    print(f"   - 테스트 이미지: {len(images)}개")
    print(f"   - 반복 횟수: {args.runs}회")
    print(f"   - 테스트 모드: {', '.join(args.modes)}")
    print()

    # 각 이미지에 대해 벤치마크
    all_benchmarks = []

    for img_idx, image in enumerate(images, 1):
        if len(images) > 1:
            print(f"\n이미지 {img_idx}/{len(images)} 테스트 중...")

        benchmarks = []

        for mode in args.modes:
            bench = benchmark_api(mode, image, runs=args.runs)
            benchmarks.append(bench)

        all_benchmarks.append(benchmarks)

    # 평균 벤치마크 계산 (배치 모드)
    if len(images) > 1:
        avg_benchmarks = []

        for mode_idx, mode in enumerate(args.modes):
            mode_results = [b[mode_idx] for b in all_benchmarks if b[mode_idx] is not None]

            if mode_results:
                avg_bench = {
                    'mode': mode,
                    'name': mode_results[0]['name'],
                    'runs': sum(b['runs'] for b in mode_results),
                    'avg_time': statistics.mean([b['avg_time'] for b in mode_results]),
                    'min_time': min(b['min_time'] for b in mode_results),
                    'max_time': max(b['max_time'] for b in mode_results),
                    'std_time': statistics.mean([b['std_time'] for b in mode_results]),
                    'consistency': statistics.mean([b['consistency'] for b in mode_results]),
                    'cost_per_1k': mode_results[0]['cost_per_1k'],
                    'results': []
                }
                avg_benchmarks.append(avg_bench)
            else:
                avg_benchmarks.append(None)

        benchmarks = avg_benchmarks
    else:
        benchmarks = all_benchmarks[0]

    # 결과 출력
    print_comparison_table(benchmarks)
    print_winner_summary(benchmarks)

    # 보고서 저장
    if args.output:
        save_benchmark_report(benchmarks, args.output)

    print("\n✅ 벤치마크 완료")


if __name__ == "__main__":
    main()
