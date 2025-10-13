"""
Lab 03: 멀티모달 감정 분석 (Multimodal Emotion Analysis)

이 실습에서는 이미지와 텍스트를 결합한 멀티모달 감정 분석을 학습합니다:
- 이미지 단독 분석
- 이미지 + 텍스트 통합 분석
- 감정 불일치 감지
- 컨텍스트 영향 분석

사용법:
    python lab03_multimodal.py --input image.jpg --text "오늘 시험에 떨어졌어요"
    python lab03_multimodal.py --input image.jpg --text "합격했어요!" --output result.json
    python lab03_multimodal.py --input image.jpg --text "..." --detect-conflict
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from modules.week08.emotion_helpers import EmotionHelper


def analyze_multimodal(
    helper: EmotionHelper,
    image_path: str,
    text: str,
    verbose: bool = True
) -> Dict:
    """
    이미지와 텍스트를 결합한 멀티모달 분석을 수행합니다.

    Args:
        helper: EmotionHelper 인스턴스
        image_path: 이미지 파일 경로
        text: 텍스트 컨텍스트
        verbose: 상세 출력 여부

    Returns:
        분석 결과 딕셔너리
    """
    if verbose:
        print(f"📷 이미지: {image_path}")
        print(f"📝 텍스트: \"{text}\"")
        print()

    # 이미지 로드
    image = Image.open(image_path)

    # 멀티모달 분석
    result = helper.analyze_multimodal(image, text)

    if verbose:
        print(f"🤖 분석 모드: {helper.mode}")

    return result


def print_comparison_results(result: Dict, detailed: bool = True):
    """
    이미지 vs 통합 분석 결과를 비교하여 출력합니다.

    Args:
        result: analyze_multimodal 결과
        detailed: 상세 정보 출력 여부
    """
    image_only = result['image_only']
    combined = result['combined']
    difference = result['difference']

    # 각각의 지배적 감정
    dominant_image = max(image_only.items(), key=lambda x: x[1])
    dominant_combined = max(combined.items(), key=lambda x: x[1])

    print("\n" + "=" * 60)
    print("📊 멀티모달 분석 결과")
    print("=" * 60)

    # 1. 이미지만 분석
    print("\n🖼️  이미지만 분석:")
    print(f"   주요 감정: {dominant_image[0].upper()} ({dominant_image[1]:.2%})")

    if detailed:
        print("   전체 감정:")
        for emotion, score in sorted(image_only.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"     - {emotion.capitalize():<10} : {score:.2%}")

    # 2. 통합 분석
    print("\n🎨 이미지 + 텍스트 통합:")
    print(f"   주요 감정: {dominant_combined[0].upper()} ({dominant_combined[1]:.2%})")

    if detailed:
        print("   전체 감정:")
        for emotion, score in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"     - {emotion.capitalize():<10} : {score:.2%}")

    # 3. 차이 분석
    print("\n🔍 텍스트 컨텍스트 영향:")

    # 큰 변화만 표시
    significant_changes = [(e, d) for e, d in difference.items() if abs(d) > 0.05]
    significant_changes.sort(key=lambda x: abs(x[1]), reverse=True)

    if significant_changes:
        for emotion, diff in significant_changes[:5]:
            direction = "↑ 증가" if diff > 0 else "↓ 감소"
            print(f"   {emotion.capitalize():<10} : {diff:+.2%} {direction}")
    else:
        print("   큰 변화 없음 (±5% 이내)")


def detect_emotion_conflict(
    result: Dict,
    threshold: float = 0.3
) -> Tuple[bool, Optional[str]]:
    """
    이미지와 텍스트 간 감정 불일치를 감지합니다.

    Args:
        result: analyze_multimodal 결과
        threshold: 불일치 판단 임계값

    Returns:
        (불일치 여부, 설명) 튜플
    """
    image_only = result['image_only']
    combined = result['combined']

    # 각각의 지배적 감정
    dominant_image = max(image_only.items(), key=lambda x: x[1])[0]
    dominant_combined = max(combined.items(), key=lambda x: x[1])[0]

    # 감정이 다른 경우
    if dominant_image != dominant_combined:
        # 신뢰도 차이 확인
        image_confidence = image_only[dominant_image]
        combined_confidence = combined[dominant_combined]

        diff = abs(image_confidence - combined.get(dominant_image, 0))

        if diff > threshold:
            explanation = (
                f"이미지는 '{dominant_image}'를 나타내지만, "
                f"텍스트 컨텍스트는 '{dominant_combined}'로 해석됩니다. "
                f"(차이: {diff:.2%})"
            )
            return True, explanation

    return False, None


def analyze_context_impact(result: Dict) -> Dict[str, any]:
    """
    텍스트 컨텍스트가 감정 분석에 미친 영향을 분석합니다.

    Args:
        result: analyze_multimodal 결과

    Returns:
        영향 분석 결과
    """
    difference = result['difference']

    # 가장 큰 변화
    max_increase = max(difference.items(), key=lambda x: x[1])
    max_decrease = min(difference.items(), key=lambda x: x[1])

    # 평균 절대 변화량
    avg_change = sum(abs(d) for d in difference.values()) / len(difference)

    # 변화 방향
    increases = {e: d for e, d in difference.items() if d > 0.05}
    decreases = {e: d for e, d in difference.items() if d < -0.05}

    return {
        'max_increase': max_increase,
        'max_decrease': max_decrease,
        'avg_change': avg_change,
        'increases': increases,
        'decreases': decreases,
        'impact_level': 'high' if avg_change > 0.15 else 'medium' if avg_change > 0.08 else 'low'
    }


def save_detailed_report(
    result: Dict,
    image_path: str,
    text: str,
    output_path: str
):
    """
    상세 분석 보고서를 JSON으로 저장합니다.

    Args:
        result: analyze_multimodal 결과
        image_path: 이미지 경로
        text: 텍스트 컨텍스트
        output_path: 저장할 파일 경로
    """
    # 불일치 감지
    has_conflict, conflict_msg = detect_emotion_conflict(result)

    # 영향 분석
    impact = analyze_context_impact(result)

    # 보고서 작성
    report = {
        'input': {
            'image': image_path,
            'text': text
        },
        'results': {
            'image_only': result['image_only'],
            'combined': result['combined'],
            'difference': result['difference']
        },
        'analysis': {
            'has_conflict': has_conflict,
            'conflict_message': conflict_msg,
            'context_impact': {
                'level': impact['impact_level'],
                'avg_change': f"{impact['avg_change']:.2%}",
                'max_increase': {
                    'emotion': impact['max_increase'][0],
                    'value': f"{impact['max_increase'][1]:+.2%}"
                },
                'max_decrease': {
                    'emotion': impact['max_decrease'][0],
                    'value': f"{impact['max_decrease'][1]:+.2%}"
                }
            }
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 상세 보고서 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Lab 03: 멀티모달 감정 분석"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 이미지 파일 경로"
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="텍스트 컨텍스트"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="결과를 저장할 JSON 파일 경로"
    )

    parser.add_argument(
        "--detect-conflict",
        action="store_true",
        help="감정 불일치 감지 활성화"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="불일치 판단 임계값 (기본값: 0.3)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="최소한의 출력만 표시"
    )

    args = parser.parse_args()

    # 입력 검증
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        return

    # 감정 인식 시스템 초기화
    if not args.quiet:
        print("🤖 멀티모달 감정 분석 시스템 초기화 중...")

    helper = EmotionHelper()

    if not args.quiet:
        print(f"✅ 초기화 완료: {helper.mode} 모드\n")

    # 멀티모달 분석
    result = analyze_multimodal(
        helper,
        str(input_path),
        args.text,
        verbose=not args.quiet
    )

    # 결과 출력
    if not args.quiet:
        print_comparison_results(result, detailed=True)

    # 불일치 감지
    if args.detect_conflict:
        print("\n" + "=" * 60)
        print("⚠️  감정 불일치 감지")
        print("=" * 60)

        has_conflict, conflict_msg = detect_emotion_conflict(result, args.threshold)

        if has_conflict:
            print(f"\n🚨 불일치 감지됨!")
            print(f"   {conflict_msg}")
        else:
            print(f"\n✅ 감정 일치 (임계값: {args.threshold:.0%})")

    # 영향 분석
    if not args.quiet:
        print("\n" + "=" * 60)
        print("📈 텍스트 컨텍스트 영향 분석")
        print("=" * 60)

        impact = analyze_context_impact(result)

        print(f"\n영향 수준: {impact['impact_level'].upper()}")
        print(f"평균 변화량: {impact['avg_change']:.2%}")
        print(f"\n가장 많이 증가한 감정: {impact['max_increase'][0].capitalize()} ({impact['max_increase'][1]:+.2%})")
        print(f"가장 많이 감소한 감정: {impact['max_decrease'][0].capitalize()} ({impact['max_decrease'][1]:+.2%})")

        if impact['increases']:
            print(f"\n증가한 감정 ({len(impact['increases'])}개):")
            for emotion, diff in sorted(impact['increases'].items(), key=lambda x: x[1], reverse=True):
                print(f"  + {emotion.capitalize():<10} : {diff:+.2%}")

        if impact['decreases']:
            print(f"\n감소한 감정 ({len(impact['decreases'])}개):")
            for emotion, diff in sorted(impact['decreases'].items(), key=lambda x: x[1]):
                print(f"  - {emotion.capitalize():<10} : {diff:+.2%}")

    # 상세 보고서 저장
    if args.output:
        save_detailed_report(result, str(input_path), args.text, args.output)

    # JSON 출력 (quiet 모드)
    if args.quiet:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
