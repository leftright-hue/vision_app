"""
Lab 02: VAD 3차원 감정 모델 (VAD 3D Emotion Model)

이 실습에서는 VAD (Valence-Arousal-Dominance) 3차원 감정 모델을 학습합니다:
- VAD 좌표 계산
- 3D 공간 시각화
- 감정 간 유사도 분석
- 감정 유사도 매트릭스 생성

사용법:
    python lab02_vad_model.py --input image.jpg
    python lab02_vad_model.py --input image.jpg --plot vad_3d.png
    python lab02_vad_model.py --similarity-matrix --output matrix.png
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from modules.week08.emotion_helpers import EmotionHelper, VADModel


def analyze_emotion_vad(
    helper: EmotionHelper,
    image_path: str,
    verbose: bool = True
) -> Tuple[str, float, Tuple[float, float, float]]:
    """
    이미지의 감정을 분석하고 VAD 좌표를 계산합니다.

    Args:
        helper: EmotionHelper 인스턴스
        image_path: 이미지 파일 경로
        verbose: 상세 출력 여부

    Returns:
        (지배적 감정, 신뢰도, VAD 좌표) 튜플
    """
    if verbose:
        print(f"📷 이미지 분석: {image_path}")

    # 이미지 로드 및 감정 분석
    image = Image.open(image_path)
    emotions = helper.analyze_basic_emotion(image)

    # 지배적 감정 찾기
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
    emotion_name = dominant_emotion[0]
    confidence = dominant_emotion[1]

    # VAD 좌표 계산
    vad = VADModel.emotion_to_vad(emotion_name)

    if verbose:
        print(f"\n🎯 감정 분석 결과:")
        print(f"  - 주요 감정: {emotion_name.upper()}")
        print(f"  - 신뢰도: {confidence:.2%}")
        print(f"\n📊 VAD 좌표:")
        print(f"  - Valence (원자가): {vad[0]:+.2f}")
        print(f"  - Arousal (각성): {vad[1]:+.2f}")
        print(f"  - Dominance (지배): {vad[2]:+.2f}")
        print()

    return emotion_name, confidence, vad


def calculate_similarity_matrix(
    emotions: Optional[List[str]] = None
) -> np.ndarray:
    """
    감정 간 유사도 매트릭스를 계산합니다.

    Args:
        emotions: 분석할 감정 목록 (None이면 모든 기본 감정)

    Returns:
        유사도 매트릭스 (numpy array)
    """
    if emotions is None:
        emotions = list(VADModel.EMOTION_VAD_MAP.keys())

    n = len(emotions)
    matrix = np.zeros((n, n))

    for i, emotion1 in enumerate(emotions):
        for j, emotion2 in enumerate(emotions):
            vad1 = VADModel.emotion_to_vad(emotion1)
            vad2 = VADModel.emotion_to_vad(emotion2)
            similarity = VADModel.calculate_similarity(vad1, vad2)
            matrix[i][j] = similarity

    return matrix


def plot_vad_3d(
    emotions_vad: Dict[str, Tuple[float, float, float]],
    output_path: Optional[str] = None,
    highlight: Optional[str] = None
):
    """
    VAD 좌표를 3D 공간에 시각화합니다.

    Args:
        emotions_vad: 감정명을 키로 하는 VAD 좌표 딕셔너리
        output_path: 저장할 이미지 경로 (None이면 화면 표시)
        highlight: 강조 표시할 감정명
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 각 감정 플롯
    for emotion, (v, a, d) in emotions_vad.items():
        # 강조 표시
        if highlight and emotion == highlight:
            ax.scatter(v, a, d, c='red', s=300, marker='*',
                      label=f'{emotion.upper()} (분석 결과)',
                      edgecolors='darkred', linewidths=2, zorder=5)
        else:
            ax.scatter(v, a, d, s=100, alpha=0.6, label=emotion.capitalize())

        # 레이블
        ax.text(v, a, d, f'  {emotion}', fontsize=9)

    # 축 설정
    ax.set_xlabel('Valence (긍정 ↔ 부정)', fontsize=11, labelpad=10)
    ax.set_ylabel('Arousal (차분 ↔ 흥분)', fontsize=11, labelpad=10)
    ax.set_zlabel('Dominance (복종 ↔ 지배)', fontsize=11, labelpad=10)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # 제목
    ax.set_title('VAD 3차원 감정 공간', fontsize=14, fontweight='bold', pad=20)

    # 범례
    ax.legend(loc='upper left', fontsize=9)

    # 그리드
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 3D 플롯 저장: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_similarity_matrix(
    matrix: np.ndarray,
    emotions: List[str],
    output_path: Optional[str] = None
):
    """
    감정 유사도 매트릭스를 히트맵으로 시각화합니다.

    Args:
        matrix: 유사도 매트릭스
        emotions: 감정 목록
        output_path: 저장할 이미지 경로 (None이면 화면 표시)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 히트맵
    im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=1)

    # 축 레이블
    ax.set_xticks(np.arange(len(emotions)))
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_xticklabels([e.capitalize() for e in emotions])
    ax.set_yticklabels([e.capitalize() for e in emotions])

    # 레이블 회전
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 셀에 값 표시
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    # 제목
    ax.set_title('감정 유사도 매트릭스 (VAD 기반)', fontsize=14, fontweight='bold', pad=15)

    # 컬러바
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('유사도 (0.0 ~ 1.0)', rotation=270, labelpad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 유사도 매트릭스 저장: {output_path}")
    else:
        plt.show()

    plt.close()


def find_similar_emotions(
    target_emotion: str,
    top_n: int = 3
) -> List[Tuple[str, float]]:
    """
    특정 감정과 유사한 감정들을 찾습니다.

    Args:
        target_emotion: 대상 감정
        top_n: 반환할 유사 감정 개수

    Returns:
        (감정명, 유사도) 튜플 리스트
    """
    target_vad = VADModel.emotion_to_vad(target_emotion)

    similarities = []

    for emotion in VADModel.EMOTION_VAD_MAP.keys():
        if emotion != target_emotion:
            emotion_vad = VADModel.emotion_to_vad(emotion)
            similarity = VADModel.calculate_similarity(target_vad, emotion_vad)
            similarities.append((emotion, similarity))

    # 유사도 내림차순 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description="Lab 02: VAD 3차원 감정 모델 분석"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="입력 이미지 파일 경로"
    )

    parser.add_argument(
        "--plot",
        type=str,
        help="3D VAD 플롯을 저장할 파일 경로"
    )

    parser.add_argument(
        "--similarity-matrix",
        action="store_true",
        help="감정 유사도 매트릭스 생성"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="유사도 매트릭스 저장 경로"
    )

    parser.add_argument(
        "--show-all-emotions",
        action="store_true",
        help="모든 기본 감정을 3D 플롯에 표시"
    )

    args = parser.parse_args()

    # 유사도 매트릭스 모드
    if args.similarity_matrix:
        print("📊 감정 유사도 매트릭스 생성 중...")

        emotions = list(VADModel.EMOTION_VAD_MAP.keys())
        matrix = calculate_similarity_matrix(emotions)

        print(f"\n✅ 매트릭스 계산 완료 ({len(emotions)}x{len(emotions)})")
        print("\n감정 유사도:")

        for i, emotion in enumerate(emotions):
            print(f"\n{emotion.capitalize()}와 유사한 감정:")
            similar = find_similar_emotions(emotion, top_n=3)
            for j, (similar_emotion, score) in enumerate(similar, 1):
                print(f"  {j}. {similar_emotion.capitalize()}: {score:.2%}")

        # 시각화
        plot_similarity_matrix(matrix, emotions, args.output)

    # 이미지 분석 모드
    elif args.input:
        input_path = Path(args.input)

        if not input_path.exists():
            print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
            return

        # 감정 인식 시스템 초기화
        print("🤖 감정 인식 시스템 초기화 중...")
        helper = EmotionHelper()
        print(f"✅ 초기화 완료: {helper.mode} 모드\n")

        # 감정 분석 및 VAD 계산
        emotion_name, confidence, vad = analyze_emotion_vad(
            helper,
            str(input_path),
            verbose=True
        )

        # 유사 감정 찾기
        print("🔍 유사한 감정:")
        similar = find_similar_emotions(emotion_name, top_n=3)

        for i, (similar_emotion, score) in enumerate(similar, 1):
            similar_vad = VADModel.emotion_to_vad(similar_emotion)
            print(f"  {i}. {similar_emotion.capitalize()}")
            print(f"     - 유사도: {score:.2%}")
            print(f"     - VAD: ({similar_vad[0]:+.2f}, {similar_vad[1]:+.2f}, {similar_vad[2]:+.2f})")

        # 3D 시각화
        if args.plot or args.show_all_emotions:
            print(f"\n📈 3D VAD 공간 시각화 중...")

            emotions_vad = {}

            if args.show_all_emotions:
                # 모든 기본 감정 표시
                for emotion in VADModel.EMOTION_VAD_MAP.keys():
                    emotions_vad[emotion] = VADModel.emotion_to_vad(emotion)
            else:
                # 분석 결과 + 유사 감정만 표시
                emotions_vad[emotion_name] = vad
                for similar_emotion, _ in similar:
                    emotions_vad[similar_emotion] = VADModel.emotion_to_vad(similar_emotion)

            plot_vad_3d(emotions_vad, args.plot, highlight=emotion_name)

    else:
        parser.print_help()
        print("\n예제:")
        print("  python lab02_vad_model.py --input image.jpg --plot vad_3d.png")
        print("  python lab02_vad_model.py --similarity-matrix --output matrix.png")


if __name__ == "__main__":
    main()
