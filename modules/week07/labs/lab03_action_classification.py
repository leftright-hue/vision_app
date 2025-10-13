"""
Lab 03: 행동 분류 (Action Classification)

이 실습에서는 사전훈련된 HuggingFace 모델을 사용한 행동 분류를 배웁니다:
- VideoMAE 모델로 행동 인식
- TimeSformer 모델 활용
- X-CLIP 모델 사용
- 다중 모델 앙상블

사용법:
    python lab03_action_classification.py --input sample.mp4 --model videomae
    python lab03_action_classification.py --input sample.mp4 --ensemble
"""

import argparse
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForVideoClassification,
        VideoMAEImageProcessor,
        VideoMAEForVideoClassification,
        TimesformerForVideoClassification
    )
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# 모델 레지스트리
MODEL_REGISTRY = {
    'videomae': 'MCG-NJU/videomae-base-finetuned-kinetics',
    'timesformer': 'facebook/timesformer-base-finetuned-k400',
    'xclip': 'microsoft/xclip-base-patch32',
}


class ActionClassifier:
    """
    HuggingFace 사전훈련 모델을 사용한 행동 분류기
    """

    def __init__(self, model_name: str = 'videomae'):
        """
        Args:
            model_name: 'videomae', 'timesformer', 'xclip' 중 선택
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers 패키지가 필요합니다: pip install transformers torch")

        if not HAS_OPENCV:
            raise ImportError("opencv-python 패키지가 필요합니다: pip install opencv-python")

        self.model_name = model_name
        self.model_id = MODEL_REGISTRY.get(model_name)

        if not self.model_id:
            raise ValueError(f"지원하지 않는 모델: {model_name}")

        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"디바이스: {self.device}")

        # 모델 로드
        print(f"모델 로딩 중: {self.model_id}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVideoClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ {model_name} 모델 로드 완료")

    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224)
    ) -> List[np.ndarray]:
        """
        비디오에서 균등하게 프레임을 추출합니다.

        Args:
            video_path: 비디오 파일 경로
            num_frames: 추출할 프레임 수
            target_size: 타겟 크기 (width, height)

        Returns:
            프레임 리스트 (RGB)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 균등하게 프레임 인덱스 선택
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 크기 조정
            frame_resized = cv2.resize(frame_rgb, target_size)

            frames.append(frame_resized)

        cap.release()

        if len(frames) < num_frames:
            # 부족한 경우 마지막 프레임 복제
            while len(frames) < num_frames:
                frames.append(frames[-1])

        return frames

    def classify(
        self,
        video_path: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        비디오의 행동을 분류합니다.

        Args:
            video_path: 비디오 파일 경로
            top_k: 반환할 상위 예측 수

        Returns:
            (label, score) 리스트
        """
        # 프레임 추출
        print(f"프레임 추출 중...")
        frames = self.extract_frames(video_path, num_frames=16)

        # 전처리
        print(f"전처리 중...")
        inputs = self.processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 추론
        print(f"추론 중...")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Top-K 선택
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_indices:
            label = self.model.config.id2label[idx]
            score = float(probs[idx])
            results.append((label, score))

        return results


class EnsembleClassifier:
    """
    다중 모델 앙상블 분류기
    """

    def __init__(self, model_names: List[str] = None):
        """
        Args:
            model_names: 앙상블할 모델 리스트 (None이면 모든 모델)
        """
        if model_names is None:
            model_names = ['videomae', 'timesformer']

        self.classifiers = {}

        for name in model_names:
            try:
                print(f"\n{name} 모델 로딩...")
                self.classifiers[name] = ActionClassifier(name)
            except Exception as e:
                print(f"⚠️ {name} 모델 로드 실패: {e}")

        if not self.classifiers:
            raise ValueError("사용 가능한 모델이 없습니다")

        print(f"\n✅ 앙상블 준비 완료 ({len(self.classifiers)}개 모델)")

    def classify(
        self,
        video_path: str,
        top_k: int = 5,
        voting: str = 'soft'
    ) -> List[Tuple[str, float]]:
        """
        앙상블 분류를 수행합니다.

        Args:
            video_path: 비디오 파일 경로
            top_k: 반환할 상위 예측 수
            voting: 'soft' (확률 평균) 또는 'hard' (다수결)

        Returns:
            (label, score) 리스트
        """
        all_predictions = {}

        # 각 모델에서 예측
        for name, classifier in self.classifiers.items():
            print(f"\n{name} 모델 예측 중...")
            preds = classifier.classify(video_path, top_k=20)  # 더 많은 후보 수집

            for label, score in preds:
                if label not in all_predictions:
                    all_predictions[label] = []
                all_predictions[label].append(score)

        # 앙상블
        ensemble_results = []

        if voting == 'soft':
            # 확률 평균
            for label, scores in all_predictions.items():
                avg_score = np.mean(scores)
                ensemble_results.append((label, avg_score))

        elif voting == 'hard':
            # 다수결 (등장 횟수)
            for label, scores in all_predictions.items():
                vote_count = len(scores)
                ensemble_results.append((label, vote_count / len(self.classifiers)))

        # 정렬
        ensemble_results.sort(key=lambda x: x[1], reverse=True)

        return ensemble_results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Lab 03: 행동 분류")
    parser.add_argument("--input", type=str, required=True,
                       help="입력 비디오 파일 경로")
    parser.add_argument("--model", type=str, default='videomae',
                       choices=['videomae', 'timesformer', 'xclip'],
                       help="사용할 모델")
    parser.add_argument("--ensemble", action="store_true",
                       help="앙상블 모드 사용")
    parser.add_argument("--top-k", type=int, default=5,
                       help="상위 K개 예측 출력")

    args = parser.parse_args()

    print(f"📹 비디오 파일: {args.input}")

    try:
        if args.ensemble:
            # 앙상블 모드
            print("\n🎭 앙상블 모드")
            classifier = EnsembleClassifier()
            results = classifier.classify(args.input, top_k=args.top_k)

        else:
            # 단일 모델 모드
            print(f"\n🤖 모델: {args.model}")
            classifier = ActionClassifier(args.model)
            results = classifier.classify(args.input, top_k=args.top_k)

        # 결과 출력
        print(f"\n{'='*60}")
        print(f"Top-{args.top_k} 예측 결과:")
        print(f"{'='*60}")

        for i, (label, score) in enumerate(results, 1):
            bar = '█' * int(score * 50)
            print(f"{i}. {label:30s} {score:.4f} {bar}")

        print(f"{'='*60}")

    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
