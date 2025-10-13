"""
Lab 04: 실시간 행동 인식 (Real-time Action Recognition)

이 실습에서는 웹캠을 사용한 실시간 행동 인식을 배웁니다:
- 프레임 버퍼를 사용한 실시간 처리
- 경량화된 모델 사용
- FPS 최적화 기법
- 비동기 처리

사용법:
    python lab04_realtime_recognition.py
    python lab04_realtime_recognition.py --model videomae --buffer-size 32
"""

import argparse
import time
from collections import deque
from typing import List, Tuple, Optional, Deque
import numpy as np
import threading
import queue

try:
    from transformers import AutoImageProcessor, AutoModelForVideoClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class RealtimeActionRecognizer:
    """
    실시간 행동 인식 클래스
    """

    def __init__(
        self,
        model_name: str = 'videomae',
        buffer_size: int = 16,
        skip_frames: int = 2,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            model_name: 모델 ID
            buffer_size: 프레임 버퍼 크기
            skip_frames: N 프레임마다 1개씩 추론 (성능 최적화)
            target_size: 프레임 크기 (width, height)
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers 패키지 필요: pip install transformers torch")

        if not HAS_OPENCV:
            raise ImportError("opencv-python 패키지 필요: pip install opencv-python")

        self.model_name = model_name
        self.buffer_size = buffer_size
        self.skip_frames = skip_frames
        self.target_size = target_size

        # 프레임 버퍼 (deque for efficiency)
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)

        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"디바이스: {self.device}")

        # 모델 로드
        print(f"모델 로딩 중: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForVideoClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # 최적화: JIT 컴파일 (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                print("✅ 모델 JIT 컴파일 완료")
            except Exception as e:
                print(f"⚠️ JIT 컴파일 실패: {e}")

        print(f"✅ 모델 로드 완료")

        # 예측 결과 캐시
        self.current_prediction: Optional[Tuple[str, float]] = None
        self.prediction_history: Deque[str] = deque(maxlen=10)

        # 성능 측정
        self.fps_history: Deque[float] = deque(maxlen=30)
        self.inference_time_history: Deque[float] = deque(maxlen=30)

        # 비동기 추론 큐
        self.inference_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.inference_thread = None
        self.running = False

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임 전처리 (RGB 변환 및 크기 조정)
        """
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 크기 조정
        frame_resized = cv2.resize(frame_rgb, self.target_size)

        return frame_resized

    def add_frame(self, frame: np.ndarray):
        """
        프레임을 버퍼에 추가합니다.
        """
        preprocessed = self.preprocess_frame(frame)
        self.frame_buffer.append(preprocessed)

    def predict(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        현재 버퍼의 프레임으로 행동을 예측합니다.

        Args:
            top_k: 상위 K개 예측

        Returns:
            (label, score) 리스트
        """
        if len(self.frame_buffer) < self.buffer_size:
            return []

        # 버퍼에서 균등하게 샘플링
        indices = np.linspace(0, len(self.frame_buffer) - 1, self.buffer_size, dtype=int)
        frames = [self.frame_buffer[i] for i in indices]

        # 전처리
        inputs = self.processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 추론
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        inference_time = time.time() - start_time
        self.inference_time_history.append(inference_time)

        # Softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Top-K
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_indices:
            label = self.model.config.id2label[idx]
            score = float(probs[idx])
            results.append((label, score))

        # 캐시 업데이트
        if results:
            self.current_prediction = results[0]
            self.prediction_history.append(results[0][0])

        return results

    def _inference_worker(self):
        """
        비동기 추론 워커 스레드
        """
        while self.running:
            try:
                # 큐에서 프레임 가져오기
                frames = self.inference_queue.get(timeout=0.1)

                if frames is None:  # 종료 신호
                    break

                # 추론
                results = self.predict(top_k=3)

                # 결과 큐에 넣기
                if not self.result_queue.full():
                    self.result_queue.put(results)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"추론 오류: {e}")

    def start_async_inference(self):
        """
        비동기 추론 시작
        """
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        print("✅ 비동기 추론 시작됨")

    def stop_async_inference(self):
        """
        비동기 추론 종료
        """
        self.running = False
        self.inference_queue.put(None)  # 종료 신호
        if self.inference_thread:
            self.inference_thread.join()
        print("✅ 비동기 추론 종료됨")

    def run(self, async_mode: bool = True):
        """
        실시간 행동 인식 실행

        Args:
            async_mode: 비동기 추론 사용 여부
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("웹캠을 열 수 없습니다")

        # 해상도 설정 (낮추면 성능 향상)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n실시간 행동 인식 시작!")
        print("종료: 'q' 키")
        print("일시정지: 'p' 키")
        print(f"버퍼 크기: {self.buffer_size} 프레임")
        print(f"Skip frames: {self.skip_frames}")
        print(f"비동기 모드: {async_mode}\n")

        if async_mode:
            self.start_async_inference()

        frame_count = 0
        last_inference_frame = 0
        paused = False

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # FPS 측정
            if len(self.fps_history) > 0:
                fps = 1.0 / (time.time() - self.fps_history[-1]) if self.fps_history else 0
                self.fps_history.append(time.time())
            else:
                fps = 0
                self.fps_history.append(time.time())

            # 프레임 버퍼에 추가
            if not paused:
                self.add_frame(frame)

            # 추론 (skip_frames 간격으로)
            if not paused and frame_count - last_inference_frame >= self.skip_frames:
                if len(self.frame_buffer) >= self.buffer_size:
                    if async_mode:
                        # 비동기 추론
                        if self.inference_queue.empty():
                            self.inference_queue.put(list(self.frame_buffer))

                        # 결과 가져오기
                        try:
                            results = self.result_queue.get_nowait()
                            self.current_prediction = results[0] if results else None
                        except queue.Empty:
                            pass
                    else:
                        # 동기 추론
                        results = self.predict(top_k=3)

                last_inference_frame = frame_count

            # 시각화
            display_frame = frame.copy()

            # 예측 결과 표시
            if self.current_prediction:
                label, score = self.current_prediction
                text = f"{label}: {score:.2f}"
                cv2.putText(
                    display_frame,
                    text,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

            # 성능 정보 표시
            avg_inference = np.mean(self.inference_time_history) if self.inference_time_history else 0
            info = [
                f"FPS: {fps:.1f}",
                f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}",
                f"Inference: {avg_inference*1000:.0f}ms"
            ]

            for i, text in enumerate(info):
                cv2.putText(
                    display_frame,
                    text,
                    (10, 80 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

            # 일시정지 표시
            if paused:
                cv2.putText(
                    display_frame,
                    "PAUSED",
                    (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    3
                )

            cv2.imshow('Real-time Action Recognition', display_frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused

            frame_count += 1

        # 정리
        if async_mode:
            self.stop_async_inference()

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n✅ 총 {frame_count} 프레임 처리됨")
        if self.inference_time_history:
            print(f"평균 추론 시간: {np.mean(self.inference_time_history)*1000:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Lab 04: 실시간 행동 인식")
    parser.add_argument("--model", type=str,
                       default='MCG-NJU/videomae-base-finetuned-kinetics',
                       help="HuggingFace 모델 ID")
    parser.add_argument("--buffer-size", type=int, default=16,
                       help="프레임 버퍼 크기")
    parser.add_argument("--skip-frames", type=int, default=2,
                       help="N 프레임마다 추론")
    parser.add_argument("--sync", action="store_true",
                       help="동기 모드 사용 (기본: 비동기)")

    args = parser.parse_args()

    try:
        recognizer = RealtimeActionRecognizer(
            model_name=args.model,
            buffer_size=args.buffer_size,
            skip_frames=args.skip_frames
        )

        recognizer.run(async_mode=not args.sync)

    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
