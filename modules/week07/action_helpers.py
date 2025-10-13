"""
행동인식 헬퍼 클래스
3-tier fallback 전략: HuggingFace Transformers → OpenCV → Simulation
"""

import numpy as np
from PIL import Image
import streamlit as st
from typing import Optional, List, Dict, Tuple, Any, Union
import warnings
import sys
import os
import traceback

# BaseImageProcessor import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.base_processor import BaseImageProcessor

# HuggingFace 모델 레지스트리
MODEL_REGISTRY = {
    'videomae': 'MCG-NJU/videomae-base-finetuned-kinetics',
    'timesformer': 'facebook/timesformer-base-finetuned-k400',
    'xclip': 'microsoft/xclip-base-patch32',
}


class VideoHelper(BaseImageProcessor):
    """
    비디오 행동인식 헬퍼 클래스
    - 1순위: HuggingFace Transformers (transformers 패키지 + VideoMAE/TimeSformer 모델)
    - 2순위: OpenCV only (비디오 처리, Optical Flow만 가능, ML 모델 없음)
    - 3순위: Simulation mode (기본 비디오 처리 시뮬레이션)
    """

    def __init__(self):
        """
        VideoHelper 초기화
        - 3-tier fallback으로 사용 가능한 모드 감지
        - 디바이스 감지 (CPU/GPU)
        """
        super().__init__()
        self.mode = None  # 'transformers', 'opencv', 'simulation'
        self.device = None  # 'cuda', 'cpu', None
        self.model = None
        self.processor = None
        self.pipeline = None

        self._initialize()

    def _initialize(self):
        """
        3-tier fallback으로 초기화
        """
        # Tier 1: Try HuggingFace Transformers
        if self._try_transformers():
            self.mode = 'transformers'
            st.success("✅ 행동인식 준비 완료 (HuggingFace Transformers)")
            return

        # Tier 2: Try OpenCV only
        if self._try_opencv():
            self.mode = 'opencv'
            st.info("ℹ️ OpenCV 모드 활성화 (비디오 처리 가능, ML 모델 미사용)\n\n"
                   "행동 분류 기능을 사용하려면:\n"
                   "```bash\n"
                   "pip install transformers torch\n"
                   "```")
            return

        # Tier 3: Fallback to simulation
        self.mode = 'simulation'
        st.warning("⚠️ 시뮬레이션 모드 (실제 비디오 처리 미사용)\n\n"
                  "실제 기능을 사용하려면:\n"
                  "```bash\n"
                  "pip install opencv-python transformers torch\n"
                  "```")

    def _try_transformers(self) -> bool:
        """
        HuggingFace Transformers로 로드 시도
        VideoMAE, TimeSformer, X-CLIP 등 비디오 모델 지원
        """
        try:
            import transformers
            import torch

            # 디바이스 감지
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                st.info("ℹ️ GPU를 사용할 수 없어 CPU로 실행합니다. (비디오 처리는 느릴 수 있음)")

            # 필요한 모듈이 있는지만 확인 (실제 모델은 나중에 로드)
            return True

        except ImportError:
            return False
        except Exception as e:
            st.error(f"Transformers 초기화 실패: {e}")
            return False

    def _try_opencv(self) -> bool:
        """
        OpenCV 사용 가능 여부 확인
        비디오 처리와 Optical Flow는 가능하지만 ML 모델은 없음
        """
        try:
            import cv2
            return True
        except ImportError:
            return False
        except Exception as e:
            return False

    def _detect_device(self) -> Optional[str]:
        """
        CUDA 디바이스 감지
        Returns:
            'cuda', 'cpu', or None
        """
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return None

    def get_mode(self) -> str:
        """
        현재 동작 모드 반환
        Returns:
            'transformers', 'opencv', 'simulation'
        """
        return self.mode

    def get_device(self) -> Optional[str]:
        """
        현재 디바이스 반환
        Returns:
            'cuda', 'cpu', or None
        """
        return self.device

    def is_available(self, feature: str) -> bool:
        """
        특정 기능 사용 가능 여부 확인

        Args:
            feature: 'video_processing', 'optical_flow', 'action_classification'

        Returns:
            bool: 기능 사용 가능 여부
        """
        if feature == 'video_processing':
            return self.mode in ['transformers', 'opencv']
        elif feature == 'optical_flow':
            return self.mode in ['transformers', 'opencv']
        elif feature == 'action_classification':
            return self.mode == 'transformers'
        else:
            return False

    def extract_frames(
        self,
        video_path: str,
        sample_rate: int = 30,
        max_frames: int = 100,
        target_size: Tuple[int, int] = (224, 224)
    ) -> List[np.ndarray]:
        """
        비디오에서 프레임 추출 (메모리 효율적)

        Args:
            video_path: 비디오 파일 경로
            sample_rate: 샘플링 레이트 (예: 30이면 30프레임당 1개 추출)
            max_frames: 최대 프레임 수 (메모리 제한)
            target_size: 출력 이미지 크기 (width, height)

        Returns:
            List[np.ndarray]: 프레임 리스트, 각 프레임 shape (H, W, 3) RGB
        """
        # Simulation mode: 랜덤 프레임 생성
        if self.mode == 'simulation':
            st.info("ℹ️ 시뮬레이션 모드: 랜덤 프레임 생성")
            frames = []
            for _ in range(min(10, max_frames)):
                frame = np.random.randint(0, 255, (target_size[1], target_size[0], 3), dtype=np.uint8)
                frames.append(frame)
            return frames

        # OpenCV or Transformers mode
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

            # 비디오 정보 가져오기
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # 실제 샘플링 레이트 계산 (max_frames 제한 고려)
            actual_sample_rate = max(sample_rate, total_frames // max_frames) if total_frames > max_frames else sample_rate

            st.info(f"📹 비디오 정보: {total_frames}프레임, {fps}fps\n"
                   f"샘플링: 매 {actual_sample_rate}프레임당 1개 추출")

            frames = []
            frame_idx = 0

            # 프레임 추출
            while len(frames) < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                # BGR → RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 리사이즈
                frame_resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)

                frames.append(frame_resized)
                frame_idx += actual_sample_rate

            cap.release()

            st.success(f"✅ {len(frames)}개 프레임 추출 완료")
            return frames

        except ImportError:
            st.error("❌ OpenCV가 설치되지 않았습니다.")
            return []
        except Exception as e:
            st.error(f"❌ 프레임 추출 실패: {e}")
            return []

    def compute_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """
        두 프레임 간 Optical Flow 계산 (Farneback 알고리즘)

        Args:
            frame1: 첫 번째 프레임 (H, W, 3) RGB
            frame2: 두 번째 프레임 (H, W, 3) RGB

        Returns:
            np.ndarray: Optical flow (H, W, 2) - [dx, dy] 모션 벡터
        """
        # Simulation mode: 랜덤 flow 생성
        if self.mode == 'simulation':
            h, w = frame1.shape[:2]
            flow = np.random.randn(h, w, 2).astype(np.float32) * 2
            return flow

        # OpenCV or Transformers mode
        try:
            import cv2

            # RGB → Grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            # Farneback Optical Flow 계산
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=0.5,    # 피라미드 스케일
                levels=3,          # 피라미드 레벨
                winsize=15,        # 윈도우 크기
                iterations=3,      # 반복 횟수
                poly_n=5,          # 다항식 확장
                poly_sigma=1.2,    # 가우시안 시그마
                flags=0
            )

            return flow

        except ImportError:
            st.error("❌ OpenCV가 설치되지 않았습니다.")
            return np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
        except Exception as e:
            st.error(f"❌ Optical Flow 계산 실패: {e}")
            return np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)

    def visualize_flow(
        self,
        flow: np.ndarray
    ) -> np.ndarray:
        """
        Optical Flow를 HSV 색상 공간으로 시각화

        Args:
            flow: Optical flow (H, W, 2) - [dx, dy] 모션 벡터

        Returns:
            np.ndarray: RGB 이미지 (H, W, 3)
        """
        try:
            import cv2

            h, w = flow.shape[:2]

            # HSV 이미지 생성
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 1] = 255  # Saturation을 최대로

            # Magnitude (크기)와 Angle (각도) 계산
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Hue: 방향 (0-180 범위)
            hsv[..., 0] = angle * 180 / np.pi / 2

            # Value: 크기 (정규화)
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # HSV → RGB 변환
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            return rgb

        except ImportError:
            st.error("❌ OpenCV가 설치되지 않았습니다.")
            return np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        except Exception as e:
            st.error(f"❌ Flow 시각화 실패: {e}")
            return np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    def classify_action(
        self,
        video_path: str,
        model_name: str = 'videomae',
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        비디오에서 행동 분류 수행

        Args:
            video_path: 비디오 파일 경로
            model_name: 사용할 모델 ('videomae', 'timesformer', 'xclip')
            top_k: 반환할 상위 예측 개수

        Returns:
            List[Tuple[str, float]]: [(행동명, 확률)] 리스트
        """
        if self.mode == 'transformers':
            return self._classify_with_transformers(video_path, model_name, top_k)
        elif self.mode == 'opencv':
            return self._classify_with_opencv(video_path, top_k)
        else:  # simulation
            return self._simulate_classification(top_k)

    def _classify_with_transformers(
        self,
        video_path: str,
        model_name: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        HuggingFace Transformers로 행동 분류 수행

        Args:
            video_path: 비디오 파일 경로
            model_name: 모델 이름
            top_k: 상위 K개 예측

        Returns:
            List[Tuple[str, float]]: 예측 결과
        """
        try:
            from transformers import AutoImageProcessor, AutoModelForVideoClassification
            import torch

            # 모델 ID 가져오기
            if model_name not in MODEL_REGISTRY:
                st.warning(f"⚠️ '{model_name}' 모델을 찾을 수 없습니다. videomae를 사용합니다.")
                model_name = 'videomae'

            model_id = MODEL_REGISTRY[model_name]

            st.info(f"🔄 모델 로딩 중... ({model_id})")

            # 프레임 추출 (VideoMAE는 16프레임 사용)
            frames = self.extract_frames(
                video_path,
                sample_rate=2,
                max_frames=16,
                target_size=(224, 224)
            )

            if len(frames) == 0:
                return [('error', 0.0)]

            # 모델 및 프로세서 로드
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForVideoClassification.from_pretrained(model_id).to(self.device)
            model.eval()

            # 프레임을 PIL 이미지로 변환
            pil_frames = [Image.fromarray(frame) for frame in frames]

            # 전처리
            inputs = processor(pil_frames, return_tensors="pt").to(self.device)

            # 추론 (타임아웃 30초)
            with st.spinner("🎬 행동 분류 중..."):
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

            # Softmax로 확률 변환
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

            # Top-K 예측
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

            results = []
            for prob, idx in zip(top_probs, top_indices):
                label = model.config.id2label[idx.item()]
                score = prob.item()
                results.append((label, score))

            st.success(f"✅ 분류 완료: {results[0][0]} ({results[0][1]:.2%})")
            return results

        except ImportError:
            st.error("❌ transformers 또는 torch가 설치되지 않았습니다.")
            return [('error', 0.0)]
        except Exception as e:
            st.error(f"❌ 분류 실패: {e}")
            traceback.print_exc()
            return [('error', 0.0)]

    def _classify_with_opencv(
        self,
        video_path: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        OpenCV 기반 간단한 행동 분류 (움직임 강도 기반)

        Args:
            video_path: 비디오 파일 경로
            top_k: 상위 K개 예측

        Returns:
            List[Tuple[str, float]]: 예측 결과
        """
        try:
            st.info("ℹ️ OpenCV 모드: 움직임 강도 기반 간단 분류")

            # 프레임 추출
            frames = self.extract_frames(video_path, sample_rate=10, max_frames=20)

            if len(frames) < 2:
                return [('static', 0.9)]

            # 연속 프레임 간 움직임 계산
            motion_scores = []
            for i in range(len(frames) - 1):
                flow = self.compute_optical_flow(frames[i], frames[i + 1])
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_score = np.mean(magnitude)
                motion_scores.append(motion_score)

            avg_motion = np.mean(motion_scores)

            # 간단한 임계값 기반 분류
            if avg_motion > 10.0:
                return [
                    ('high_activity', 0.8),
                    ('running', 0.6),
                    ('jumping', 0.4),
                    ('walking', 0.2),
                    ('static', 0.1)
                ][:top_k]
            elif avg_motion > 5.0:
                return [
                    ('moderate_activity', 0.8),
                    ('walking', 0.7),
                    ('gesturing', 0.5),
                    ('running', 0.3),
                    ('static', 0.2)
                ][:top_k]
            else:
                return [
                    ('low_activity', 0.9),
                    ('static', 0.7),
                    ('sitting', 0.6),
                    ('standing', 0.4),
                    ('walking', 0.1)
                ][:top_k]

        except Exception as e:
            st.error(f"❌ OpenCV 분류 실패: {e}")
            return [('error', 0.0)]

    def _simulate_classification(self, top_k: int) -> List[Tuple[str, float]]:
        """
        시뮬레이션 모드: 더미 분류 결과 반환

        Args:
            top_k: 상위 K개 예측

        Returns:
            List[Tuple[str, float]]: 더미 예측 결과
        """
        st.info("ℹ️ 시뮬레이션 모드: 더미 분류 결과")

        dummy_results = [
            ('walking', 0.85),
            ('running', 0.10),
            ('jumping', 0.03),
            ('sitting', 0.01),
            ('standing', 0.01)
        ]

        return dummy_results[:top_k]

    def count_exercise_reps(
        self,
        video_path: str,
        exercise_type: str = 'pushup'
    ) -> Dict[str, Any]:
        """
        비디오에서 운동 반복 횟수 카운트 (MediaPipe Pose 사용)

        Args:
            video_path: 비디오 파일 경로
            exercise_type: 운동 종류 ('pushup', 'squat', 'jumping_jack')

        Returns:
            Dict: {'count': int, 'angle_history': List[float], 'confidence': float}
        """
        # Simulation mode
        if self.mode == 'simulation':
            st.info(f"ℹ️ 시뮬레이션 모드: {exercise_type} 카운트")
            count = np.random.randint(5, 20)
            angles = np.random.uniform(60, 160, size=count*2).tolist()
            return {
                'count': count,
                'angle_history': angles,
                'confidence': 0.85
            }

        # MediaPipe 사용 시도
        try:
            import mediapipe as mp

            st.info(f"🏋️ MediaPipe Pose로 {exercise_type} 카운트 중...")

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # 프레임 추출
            frames = self.extract_frames(video_path, sample_rate=2, max_frames=50)

            angles = []
            count = 0
            prev_angle = None
            in_down_position = False

            for frame in frames:
                # RGB로 변환 (MediaPipe는 RGB 입력)
                results = pose.process(frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # 운동별 각도 계산
                    if exercise_type == 'pushup':
                        # 팔꿈치 각도 (어깨-팔꿈치-손목)
                        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

                        angle = self._calculate_angle(
                            (shoulder.x, shoulder.y),
                            (elbow.x, elbow.y),
                            (wrist.x, wrist.y)
                        )

                    elif exercise_type == 'squat':
                        # 무릎 각도 (엉덩이-무릎-발목)
                        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

                        angle = self._calculate_angle(
                            (hip.x, hip.y),
                            (knee.x, knee.y),
                            (ankle.x, ankle.y)
                        )

                    else:  # jumping_jack
                        # 팔 각도 (어깨-팔꿈치-손목)
                        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

                        angle = self._calculate_angle(
                            (shoulder.x, shoulder.y),
                            (elbow.x, elbow.y),
                            (wrist.x, wrist.y)
                        )

                    angles.append(angle)

                    # 반복 카운트 로직 (각도 변화 감지)
                    if prev_angle is not None:
                        # Down position 감지 (각도가 작아짐)
                        if angle < 100 and not in_down_position:
                            in_down_position = True
                        # Up position 감지 (각도가 커짐) → 1회 카운트
                        elif angle > 140 and in_down_position:
                            count += 1
                            in_down_position = False

                    prev_angle = angle

            pose.close()

            st.success(f"✅ {exercise_type} {count}회 카운트 완료")

            return {
                'count': count,
                'angle_history': angles,
                'confidence': 0.9 if len(angles) > 10 else 0.5
            }

        except ImportError:
            st.warning("⚠️ MediaPipe가 설치되지 않아 시뮬레이션 모드로 동작합니다.")
            count = np.random.randint(5, 15)
            angles = np.random.uniform(60, 160, size=count*2).tolist()
            return {
                'count': count,
                'angle_history': angles,
                'confidence': 0.5
            }
        except Exception as e:
            st.error(f"❌ 운동 카운트 실패: {e}")
            return {'count': 0, 'angle_history': [], 'confidence': 0.0}

    def _calculate_angle(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        point3: Tuple[float, float]
    ) -> float:
        """
        3개 점으로 각도 계산

        Args:
            point1: 첫 번째 점 (x, y)
            point2: 중간 점 (x, y) - 각도의 꼭지점
            point3: 세 번째 점 (x, y)

        Returns:
            float: 각도 (도 단위, 0-180)
        """
        # 벡터 계산
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

        # 내적과 norm으로 각도 계산
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        비디오 메타데이터 추출

        Args:
            video_path: 비디오 파일 경로

        Returns:
            Dict: {'fps': float, 'duration': float, 'resolution': Tuple[int, int], 'frame_count': int}
        """
        if self.mode == 'simulation':
            return {
                'fps': 30.0,
                'duration': 10.0,
                'resolution': (1920, 1080),
                'frame_count': 300
            }

        try:
            import cv2

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0.0

            cap.release()

            return {
                'fps': fps,
                'duration': duration,
                'resolution': (width, height),
                'frame_count': frame_count
            }

        except Exception as e:
            st.error(f"❌ 비디오 정보 추출 실패: {e}")
            return {
                'fps': 0.0,
                'duration': 0.0,
                'resolution': (0, 0),
                'frame_count': 0
            }

    def save_temp_video(self, uploaded_bytes: bytes) -> Optional[str]:
        """
        업로드된 비디오를 임시 파일로 저장

        Args:
            uploaded_bytes: 업로드된 비디오 바이트

        Returns:
            Optional[str]: 임시 파일 경로, 실패 시 None
        """
        try:
            import tempfile

            # 임시 파일 생성 (자동 삭제하지 않음)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_bytes)
                tmp_path = tmp_file.name

            st.success(f"✅ 비디오 저장 완료: {os.path.basename(tmp_path)}")
            return tmp_path

        except Exception as e:
            st.error(f"❌ 비디오 저장 실패: {e}")
            return None


@st.cache_resource
def get_video_helper() -> VideoHelper:
    """
    캐시된 VideoHelper 인스턴스 반환 (싱글톤 패턴)

    Returns:
        VideoHelper: 캐시된 헬퍼 인스턴스
    """
    return VideoHelper()
