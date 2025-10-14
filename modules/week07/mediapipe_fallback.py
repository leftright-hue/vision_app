"""
MediaPipe 대체 기능 모듈
Python 3.13에서 MediaPipe가 지원되지 않을 때 OpenCV 기반 대체 기능 제공
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import streamlit as st


class MediaPipeFallback:
    """MediaPipe 대체 기능 클래스"""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.load_cascades()
    
    def load_cascades(self):
        """OpenCV Haar Cascade 로드"""
        try:
            # 얼굴 검출기
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            # 눈 검출기
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        except Exception as e:
            st.warning(f"Cascade 로드 실패: {e}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """얼굴 검출 (MediaPipe 대체)"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_pose_simulation(self, image: np.ndarray) -> dict:
        """포즈 검출 시뮬레이션 (MediaPipe 대체)"""
        height, width = image.shape[:2]
        
        # 가상의 관절 포인트 (시뮬레이션)
        landmarks = {
            'nose': (width//2, height//4),
            'left_shoulder': (width//3, height//2),
            'right_shoulder': (width*2//3, height//2),
            'left_elbow': (width//4, height*3//4),
            'right_elbow': (width*3//4, height*3//4),
            'left_wrist': (width//5, height*4//5),
            'right_wrist': (width*4//5, height*4//5),
        }
        
        return {
            'landmarks': landmarks,
            'visibility': {key: 0.8 for key in landmarks.keys()},
            'simulation': True
        }
    
    def count_exercise_reps_simulation(self, exercise_type: str, video_duration: float = 30.0) -> dict:
        """운동 횟수 카운팅 시뮬레이션 (개선된 버전)"""
        import random
        
        # 운동별 현실적인 범위 설정
        exercise_ranges = {
            'pushup': {'min': 5, 'max': 20, 'avg_angle': (60, 120), 'difficulty': 0.8},
            'squat': {'min': 8, 'max': 25, 'avg_angle': (45, 135), 'difficulty': 0.7}, 
            'jumping_jack': {'min': 10, 'max': 30, 'avg_angle': (30, 150), 'difficulty': 0.6},
            'plank': {'min': 1, 'max': 3, 'avg_angle': (170, 180), 'difficulty': 0.9}
        }
        
        # 기본값 설정
        if exercise_type not in exercise_ranges:
            exercise_type = 'pushup'
        
        config = exercise_ranges[exercise_type]
        
        # 비디오 길이에 따른 현실적인 횟수 계산
        duration_factor = min(video_duration / 30.0, 2.0)  # 최대 2배
        base_reps = random.randint(config['min'], config['max'])
        rep_count = int(base_reps * duration_factor)
        
        # 운동 난이도에 따른 신뢰도 조정
        base_confidence = config['difficulty']
        confidence = base_confidence + random.uniform(-0.1, 0.1)
        confidence = max(0.5, min(0.95, confidence))
        
        # 현실적인 각도 범위
        angle_range = config['avg_angle']
        avg_angle = random.uniform(angle_range[0], angle_range[1])
        
        return {
            'exercise': exercise_type,
            'reps': rep_count,
            'confidence': confidence,
            'feedback': f"{exercise_type} {rep_count}회 감지 (시뮬레이션 - 실제 결과와 다를 수 있음)",
            'simulation': True,
            'warning': "⚠️ 시뮬레이션 결과입니다. 정확한 분석은 MediaPipe(Python 3.11) 필요",
            'details': {
                'avg_angle': avg_angle,
                'completion_rate': random.uniform(0.75, 0.95),
                'form_score': confidence * random.uniform(0.9, 1.1),
                'duration': video_duration,
                'reps_per_minute': (rep_count / video_duration) * 60 if video_duration > 0 else 0
            }
        }


def get_mediapipe_fallback():
    """MediaPipe 대체 객체 반환"""
    return MediaPipeFallback()


def check_mediapipe_availability():
    """MediaPipe 사용 가능 여부 확인"""
    try:
        import mediapipe
        return True, mediapipe.__version__
    except ImportError:
        return False, None


def safe_mediapipe_import():
    """안전한 MediaPipe 임포트"""
    try:
        import mediapipe as mp
        return mp, True
    except ImportError:
        st.info("""
        💡 **OpenCV 기반 대체 기능 사용**
        
        MediaPipe가 설치되지 않았지만 OpenCV를 사용한 기본 기능을 제공합니다:
        - 얼굴 검출 (Haar Cascade)
        - 포즈 추정 시뮬레이션  
        - 운동 카운팅 시뮬레이션
        
        더 정확한 기능을 원한다면 Python 3.11 환경에서 MediaPipe를 사용하세요.
        """)
        return get_mediapipe_fallback(), False


def draw_face_landmarks_opencv(image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """OpenCV를 사용한 얼굴 랜드마크 그리기"""
    result = image.copy()
    
    for (x, y, w, h) in faces:
        # 얼굴 박스
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 가상의 랜드마크 포인트
        center_x, center_y = x + w//2, y + h//2
        
        # 눈 위치 추정
        left_eye = (x + w//3, y + h//3)
        right_eye = (x + w*2//3, y + h//3)
        
        # 코 위치 추정
        nose = (center_x, y + h//2)
        
        # 입 위치 추정
        mouth = (center_x, y + h*3//4)
        
        # 포인트 그리기
        points = [left_eye, right_eye, nose, mouth]
        for point in points:
            cv2.circle(result, point, 3, (0, 255, 0), -1)
    
    return result