"""
Week 7: 행동인식과 Action Recognition

이 모듈은 비디오에서 행동을 인식하는 이론과 실습을 다룹니다:
- 행동인식 개념과 아키텍처 (3D CNN, Two-Stream, Transformer)
- 비디오 처리 기초 (프레임 추출, Optical Flow)
- 사전훈련 모델 활용 (HuggingFace VideoMAE, TimeSformer)
- 실시간 행동 인식 및 실전 응용 (운동 카운터 등)
"""

from .action_recognition_module import ActionRecognitionModule

__all__ = ['ActionRecognitionModule']
