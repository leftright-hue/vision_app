"""
Week 8: 고급 감정 인식 (Advanced Emotion Recognition)

이 모듈은 멀티모달 API를 활용한 고급 감정 인식을 다룹니다:
- Google Gemini API를 사용한 얼굴 감정 분석
- OpenAI GPT-4o API를 사용한 멀티모달 감정 해석
- VAD (Valence-Arousal-Dominance) 3차원 감정 모델
- 멀티모달 분석 (이미지 + 텍스트)
- 시계열 감정 변화 추적 및 시각화
"""

from .emotion_recognition_module import EmotionRecognitionModule

__all__ = ['EmotionRecognitionModule']
