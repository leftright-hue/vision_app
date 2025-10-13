"""
Emotion Recognition Helper Classes

이 모듈은 고급 감정 인식을 위한 헬퍼 클래스들을 제공합니다:
- EmotionHelper: 멀티모달 API를 사용한 감정 분석 (3-tier fallback)
- VADModel: Valence-Arousal-Dominance 3차원 감정 모델
- EmotionTimeSeries: 시계열 감정 분석 및 추적
"""

import os
import io
import base64
import json
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Deque
import numpy as np
from PIL import Image
import streamlit as st
from dotenv import load_dotenv

from core.base_processor import BaseImageProcessor

# .env 파일 로드
load_dotenv()


class EmotionHelper(BaseImageProcessor):
    """
    멀티모달 API를 사용한 감정 분석 헬퍼 클래스

    3-tier fallback 전략:
    1. Google Gemini API (primary)
    2. OpenAI GPT-4o API (secondary)
    3. Simulation mode (fallback)
    """

    def __init__(self):
        """EmotionHelper 초기화"""
        super().__init__()
        self.mode: Optional[str] = None  # 'gemini', 'openai', 'simulation'
        self.gemini_model = None
        self.openai_client = None

        # API 초기화
        self._initialize_apis()

    def _initialize_apis(self):
        """
        3-tier fallback으로 API 초기화

        우선순위:
        1. Gemini API
        2. OpenAI API
        3. Simulation mode
        """
        # Try Gemini first
        if self._try_gemini():
            self.mode = 'gemini'
            st.success('✅ Google Gemini API 연결 성공')
            return

        # Try OpenAI as fallback
        if self._try_openai():
            self.mode = 'openai'
            st.success('✅ OpenAI GPT-4o API 연결 성공')
            return

        # Use simulation mode as last resort
        self.mode = 'simulation'
        st.warning('⚠️ API 키가 없습니다. 시뮬레이션 모드로 실행합니다.')

    def _try_gemini(self) -> bool:
        """
        Google Gemini API 연결 시도

        Returns:
            bool: 연결 성공 여부
        """
        try:
            import google.generativeai as genai

            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key or api_key == 'your_api_key_here':
                return False

            # Gemini API 설정
            genai.configure(api_key=api_key)

            # 모델 생성
            model_name = os.getenv('GENERATION_MODEL', 'gemini-2.5-pro')
            self.gemini_model = genai.GenerativeModel(model_name)

            return True

        except ImportError:
            st.warning('⚠️ google-generativeai 패키지가 설치되지 않았습니다.')
            return False
        except Exception as e:
            st.warning(f'⚠️ Gemini API 초기화 실패: {str(e)}')
            return False

    def _try_openai(self) -> bool:
        """
        OpenAI GPT-4o API 연결 시도

        Returns:
            bool: 연결 성공 여부
        """
        try:
            from openai import OpenAI

            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key or api_key.startswith('your_'):
                return False

            # OpenAI 클라이언트 생성
            self.openai_client = OpenAI(api_key=api_key)

            return True

        except ImportError:
            st.warning('⚠️ openai 패키지가 설치되지 않았습니다.')
            return False
        except Exception as e:
            st.warning(f'⚠️ OpenAI API 초기화 실패: {str(e)}')
            return False

    def get_mode(self) -> str:
        """
        현재 동작 모드 반환

        Returns:
            str: 'gemini', 'openai', 또는 'simulation'
        """
        return self.mode or 'unknown'

    def is_available(self) -> bool:
        """
        API 사용 가능 여부 확인

        Returns:
            bool: API가 사용 가능하면 True
        """
        return self.mode in ['gemini', 'openai']

    def get_status_message(self) -> str:
        """
        현재 상태 메시지 반환

        Returns:
            str: 상태 설명 메시지
        """
        if self.mode == 'gemini':
            return '🟢 Google Gemini API 사용 중'
        elif self.mode == 'openai':
            return '🟡 OpenAI GPT-4o API 사용 중'
        elif self.mode == 'simulation':
            return '🔴 시뮬레이션 모드 (API 키 필요)'
        else:
            return '⚫ 알 수 없는 상태'

    def analyze_basic_emotion(
        self,
        image: Image.Image,
        prompt: Optional[str] = None
    ) -> Dict[str, float]:
        """
        이미지에서 기본 감정을 분석합니다.

        7가지 기본 감정을 분석하여 각 감정의 신뢰도를 반환합니다:
        - happy (행복)
        - sad (슬픔)
        - angry (분노)
        - fear (공포)
        - surprise (놀람)
        - disgust (혐오)
        - neutral (중립)

        Args:
            image: 분석할 PIL 이미지
            prompt: 추가 컨텍스트 프롬프트 (선택 사항)

        Returns:
            Dict[str, float]: 감정명: 신뢰도 (0.0-1.0) 딕셔너리
        """
        if self.mode == 'gemini':
            return self._analyze_with_gemini(image, prompt)
        elif self.mode == 'openai':
            return self._analyze_with_openai(image, prompt)
        else:
            return self._simulate_emotion()

    def _analyze_with_gemini(
        self,
        image: Image.Image,
        prompt: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Google Gemini API를 사용하여 감정 분석

        Args:
            image: 분석할 PIL 이미지
            prompt: 추가 컨텍스트 프롬프트

        Returns:
            Dict[str, float]: 감정 신뢰도 딕셔너리
        """
        import re

        try:
            # 프롬프트 구성
            analysis_prompt = '''이미지 속 사람의 감정을 분석하고 다음 JSON 형식으로만 반환하세요.
다른 설명 없이 JSON만 출력해주세요:

{
  "happy": 0.0,
  "sad": 0.0,
  "angry": 0.0,
  "fear": 0.0,
  "surprise": 0.0,
  "disgust": 0.0,
  "neutral": 0.0
}

각 값은 0.0에서 1.0 사이의 신뢰도입니다.'''

            if prompt:
                analysis_prompt += f'\n\n추가 컨텍스트: {prompt}'

            # 이미지 크기 최적화 (API 비용 절감)
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                image = self.resize_image(image, (max_size, max_size))

            # Gemini API 호출
            response = self.gemini_model.generate_content([analysis_prompt, image])

            # JSON 파싱
            return self._parse_emotion_response(response.text)

        except Exception as e:
            st.error(f'❌ Gemini API 호출 실패: {str(e)}')
            return self._simulate_emotion()

    def _parse_emotion_response(self, text: str) -> Dict[str, float]:
        """
        API 응답 텍스트에서 감정 데이터 파싱

        Args:
            text: API 응답 텍스트

        Returns:
            Dict[str, float]: 감정 신뢰도 딕셔너리
        """
        import re

        try:
            # JSON 블록 찾기 (```json ... ``` 또는 { ... } 형태)
            json_match = re.search(r'```json\s*(\{[^`]+\})\s*```', text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)

            if json_match:
                json_str = json_match.group(1) if '```' in text else json_match.group(0)
                emotion_data = json.loads(json_str)

                # 7가지 기본 감정이 모두 있는지 확인
                required_emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
                if all(emotion in emotion_data for emotion in required_emotions):
                    # 값이 0.0-1.0 범위인지 확인
                    for emotion in required_emotions:
                        if not isinstance(emotion_data[emotion], (int, float)):
                            raise ValueError(f'Invalid value type for {emotion}')
                        emotion_data[emotion] = max(0.0, min(1.0, float(emotion_data[emotion])))

                    return emotion_data

        except (json.JSONDecodeError, ValueError) as e:
            st.warning(f'⚠️ JSON 파싱 실패: {str(e)}')

        # 파싱 실패 시 기본값 반환
        return {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0,
            'neutral': 1.0
        }

    def _analyze_with_openai(
        self,
        image: Image.Image,
        prompt: Optional[str] = None
    ) -> Dict[str, float]:
        """
        OpenAI GPT-4o API를 사용하여 감정 분석

        Args:
            image: 분석할 PIL 이미지
            prompt: 추가 컨텍스트 프롬프트

        Returns:
            Dict[str, float]: 감정 신뢰도 딕셔너리
        """
        try:
            # 이미지를 base64로 변환
            image_base64 = self._image_to_base64(image)

            # 프롬프트 구성
            analysis_prompt = '''이미지 속 사람의 감정을 분석하고 다음 JSON 형식으로만 반환하세요.
다른 설명 없이 JSON만 출력해주세요:

{
  "happy": 0.0,
  "sad": 0.0,
  "angry": 0.0,
  "fear": 0.0,
  "surprise": 0.0,
  "disgust": 0.0,
  "neutral": 0.0
}

각 값은 0.0에서 1.0 사이의 신뢰도입니다.'''

            if prompt:
                analysis_prompt += f'\n\n추가 컨텍스트: {prompt}'

            # GPT-4o API 호출
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            # JSON 파싱
            return self._parse_emotion_response(response.choices[0].message.content)

        except Exception as e:
            st.error(f'❌ OpenAI API 호출 실패: {str(e)}')
            return self._simulate_emotion()

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        PIL 이미지를 base64 문자열로 변환

        OpenAI API는 base64 인코딩된 이미지를 요구합니다.

        Args:
            image: 변환할 PIL 이미지

        Returns:
            str: base64 인코딩된 이미지 문자열
        """
        # 이미지 크기 최적화 (API 비용 절감)
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            # 비율 유지하며 리사이즈
            ratio = max_size / max(image.width, image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # JPEG로 변환하여 base64 인코딩
        buffered = io.BytesIO()

        # RGB로 변환 (JPEG는 RGBA 지원 안 함)
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image

        image.save(buffered, format="JPEG", quality=85)
        image_bytes = buffered.getvalue()

        return base64.b64encode(image_bytes).decode('utf-8')

    def _simulate_emotion(self) -> Dict[str, float]:
        """
        시뮬레이션 모드에서 랜덤 감정 생성

        테스트 및 데모 목적으로 사용됩니다.

        Returns:
            Dict[str, float]: 랜덤 감정 신뢰도 딕셔너리
        """
        import random

        # 랜덤 감정 생성
        emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
        values = [random.random() for _ in emotions]

        # 정규화 (합계가 1.0이 되도록)
        total = sum(values)
        normalized = {emotion: value / total for emotion, value in zip(emotions, values)}

        return normalized

    def analyze_multimodal(
        self,
        image: Image.Image,
        text: str
    ) -> Dict[str, Any]:
        """
        이미지와 텍스트를 함께 분석 (멀티모달 분석)

        이미지만 분석한 결과와 텍스트 컨텍스트를 함께 고려한 결과를 비교합니다.

        Args:
            image: 분석할 PIL 이미지
            text: 추가 컨텍스트 텍스트

        Returns:
            Dict[str, Any]: {
                'image_only': 이미지만 분석한 감정,
                'combined': 이미지 + 텍스트 통합 분석,
                'text': 입력 텍스트,
                'difference': 두 분석 결과의 차이
            }
        """
        # 1. 이미지만 분석
        image_only = self.analyze_basic_emotion(image)

        # 2. 이미지 + 텍스트 통합 분석
        if self.mode in ['gemini', 'openai']:
            prompt = f'''이미지와 다음 텍스트를 함께 고려하여 감정을 분석하세요.

텍스트 내용: "{text}"

이미지의 감정과 텍스트의 감정을 종합하여 최종 감정을 판단하세요.'''

            combined = self.analyze_basic_emotion(image, prompt)
        else:
            # 시뮬레이션 모드에서는 이미지 결과 재사용
            combined = image_only.copy()

        # 3. 차이 계산
        difference = {}
        for emotion in image_only.keys():
            diff = combined.get(emotion, 0.0) - image_only.get(emotion, 0.0)
            difference[emotion] = diff

        return {
            'image_only': image_only,
            'combined': combined,
            'text': text,
            'difference': difference
        }


@st.cache_resource
def get_emotion_helper() -> EmotionHelper:
    """
    EmotionHelper 싱글톤 인스턴스 반환

    Streamlit의 @st.cache_resource를 사용하여
    세션 전체에서 하나의 인스턴스만 생성합니다.

    Returns:
        EmotionHelper: 감정 분석 헬퍼 인스턴스
    """
    return EmotionHelper()


class VADModel:
    """
    VAD (Valence-Arousal-Dominance) 3차원 감정 모델

    감정을 3차원 공간에 매핑하여 표현합니다:
    - Valence (가): 긍정적 ↔ 부정적 (-1.0 ~ 1.0)
    - Arousal (각성): 차분함 ↔ 흥분 (-1.0 ~ 1.0)
    - Dominance (지배): 복종 ↔ 지배 (-1.0 ~ 1.0)
    """

    # 기본 감정의 VAD 좌표 매핑
    EMOTION_VAD_MAP: Dict[str, Tuple[float, float, float]] = {
        'happy': (0.8, 0.5, 0.6),       # 긍정적, 중간 각성, 약간 지배적
        'sad': (-0.7, -0.6, -0.5),      # 부정적, 낮은 각성, 복종적
        'angry': (-0.5, 0.7, 0.8),      # 부정적, 높은 각성, 매우 지배적
        'fear': (-0.6, 0.7, -0.6),      # 부정적, 높은 각성, 복종적
        'surprise': (0.2, 0.8, 0.0),    # 약간 긍정, 매우 높은 각성, 중립
        'disgust': (-0.6, 0.4, 0.3),    # 부정적, 중간 각성, 약간 지배적
        'neutral': (0.0, 0.0, 0.0),     # 중립
        'calm': (0.3, -0.5, 0.2),       # 약간 긍정, 낮은 각성, 약간 지배적
    }

    @staticmethod
    def emotion_to_vad(emotion: str) -> Tuple[float, float, float]:
        """
        감정명을 VAD 좌표로 변환

        Args:
            emotion: 감정명 (예: 'happy', 'sad')

        Returns:
            Tuple[float, float, float]: (valence, arousal, dominance)
        """
        return VADModel.EMOTION_VAD_MAP.get(emotion.lower(), (0.0, 0.0, 0.0))

    @staticmethod
    def vad_to_emotion(
        valence: float,
        arousal: float,
        dominance: float
    ) -> str:
        """
        VAD 좌표를 가장 가까운 감정으로 역변환

        Args:
            valence: Valence 값 (-1.0 ~ 1.0)
            arousal: Arousal 값 (-1.0 ~ 1.0)
            dominance: Dominance 값 (-1.0 ~ 1.0)

        Returns:
            str: 가장 가까운 감정명
        """
        min_distance = float('inf')
        closest_emotion = 'neutral'

        # 모든 감정과의 거리 계산
        for emotion, (v, a, d) in VADModel.EMOTION_VAD_MAP.items():
            distance = np.sqrt(
                (valence - v) ** 2 +
                (arousal - a) ** 2 +
                (dominance - d) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_emotion = emotion

        return closest_emotion

    @staticmethod
    def calculate_similarity(
        vad1: Tuple[float, float, float],
        vad2: Tuple[float, float, float]
    ) -> float:
        """
        두 VAD 좌표 간의 유사도 계산

        유사도는 유클리드 거리를 반전시켜 0.0-1.0 범위로 정규화합니다.

        Args:
            vad1: 첫 번째 VAD 좌표
            vad2: 두 번째 VAD 좌표

        Returns:
            float: 유사도 (0.0 = 완전히 다름, 1.0 = 동일)
        """
        # 유클리드 거리 계산
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(vad1, vad2)))

        # 3차원 공간에서 최대 거리는 sqrt(3) * 2 = 3.46
        # 정규화: 1.0 - (distance / max_distance)
        max_distance = np.sqrt(3 * (2.0 ** 2))  # sqrt(12) = 3.46
        similarity = max(0.0, 1.0 - (distance / max_distance))

        return similarity

    @staticmethod
    def visualize_3d(
        vad_points: List[Tuple[float, float, float]],
        labels: Optional[List[str]] = None,
        title: str = 'VAD 3D Emotion Space'
    ):
        """
        VAD 좌표를 3D 산점도로 시각화

        Args:
            vad_points: VAD 좌표 리스트
            labels: 각 점의 레이블 (선택 사항)
            title: 그래프 제목

        Returns:
            matplotlib.figure.Figure: 생성된 Figure 객체
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 점 그리기
        for i, (v, a, d) in enumerate(vad_points):
            label = labels[i] if labels and i < len(labels) else f'Point {i+1}'
            ax.scatter(v, a, d, s=200, alpha=0.6, label=label)

        # 축 설정
        ax.set_xlabel('Valence (긍정 ↔ 부정)', fontsize=12)
        ax.set_ylabel('Arousal (각성)', fontsize=12)
        ax.set_zlabel('Dominance (지배)', fontsize=12)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        # 제목 및 범례
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        # 그리드
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def get_emotion_description(emotion: str) -> str:
        """
        감정에 대한 설명 반환

        Args:
            emotion: 감정명

        Returns:
            str: 감정 설명
        """
        descriptions = {
            'happy': '😊 행복: 긍정적이고 즐거운 감정',
            'sad': '😢 슬픔: 부정적이고 우울한 감정',
            'angry': '😠 분노: 부정적이고 격렬한 감정',
            'fear': '😨 공포: 위협을 느끼는 부정적 감정',
            'surprise': '😲 놀람: 예상치 못한 자극에 대한 반응',
            'disgust': '🤢 혐오: 불쾌하고 거부감이 드는 감정',
            'neutral': '😐 중립: 특별한 감정이 없는 상태',
            'calm': '😌 평온: 차분하고 안정된 감정',
        }
        return descriptions.get(emotion.lower(), f'{emotion}: 알 수 없는 감정')


class EmotionTimeSeries:
    """
    시계열 감정 분석 및 추적 클래스

    여러 프레임에 걸친 감정 변화를 추적하고 분석합니다.
    """

    def __init__(self, window_size: int = 100):
        """
        EmotionTimeSeries 초기화

        Args:
            window_size: 저장할 최대 프레임 수 (메모리 제한)
        """
        self.history: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.window_size = window_size

    def add_frame(
        self,
        emotion_data: Dict[str, float],
        timestamp: Optional[float] = None
    ):
        """
        프레임별 감정 데이터 추가

        Args:
            emotion_data: 감정 신뢰도 딕셔너리
            timestamp: 타임스탬프 (None이면 현재 시각)
        """
        import time

        if timestamp is None:
            timestamp = time.time()

        self.history.append({
            'timestamp': timestamp,
            'emotions': emotion_data.copy()
        })

    def get_trend(self, emotion: str = 'happy') -> str:
        """
        특정 감정의 변화 추세 분석

        최근 5개 프레임의 선형 회귀 기울기로 추세를 판단합니다.

        Args:
            emotion: 분석할 감정명

        Returns:
            str: 'increasing', 'decreasing', 또는 'stable'
        """
        if len(self.history) < 5:
            return 'stable'

        # 최근 5개 프레임의 감정 값 추출
        values = [frame['emotions'].get(emotion, 0.0) for frame in list(self.history)[-5:]]

        # 선형 회귀로 기울기 계산
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # 기울기로 추세 판단
        if slope > 0.05:
            return 'increasing'
        elif slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'

    def detect_change_points(self, threshold: float = 0.3) -> List[int]:
        """
        급격한 감정 변화 지점 감지

        연속된 프레임 간 감정 변화가 threshold를 초과하는 지점을 찾습니다.

        Args:
            threshold: 변화 감지 임계값 (0.0-1.0)

        Returns:
            List[int]: 변화 지점 인덱스 리스트
        """
        if len(self.history) < 2:
            return []

        change_points = []

        for i in range(1, len(self.history)):
            prev_emotions = self.history[i - 1]['emotions']
            curr_emotions = self.history[i]['emotions']

            # 모든 감정의 변화량 계산
            max_change = 0.0
            for emotion in prev_emotions.keys():
                change = abs(curr_emotions.get(emotion, 0.0) - prev_emotions.get(emotion, 0.0))
                max_change = max(max_change, change)

            # 임계값 초과 시 변화점으로 기록
            if max_change > threshold:
                change_points.append(i)

        return change_points

    def visualize_timeline(
        self,
        emotions: Optional[List[str]] = None,
        title: str = 'Emotion Timeline'
    ):
        """
        시계열 감정 그래프 생성

        Args:
            emotions: 표시할 감정 리스트 (None이면 주요 3개)
            title: 그래프 제목

        Returns:
            matplotlib.figure.Figure: 생성된 Figure 객체
        """
        import matplotlib.pyplot as plt

        if not self.history:
            return None

        if emotions is None:
            emotions = ['happy', 'sad', 'angry']

        fig, ax = plt.subplots(figsize=(14, 7))

        # 각 감정의 시계열 데이터 플롯
        for emotion in emotions:
            values = [frame['emotions'].get(emotion, 0.0) for frame in self.history]
            ax.plot(values, label=emotion.capitalize(), linewidth=2, marker='o', markersize=4)

        # 변화점 표시
        change_points = self.detect_change_points()
        if change_points:
            for cp in change_points:
                ax.axvline(x=cp, color='red', linestyle='--', alpha=0.3, linewidth=1)

        # 축 및 레이블 설정
        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('Confidence (0-1)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        return fig

    def export_to_csv(self, filepath: str):
        """
        감정 데이터를 CSV 파일로 내보내기

        Args:
            filepath: 저장할 CSV 파일 경로
        """
        if not self.history:
            raise ValueError("No data to export")

        # 데이터 변환
        data = []
        for frame in self.history:
            row = {'timestamp': frame['timestamp']}
            row.update(frame['emotions'])
            data.append(row)

        # pandas DataFrame으로 변환 후 CSV 저장
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        except ImportError:
            # pandas 없으면 기본 CSV 작성
            import csv
            with open(filepath, 'w', newline='') as f:
                if data:
                    fieldnames = list(data[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)

    def get_summary(self) -> Dict[str, Any]:
        """
        시계열 데이터 요약 통계

        Returns:
            Dict[str, Any]: 통계 정보 딕셔너리
        """
        if not self.history:
            return {}

        # 모든 감정의 평균값 계산
        all_emotions = set()
        for frame in self.history:
            all_emotions.update(frame['emotions'].keys())

        summary = {}
        for emotion in all_emotions:
            values = [frame['emotions'].get(emotion, 0.0) for frame in self.history]
            summary[emotion] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': self.get_trend(emotion)
            }

        return summary
