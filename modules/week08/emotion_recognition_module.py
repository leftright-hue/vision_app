"""
Week 8: 고급 감정 인식 (Advanced Emotion Recognition)

멀티모달 API를 활용한 고급 감정 인식 Streamlit 모듈
"""

import streamlit as st
from typing import Dict, List, Optional, Any
from PIL import Image
import numpy as np

from core.base_processor import BaseImageProcessor
from .emotion_helpers import get_emotion_helper, VADModel, EmotionTimeSeries


class EmotionRecognitionModule(BaseImageProcessor):
    """
    고급 감정 인식 모듈

    Google Gemini, OpenAI GPT-4o API를 사용하여 다음 기능을 제공합니다:
    - 7가지 기본 감정 인식
    - VAD 3차원 감정 모델
    - 멀티모달 분석 (이미지 + 텍스트)
    - 시계열 감정 추적
    """

    def __init__(self):
        """EmotionRecognitionModule 초기화"""
        super().__init__()
        self.name = 'Week 8: Emotion Recognition'

    def render(self):
        """메인 렌더링 메서드"""
        st.title('🎭 Week 8: 고급 감정 인식')

        st.markdown("""
        Google Gemini와 OpenAI GPT-4o API를 사용한 고급 감정 인식을 학습합니다.
        """)

        # 환경 체크
        self._display_environment_status()

        # 5개 탭 생성
        tabs = st.tabs([
            '📚 개념 소개',
            '😊 기본 감정 인식',
            '📊 VAD 모델',
            '🎨 멀티모달 분석',
            '📈 시계열 분석'
        ])

        with tabs[0]:
            self.render_theory()

        with tabs[1]:
            self.render_basic_emotion()

        with tabs[2]:
            self.render_vad_model()

        with tabs[3]:
            self.render_multimodal()

        with tabs[4]:
            self.render_timeseries()

    def _display_environment_status(self):
        """환경 상태 표시"""
        status = self._check_environment()

        cols = st.columns(3)

        with cols[0]:
            if status.get('gemini'):
                st.success('✅ Google Gemini 사용 가능')
            else:
                st.warning('⚠️ Gemini 패키지 없음')

        with cols[1]:
            if status.get('openai'):
                st.success('✅ OpenAI 사용 가능')
            else:
                st.warning('⚠️ OpenAI 패키지 없음')

        with cols[2]:
            if status.get('plotly'):
                st.success('✅ Plotly 사용 가능')
            else:
                st.warning('⚠️ Plotly 패키지 없음')

    def _check_environment(self) -> Dict[str, bool]:
        """패키지 설치 여부 확인"""
        status = {}

        # Google Gemini
        try:
            import google.generativeai
            status['gemini'] = True
        except ImportError:
            status['gemini'] = False

        # OpenAI
        try:
            import openai
            status['openai'] = True
        except ImportError:
            status['openai'] = False

        # Plotly
        try:
            import plotly
            status['plotly'] = True
        except ImportError:
            status['plotly'] = False

        return status

    def render_theory(self):
        """Tab 1: 개념 소개"""
        st.header('📚 고급 감정 인식 개념')

        st.markdown("""
        ### 🎭 감정 인식의 발전

        감정 인식은 단순한 분류에서 연속적 모델로 발전해왔습니다.
        """)

        # 기본 감정 vs 복잡한 감정
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('📝 Ekman의 6가지 기본 감정')
            st.markdown("""
            Paul Ekman이 제안한 문화 보편적 감정:
            - 😊 **Happy** (행복)
            - 😢 **Sad** (슬픔)
            - 😠 **Angry** (분노)
            - 😨 **Fear** (공포)
            - 😲 **Surprise** (놀람)
            - 🤢 **Disgust** (혐오)

            **장점**: 단순하고 명확
            **단점**: 복잡한 감정 표현 불가
            """)

        with col2:
            st.subheader('🌈 Plutchik의 감정 바퀴')
            st.markdown("""
            Robert Plutchik의 8가지 기본 + 24가지 복합 감정:
            - 8가지 기본 감정
            - 강도에 따른 변화
            - 복합 감정 (예: 사랑 = 기쁨 + 신뢰)

            **장점**: 복잡한 감정 표현 가능
            **단점**: 여전히 이산적
            """)

        st.markdown('---')

        # VAD 모델 설명
        st.subheader('🎯 VAD 3차원 감정 모델')

        st.markdown("""
        **VAD (Valence-Arousal-Dominance)** 모델은 감정을 3차원 연속 공간으로 표현합니다:

        - **Valence (원자가)**: 긍정적 ↔ 부정적 (-1.0 ~ 1.0)
        - **Arousal (각성)**: 차분함 ↔ 흥분 (-1.0 ~ 1.0)
        - **Dominance (지배)**: 복종 ↔ 지배 (-1.0 ~ 1.0)

        **장점**:
        - 연속적이고 미묘한 감정 표현
        - 무한한 감정 상태 표현 가능
        - 감정 간 유사도 계산 가능
        """)

        # VAD 3D 시각화
        st.subheader('🔬 VAD 공간 시각화')

        emotions_to_show = st.multiselect(
            '표시할 감정 선택',
            ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral', 'calm'],
            default=['happy', 'sad', 'angry', 'fear']
        )

        if emotions_to_show:
            vad_points = [VADModel.emotion_to_vad(e) for e in emotions_to_show]
            fig = VADModel.visualize_3d(vad_points, emotions_to_show, '기본 감정의 VAD 좌표')
            st.pyplot(fig)

            # 감정 설명
            st.markdown('#### 선택한 감정의 설명')
            for emotion in emotions_to_show:
                st.write(VADModel.get_emotion_description(emotion))

        st.markdown('---')

        # 멀티모달 분석
        st.subheader('🎨 멀티모달 감정 분석')

        st.markdown("""
        **멀티모달 분석**은 여러 정보 소스를 통합하여 더 정확한 감정 인식을 수행합니다:

        - 🖼️ **이미지**: 얼굴 표정, 몸짓
        - 📝 **텍스트**: 컨텍스트, 상황 설명
        - 🎤 **음성**: 톤, 억양 (미구현)
        - 🎬 **비디오**: 시간적 변화 (시계열 분석)

        이미지만으로는 애매한 감정도 텍스트 컨텍스트와 함께 분석하면 정확도가 향상됩니다.

        **예시**:
        - 이미지: 웃는 얼굴 → "행복"
        - 텍스트: "오늘 시험에 떨어졌어요" → "슬픔"
        - 통합: 억지로 웃는 "슬픔 + 강요된 행복"
        """)

    def render_basic_emotion(self):
        """Tab 2: 기본 감정 인식"""
        st.header('😊 기본 감정 인식')

        st.markdown("""
        이미지에서 7가지 기본 감정을 인식합니다.
        """)

        # EmotionHelper 가져오기
        helper = get_emotion_helper()
        st.info(f'🤖 {helper.get_status_message()}')

        # 이미지 업로드
        uploaded_file = st.file_uploader(
            '감정을 분석할 이미지를 업로드하세요',
            type=['png', 'jpg', 'jpeg', 'webp'],
            help='얼굴이 명확히 보이는 이미지를 선택하세요'
        )

        if uploaded_file is not None:
            # 이미지 로드
            image = Image.open(uploaded_file)

            # 2열 레이아웃
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader('입력 이미지')
                st.image(image, use_container_width=True)

                # 이미지 정보
                with st.expander('이미지 정보'):
                    stats = self.get_image_stats(image)
                    st.write(f"**크기**: {stats['width']} x {stats['height']}")
                    st.write(f"**모드**: {stats['mode']}")

            with col2:
                st.subheader('감정 분석 결과')

                # 분석 버튼
                if st.button('🔍 감정 분석 시작', type='primary', use_container_width=True):
                    with st.spinner('AI가 감정을 분석하고 있습니다...'):
                        result = helper.analyze_basic_emotion(image)

                    # 결과 저장
                    st.session_state['emotion_result'] = result

                # 결과 표시
                if 'emotion_result' in st.session_state:
                    result = st.session_state['emotion_result']

                    # 상위 3개 감정
                    st.markdown('#### 🏆 Top 3 감정')
                    sorted_emotions = sorted(result.items(), key=lambda x: x[1], reverse=True)

                    for i, (emotion, score) in enumerate(sorted_emotions[:3], 1):
                        st.metric(
                            label=f"{i}. {emotion.capitalize()}",
                            value=f"{score:.2%}"
                        )

                    # 바 차트
                    st.markdown('#### 📊 전체 감정 분석')

                    try:
                        import plotly.graph_objects as go

                        fig = go.Figure([
                            go.Bar(
                                x=list(result.keys()),
                                y=list(result.values()),
                                marker_color='lightblue',
                                text=[f'{v:.2%}' for v in result.values()],
                                textposition='outside'
                            )
                        ])

                        fig.update_layout(
                            title='7가지 감정 신뢰도',
                            xaxis_title='감정',
                            yaxis_title='신뢰도',
                            yaxis=dict(range=[0, 1]),
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    except ImportError:
                        # Plotly 없으면 streamlit bar_chart 사용
                        st.bar_chart(result)

                    # VAD 좌표
                    st.markdown('#### 🎯 VAD 좌표')
                    dominant_emotion = max(result.items(), key=lambda x: x[1])[0]
                    v, a, d = VADModel.emotion_to_vad(dominant_emotion)

                    vad_cols = st.columns(3)
                    vad_cols[0].metric('Valence', f'{v:.2f}')
                    vad_cols[1].metric('Arousal', f'{a:.2f}')
                    vad_cols[2].metric('Dominance', f'{d:.2f}')

        else:
            st.info('👆 이미지를 업로드하여 감정 분석을 시작하세요')

    def render_vad_model(self):
        """Tab 3: VAD 모델"""
        st.header('📊 VAD 3차원 감정 모델')

        st.markdown("""
        VAD 모델을 사용하여 감정을 3차원 공간에 매핑합니다.
        이미지의 주요 감정을 VAD 좌표로 변환하고 유사한 감정을 찾을 수 있습니다.
        """)

        # EmotionHelper 가져오기
        helper = get_emotion_helper()
        st.info(f'🤖 {helper.get_status_message()}')

        # 이미지 업로드
        uploaded_file = st.file_uploader(
            'VAD 분석할 이미지를 업로드하세요',
            type=['png', 'jpg', 'jpeg', 'webp'],
            key='vad_upload'
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader('입력 이미지')
                st.image(image, use_container_width=True)

            with col2:
                st.subheader('VAD 분석')

                if st.button('🎯 VAD 분석 시작', type='primary', use_container_width=True):
                    with st.spinner('감정을 분석하고 VAD 좌표를 계산 중...'):
                        # 기본 감정 분석
                        emotions = helper.analyze_basic_emotion(image)

                        # 지배적 감정 찾기
                        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                        dominant_score = emotions[dominant_emotion]

                        # VAD 좌표 계산
                        vad = VADModel.emotion_to_vad(dominant_emotion)

                        # 결과 저장
                        st.session_state['vad_result'] = {
                            'emotions': emotions,
                            'dominant': dominant_emotion,
                            'score': dominant_score,
                            'vad': vad
                        }

            # 결과 표시
            if 'vad_result' in st.session_state:
                result = st.session_state['vad_result']

                st.markdown('---')

                # 주요 정보
                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.markdown('#### 🏆 주요 감정')
                    st.success(f"**{result['dominant'].upper()}** ({result['score']:.2%})")
                    st.write(VADModel.get_emotion_description(result['dominant']))

                with info_col2:
                    st.markdown('#### 🎯 VAD 좌표')
                    v, a, d = result['vad']

                    vad_metrics = st.columns(3)
                    vad_metrics[0].metric('Valence', f'{v:+.2f}')
                    vad_metrics[1].metric('Arousal', f'{a:+.2f}')
                    vad_metrics[2].metric('Dominance', f'{d:+.2f}')

                # 3D 시각화
                st.markdown('#### 🔬 3D VAD 공간 시각화')

                # 기본 감정들과 함께 표시
                base_emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
                base_vad_points = [VADModel.emotion_to_vad(e) for e in base_emotions]
                base_vad_points.append(result['vad'])

                labels = base_emotions + [f"{result['dominant']} (분석 결과)"]

                fig = VADModel.visualize_3d(
                    base_vad_points,
                    labels,
                    f"{result['dominant'].capitalize()} 감정의 VAD 공간 위치"
                )
                st.pyplot(fig)

                # 유사 감정 찾기
                st.markdown('#### 🔍 유사한 감정')

                similarities = []
                for emotion in VADModel.EMOTION_VAD_MAP.keys():
                    if emotion != result['dominant']:
                        emotion_vad = VADModel.emotion_to_vad(emotion)
                        similarity = VADModel.calculate_similarity(result['vad'], emotion_vad)
                        similarities.append((emotion, similarity))

                # 상위 3개 유사 감정
                similarities.sort(key=lambda x: x[1], reverse=True)

                sim_cols = st.columns(3)
                for i, (emotion, sim) in enumerate(similarities[:3]):
                    with sim_cols[i]:
                        st.metric(
                            label=emotion.capitalize(),
                            value=f'{sim:.1%} 유사',
                            help=VADModel.get_emotion_description(emotion)
                        )

        else:
            st.info('👆 이미지를 업로드하여 VAD 분석을 시작하세요')

    def render_multimodal(self):
        """Tab 4: 멀티모달 분석"""
        st.header('🎨 멀티모달 감정 분석')

        st.markdown("""
        이미지와 텍스트를 함께 분석하여 더 정확한 감정 인식을 수행합니다.

        **사용 사례**:
        - 이미지만으로는 애매한 감정을 텍스트 컨텍스트로 명확히
        - SNS 게시물 (이미지 + 캡션) 감정 분석
        - 상황 설명과 함께 표정 해석
        """)

        # EmotionHelper 가져오기
        helper = get_emotion_helper()
        st.info(f'🤖 {helper.get_status_message()}')

        # 입력 영역
        st.subheader('📥 입력')

        uploaded_file = st.file_uploader(
            '이미지를 업로드하세요',
            type=['png', 'jpg', 'jpeg', 'webp'],
            key='multimodal_upload'
        )

        text_context = st.text_area(
            '추가 텍스트 컨텍스트 (선택 사항)',
            placeholder='예: "오늘 시험에 합격했어요!" 또는 "면접에서 떨어졌습니다..."',
            help='이미지와 함께 고려할 상황이나 텍스트를 입력하세요',
            height=100
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader('입력 이미지')
                st.image(image, use_container_width=True)

                if text_context:
                    st.info(f'📝 텍스트: "{text_context}"')

            with col2:
                st.subheader('분석 옵션')

                analysis_mode = st.radio(
                    '분석 모드 선택',
                    ['이미지만 분석', '이미지 + 텍스트 통합 분석', '비교 분석 (양쪽 모두)'],
                    help='원하는 분석 방식을 선택하세요'
                )

                if st.button('🚀 분석 시작', type='primary', use_container_width=True):
                    with st.spinner('멀티모달 분석 중...'):
                        if analysis_mode == '이미지만 분석' or not text_context:
                            # 이미지만 분석
                            result = helper.analyze_basic_emotion(image)
                            st.session_state['multimodal_result'] = {
                                'mode': 'image_only',
                                'image_only': result
                            }

                        elif analysis_mode == '이미지 + 텍스트 통합 분석':
                            # 통합 분석
                            result = helper.analyze_multimodal(image, text_context)
                            st.session_state['multimodal_result'] = {
                                'mode': 'combined',
                                'combined': result['combined'],
                                'text': text_context
                            }

                        else:  # 비교 분석
                            # 양쪽 모두
                            result = helper.analyze_multimodal(image, text_context)
                            st.session_state['multimodal_result'] = {
                                'mode': 'compare',
                                **result
                            }

            # 결과 표시
            if 'multimodal_result' in st.session_state:
                result = st.session_state['multimodal_result']

                st.markdown('---')
                st.subheader('📊 분석 결과')

                if result['mode'] == 'image_only':
                    # 이미지만
                    st.markdown('#### 이미지 감정 분석')
                    emotions = result['image_only']

                    top3 = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    cols = st.columns(3)
                    for i, (emotion, score) in enumerate(top3):
                        cols[i].metric(f"{i+1}. {emotion.capitalize()}", f"{score:.2%}")

                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure([go.Bar(x=list(emotions.keys()), y=list(emotions.values()))])
                        fig.update_layout(title='감정 분석 결과', yaxis_title='신뢰도')
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.bar_chart(emotions)

                elif result['mode'] == 'combined':
                    # 통합 분석
                    st.markdown('#### 이미지 + 텍스트 통합 분석')
                    st.caption(f'텍스트: "{result["text"]}"')

                    emotions = result['combined']

                    top3 = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    cols = st.columns(3)
                    for i, (emotion, score) in enumerate(top3):
                        cols[i].metric(f"{i+1}. {emotion.capitalize()}", f"{score:.2%}")

                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure([go.Bar(x=list(emotions.keys()), y=list(emotions.values()))])
                        fig.update_layout(title='통합 감정 분석', yaxis_title='신뢰도')
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.bar_chart(emotions)

                else:  # 비교 분석
                    st.markdown('#### 📊 비교 분석: 이미지 vs 통합')

                    comp_col1, comp_col2 = st.columns(2)

                    with comp_col1:
                        st.markdown('**🖼️ 이미지만**')
                        image_emotions = result['image_only']
                        top_image = max(image_emotions.items(), key=lambda x: x[1])
                        st.success(f"**{top_image[0].upper()}** ({top_image[1]:.2%})")

                        for emotion, score in sorted(image_emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
                            st.write(f"- {emotion}: {score:.2%}")

                    with comp_col2:
                        st.markdown('**🎨 이미지 + 텍스트**')
                        st.caption(f'"{result["text"]}"')
                        combined_emotions = result['combined']
                        top_combined = max(combined_emotions.items(), key=lambda x: x[1])
                        st.success(f"**{top_combined[0].upper()}** ({top_combined[1]:.2%})")

                        for emotion, score in sorted(combined_emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
                            st.write(f"- {emotion}: {score:.2%}")

                    # 차이 분석
                    st.markdown('#### 🔍 차이 분석')

                    difference = result.get('difference', {})
                    if difference:
                        st.write('텍스트 컨텍스트 추가로 인한 변화:')

                        # 가장 큰 변화
                        sorted_diff = sorted(difference.items(), key=lambda x: abs(x[1]), reverse=True)

                        diff_cols = st.columns(3)
                        for i, (emotion, diff) in enumerate(sorted_diff[:3]):
                            if abs(diff) > 0.01:
                                with diff_cols[i]:
                                    delta_color = "normal" if diff > 0 else "inverse"
                                    st.metric(
                                        emotion.capitalize(),
                                        f"{diff:+.2%}",
                                        delta=f"{'증가' if diff > 0 else '감소'}"
                                    )

        else:
            st.info('👆 이미지를 업로드하여 멀티모달 분석을 시작하세요')

        # 예시
        with st.expander('💡 멀티모달 분석 예시'):
            st.markdown("""
            **예시 1: 억지 미소**
            - 이미지: 웃는 표정 → "happy"
            - 텍스트: "오늘 직장에서 해고당했어요..."
            - 통합: "sad" (텍스트 컨텍스트가 실제 감정을 드러냄)

            **예시 2: 긍정적 문맥**
            - 이미지: 평범한 표정 → "neutral"
            - 텍스트: "드디어 합격 통지를 받았습니다!"
            - 통합: "happy" (긍정적 상황 반영)
            """)

    def render_timeseries(self):
        """Tab 5: 시계열 분석"""
        st.header('📈 시계열 감정 분석')

        st.markdown("""
        여러 이미지 또는 비디오를 순차적으로 분석하여 감정 변화를 추적합니다.

        **사용 사례**:
        - 비디오 프레임별 감정 변화 추적
        - 시간에 따른 감정 트렌드 분석
        - 급격한 감정 변화 시점 탐지
        """)

        # EmotionHelper 가져오기
        helper = get_emotion_helper()
        st.info(f'🤖 {helper.get_status_message()}')

        # 입력 타입 선택
        input_type = st.radio(
            '입력 타입을 선택하세요',
            ['📁 이미지 파일 (여러 개)', '🎬 비디오 파일'],
            horizontal=True
        )

        uploaded_files = None
        video_frames = None

        if input_type == '📁 이미지 파일 (여러 개)':
            # 다중 이미지 업로드
            uploaded_files = st.file_uploader(
                '이미지 여러 개를 업로드하세요 (시간 순서대로)',
                type=['png', 'jpg', 'jpeg', 'webp'],
                accept_multiple_files=True,
                key='timeseries_upload',
                help='분석할 이미지들을 시간 순서대로 선택하세요'
            )

        else:  # 비디오 파일
            # OpenCV 체크
            try:
                import cv2
                HAS_OPENCV = True
            except ImportError:
                HAS_OPENCV = False
                st.error('⚠️ 비디오 처리를 위해 OpenCV가 필요합니다. `pip install opencv-python`을 실행하세요.')

            if HAS_OPENCV:
                uploaded_video = st.file_uploader(
                    '비디오 파일을 업로드하세요',
                    type=['mp4', 'avi', 'mov', 'mkv'],
                    key='video_upload',
                    help='감정 변화를 분석할 비디오 파일을 선택하세요'
                )

                if uploaded_video is not None:
                    # 비디오 옵션
                    st.subheader('🎬 비디오 처리 옵션')

                    col1, col2 = st.columns(2)
                    with col1:
                        sample_rate = st.slider(
                            '샘플링 비율 (N 프레임마다 1개)',
                            min_value=1,
                            max_value=60,
                            value=30,
                            help='30이면 30프레임마다 1개씩 추출 (FPS 30일 때 1초마다 1장)'
                        )

                    with col2:
                        max_frames = st.number_input(
                            '최대 프레임 수',
                            min_value=10,
                            max_value=500,
                            value=100,
                            help='추출할 최대 프레임 개수 (API 비용 절감)'
                        )

                    if st.button('🎬 비디오에서 프레임 추출', type='primary'):
                        with st.spinner('비디오에서 프레임을 추출하고 있습니다...'):
                            try:
                                import tempfile
                                import os

                                # 임시 파일로 저장
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                                    tmp_file.write(uploaded_video.read())
                                    tmp_path = tmp_file.name

                                # OpenCV로 프레임 추출
                                cap = cv2.VideoCapture(tmp_path)

                                if not cap.isOpened():
                                    st.error('비디오를 열 수 없습니다')
                                else:
                                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    fps = cap.get(cv2.CAP_PROP_FPS)

                                    st.info(f'📹 비디오 정보: 총 {total_frames} 프레임, {fps:.2f} FPS')

                                    frames = []
                                    frame_idx = 0
                                    saved_count = 0

                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    while True:
                                        ret, frame = cap.read()
                                        if not ret:
                                            break

                                        # 샘플링
                                        if frame_idx % sample_rate == 0:
                                            # BGR → RGB 변환
                                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                            pil_image = Image.fromarray(rgb_frame)
                                            frames.append(pil_image)
                                            saved_count += 1

                                            status_text.text(f'프레임 추출 중... {saved_count}개')
                                            progress_bar.progress(min(1.0, saved_count / max_frames))

                                            if saved_count >= max_frames:
                                                break

                                        frame_idx += 1

                                    cap.release()
                                    os.unlink(tmp_path)  # 임시 파일 삭제

                                    status_text.empty()
                                    progress_bar.empty()

                                    video_frames = frames
                                    st.session_state['video_frames'] = frames
                                    st.success(f'✅ 총 {len(frames)}개 프레임 추출 완료!')

                            except Exception as e:
                                st.error(f'비디오 처리 중 오류 발생: {e}')
                                import traceback
                                st.text(traceback.format_exc())

                # 세션 상태에서 프레임 가져오기
                if 'video_frames' in st.session_state:
                    video_frames = st.session_state['video_frames']

        # 이미지/비디오 데이터 확인
        images_to_analyze = None

        if uploaded_files:
            images_to_analyze = uploaded_files
            st.success(f'✅ {len(uploaded_files)}개 이미지가 업로드되었습니다')

            # 업로드된 이미지 미리보기
            with st.expander(f'📁 업로드된 이미지 미리보기 ({len(uploaded_files)}개)'):
                preview_cols = st.columns(min(5, len(uploaded_files)))
                for i, file in enumerate(uploaded_files[:5]):
                    with preview_cols[i]:
                        image = Image.open(file)
                        st.image(image, caption=f'이미지 {i+1}', use_container_width=True)
                if len(uploaded_files) > 5:
                    st.caption(f'... 외 {len(uploaded_files) - 5}개 이미지')

        elif video_frames:
            images_to_analyze = video_frames
            st.success(f'✅ 비디오에서 {len(video_frames)}개 프레임 추출됨')

            # 추출된 프레임 미리보기
            with st.expander(f'🎬 추출된 프레임 미리보기 ({len(video_frames)}개)'):
                preview_cols = st.columns(min(5, len(video_frames)))
                for i, frame in enumerate(video_frames[:5]):
                    with preview_cols[i]:
                        st.image(frame, caption=f'프레임 {i+1}', use_container_width=True)
                if len(video_frames) > 5:
                    st.caption(f'... 외 {len(video_frames) - 5}개 프레임')

        # 분석 버튼
        if images_to_analyze:
            if st.button('🔍 시계열 분석 시작', type='primary', use_container_width=True):
                # EmotionTimeSeries 객체 생성
                timeseries = EmotionTimeSeries(window_size=len(images_to_analyze))

                # 프로그레스 바
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 각 이미지/프레임 분석
                for i, item in enumerate(images_to_analyze):
                    status_text.text(f'프레임 {i+1}/{len(images_to_analyze)} 분석 중...')

                    # 이미지 로드
                    if isinstance(item, Image.Image):
                        # 비디오 프레임 (이미 PIL Image)
                        image = item
                    else:
                        # 업로드된 파일
                        image = Image.open(item)

                    # 감정 분석
                    emotions = helper.analyze_basic_emotion(image)

                    # 타임스탬프와 함께 추가
                    timeseries.add_frame(emotions, timestamp=i)

                    # 프로그레스 업데이트
                    progress_bar.progress((i + 1) / len(images_to_analyze))

                status_text.empty()
                progress_bar.empty()

                # 결과 저장
                st.session_state['timeseries_result'] = timeseries

                st.success(f'✅ {len(images_to_analyze)}개 프레임 분석 완료!')

            # 결과 표시
            if 'timeseries_result' in st.session_state:
                timeseries = st.session_state['timeseries_result']

                st.markdown('---')

                # 요약 정보
                st.subheader('📊 분석 요약')

                summary = timeseries.get_summary()

                summary_cols = st.columns(4)
                summary_cols[0].metric('총 프레임 수', summary['total_frames'])
                summary_cols[1].metric('지배적 감정', summary['dominant_emotion'].capitalize())
                summary_cols[2].metric('평균 신뢰도', f"{summary['avg_confidence']:.2%}")
                summary_cols[3].metric('감정 변화점', len(summary['change_points']))

                # 시계열 그래프
                st.subheader('📈 감정 변화 타임라인')

                try:
                    fig = timeseries.visualize_timeline()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f'시각화 중 오류 발생: {e}')

                # 트렌드 분석
                st.subheader('📉 감정 트렌드')

                trend_cols = st.columns(4)
                emotions_to_check = ['happy', 'sad', 'angry', 'fear']

                for i, emotion in enumerate(emotions_to_check):
                    trend = timeseries.get_trend(emotion)
                    trend_emoji = {
                        'increasing': '📈 상승',
                        'decreasing': '📉 하락',
                        'stable': '➡️ 안정'
                    }
                    with trend_cols[i]:
                        st.write(f"**{emotion.capitalize()}**")
                        st.write(trend_emoji.get(trend, trend))

                # 변화점 분석
                st.subheader('🔍 감정 변화점 탐지')

                change_points = timeseries.detect_change_points(threshold=0.3)

                if change_points:
                    st.write(f'감정이 크게 변화한 시점: **{len(change_points)}개 발견**')

                    # 변화점 상세 정보
                    with st.expander('변화점 상세 정보 보기'):
                        for idx in change_points:
                            if idx < len(timeseries.history):
                                frame = timeseries.history[idx]
                                prev_frame = timeseries.history[idx - 1] if idx > 0 else None

                                st.markdown(f'**프레임 {idx + 1}번**')

                                if prev_frame:
                                    # 이전 프레임과 비교
                                    prev_dominant = max(prev_frame['emotions'].items(), key=lambda x: x[1])[0]
                                    curr_dominant = max(frame['emotions'].items(), key=lambda x: x[1])[0]

                                    change_cols = st.columns(2)
                                    with change_cols[0]:
                                        st.write(f'이전: {prev_dominant.capitalize()}')
                                    with change_cols[1]:
                                        st.write(f'→ {curr_dominant.capitalize()}')

                                st.markdown('---')
                else:
                    st.info('감정 변화가 안정적입니다 (큰 변화점 없음)')

                # CSV 내보내기
                st.subheader('💾 데이터 내보내기')

                if st.button('📥 CSV 파일로 내보내기'):
                    import tempfile

                    try:
                        # 임시 파일 생성
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as f:
                            timeseries.export_to_csv(f.name)

                            # 파일 읽기
                            with open(f.name, 'r', encoding='utf-8') as csv_file:
                                csv_content = csv_file.read()

                            # 다운로드 버튼
                            st.download_button(
                                label='📥 CSV 다운로드',
                                data=csv_content,
                                file_name='emotion_timeseries.csv',
                                mime='text/csv',
                                use_container_width=True
                            )

                            st.success('✅ CSV 파일이 준비되었습니다!')

                    except Exception as e:
                        st.error(f'CSV 내보내기 실패: {e}')

        else:
            st.info('👆 여러 이미지 또는 비디오를 업로드하여 시계열 분석을 시작하세요')

            # 사용 팁
            with st.expander('💡 시계열 분석 사용 팁'):
                st.markdown("""
                **최적의 결과를 위한 팁**:

                **이미지 파일 모드**:
                1. **이미지 순서**: 시간 순서대로 이미지를 선택하세요
                2. **프레임 수**: 최소 3개 이상의 이미지를 업로드하세요
                3. **일관성**: 비슷한 조명과 각도의 이미지가 좋습니다
                4. **얼굴 가시성**: 얼굴이 명확히 보이는 이미지를 선택하세요

                **비디오 파일 모드**:
                1. **샘플링 비율**: FPS가 30이면 sample_rate=30으로 설정 시 1초마다 1프레임 추출
                2. **최대 프레임**: API 비용을 고려하여 적절한 프레임 수 설정 (권장: 50-100개)
                3. **비디오 형식**: MP4, AVI, MOV, MKV 지원 (H.264 코덱 권장)
                4. **비디오 길이**: 긴 비디오는 샘플링 비율을 높여서 프레임 수 조절

                **사용 예시**:
                - 인터뷰 비디오에서 감정 변화 추적
                - 프레젠테이션 중 청중 반응 분석
                - 강의 영상에서 학생들의 집중도 측정
                - 상담 세션 중 내담자 감정 변화 모니터링
                - 유튜브 영상에서 인물의 감정 타임라인 생성
                """)

