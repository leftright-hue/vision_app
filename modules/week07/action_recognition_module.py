"""
Week 7: 행동인식 (Action Recognition)
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import io
import os

from core.base_processor import BaseImageProcessor
from .action_helpers import get_video_helper


class ActionRecognitionModule(BaseImageProcessor):
    """Week 7: 행동인식 모듈"""

    def __init__(self):
        super().__init__()
        self.name = "Week 7: Action Recognition"

    def render(self):
        """메인 렌더링 함수"""
        st.title("🎬 Week 7: 행동인식 (Action Recognition)")

        st.markdown("""
        ## 학습 목표
        - **이론**: 행동인식 개념, 3D CNN, Two-Stream, Transformer 아키텍처 이해
        - **실습**: 비디오 처리, Optical Flow, HuggingFace 모델 활용
        - **응용**: 실시간 행동 인식, 운동 카운터 제작
        - **실전**: MediaPipe와 Google Video Intelligence API 활용
        """)

        # 환경 체크
        self._check_environment()

        # 7개 탭 구성
        tabs = st.tabs([
            "📚 개념 소개",
            "🎬 비디오 처리 기초",
            "🤖 사전훈련 모델",
            "📹 실시간 인식",
            "💼 실전 응용",
            "🔧 MediaPipe (Open Source)",
            "☁️ Google Video Intelligence (Cloud)"
        ])

        with tabs[0]:
            self.render_theory()

        with tabs[1]:
            self.render_video_basics()

        with tabs[2]:
            self.render_pretrained_models()

        with tabs[3]:
            self.render_realtime()

        with tabs[4]:
            self.render_applications()

        with tabs[5]:
            self.render_mediapipe_realtime()

        with tabs[6]:
            self.render_google_video_intelligence()

    def _check_environment(self):
        """환경 체크 및 설정"""
        with st.expander("🔧 환경 설정 확인", expanded=False):
            st.markdown("""
            ### 필요한 패키지
            - `transformers`: HuggingFace 모델 (VideoMAE, TimeSformer)
            - `torch`: PyTorch 백엔드
            - `opencv-python`: 비디오 처리, Optical Flow
            - `mediapipe`: 운동 카운터 (선택적)

            ### 3-Tier Fallback 전략
            1. **Transformers** (권장): `pip install transformers torch`
            2. **OpenCV only**: `pip install opencv-python`
            3. **Simulation Mode**: 패키지 없이 기본 기능 사용
            """)

            issues = []

            # Check transformers
            try:
                import transformers
                st.success(f"✅ transformers {transformers.__version__}")
            except ImportError:
                issues.append("transformers")
                st.warning("⚠️ transformers 미설치")

            # Check torch
            try:
                import torch
                device = "GPU" if torch.cuda.is_available() else "CPU"
                st.success(f"✅ torch {torch.__version__} ({device})")
            except ImportError:
                issues.append("torch")
                st.warning("⚠️ torch 미설치")

            # Check opencv
            try:
                import cv2
                st.success(f"✅ opencv-python {cv2.__version__}")
            except ImportError:
                issues.append("opencv-python")
                st.warning("⚠️ opencv-python 미설치")

            # Check mediapipe
            try:
                import mediapipe
                st.success(f"✅ mediapipe {mediapipe.__version__}")
            except ImportError:
                st.info("ℹ️ mediapipe 미설치 (운동 카운터 기능 제한)")

            if issues:
                st.info(f"""
                ### 🔧 설치 방법
                ```bash
                pip install transformers torch opencv-python mediapipe
                ```

                시뮬레이션 모드에서도 기본 기능은 사용 가능합니다.
                """)

    # ==================== Tab 1: 개념 소개 ====================

    def render_theory(self):
        """행동인식 이론 설명"""
        st.header("📚 행동인식 개념 소개")

        st.markdown("""
        ## 1. 행동인식이란?

        **행동인식(Action Recognition)**은 비디오에서 사람이나 객체의 행동을 자동으로 분류하는 기술입니다.

        **비유**: 행동인식은 "영화 감독이 장면을 보고 무슨 일이 일어나는지 이해하는 것"과 같아요!

        일반 이미지 분류는 사진 한 장만 보지만, 행동인식은 **연속된 프레임들을 보고 시간적 변화를 이해**합니다.

        ---

        ### 1.1 이미지 vs 비디오 데이터

        | 특성 | 이미지 | 비디오 |
        |------|--------|--------|
        | **차원** | 2D (H × W × C) | 3D (T × H × W × C) |
        | **정보** | 공간 (Spatial) | 공간 + 시간 (Spatiotemporal) |
        | **예시** | "고양이" | "고양이가 뛰어오른다" |
        | **크기** | 작음 (~MB) | 큼 (~GB) |

        💡 *T = 시간(프레임 수), H = 높이, W = 너비, C = 채널(RGB)*

        ---

        ## 2. 주요 아키텍처

        ### 2.1 3D CNN (C3D)

        **아이디어**: Conv2D를 Conv3D로 확장하여 시간 차원도 함께 처리

        **비유**: 2D는 사진 한 장을 보는 것, 3D는 연속된 사진들(영화)을 한번에 보는 것

        **특징**:
        - 입력: (T × H × W) 비디오 클립
        - Conv3D 필터: (t × h × w) 크기
        - 시공간 특징 동시 추출

        **장점**: 간단하고 효과적
        **단점**: 메모리 사용량 많음

        ---

        ### 2.2 Two-Stream Networks

        **아이디어**: 공간 정보(RGB)와 시간 정보(Optical Flow)를 별도 처리 후 결합

        **비유**: 한 눈으로는 "무엇이 있는지"(외형), 다른 눈으로는 "어떻게 움직이는지"(동작)를 본다

        **구조**:
        1. **Spatial Stream (공간)**: RGB 프레임 → 외형 인식
        2. **Temporal Stream (시간)**: Optical Flow → 움직임 인식
        3. **Fusion**: 두 스트림 결과 결합

        **장점**: 외형과 동작 명확히 분리
        **단점**: Optical Flow 계산 비용 높음

        ---

        ### 2.3 Video Transformer

        **아이디어**: Transformer를 비디오에 적용하여 장거리 의존성 포착

        **대표 모델**:

        #### **TimeSformer** (Facebook AI, 2021)
        - **Divided Space-Time Attention**
        - 공간 Attention + 시간 Attention 분리
        - 효율적이고 확장 가능

        #### **VideoMAE** (2022)
        - **Masked Autoencoding**
        - 프레임 일부를 가리고 복원하며 학습
        - 적은 데이터로도 고성능

        #### **X-CLIP** (Microsoft, 2022)
        - CLIP을 비디오로 확장
        - 텍스트-비디오 정렬
        - Zero-shot 행동 인식 가능

        **장점**:
        - 장거리 의존성 포착 (먼 프레임 간 관계)
        - Pre-training으로 고성능

        **단점**:
        - 계산 비용 높음
        - 많은 메모리 필요

        ---

        ## 3. Optical Flow (광학 흐름)

        **정의**: 연속된 두 프레임 사이의 픽셀 이동 벡터

        **비유**: 물결이 어느 방향으로 흐르는지 화살표로 표시하는 것과 같아요

        ### 3.1 Farneback 알고리즘

        **특징**:
        - Dense Optical Flow (모든 픽셀의 움직임 계산)
        - 다항식 근사 사용
        - OpenCV로 쉽게 사용 가능

        **시각화**:
        - **Hue (색상)**: 움직임 방향
        - **Value (밝기)**: 움직임 크기
        - **결과**: 컬러풀한 움직임 맵

        💡 *빨강=오른쪽, 파랑=왼쪽, 초록=아래, 노랑=위 움직임*

        ---

        ## 4. 행동인식의 응용

        ### 4.1 스포츠 분석
        - **예시**: 축구 하이라이트 자동 생성, 골 장면 감지
        - **사용자**: 방송사, 스포츠 팀

        ### 4.2 보안 및 감시
        - **예시**: 이상 행동 감지 (싸움, 쓰러짐)
        - **사용자**: 보안 업체, 공항, 병원

        ### 4.3 HCI (Human-Computer Interaction)
        - **예시**: 제스처 인식, 수화 번역
        - **사용자**: 게임, VR/AR, 접근성 도구

        ### 4.4 자율주행
        - **예시**: 보행자 행동 예측 (건너려는지, 멈출지)
        - **사용자**: 자동차 회사

        ### 4.5 헬스케어
        - **예시**: 운동 자세 교정, 재활 모니터링
        - **사용자**: 피트니스 앱, 병원

        ### 4.6 콘텐츠 제작
        - **예시**: 영상 자동 편집, 하이라이트 추출
        - **사용자**: 유튜버, 방송 PD

        ---

        ## 참고 자료
        - [C3D 논문](https://arxiv.org/abs/1412.0767)
        - [Two-Stream 논문](https://arxiv.org/abs/1406.2199)
        - [TimeSformer 논문](https://arxiv.org/abs/2102.05095)
        - [VideoMAE 논문](https://arxiv.org/abs/2203.12602)
        """)

    # ==================== Tab 2: 비디오 처리 기초 ====================

    def render_video_basics(self):
        """비디오 처리 기초"""
        st.header("🎬 비디오 처리 기초")

        st.markdown("""
        이 탭에서는 비디오 처리의 기초를 실습합니다:
        1. **프레임 추출**: 비디오 → 이미지 시퀀스
        2. **Optical Flow**: 프레임 간 움직임 계산
        3. **시각화**: 움직임을 색상으로 표현
        """)

        # VideoHelper 로드
        helper = get_video_helper()
        st.markdown(f"**현재 모드**: `{helper.get_mode()}`")

        # 비디오 업로드
        uploaded = st.file_uploader(
            "비디오 파일 선택 (.mp4, .avi)",
            type=['mp4', 'avi', 'mov'],
            key="video_basics_upload"
        )

        if uploaded:
            # 임시 파일로 저장
            video_bytes = uploaded.read()
            temp_path = helper.save_temp_video(video_bytes)

            if temp_path:
                # 비디오 정보
                info = helper.get_video_info(temp_path)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("FPS", f"{info['fps']:.1f}")
                col2.metric("Duration", f"{info['duration']:.1f}s")
                col3.metric("Resolution", f"{info['resolution'][0]}×{info['resolution'][1]}")
                col4.metric("Frames", info['frame_count'])

                # 기능 선택
                demo_type = st.radio(
                    "데모 선택",
                    ["프레임 추출", "Optical Flow"],
                    horizontal=True
                )

                if demo_type == "프레임 추출":
                    self._demo_frame_extraction(temp_path, helper)
                else:
                    self._demo_optical_flow(temp_path, helper)

    def _demo_frame_extraction(self, video_path: str, helper):
        """프레임 추출 데모"""
        st.subheader("📸 프레임 추출")

        st.markdown("""
        **파라미터**:
        - **sample_rate**: 매 N 프레임당 1개 추출 (메모리 절약)
        - **max_frames**: 최대 프레임 수 (기본 100개)
        """)

        col1, col2 = st.columns(2)
        with col1:
            sample_rate = st.slider("Sample Rate", 1, 60, 30,
                                   help="높을수록 적게 추출 (빠름)")
        with col2:
            max_frames = st.slider("Max Frames", 10, 200, 50,
                                  help="메모리 제한")

        if st.button("🎬 프레임 추출", type="primary"):
            with st.spinner("프레임 추출 중..."):
                frames = helper.extract_frames(
                    video_path,
                    sample_rate=sample_rate,
                    max_frames=max_frames
                )

                if frames:
                    st.success(f"✅ {len(frames)}개 프레임 추출 완료")

                    # 일부 프레임 표시
                    num_display = min(6, len(frames))
                    cols = st.columns(3)

                    for i in range(num_display):
                        with cols[i % 3]:
                            st.image(frames[i], caption=f"Frame {i}", use_container_width=True)

    def _demo_optical_flow(self, video_path: str, helper):
        """Optical Flow 데모"""
        st.subheader("🌊 Optical Flow")

        st.markdown("""
        **설명**: Farneback 알고리즘으로 연속 프레임 간 움직임을 계산합니다.

        **색상 의미**:
        - 🔴 빨강: 오른쪽 이동
        - 🔵 파랑: 왼쪽 이동
        - 🟢 초록: 아래 이동
        - 🟡 노랑: 위 이동
        - 밝기: 움직임 속도
        """)

        if st.button("🌊 Optical Flow 계산", type="primary"):
            with st.spinner("처리 중..."):
                # 프레임 추출
                frames = helper.extract_frames(video_path, sample_rate=5, max_frames=20)

                if len(frames) >= 2:
                    # 첫 2개 프레임으로 Flow 계산
                    flow = helper.compute_optical_flow(frames[0], frames[1])
                    flow_vis = helper.visualize_flow(flow)

                    # 시각화
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(frames[0])
                    axes[0].set_title("Frame t")
                    axes[0].axis('off')

                    axes[1].imshow(frames[1])
                    axes[1].set_title("Frame t+1")
                    axes[1].axis('off')

                    axes[2].imshow(flow_vis)
                    axes[2].set_title("Optical Flow")
                    axes[2].axis('off')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # 통계
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    st.info(f"""
                    **움직임 통계**:
                    - 평균 이동 거리: {np.mean(magnitude):.2f} 픽셀
                    - 최대 이동 거리: {np.max(magnitude):.2f} 픽셀
                    """)

    # ==================== Tab 3: 사전훈련 모델 ====================

    def render_pretrained_models(self):
        """사전훈련 모델 활용"""
        st.header("🤖 사전훈련 모델 활용")

        st.markdown("""
        HuggingFace의 사전훈련 모델로 행동을 분류합니다.

        ### 지원 모델
        - **VideoMAE**: Masked Autoencoding, Kinetics-400 데이터셋
        - **TimeSformer**: Divided Space-Time Attention
        - **X-CLIP**: CLIP 기반 비디오 모델

        ⚠️ **주의**: 모델 다운로드에 시간이 걸릴 수 있습니다 (~1GB).
        """)

        helper = get_video_helper()

        # 모델 사용 불가능 경고
        if not helper.is_available('action_classification'):
            st.warning("""
            ⚠️ **Transformers 모드 필요**

            행동 분류를 사용하려면 transformers와 torch를 설치하세요:
            ```bash
            pip install transformers torch
            ```

            현재는 시뮬레이션 모드로 동작합니다.
            """)

        # 모델 선택
        model_name = st.selectbox(
            "모델 선택",
            ["videomae", "timesformer", "xclip"],
            help="VideoMAE 권장 (빠르고 정확)"
        )

        # 비디오 업로드
        uploaded = st.file_uploader(
            "비디오 파일",
            type=['mp4', 'avi', 'mov'],
            key="model_upload"
        )

        if uploaded:
            video_bytes = uploaded.read()
            temp_path = helper.save_temp_video(video_bytes)

            if temp_path:
                # 비디오 미리보기
                st.video(uploaded)

                if st.button("🎬 행동 분류", type="primary"):
                    with st.spinner("모델 로딩 및 추론 중... (최대 1분)"):
                        results = helper.classify_action(temp_path, model_name, top_k=5)

                        # 결과 표시
                        st.subheader("🎯 예측 결과")

                        if results and results[0][0] != 'error':
                            for i, (label, score) in enumerate(results):
                                st.metric(
                                    f"#{i+1} {label}",
                                    f"{score*100:.1f}%"
                                )

                                # 프로그레스 바
                                st.progress(score)

                            # 설명
                            top_label = results[0][0]
                            st.success(f"✅ 가장 가능성 높은 행동: **{top_label}**")
                        else:
                            st.error("❌ 분류 실패. 로그를 확인하세요.")

    # ==================== Tab 4: 실시간 인식 ====================

    def render_realtime(self):
        """실시간 행동 인식 안내"""
        st.header("📹 실시간 행동 인식")

        st.markdown("""
        ## 실시간 웹캠 행동 인식

        실시간 웹캠 처리는 **lab 파일**에서 제공됩니다.

        ### 실행 방법

        1. 터미널에서 lab 파일 실행:
        ```bash
        cd modules/week07/labs
        python lab04_realtime_recognition.py
        ```

        2. 웹캠이 자동으로 열립니다

        3. 키보드 조작:
        - **SPACE**: 프레임 저장
        - **ESC**: 종료

        ---

        ### 실시간 처리 최적화 팁

        #### 1. 프레임 샘플링
        - **문제**: 60fps로 모든 프레임 처리 시 느림
        - **해결**: 매 5프레임당 1개만 처리 (12fps)

        #### 2. 해상도 다운샘플링
        - **문제**: 1080p 처리 시 느림
        - **해결**: 480p 또는 360p로 리사이즈

        #### 3. 프레임 버퍼링
        - **문제**: 단일 프레임으로는 행동 판단 어려움
        - **해결**: 최근 16프레임을 버퍼에 저장

        #### 4. 비동기 처리
        - **문제**: 모델 추론 중 프레임 드롭
        - **해결**: 별도 스레드에서 추론

        ---

        ### 성능 벤치마크

        | 설정 | FPS | 정확도 | 메모리 |
        |------|-----|--------|--------|
        | CPU, 1080p, 모든 프레임 | ~5 | 높음 | ~2GB |
        | CPU, 480p, 5프레임 샘플 | ~20 | 중간 | ~1GB |
        | GPU, 1080p, 모든 프레임 | ~30 | 높음 | ~3GB |
        | GPU, 480p, 5프레임 샘플 | ~60 | 중간 | ~1.5GB |

        💡 **권장**: GPU + 480p + 5프레임 샘플링

        ---

        ## Lab 파일 목록

        ### lab04_realtime_recognition.py
        OpenCV 기반 실시간 웹캠 행동 인식

        **기능**:
        - 웹캠 입력 처리
        - 프레임 버퍼링
        - 행동 분류 (간단한 움직임 기반)
        - FPS 표시

        ### 실행 예시
        ```python
        # lab04_realtime_recognition.py
        import cv2
        from action_helpers import get_video_helper

        helper = get_video_helper()
        cap = cv2.VideoCapture(0)

        frame_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 버퍼에 추가
            frame_buffer.append(frame)
            if len(frame_buffer) > 16:
                frame_buffer.pop(0)

            # 16프레임 모이면 분류
            if len(frame_buffer) == 16:
                # 간단한 움직임 기반 분류
                pass

            cv2.imshow('Realtime Recognition', frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()
        ```
        """)

        st.info("""
        💡 **Streamlit 제한 사항**

        Streamlit은 웹 기반이라 직접 웹캠 접근이 제한적입니다.
        실시간 처리는 **별도 Python 스크립트**로 실행하세요.
        """)

    # ==================== Tab 5: 실전 응용 ====================

    def render_applications(self):
        """실전 응용 예제"""
        st.header("💼 실전 응용")

        st.markdown("""
        ## 행동인식 활용 사례

        실전에서 행동인식을 어떻게 활용하는지 실습합니다.
        """)

        app_type = st.selectbox(
            "응용 예제 선택",
            [
                "운동 카운터 (푸시업/스쿼트)",
                "제스처 인식",
                "이상 행동 감지"
            ]
        )

        if app_type == "운동 카운터 (푸시업/스쿼트)":
            self._app_exercise_counter()
        elif app_type == "제스처 인식":
            self._app_gesture()
        else:
            self._app_anomaly()

    def _app_exercise_counter(self):
        """운동 카운터 앱"""
        st.subheader("🏋️ 운동 카운터")

        st.markdown("""
        **기능**: MediaPipe Pose로 관절 각도를 추적하여 운동 반복 횟수 자동 카운트

        **지원 운동**:
        - 푸시업 (Pushup): 팔꿈치 각도
        - 스쿼트 (Squat): 무릎 각도
        - 점프잭 (Jumping Jack): 팔/다리 각도
        """)

        helper = get_video_helper()

        # MediaPipe 미설치 경고
        try:
            import mediapipe
        except ImportError:
            st.warning("""
            ⚠️ **MediaPipe 미설치**

            운동 카운터를 사용하려면:
            ```bash
            pip install mediapipe
            ```

            현재는 시뮬레이션 모드로 동작합니다.
            """)

        # 운동 선택
        exercise_type = st.selectbox(
            "운동 종류",
            ["pushup", "squat", "jumping_jack"]
        )

        # 비디오 업로드
        uploaded = st.file_uploader(
            "운동 비디오 업로드",
            type=['mp4', 'avi', 'mov'],
            key="exercise_upload"
        )

        if uploaded:
            video_bytes = uploaded.read()
            temp_path = helper.save_temp_video(video_bytes)

            if temp_path:
                st.video(uploaded)

                if st.button("🏋️ 운동 카운트", type="primary"):
                    with st.spinner("관절 추적 및 카운트 중..."):
                        result = helper.count_exercise_reps(temp_path, exercise_type)

                        # 결과 표시
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("반복 횟수", f"{result['count']}회")

                        with col2:
                            st.metric("신뢰도", f"{result['confidence']*100:.1f}%")

                        # 각도 그래프
                        if result['angle_history']:
                            st.subheader("📊 관절 각도 변화")

                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(result['angle_history'], linewidth=2)
                            ax.axhline(y=100, color='r', linestyle='--', label='Down position')
                            ax.axhline(y=140, color='g', linestyle='--', label='Up position')
                            ax.set_xlabel("프레임")
                            ax.set_ylabel("각도 (도)")
                            ax.set_title(f"{exercise_type} 관절 각도 변화")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close()

                        st.success(f"✅ {exercise_type} {result['count']}회 완료!")

    def _app_gesture(self):
        """제스처 인식"""
        st.subheader("👋 제스처 인식")

        st.markdown("""
        **개념**: 손동작을 인식하여 컴퓨터를 제어

        **응용 분야**:
        - 스마트 홈 제어 (손짓으로 조명 조절)
        - 프레젠테이션 제어 (슬라이드 넘기기)
        - VR/AR 인터페이스

        **주요 제스처**:
        - 👍 좋아요 (Thumbs Up)
        - 👋 손 흔들기 (Wave)
        - ✌️ V 사인 (Peace)
        - 👊 주먹 (Fist)
        - ✋ 정지 (Stop)

        ---

        💡 **구현 팁**:

        1. **MediaPipe Hands** 사용
        2. 21개 손 랜드마크 추적
        3. 손가락 관절 각도로 제스처 분류
        4. 시간적 일관성 체크 (떨림 방지)

        ---

        이 기능은 **lab05_practical_apps.py**에서 구현됩니다.
        """)

        st.info("""
        📝 **실습 과제**

        lab05_practical_apps.py를 실행하여 제스처 인식을 직접 구현해보세요!
        """)

    def _app_anomaly(self):
        """이상 행동 감지"""
        st.subheader("⚠️ 이상 행동 감지")

        st.markdown("""
        **목적**: CCTV 영상에서 위험 상황 자동 감지

        **감지 대상**:
        - 🚨 폭력 (싸움, 폭행)
        - 🏃 급격한 움직임 (도망, 추격)
        - 😵 쓰러짐 (낙상, 응급상황)
        - 🔥 화재/연기
        """)

        # 분석 방법 선택
        detection_method = st.radio(
            "감지 방법 선택",
            ["Optical Flow (임계값 기반)", "MediaPipe (포즈 기반)", "Google Video Intelligence (AI 기반)"],
            horizontal=True
        )

        # 비디오 업로드
        uploaded_video = st.file_uploader(
            "🎥 CCTV 또는 보안 영상 업로드",
            type=['mp4', 'avi', 'mov'],
            key="anomaly_upload"
        )

        if uploaded_video:
            # 비디오 미리보기
            st.video(uploaded_video)

            # VideoHelper 로드
            helper = get_video_helper()

            # 임시 파일 저장
            video_bytes = uploaded_video.read()
            temp_path = helper.save_temp_video(video_bytes)

            if temp_path:
                # 파라미터 설정
                col1, col2, col3 = st.columns(3)

                with col1:
                    sample_rate = st.slider(
                        "프레임 샘플링 Rate",
                        min_value=1, max_value=30, value=5,
                        help="낮을수록 더 많은 프레임 분석 (정확도↑, 속도↓)"
                    )

                with col2:
                    if detection_method == "Optical Flow (임계값 기반)":
                        motion_threshold = st.slider(
                            "움직임 임계값 (픽셀)",
                            min_value=5, max_value=30, value=15,
                            help="이 값 이상의 움직임을 이상 행동으로 판단"
                        )
                    elif detection_method == "MediaPipe (포즈 기반)":
                        fall_threshold = st.slider(
                            "낙상 감지 임계값",
                            min_value=0.1, max_value=1.0, value=0.5,
                            help="신체 중심점 변화량 임계값"
                        )
                    else:  # Google Video Intelligence
                        confidence_threshold = st.slider(
                            "신뢰도 임계값",
                            min_value=0.5, max_value=1.0, value=0.7,
                            help="이 값 이상의 신뢰도만 표시"
                        )

                with col3:
                    max_frames = st.number_input(
                        "최대 분석 프레임",
                        min_value=10, max_value=500, value=100,
                        help="메모리 제한을 위한 최대 프레임 수"
                    )

                # 분석 시작
                if st.button("🔍 이상 행동 분석 시작", type="primary", key="analyze_anomaly"):
                    with st.spinner(f"{detection_method} 방법으로 분석 중..."):
                        if detection_method == "Optical Flow (임계값 기반)":
                            results = self._analyze_with_optical_flow(
                                helper, temp_path, sample_rate, motion_threshold, max_frames
                            )
                        elif detection_method == "MediaPipe (포즈 기반)":
                            results = self._analyze_with_mediapipe(
                                helper, temp_path, sample_rate, fall_threshold, max_frames
                            )
                        else:  # Google Video Intelligence
                            results = self._analyze_with_google_api(
                                helper, temp_path, confidence_threshold
                            )

                        # 결과 표시
                        self._display_anomaly_results(results, detection_method)

        # 방법론 설명
        with st.expander("📚 감지 방법 상세 설명", expanded=False):
            st.markdown("""
            ### 1. Optical Flow (임계값 기반)
            **원리**: 연속 프레임 간 픽셀 이동량 계산
            - 정상: 평균 움직임 < 임계값
            - 이상: 평균 움직임 > 임계값
            - **장점**: 빠르고 간단
            - **단점**: 정확도 제한적

            ### 2. MediaPipe (포즈 기반)
            **원리**: 신체 포즈 추적을 통한 이상 감지
            - 낙상: 신체 중심점 급격한 하강
            - 폭력: 빠른 팔 움직임 패턴
            - **장점**: 구체적인 행동 구분 가능
            - **단점**: 사람이 보여야 함

            ### 3. Google Video Intelligence (AI 기반)
            **원리**: 사전 학습된 AI 모델 활용
            - 400+ 행동 레이블 인식
            - 높은 정확도
            - **장점**: 다양한 상황 인식
            - **단점**: API 비용 발생
            """)

        st.warning("""
        ⚠️ **윤리적 고려사항**

        CCTV 기반 이상 행동 감지는 **프라이버시 침해** 우려가 있습니다.
        - 명확한 동의 필요
        - 얼굴 익명화
        - 법적 규제 준수
        """)

    def _analyze_with_optical_flow(self, helper, video_path, sample_rate, threshold, max_frames):
        """Optical Flow 기반 이상 행동 분석"""
        try:
            import cv2
            import numpy as np

            # 프레임 추출
            frames = helper.extract_frames(video_path, sample_rate=sample_rate, max_frames=max_frames)

            motion_scores = []
            anomaly_frames = []

            # 프레임 간 움직임 계산
            for i in range(len(frames) - 1):
                # Optical Flow 계산
                flow = helper.compute_optical_flow(frames[i], frames[i+1])
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                avg_motion = np.mean(magnitude)

                motion_scores.append(avg_motion)

                # 이상 감지
                if avg_motion > threshold:
                    anomaly_frames.append({
                        'frame': i,
                        'motion': avg_motion,
                        'type': 'High Motion Detected'
                    })

            # 전체 통계
            avg_motion = np.mean(motion_scores) if motion_scores else 0
            max_motion = np.max(motion_scores) if motion_scores else 0

            return {
                'status': 'completed',
                'avg_motion': avg_motion,
                'max_motion': max_motion,
                'motion_scores': motion_scores,
                'anomalies': anomaly_frames,
                'total_frames': len(frames),
                'anomaly_count': len(anomaly_frames)
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _analyze_with_mediapipe(self, helper, video_path, sample_rate, fall_threshold, max_frames):
        """MediaPipe 기반 포즈 분석"""
        try:
            import mediapipe as mp
            import numpy as np

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # 프레임 추출
            frames = helper.extract_frames(video_path, sample_rate=sample_rate, max_frames=max_frames)

            pose_history = []
            anomaly_frames = []

            for i, frame in enumerate(frames):
                # RGB 변환 및 포즈 감지
                rgb_frame = np.array(frame)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    # 신체 중심점 계산 (엉덩이 중심)
                    left_hip = results.pose_landmarks.landmark[23]
                    right_hip = results.pose_landmarks.landmark[24]
                    center_y = (left_hip.y + right_hip.y) / 2

                    pose_history.append(center_y)

                    # 낙상 감지 (급격한 Y 변화)
                    if len(pose_history) > 1:
                        y_change = abs(pose_history[-1] - pose_history[-2])
                        if y_change > fall_threshold:
                            anomaly_frames.append({
                                'frame': i,
                                'change': y_change,
                                'type': 'Potential Fall Detected'
                            })

            pose.close()

            return {
                'status': 'completed',
                'pose_detected': len(pose_history),
                'anomalies': anomaly_frames,
                'total_frames': len(frames),
                'anomaly_count': len(anomaly_frames)
            }

        except ImportError:
            return {'status': 'error', 'message': 'MediaPipe not installed'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _analyze_with_google_api(self, helper, video_path, confidence_threshold):
        """Google Video Intelligence API 분석 (시뮬레이션)"""
        # 실제 구현시 Google API 호출
        # 여기서는 시뮬레이션
        import random

        # 시뮬레이션 결과
        potential_anomalies = [
            {'label': 'fighting', 'confidence': 0.85, 'start': 2.5, 'end': 4.2},
            {'label': 'running', 'confidence': 0.72, 'start': 5.0, 'end': 6.5},
            {'label': 'falling', 'confidence': 0.68, 'start': 8.0, 'end': 8.5},
            {'label': 'person walking', 'confidence': 0.95, 'start': 0.0, 'end': 2.0}
        ]

        # 신뢰도 임계값 필터
        anomalies = [a for a in potential_anomalies
                    if a['confidence'] >= confidence_threshold
                    and a['label'] in ['fighting', 'running', 'falling']]

        return {
            'status': 'completed',
            'anomalies': anomalies,
            'total_labels': len(potential_anomalies),
            'anomaly_count': len(anomalies)
        }

    def _display_anomaly_results(self, results, method):
        """이상 행동 분석 결과 표시"""
        if results['status'] == 'error':
            st.error(f"❌ 분석 실패: {results['message']}")
            return

        # 요약 메트릭
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("분석 방법", method.split(" ")[0])

        with col2:
            st.metric("이상 감지", f"{results['anomaly_count']}건")

        with col3:
            if 'total_frames' in results:
                st.metric("분석 프레임", results['total_frames'])
            else:
                st.metric("분석 완료", "✅")

        # 이상 행동 감지 결과
        if results['anomaly_count'] > 0:
            st.error(f"🚨 **이상 행동 감지됨!** ({results['anomaly_count']}건)")

            # 상세 결과
            with st.expander("🔍 상세 분석 결과", expanded=True):
                if method.startswith("Optical Flow"):
                    st.subheader("움직임 분석")
                    for anomaly in results['anomalies']:
                        st.warning(f"Frame {anomaly['frame']}: {anomaly['type']} (움직임: {anomaly['motion']:.2f} 픽셀)")

                    # 움직임 그래프
                    if 'motion_scores' in results and results['motion_scores']:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(results['motion_scores'], linewidth=2)
                        ax.axhline(y=15, color='r', linestyle='--', label='이상 임계값')
                        ax.set_xlabel("프레임")
                        ax.set_ylabel("평균 움직임 (픽셀)")
                        ax.set_title("프레임별 움직임 분석")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()

                elif method.startswith("MediaPipe"):
                    st.subheader("포즈 기반 감지")
                    for anomaly in results['anomalies']:
                        st.warning(f"Frame {anomaly['frame']}: {anomaly['type']} (변화량: {anomaly['change']:.3f})")

                else:  # Google API
                    st.subheader("AI 기반 감지")
                    for anomaly in results['anomalies']:
                        st.warning(
                            f"⚠️ {anomaly['label'].upper()} "
                            f"(신뢰도: {anomaly['confidence']:.1%}) "
                            f"시간: {anomaly['start']:.1f}초 - {anomaly['end']:.1f}초"
                        )
        else:
            st.success("✅ 정상 - 이상 행동이 감지되지 않았습니다.")

            if method.startswith("Optical Flow") and 'avg_motion' in results:
                st.info(f"평균 움직임: {results['avg_motion']:.2f} 픽셀 (정상 범위)")

    # ==================== Tab 6: MediaPipe 실시간 ====================

    def render_mediapipe_realtime(self):
        """MediaPipe를 이용한 실시간 행동 인식"""
        from .action_recognition_realtime import RealtimeActionRecognitionModule

        # 실시간 모듈 인스턴스 생성
        realtime_module = RealtimeActionRecognitionModule()

        # MediaPipe 탭 렌더링
        realtime_module.render_mediapipe_tab()

    # ==================== Tab 7: Google Video Intelligence ====================

    def render_google_video_intelligence(self):
        """Google Video Intelligence API를 이용한 비디오 분석"""
        from .action_recognition_realtime import RealtimeActionRecognitionModule

        # 실시간 모듈 인스턴스 생성
        realtime_module = RealtimeActionRecognitionModule()

        # Google Video Intelligence 탭 렌더링
        realtime_module.render_google_cloud_tab()
