"""
Week 7: Real-time Action Recognition Module
MediaPipe (Open Source) and Google Video Intelligence API (Cloud) Implementation
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import io
import base64

# Core imports
from core.base_processor import BaseImageProcessor


class RealtimeActionRecognitionModule(BaseImageProcessor):
    """Week 7: 실시간 행동인식 모듈"""

    def __init__(self):
        super().__init__()
        self.name = "Week 7: Realtime Action Recognition"
        self.mediapipe_available = self._check_mediapipe()
        self.google_cloud_available = self._check_google_cloud()

    def _check_mediapipe(self) -> bool:
        """MediaPipe 설치 확인"""
        try:
            import mediapipe as mp
            return True
        except ImportError:
            return False

    def _check_google_cloud(self) -> bool:
        """Google Cloud Video Intelligence API 설치 확인"""
        try:
            from google.cloud import videointelligence
            return True
        except ImportError:
            return False

    def render(self):
        """메인 렌더링 함수"""
        st.title("🎬 Week 7: 실시간 행동인식 (Real-time Action Recognition)")

        st.markdown("""
        ## 학습 목표
        - **Open Source**: MediaPipe를 활용한 실시간 행동 인식
        - **Cloud API**: Google Video Intelligence API를 활용한 비디오 분석
        - **실습**: 두 가지 접근 방식을 통한 행동 인식 구현
        """)

        # 환경 체크
        self._check_environment()

        # 2개 탭 구성
        tabs = st.tabs([
            "🔧 Open Source: MediaPipe",
            "☁️ Cloud: Google Video Intelligence"
        ])

        with tabs[0]:
            self.render_mediapipe_tab()

        with tabs[1]:
            self.render_google_cloud_tab()

    def _check_environment(self):
        """환경 체크 및 설정"""
        with st.expander("🔧 환경 설정 확인", expanded=False):
            st.markdown("""
            ### 필요한 패키지

            **MediaPipe (Open Source)**:
            ```bash
            pip install mediapipe opencv-python numpy
            ```

            **Google Cloud Video Intelligence**:
            ```bash
            pip install google-cloud-videointelligence
            # Google Cloud 인증 설정 필요
            ```
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("MediaPipe Status")
                if self.mediapipe_available:
                    st.success("✅ MediaPipe 설치됨")
                else:
                    st.error("❌ MediaPipe 미설치")

            with col2:
                st.subheader("Google Cloud Status")
                if self.google_cloud_available:
                    st.success("✅ Google Cloud SDK 설치됨")
                else:
                    st.warning("⚠️ Google Cloud SDK 미설치")

    # ==================== MediaPipe Tab ====================

    def render_mediapipe_tab(self):
        """MediaPipe를 이용한 행동 인식"""
        st.header("🔧 Open Source: MediaPipe Action Recognition")

        st.markdown("""
        ### MediaPipe란?
        Google에서 개발한 오픈소스 프레임워크로, 실시간 비디오 분석을 위한 다양한 ML 솔루션 제공

        ### 주요 기능
        - **Pose Detection**: 33개 신체 랜드마크 추적
        - **Hand Tracking**: 21개 손 랜드마크 추적
        - **Face Detection**: 468개 얼굴 랜드마크 추적
        - **Holistic**: Pose + Hand + Face 통합 추적

        ### 행동 인식 응용
        - 운동 자세 분석 (스쿼트, 푸시업 카운팅)
        - 제스처 인식 (수화, 손동작 명령)
        - 넘어짐 감지 (안전 모니터링)
        - 스포츠 동작 분석
        """)

        if not self.mediapipe_available:
            st.error("""
            ❌ MediaPipe가 설치되지 않았습니다!

            설치 방법:
            ```bash
            pip install mediapipe opencv-python
            ```
            """)
            return

        # MediaPipe 모드 선택
        detection_mode = st.selectbox(
            "검출 모드 선택",
            ["Pose Detection (전신 포즈)", "Hand Tracking (손 제스처)", "Holistic (통합)"]
        )

        # 행동 유형 선택
        if detection_mode == "Pose Detection (전신 포즈)":
            action_type = st.selectbox(
                "인식할 행동",
                ["운동 카운팅 (스쿼트/푸시업)", "넘어짐 감지", "요가 자세 인식"]
            )
        elif detection_mode == "Hand Tracking (손 제스처)":
            action_type = st.selectbox(
                "인식할 제스처",
                ["기본 제스처 (주먹/가위/바위)", "숫자 카운팅 (1-5)", "방향 지시"]
            )
        else:
            action_type = "통합 분석"

        # 비디오 입력
        st.subheader("📹 비디오 입력")
        input_source = st.radio(
            "입력 소스 선택",
            ["파일 업로드", "샘플 비디오", "웹캠 (별도 실행)"]
        )

        if input_source == "파일 업로드":
            uploaded_file = st.file_uploader(
                "비디오 파일 선택",
                type=['mp4', 'avi', 'mov'],
                key="mediapipe_upload"
            )

            if uploaded_file:
                # 임시 파일 저장
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name

                # 비디오 미리보기
                st.video(uploaded_file)

                if st.button("🎬 MediaPipe 분석 시작", type="primary", key="mp_analyze"):
                    self._process_with_mediapipe(video_path, detection_mode, action_type)

        elif input_source == "샘플 비디오":
            st.info("샘플 비디오를 생성합니다...")
            if st.button("🎥 샘플 비디오 생성 및 분석", key="mp_sample"):
                video_path = self._create_sample_video()
                if video_path:
                    self._process_with_mediapipe(video_path, detection_mode, action_type)

        else:
            st.markdown("""
            ### 웹캠 실시간 처리

            웹캠을 이용한 실시간 처리는 별도 Python 스크립트로 실행하세요:

            ```python
            import cv2
            import mediapipe as mp

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose()
            mp_drawing = mp.solutions.drawing_utils

            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # MediaPipe 처리
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 랜드마크 그리기
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow('MediaPipe Pose', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            cap.release()
            cv2.destroyAllWindows()
            ```
            """)

    def _process_with_mediapipe(self, video_path: str, detection_mode: str, action_type: str):
        """MediaPipe로 비디오 처리"""
        import mediapipe as mp
        import cv2

        st.info(f"🔄 처리 중... 모드: {detection_mode}, 행동: {action_type}")

        # MediaPipe 초기화
        mp_drawing = mp.solutions.drawing_utils

        if "Pose" in detection_mode:
            mp_pose = mp.solutions.pose
            detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            connections = mp_pose.POSE_CONNECTIONS
        elif "Hand" in detection_mode:
            mp_hands = mp.solutions.hands
            detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            connections = mp_hands.HAND_CONNECTIONS
        else:
            mp_holistic = mp.solutions.holistic
            detector = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            connections = None

        # 비디오 처리
        cap = cv2.VideoCapture(video_path)

        # 프레임 샘플링 (매 5프레임마다 처리)
        frame_count = 0
        processed_frames = []
        landmarks_history = []
        action_counts = {"count": 0, "state": "neutral"}

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 프레임 샘플링
            if frame_count % 5 != 0:
                continue

            # 진행률 업데이트
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"프레임 {frame_count}/{total_frames} 처리 중...")

            # RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe 처리
            results = detector.process(rgb_frame)

            # 결과 시각화
            annotated_frame = frame.copy()

            if "Pose" in detection_mode and results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, results.pose_landmarks, connections)
                landmarks_history.append(results.pose_landmarks)

                # 운동 카운팅 로직
                if "운동" in action_type:
                    action_counts = self._count_exercise(
                        results.pose_landmarks, action_counts)

            elif "Hand" in detection_mode and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, hand_landmarks, connections)
                landmarks_history.append(hand_landmarks)

            elif "Holistic" in detection_mode:
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, results.pose_landmarks,
                        mp.solutions.holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, results.left_hand_landmarks,
                        mp.solutions.holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, results.right_hand_landmarks,
                        mp.solutions.holistic.HAND_CONNECTIONS)

            # 일부 프레임 저장 (메모리 절약)
            if len(processed_frames) < 10:
                processed_frames.append(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        cap.release()
        detector.close()

        # 결과 표시
        st.success("✅ MediaPipe 분석 완료!")

        # 메트릭 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("처리된 프레임", f"{frame_count}")
        with col2:
            st.metric("검출된 랜드마크", f"{len(landmarks_history)}")
        with col3:
            if "운동" in action_type:
                st.metric("운동 횟수", f"{action_counts['count']}회")

        # 처리된 프레임 표시
        if processed_frames:
            st.subheader("📸 처리된 프레임 샘플")
            cols = st.columns(3)
            for i, frame in enumerate(processed_frames[:6]):
                with cols[i % 3]:
                    st.image(frame, caption=f"Frame {i+1}", use_container_width=True)

        # 상세 분석 결과
        with st.expander("📊 상세 분석 결과"):
            st.json({
                "detection_mode": detection_mode,
                "action_type": action_type,
                "total_frames": total_frames,
                "processed_frames": frame_count // 5,
                "landmarks_detected": len(landmarks_history),
                "action_counts": action_counts if "운동" in action_type else "N/A"
            })

    def _count_exercise(self, landmarks, counts):
        """운동 카운팅 로직 (간단한 예시)"""
        # 무릎 각도 계산 (스쿼트 예시)
        import math

        def calculate_angle(a, b, c):
            """세 점 사이의 각도 계산"""
            ang = math.degrees(
                math.atan2(c.y - b.y, c.x - b.x) -
                math.atan2(a.y - b.y, a.x - b.x)
            )
            return ang + 360 if ang < 0 else ang

        # 왼쪽 무릎 각도 (엉덩이-무릎-발목)
        hip = landmarks.landmark[23]  # LEFT_HIP
        knee = landmarks.landmark[25]  # LEFT_KNEE
        ankle = landmarks.landmark[27]  # LEFT_ANKLE

        angle = calculate_angle(hip, knee, ankle)

        # 스쿼트 카운팅 로직
        if angle < 90:  # 무릎이 90도 이하로 굽혀짐
            if counts["state"] == "up":
                counts["count"] += 1
                counts["state"] = "down"
        elif angle > 160:  # 무릎이 거의 펴짐
            counts["state"] = "up"

        return counts

    # ==================== Google Cloud Tab ====================

    def render_google_cloud_tab(self):
        """Google Video Intelligence API를 이용한 행동 인식"""
        st.header("☁️ Cloud: Google Video Intelligence API")

        st.markdown("""
        ### Google Video Intelligence API란?
        Google Cloud의 비디오 분석 서비스로, 머신러닝을 통해 비디오 콘텐츠를 자동으로 분석

        ### 주요 기능
        - **Label Detection**: 비디오 내 객체, 장소, 활동 감지
        - **Shot Change Detection**: 장면 전환 감지
        - **Explicit Content Detection**: 부적절한 콘텐츠 감지
        - **Speech Transcription**: 음성 텍스트 변환
        - **Object Tracking**: 객체 추적
        - **Face Detection**: 얼굴 감지 및 감정 분석
        - **Person Detection**: 사람 감지 및 추적
        - **Logo Recognition**: 로고 인식

        ### 행동 인식 관련 기능
        - 400+ 가지 사전 정의된 행동 레이블
        - 시간별 행동 구간 감지
        - 신뢰도 점수 제공
        """)

        # API 설정 안내
        with st.expander("🔑 API 설정 가이드", expanded=False):
            st.markdown("""
            ### 1. Google Cloud 프로젝트 설정
            1. [Google Cloud Console](https://console.cloud.google.com) 접속
            2. 새 프로젝트 생성 또는 기존 프로젝트 선택
            3. Video Intelligence API 활성화

            ### 2. 인증 설정
            ```bash
            # 서비스 계정 키 생성 후
            export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"

            # 또는 Python 코드에서
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/key.json'
            ```

            ### 3. 패키지 설치
            ```bash
            pip install google-cloud-videointelligence
            ```

            ### 4. 무료 한도
            - 매월 처음 1000분 무료
            - 이후 분당 $0.10 ~ $0.15
            """)

        # API 키 입력
        st.subheader("🔑 API 인증")

        auth_method = st.radio(
            "인증 방법",
            ["환경 변수 (권장)", "직접 입력", "시뮬레이션 모드"]
        )

        api_ready = False

        if auth_method == "환경 변수 (권장)":
            if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                st.success("✅ 환경 변수에서 인증 정보를 찾았습니다.")
                api_ready = True
            else:
                st.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

        elif auth_method == "직접 입력":
            api_key_file = st.file_uploader(
                "서비스 계정 키 JSON 파일 업로드",
                type=['json'],
                key="gcp_key"
            )
            if api_key_file:
                # 임시로 저장
                key_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
                key_path.write(api_key_file.read())
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path.name
                st.success("✅ API 키가 로드되었습니다.")
                api_ready = True

        else:  # 시뮬레이션 모드
            st.info("📝 시뮬레이션 모드: 실제 API 호출 없이 예시 결과를 보여줍니다.")
            api_ready = True

        # 분석 기능 선택
        st.subheader("🎯 분석 기능 선택")

        features = st.multiselect(
            "분석할 기능 선택 (복수 선택 가능)",
            [
                "LABEL_DETECTION (행동/객체 레이블)",
                "SHOT_CHANGE_DETECTION (장면 전환)",
                "EXPLICIT_CONTENT_DETECTION (부적절 콘텐츠)",
                "PERSON_DETECTION (사람 감지)",
                "FACE_DETECTION (얼굴 감지)",
                "OBJECT_TRACKING (객체 추적)"
            ],
            default=["LABEL_DETECTION (행동/객체 레이블)"]
        )

        # 비디오 입력
        st.subheader("📹 비디오 입력")

        input_method = st.radio(
            "입력 방법",
            ["파일 업로드", "Google Cloud Storage URI", "YouTube URL (제한적)"]
        )

        video_input = None

        if input_method == "파일 업로드":
            uploaded_file = st.file_uploader(
                "비디오 파일 선택",
                type=['mp4', 'avi', 'mov'],
                key="gcp_upload"
            )
            if uploaded_file:
                st.video(uploaded_file)
                video_input = uploaded_file

        elif input_method == "Google Cloud Storage URI":
            gcs_uri = st.text_input(
                "GCS URI 입력",
                placeholder="gs://bucket-name/video-file.mp4"
            )
            if gcs_uri:
                video_input = gcs_uri

        else:
            st.info("YouTube URL 분석은 제한적입니다. 다운로드 후 업로드를 권장합니다.")

        # 분석 시작
        if video_input and api_ready and features:
            if st.button("🚀 Google Video Intelligence 분석 시작", type="primary", key="gcp_analyze"):
                if auth_method == "시뮬레이션 모드":
                    self._simulate_google_analysis(features)
                else:
                    self._process_with_google_api(video_input, features, input_method)

    def _simulate_google_analysis(self, features):
        """Google Video Intelligence API 시뮬레이션"""
        st.info("🔄 시뮬레이션 모드로 분석 중...")

        # 프로그레스 바
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.02)

        st.success("✅ 분석 완료 (시뮬레이션)")

        # 시뮬레이션 결과
        if "LABEL_DETECTION" in str(features):
            st.subheader("🏷️ 레이블 감지 결과")

            # 샘플 레이블
            labels = [
                {"name": "walking", "confidence": 0.92, "segments": [(0, 5), (10, 15)]},
                {"name": "running", "confidence": 0.87, "segments": [(5, 10)]},
                {"name": "jumping", "confidence": 0.75, "segments": [(15, 18)]},
                {"name": "person", "confidence": 0.95, "segments": [(0, 20)]},
                {"name": "outdoor", "confidence": 0.88, "segments": [(0, 20)]}
            ]

            for label in labels:
                with st.expander(f"{label['name']} (신뢰도: {label['confidence']:.1%})"):
                    st.write(f"감지된 구간: {label['segments']}")

                    # 타임라인 시각화
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 1))
                    for seg in label['segments']:
                        ax.barh(0, seg[1] - seg[0], left=seg[0], height=0.5,
                               color='blue', alpha=0.6)
                    ax.set_xlim(0, 20)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_xlabel("시간 (초)")
                    ax.set_yticks([])
                    ax.set_title(f"{label['name']} 타임라인")
                    st.pyplot(fig)
                    plt.close()

        if "SHOT_CHANGE_DETECTION" in str(features):
            st.subheader("🎬 장면 전환 감지")
            st.write("감지된 장면 전환 시점: 3.5초, 7.2초, 12.1초, 16.8초")

        if "PERSON_DETECTION" in str(features):
            st.subheader("👤 사람 감지")
            st.write("감지된 사람 수: 2명")
            st.write("추적 ID: Person_1 (0-20초), Person_2 (5-15초)")

    def _process_with_google_api(self, video_input, features, input_method):
        """실제 Google Video Intelligence API 호출"""
        if not self.google_cloud_available:
            st.error("Google Cloud Video Intelligence 패키지가 설치되지 않았습니다.")
            return

        try:
            from google.cloud import videointelligence

            st.info("🔄 Google Video Intelligence API 호출 중...")

            # 클라이언트 초기화
            video_client = videointelligence.VideoIntelligenceServiceClient()

            # 기능 매핑
            feature_map = {
                "LABEL_DETECTION": videointelligence.Feature.LABEL_DETECTION,
                "SHOT_CHANGE_DETECTION": videointelligence.Feature.SHOT_CHANGE_DETECTION,
                "EXPLICIT_CONTENT_DETECTION": videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
                "PERSON_DETECTION": videointelligence.Feature.PERSON_DETECTION,
                "FACE_DETECTION": videointelligence.Feature.FACE_DETECTION,
                "OBJECT_TRACKING": videointelligence.Feature.OBJECT_TRACKING
            }

            selected_features = []
            for f in features:
                key = f.split(" ")[0]
                if key in feature_map:
                    selected_features.append(feature_map[key])

            # 입력 준비
            if input_method == "Google Cloud Storage URI":
                input_uri = video_input
            else:
                # 파일 업로드의 경우 바이트로 변환
                input_content = video_input.read()

            # API 호출
            if input_method == "Google Cloud Storage URI":
                operation = video_client.annotate_video(
                    request={
                        "features": selected_features,
                        "input_uri": input_uri
                    }
                )
            else:
                operation = video_client.annotate_video(
                    request={
                        "features": selected_features,
                        "input_content": input_content
                    }
                )

            st.info("⏳ 분석 중... (1-2분 소요)")

            # 결과 대기
            result = operation.result(timeout=180)

            st.success("✅ 분석 완료!")

            # 결과 파싱 및 표시
            self._display_google_results(result)

        except Exception as e:
            st.error(f"❌ API 호출 실패: {str(e)}")
            st.info("API 키 설정과 권한을 확인하세요.")

    def _display_google_results(self, result):
        """Google API 결과 표시"""
        # 세그먼트별 주석
        for annotation in result.annotation_results:

            # 레이블 감지 결과
            if annotation.segment_label_annotations:
                st.subheader("🏷️ 감지된 행동/객체 레이블")

                for label in annotation.segment_label_annotations[:10]:  # 상위 10개
                    entity = label.entity
                    confidence = label.segments[0].confidence if label.segments else 0

                    with st.expander(f"{entity.description} (신뢰도: {confidence:.1%})"):
                        st.write(f"Entity ID: {entity.entity_id}")

                        # 세그먼트 정보
                        for segment in label.segments:
                            start_time = segment.segment.start_time_offset.total_seconds()
                            end_time = segment.segment.end_time_offset.total_seconds()
                            st.write(f"시간: {start_time:.1f}초 - {end_time:.1f}초")

            # 장면 전환 감지
            if annotation.shot_annotations:
                st.subheader("🎬 장면 전환")
                shot_times = []
                for shot in annotation.shot_annotations:
                    start = shot.start_time_offset.total_seconds()
                    end = shot.end_time_offset.total_seconds()
                    shot_times.append((start, end))
                st.write(f"총 {len(shot_times)}개 장면 감지")
                st.write(shot_times[:5])  # 처음 5개만 표시

            # 명시적 콘텐츠 감지
            if annotation.explicit_annotation:
                st.subheader("🔞 명시적 콘텐츠")
                for frame in annotation.explicit_annotation.frames[:5]:
                    time = frame.time_offset.total_seconds()
                    level = frame.pornography_likelihood.name
                    st.write(f"{time:.1f}초: {level}")

    def _create_sample_video(self):
        """간단한 샘플 비디오 생성"""
        try:
            import cv2
            import numpy as np

            # 임시 파일 생성
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_path = temp_file.name

            # 비디오 라이터 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 20.0, (640, 480))

            # 간단한 애니메이션 생성 (움직이는 원)
            for i in range(100):
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

                # 움직이는 원 그리기
                x = int(320 + 200 * np.sin(i * 0.1))
                y = int(240 + 100 * np.cos(i * 0.1))
                cv2.circle(frame, (x, y), 30, (255, 0, 0), -1)

                # 텍스트 추가
                cv2.putText(frame, f"Frame {i}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                out.write(frame)

            out.release()
            st.success(f"✅ 샘플 비디오 생성 완료: {temp_path}")
            return temp_path

        except Exception as e:
            st.error(f"샘플 비디오 생성 실패: {str(e)}")
            return None


# Streamlit 앱 실행을 위한 메인 함수
def main():
    module = RealtimeActionRecognitionModule()
    module.render()


if __name__ == "__main__":
    main()