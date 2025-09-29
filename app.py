"""
Smart Vision App - 모듈식 통합 웹 인터페이스
주차별 학습 모듈을 통합한 메인 애플리케이션
"""

import os
import streamlit as st
from PIL import Image
import sys

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
from modules.week02_cnn.cnn_module import CNNModule
from modules.week03.transfer_learning_module import TransferLearningModule

# 페이지 설정
st.set_page_config(
    page_title="Smart Vision App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SmartVisionApp:
    """메인 애플리케이션 클래스"""

    def __init__(self):
        self.modules = {
            'Week 2: CNN': CNNModule(),
            'Week 3: Transfer Learning': TransferLearningModule(),
            # Week 4 등은 나중에 추가
        }

    def run(self):
        """애플리케이션 실행"""
        # 사이드바
        with st.sidebar:
            st.title("🎯 Smart Vision App")
            st.markdown("---")

            # 모듈 선택
            st.header("📚 학습 모듈")
            selected_module = st.selectbox(
                "모듈 선택",
                list(self.modules.keys())
            )

            st.markdown("---")

            # 앱 정보
            st.header("ℹ️ 정보")
            st.info("""
            **Smart Vision App**

            AI 비전 학습 통합 플랫폼

            - Week 2: CNN과 이미지 처리 ✅
            - Week 3: Transfer Learning ✅
            - Week 4: 멀티모달 AI (예정)
            """)

            # 리소스 링크
            st.header("🔗 리소스")
            st.markdown("""
            - [HuggingFace](https://huggingface.co)
            - [PyTorch](https://pytorch.org)
            - [OpenCV Docs](https://docs.opencv.org)
            """)

        # 메인 컨텐츠
        if selected_module in self.modules:
            self.modules[selected_module].render()
        else:
            # 홈 페이지
            self.render_home()

    def render_home(self):
        """홈 페이지 렌더링"""
        st.title("🎯 Smart Vision App")
        st.markdown("### AI 비전 학습 통합 플랫폼")

        st.markdown("---")

        # 소개
        col1, col2 = st.columns(2)

        with col1:
            st.header("🚀 주요 기능")
            st.markdown("""
            - **이미지 처리**: 필터링, 변환, 분석
            - **CNN 학습**: 신경망 구조 이해와 시각화
            - **AI 모델**: HuggingFace 사전훈련 모델 활용
            - **통합 분석**: 다양한 기법을 결합한 종합 분석
            """)

            st.header("📈 학습 진도")
            progress_data = {
                "Week 2: CNN": 100,
                "Week 3: Transfer Learning": 100,
                "Week 4: Multimodal AI": 0,
            }

            for week, progress in progress_data.items():
                st.write(f"**{week}**")
                st.progress(progress / 100)

        with col2:
            st.header("🎓 학습 모듈")

            st.subheader("✅ Week 2: CNN과 디지털 이미지")
            st.markdown("""
            - 디지털 이미지의 구조
            - Convolution 연산
            - CNN 아키텍처
            - HuggingFace 활용
            """)

            st.subheader("✅ Week 3: Transfer Learning")
            st.markdown("""
            - 사전훈련 모델 활용
            - Fine-tuning 기법
            - Multi-modal API 비교
            - CLIP 기반 검색
            """)

            st.subheader("🔜 Week 4: Multimodal AI")
            st.markdown("""
            - 이미지-텍스트 통합
            - CLIP 모델
            - 비전-언어 태스크
            """)

        st.markdown("---")

        # 빠른 시작
        st.header("⚡ 빠른 시작")

        quick_start_col1, quick_start_col2, quick_start_col3 = st.columns(3)

        with quick_start_col1:
            if st.button("🔬 이미지 분석 시작", use_container_width=True):
                st.session_state['selected_module'] = 'Week 2: CNN'
                st.rerun()

        with quick_start_col2:
            if st.button("🎨 필터 적용하기", use_container_width=True):
                st.session_state['selected_module'] = 'Week 2: CNN'
                st.session_state['selected_tab'] = 'filtering'
                st.rerun()

        with quick_start_col3:
            if st.button("🤖 AI 모델 테스트", use_container_width=True):
                st.session_state['selected_module'] = 'Week 2: CNN'
                st.session_state['selected_tab'] = 'huggingface'
                st.rerun()

def main():
    """메인 함수"""
    app = SmartVisionApp()
    app.run()

if __name__ == "__main__":
    main()