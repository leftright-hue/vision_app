"""
Smart Vision App - 모듈식 통합 웹 인터페이스
주차별 학습 모듈을 통합한 메인 애플리케이션
"""

import os
import streamlit as st
from PIL import Image
import sys
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Configure matplotlib font for Korean glyphs early to avoid missing glyph warnings
try:
    from core.matplotlib_fonts import set_korean_font
    set_korean_font()
except Exception:
    # Non-fatal: proceed if font setup fails
    pass

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트 (무거운 의존성은 지연 로드)
def _try_import_class(module_path: str, class_name: str):
    """Try to import a class and return it, otherwise return None and print the error.

    This allows the Streamlit app to start even when heavy deps (torch, torchvision)
    are not installed. Modules will be enabled only if import succeeds.
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as e:
        # Print to console for debugging; Streamlit will show warnings in the UI when appropriate
        import traceback
        traceback.print_exc()
        return None


# lazy imports for classroom modules
CNNModule = _try_import_class('modules.week02_cnn.cnn_module', 'CNNModule')
TransferLearningModule = _try_import_class('modules.week03.transfer_learning_module', 'TransferLearningModule')
VisionTransformerModule = _try_import_class('modules.week04.vision_transformer_module', 'VisionTransformerModule')
ObjectDetectionModule = _try_import_class('modules.week05.object_detection_module', 'ObjectDetectionModule')

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
        # 저장: 모듈의 클래스 레퍼런스 또는 None
        # 인스턴스화는 렌더링 시점에 수행하여 불필요한 초기화를 피합니다.
        self.modules = {
            'Week 2: CNN': CNNModule,
            'Week 3: Transfer Learning': TransferLearningModule,
            'Week 4: Vision Transformer': VisionTransformerModule,
            'Week 5: Object Detection': ObjectDetectionModule,
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
            - Week 4: Vision Transformer ✅
            - Week 5: Object Detection ✅
            """)

            # API 사용 안내
            st.header("🤖 API 사용")
            import os
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key and api_key != 'your_api_key_here':
                st.success("✅ Google Gemini API 연결됨")
                st.caption("실전 프로젝트에서 실제 API를 사용할 수 있습니다.")
            else:
                st.warning("⚠️ API Key 미설정")
                st.caption(".env 파일에 GOOGLE_API_KEY를 설정하세요.")
                with st.expander("API 키 설정 방법"):
                    st.markdown("""
                    1. [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키 발급
                    2. `.env` 파일에 키 추가:
                    ```
                    GOOGLE_API_KEY=your_api_key_here
                    ```
                    3. 앱 재시작
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
            module_cls = self.modules[selected_module]
            if module_cls is None:
                st.warning(
                    "선택한 모듈은 현재 사용 불가합니다. 필요한 패키지가 설치되어 있지 않거나 import 중 오류가 발생했습니다.\n"
                    "`pip install -r requirements.txt`로 의존성을 설치한 뒤 재시작하세요."
                )
            else:
                try:
                    module = module_cls()
                    module.render()
                except Exception as e:
                    st.error(f"모듈을 초기화하거나 렌더링하는 중 오류가 발생했습니다: {e}")
                    import traceback
                    st.text(traceback.format_exc())
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
                "Week 4: Vision Transformer": 100,
                "Week 5: Object Detection": 100,
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

            st.subheader("✅ Week 4: Vision Transformer")
            st.markdown("""
            - Self-Attention 메커니즘
            - Vision Transformer (ViT)
            - DINO 자기지도학습
            - 최신 모델 비교
            """)

            st.subheader("✅ Week 5: Object Detection")
            st.markdown("""
            - R-CNN 계열 발전사
            - YOLO 아키텍처
            - IoU & mAP 평가지표
            - 실시간 객체 탐지
            """)

        st.markdown("---")

        # 빠른 시작
        st.header("⚡ 빠른 시작")

        quick_start_col1, quick_start_col2, quick_start_col3 = st.columns(3)

        with quick_start_col1:
            if st.button("🔬 이미지 분석 시작", width='stretch'):
                st.session_state['selected_module'] = 'Week 2: CNN'
                st.rerun()

        with quick_start_col2:
            if st.button("🎨 필터 적용하기", width='stretch'):
                st.session_state['selected_module'] = 'Week 2: CNN'
                st.session_state['selected_tab'] = 'filtering'
                st.rerun()

        with quick_start_col3:
            if st.button("🤖 AI 모델 테스트", width='stretch'):
                st.session_state['selected_module'] = 'Week 2: CNN'
                st.session_state['selected_tab'] = 'huggingface'
                st.rerun()

def main():
    """메인 함수"""
    app = SmartVisionApp()
    app.run()

if __name__ == "__main__":
    main()