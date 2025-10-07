"""
Week 5: 객체 탐지와 YOLO 모듈
객체 탐지 이론, R-CNN 계열, YOLO 아키텍처, 실전 프로젝트
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import google.generativeai as genai

class ObjectDetectionModule:
    def _check_environment(self):
        """환경 체크 및 자동 설정"""
        import sys
        import subprocess

        issues = []

        # Python 버전 체크
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 12:
            # Python 3.12+ 감지
            try:
                import mediapipe
                import streamlit_webrtc
            except ImportError as e:
                issues.append(f"❌ 필수 패키지 누락: {str(e)}")

                st.warning("""
                **환경 설정이 필요합니다.**

                Python 3.12 이상에서는 일부 패키지의 호환성 문제가 있을 수 있습니다.
                """)

                if st.button("🔧 자동으로 환경 설정하기", key="auto_setup_env"):
                    with st.spinner("환경을 설정하는 중..."):
                        try:
                            # requirements.txt 업데이트
                            st.info("requirements.txt를 업데이트하는 중...")

                            # 필수 패키지 설치
                            packages_to_install = [
                                "mediapipe>=0.10.0,<0.11",
                                "streamlit-webrtc>=0.63.0",
                                "numpy>=1.26.0,<2.0"
                            ]

                            for package in packages_to_install:
                                st.info(f"설치 중: {package}")
                                result = subprocess.run(
                                    [sys.executable, "-m", "pip", "install", package],
                                    capture_output=True,
                                    text=True
                                )
                                if result.returncode == 0:
                                    st.success(f"✅ {package} 설치 완료")
                                else:
                                    st.error(f"❌ {package} 설치 실패: {result.stderr}")

                            st.success("✅ 환경 설정 완료! 페이지를 새로고침하세요.")
                            st.balloons()

                        except Exception as e:
                            st.error(f"환경 설정 중 오류 발생: {e}")
                            st.code("""
# 수동 설치 방법:
python -m pip install mediapipe>=0.10.0,<0.11
python -m pip install streamlit-webrtc>=0.63.0
python -m pip install numpy>=1.26.0,<2.0
                            """, language="bash")

                return False

        elif python_version.major == 3 and python_version.minor == 13:
            st.error("""
            ⚠️ **Python 버전 호환성 문제**

            현재 Python 3.13을 사용 중입니다.
            mediapipe는 Python 3.12 이하에서만 지원됩니다.

            **해결 방법:**
            1. Python 3.12 설치
            2. 새 가상환경 생성:
            ```bash
            py -3.12 -m venv venv
            venv\\Scripts\\activate
            pip install -r requirements.txt
            ```
            """)
            return False

        return True

    def _ensure_yolo_model(self, model_path="yolov8n.pt"):
        import os
        import requests
        import streamlit as st

        url = f"https://huggingface.co/ultralytics/yolov8/resolve/main/{model_path}"
        if not os.path.exists(model_path):
            with st.spinner(f"'{model_path}' 모델을 다운로드합니다... (약 6MB)"):
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(model_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    st.success(f"'{model_path}' 다운로드 완료!")
                except Exception as e:
                    st.error(f"모델 다운로드 실패: {e}")
                    return False
        return True

    def __init__(self):
        self.name = "Week 5: Object Detection & YOLO"

    def render(self):
        st.title("🎯 Week 5: 객체 탐지와 YOLO")
        st.markdown("**객체 탐지의 이론부터 YOLO 실전 구현까지**")

        # 환경 체크
        if not self._check_environment():
            st.warning("⚠️ 환경 설정 후 계속 진행하세요.")
            return

        tabs = st.tabs([
            "📖 이론",
            "🔍 IoU & mAP",
            "🏗️ R-CNN 계열",
            "⚡ YOLO 발전사",
            "🎨 NMS",
            "💻 실전 프로젝트"
        ])

        with tabs[0]:
            self.render_theory()

        with tabs[1]:
            self.render_iou_map()

        with tabs[2]:
            self.render_rcnn()

        with tabs[3]:
            self.render_yolo()

        with tabs[4]:
            self.render_nms()

        with tabs[5]:
            self.render_projects()

    def render_theory(self):
        """객체 탐지 기초 이론"""
        st.header("📖 객체 탐지 기초 이론")

        theory_tabs = st.tabs(["개요", "구성 요소", "평가 지표"])

        with theory_tabs[0]:
            st.subheader("1. 객체 탐지란?")

            st.markdown("""
            ### 정의
            **객체 탐지(Object Detection)**: 이미지에서 관심 있는 객체들을 찾아내고,
            각 객체의 위치를 바운딩 박스로 표시하며, 객체의 클래스를 분류하는 작업

            ### 이미지 분류 vs 객체 탐지
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.info("""
                **이미지 분류 (Classification)**
                - 입력: 이미지
                - 출력: 클래스 라벨 (예: "고양이")
                - 목적: "무엇인가?"
                """)

            with col2:
                st.success("""
                **객체 탐지 (Detection)**
                - 입력: 이미지
                - 출력: 클래스 + 위치 + 신뢰도
                - 목적: "무엇이 어디에?"
                """)

            st.markdown("### 객체 탐지의 도전과제")

            challenges = {
                "다중 객체": "하나의 이미지에 여러 객체가 존재",
                "다양한 크기": "같은 클래스라도 크기가 다양함",
                "가려짐(Occlusion)": "객체들이 서로 겹쳐있음",
                "배경 복잡성": "복잡한 배경에서 객체 구분",
                "실시간 처리": "빠른 추론 속도 요구"
            }

            for challenge, description in challenges.items():
                st.markdown(f"**{challenge}**: {description}")

        with theory_tabs[1]:
            st.subheader("2. 핵심 구성 요소")

            st.markdown("### 1) 바운딩 박스 (Bounding Box)")

            st.code("""
# 바운딩 박스 표현 방식들
bbox_formats = {
    "xyxy": [x_min, y_min, x_max, y_max],           # 좌상단, 우하단
    "xywh": [x_center, y_center, width, height],    # 중심점과 크기
    "cxcywh": [cx, cy, w, h],                       # 정규화된 중심점
}
            """, language="python")

            st.markdown("### 2) 신뢰도 점수 (Confidence Score)")
            st.latex(r"\text{Confidence} = P(\text{object}) \times \text{IoU}(\text{pred}, \text{true})")

            st.markdown("""
            - **P(object)**: 해당 위치에 객체가 있을 확률
            - **IoU**: 예측 박스와 실제 박스의 겹침 정도
            """)

            st.markdown("### 3) 클래스 확률 (Class Probability)")
            st.code("""
# 각 클래스에 대한 확률 분포
class_probs = softmax([logit_cat, logit_dog, logit_car, ...])
# 예: [0.7, 0.2, 0.05, 0.03, 0.02]
            """, language="python")

        with theory_tabs[2]:
            st.subheader("3. 평가 지표")

            st.markdown("### IoU (Intersection over Union)")

            st.latex(r"\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}")

            st.markdown("""
            **IoU 해석:**
            - IoU > 0.5: "좋은" 탐지
            - IoU > 0.7: "매우 좋은" 탐지
            - IoU > 0.9: "거의 완벽한" 탐지
            """)

            # IoU 시뮬레이션
            st.markdown("#### IoU 계산 시뮬레이션")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Ground Truth Box**")
                gt_x1 = st.slider("GT X1", 0, 100, 20, key="gt_x1")
                gt_y1 = st.slider("GT Y1", 0, 100, 20, key="gt_y1")
                gt_x2 = st.slider("GT X2", 0, 100, 60, key="gt_x2")
                gt_y2 = st.slider("GT Y2", 0, 100, 60, key="gt_y2")

            with col2:
                st.markdown("**Predicted Box**")
                pred_x1 = st.slider("Pred X1", 0, 100, 25, key="pred_x1")
                pred_y1 = st.slider("Pred Y1", 0, 100, 25, key="pred_y1")
                pred_x2 = st.slider("Pred X2", 0, 100, 65, key="pred_x2")
                pred_y2 = st.slider("Pred Y2", 0, 100, 65, key="pred_y2")

            # IoU 계산
            iou = self.calculate_iou(
                [gt_x1, gt_y1, gt_x2, gt_y2],
                [pred_x1, pred_y1, pred_x2, pred_y2]
            )

            st.metric("IoU", f"{iou:.3f}")

            # 시각화
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect('equal')

            # Ground Truth (파란색)
            gt_rect = plt.Rectangle((gt_x1, gt_y1), gt_x2-gt_x1, gt_y2-gt_y1,
                                    linewidth=2, edgecolor='blue', facecolor='none',
                                    label='Ground Truth')
            ax.add_patch(gt_rect)

            # Prediction (빨간색)
            pred_rect = plt.Rectangle((pred_x1, pred_y1), pred_x2-pred_x1, pred_y2-pred_y1,
                                      linewidth=2, edgecolor='red', facecolor='none',
                                      label='Prediction')
            ax.add_patch(pred_rect)

            ax.legend()
            ax.set_title(f'IoU = {iou:.3f}')
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close()

            st.markdown("### mAP (mean Average Precision)")
            st.markdown("""
            **mAP 변형들:**
            - **mAP@0.5**: IoU 임계값 0.5에서의 mAP
            - **mAP@0.5:0.95**: IoU 0.5부터 0.95까지 0.05 간격으로 평균
            - **mAP@small/medium/large**: 객체 크기별 mAP
            """)

    def render_iou_map(self):
        """IoU와 mAP 상세 설명"""
        st.header("🔍 IoU & mAP 심화")

        iou_tabs = st.tabs(["IoU 계산", "Precision-Recall", "mAP"])

        with iou_tabs[0]:
            st.subheader("IoU 계산 실습")

            st.code("""
def calculate_iou(box1, box2):
    # 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 각 박스의 면적
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 합집합 면적
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0
            """, language="python")

        with iou_tabs[1]:
            st.subheader("Precision-Recall 곡선")

            st.markdown("""
            ### Precision과 Recall
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
                st.info("정밀도: 예측한 것 중 실제로 맞은 비율")

            with col2:
                st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
                st.info("재현율: 실제 객체 중 찾아낸 비율")

            # PR 곡선 시뮬레이션
            st.markdown("#### PR 곡선 예시")

            # 샘플 데이터 생성
            recall = np.linspace(0, 1, 100)
            precision = 1 - recall * 0.3 + np.random.normal(0, 0.05, 100)
            precision = np.clip(precision, 0, 1)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, 'b-', linewidth=2)
            ax.fill_between(recall, 0, precision, alpha=0.3)
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title('Precision-Recall Curve', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # AP 계산 (곡선 아래 면적)
            ap = np.trapz(precision, recall)
            ax.text(0.6, 0.9, f'AP = {ap:.3f}', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            st.pyplot(fig)
            plt.close()

        with iou_tabs[2]:
            st.subheader("mAP 계산")

            st.markdown("""
            ### Average Precision (AP)

            AP는 Precision-Recall 곡선 아래의 면적입니다.
            """)

            st.code("""
def calculate_ap(precisions, recalls):
    # 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):  # 0.0, 0.1, ..., 1.0
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap

def calculate_map(all_aps):
    # 모든 클래스의 AP 평균
    return np.mean(all_aps)
            """, language="python")

            st.markdown("### mAP@0.5:0.95")
            st.markdown("""
            COCO 데이터셋에서 사용하는 주요 지표:
            - IoU 임계값을 0.5부터 0.95까지 0.05 간격으로 변경
            - 각 임계값에서 AP 계산
            - 모든 AP의 평균 계산
            """)

    def render_rcnn(self):
        """R-CNN 계열 설명"""
        st.header("🏗️ R-CNN 계열의 발전")

        rcnn_tabs = st.tabs(["R-CNN", "Fast R-CNN", "Faster R-CNN", "비교"])

        with rcnn_tabs[0]:
            st.subheader("R-CNN (2014)")

            st.markdown("""
            ### 핵심 아이디어
            1. **Region Proposal**: Selective Search로 객체가 있을 만한 영역 제안
            2. **CNN Feature Extraction**: 각 영역에서 CNN으로 특징 추출
            3. **Classification**: SVM으로 객체 분류
            """)

            st.code("""
┌─────────────────────────────────────────────────────────────┐
│                     R-CNN 구조 (2014)                        │
└─────────────────────────────────────────────────────────────┘

입력 이미지
    │
    ▼
┌───────────────────┐
│ Selective Search  │  ← 2000개 Region Proposal 생성
└───────────────────┘
    │
    ▼
┌───────────────────┐
│  Region Warping   │  ← 각 영역을 227×227로 크기 조정
└───────────────────┘
    │
    ▼
┌───────────────────┐
│   CNN (AlexNet)   │  ← 각 영역마다 4096차원 특징 추출
└───────────────────┘
    │
    ├─────────────────┬─────────────────┐
    ▼                 ▼                 ▼
┌────────┐      ┌────────┐      ┌──────────────┐
│ SVM #1 │      │ SVM #2 │ ...  │ Bbox Regress │
└────────┘      └────────┘      └──────────────┘
    │                                   │
    └───────────────┬───────────────────┘
                    ▼
            최종 탐지 결과
""", language="text")

            st.warning("""
            **R-CNN의 한계:**
            - ⏱️ 속도: 이미지당 47초 (GPU 기준)
            - 💾 메모리: 각 영역마다 CNN 연산 필요
            - 🔧 복잡성: 3단계 파이프라인
            """)

        with rcnn_tabs[1]:
            st.subheader("Fast R-CNN (2015)")

            st.markdown("""
            ### 주요 개선사항
            1. **전체 이미지 CNN**: 이미지 전체에 한 번만 CNN 적용
            2. **RoI Pooling**: 다양한 크기의 영역을 고정 크기로 변환
            3. **Multi-task Loss**: 분류와 바운딩 박스 회귀를 동시에 학습
            """)

            st.code("""
┌─────────────────────────────────────────────────────────────┐
│                   Fast R-CNN 구조 (2015)                     │
└─────────────────────────────────────────────────────────────┘

입력 이미지
    │
    ├──────────────────┐
    │                  │
    ▼                  ▼
┌────────┐     ┌──────────────┐
│  CNN   │     │   Selective  │
│(VGG16) │     │    Search    │
└────────┘     └──────────────┘
    │                  │
    │      Feature Map │
    │                  │
    └────────┬─────────┘
             ▼
    ┌─────────────────┐
    │  RoI Pooling    │  ← 각 Region을 7×7 고정 크기로
    └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │   FC Layers     │
    └─────────────────┘
             │
             ├──────────────┐
             ▼              ▼
    ┌──────────────┐  ┌─────────────┐
    │ Softmax (분류)│  │ Bbox Regress│
    └──────────────┘  └─────────────┘
             │              │
             └──────┬───────┘
                    ▼
            최종 탐지 결과
""", language="text")

            st.success("""
            **성능 개선:**
            - 속도: 이미지당 2.3초 (9배 빠름)
            - 정확도: mAP 66% (R-CNN 대비 향상)
            """)

            st.code("""
# RoI Pooling 핵심 개념
def roi_pooling(feature_map, roi, output_size=(7, 7)):
    x1, y1, x2, y2 = roi
    roi_feature = feature_map[:, :, y1:y2, x1:x2]
    pooled = adaptive_max_pool2d(roi_feature, output_size)
    return pooled
            """, language="python")

        with rcnn_tabs[2]:
            st.subheader("Faster R-CNN (2015)")

            st.markdown("""
            ### 혁신적 아이디어: RPN (Region Proposal Network)

            Selective Search를 신경망으로 대체!
            """)

            st.code("""
┌─────────────────────────────────────────────────────────────┐
│                 Faster R-CNN 구조 (2015)                     │
└─────────────────────────────────────────────────────────────┘

입력 이미지
    │
    ▼
┌────────────────┐
│  CNN (VGG16)   │
└────────────────┘
    │
    │ Feature Map
    │
    ├────────────────────┬─────────────────┐
    ▼                    ▼                 │
┌────────────┐    ┌──────────────┐        │
│    RPN     │    │  RoI Pooling │◄───────┘
│(Region     │    │              │
│ Proposal   │    └──────────────┘
│ Network)   │            │
└────────────┘            ▼
    │            ┌─────────────────┐
    │            │   FC Layers     │
    │            └─────────────────┘
    │                     │
    │                     ├──────────────┐
    │                     ▼              ▼
    │            ┌──────────────┐  ┌─────────────┐
    └───────────►│ Softmax (분류)│  │ Bbox Regress│
                 └──────────────┘  └─────────────┘
                         │              │
                         └──────┬───────┘
                                ▼
                        최종 탐지 결과

【RPN 상세 구조】
Feature Map → 3×3 Conv → 1×1 Conv ─┬─ 2k scores (obj/not obj)
                                   └─ 4k coords (bbox regression)
                                      ↓
                                   9개 Anchors per position
""", language="text")

            st.markdown("""
            #### 앵커 (Anchor) 개념
            - 특징 맵의 각 위치에 미리 정의된 박스들을 배치
            - 3개 스케일 × 3개 비율 = 9개 앵커 per position
            """)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**스케일**\n8, 16, 32")
            with col2:
                st.info("**비율**\n0.5, 1.0, 2.0")
            with col3:
                st.info("**앵커 수**\n9개/위치")

            st.success("""
            **성능 개선:**
            - 속도: 이미지당 0.2초 (실시간 처리 가능!)
            - 정확도: mAP 73.2%
            - End-to-End: 전체 네트워크를 한 번에 학습
            """)

        with rcnn_tabs[3]:
            st.subheader("R-CNN 계열 비교")

            comparison_data = {
                "모델": ["R-CNN", "Fast R-CNN", "Faster R-CNN"],
                "속도 (초/이미지)": [47, 2.3, 0.2],
                "mAP (%)": [62, 66, 73.2],
                "Region Proposal": ["Selective Search", "Selective Search", "RPN"],
                "End-to-End": ["❌", "부분", "✅"]
            }

            st.table(comparison_data)

            st.markdown("### Two-stage Detector의 특징")

            col1, col2 = st.columns(2)

            with col1:
                st.success("""
                **장점**
                - 높은 정확도
                - 안정적 성능
                - 작은 객체 탐지
                """)

            with col2:
                st.warning("""
                **단점**
                - 느린 속도
                - 복잡한 구조
                - 높은 메모리 사용량
                """)

    def render_yolo(self):
        """YOLO 아키텍처 설명"""
        st.header("⚡ YOLO (You Only Look Once)")

        yolo_tabs = st.tabs(["YOLOv1", "YOLOv2/v3", "YOLOv4/v5", "YOLOv8"])

        with yolo_tabs[0]:
            st.subheader("YOLOv1 (2016): 혁신의 시작")

            st.markdown("""
            ### 핵심 개념
            > "객체 탐지를 회귀 문제로!"

            - 이미지를 S×S 그리드로 분할 (S=7)
            - 각 그리드 셀이 B개의 바운딩 박스 예측 (B=2)
            - 한 번의 forward pass로 모든 객체 탐지
            """)

            # YOLO 그리드 시각화
            st.markdown("#### 그리드 분할 시각화")

            grid_size = st.slider("그리드 크기 (S×S)", 3, 13, 7)

            fig, ax = plt.subplots(figsize=(8, 8))

            # 그리드 그리기
            for i in range(grid_size + 1):
                ax.axhline(i, color='gray', linewidth=0.5)
                ax.axvline(i, color='gray', linewidth=0.5)

            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_aspect('equal')
            ax.set_title(f'{grid_size}×{grid_size} Grid', fontsize=14)
            ax.invert_yaxis()

            st.pyplot(fig)
            plt.close()

            st.markdown("""
            ### YOLO 출력 텐서
            - 크기: S × S × (B × 5 + C)
            - 7 × 7 × 30 (S=7, B=2, C=20)
            - 각 박스: [x, y, w, h, confidence]
            - 각 셀: C개의 클래스 확률
            """)

        with yolo_tabs[1]:
            st.subheader("YOLOv2/v3: 성능 개선")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### YOLOv2 (2017)")
                st.info("""
                **주요 개선:**
                - Batch Normalization
                - High Resolution (448×448)
                - Anchor Boxes 도입
                - K-means 앵커 클러스터링
                """)

            with col2:
                st.markdown("### YOLOv3 (2018)")
                st.success("""
                **주요 개선:**
                - 다중 스케일 검출 (3개)
                - Darknet-53 백본
                - Feature Pyramid Network
                - 9개 앵커 박스
                """)

            st.markdown("#### 다중 스케일 검출")

            scales = {
                "13×13": "큰 객체",
                "26×26": "중간 객체",
                "52×52": "작은 객체"
            }

            cols = st.columns(3)
            for i, (scale, description) in enumerate(scales.items()):
                with cols[i]:
                    st.metric(scale, description)

        with yolo_tabs[2]:
            st.subheader("YOLOv4/v5: 최적화")

            st.markdown("### YOLOv4 (2020)")
            st.markdown("""
            **주요 기술:**
            1. **CSPDarknet53**: Cross Stage Partial 연결
            2. **PANet**: Path Aggregation Network
            3. **Mosaic Augmentation**: 4개 이미지 조합
            4. **CIoU Loss**: Complete IoU 손실
            """)

            st.markdown("### YOLOv5 (2020)")
            st.markdown("""
            **실용성 강화:**
            - PyTorch 구현
            - AutoAnchor (자동 앵커 최적화)
            - Model Scaling (n, s, m, l, x)
            - 쉬운 사용성
            """)

            # YOLOv5 모델 크기 비교
            st.markdown("#### YOLOv5 모델 스케일")

            model_sizes = {
                "모델": ["YOLOv5n", "YOLOv5s", "YOLOv5m", "YOLOv5l", "YOLOv5x"],
                "파라미터 (M)": [1.9, 7.2, 21.2, 46.5, 86.7],
                "FLOPs (G)": [4.5, 16.5, 49.0, 109.1, 205.7],
                "속도 (ms)": [6.3, 6.4, 8.2, 10.1, 12.1]
            }

            st.table(model_sizes)

        with yolo_tabs[3]:
            st.subheader("YOLOv8 (2023): 최신 기술")

            st.markdown("""
            ### 혁신적 개선
            1. **Anchor-Free**: 앵커 박스 없이 직접 예측
            2. **Decoupled Head**: 분류와 회귀 헤드 분리
            3. **C2f 모듈**: 새로운 백본 구조
            4. **Advanced Augmentation**: MixUp, CutMix
            """)

            st.success("""
            **주요 특징:**
            - 더 빠른 속도
            - 더 높은 정확도
            - 쉬운 학습 및 배포
            - 다양한 태스크 지원 (Detection, Segmentation, Classification, Pose)
            """)

            st.code("""
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 학습
model.train(data='dataset.yaml', epochs=100)

# 추론
results = model.predict('image.jpg')
            """, language="python")

    def render_nms(self):
        """NMS 설명 및 시뮬레이션"""
        st.header("🎨 NMS (Non-Maximum Suppression)")

        st.markdown("""
        ### 🤔 문제 상황

        **예시**: 사진 속 강아지를 탐지할 때
        - AI가 같은 강아지에게 박스를 **10개나 그림** 📦📦📦📦📦
        - 겹치는 박스들이 너무 많아서 지저분함

        ### 💡 NMS가 하는 일

        **"중복된 박스를 정리해서 가장 좋은 박스 1개만 남긴다!"**

        **비유**:
        - 시험 답안지에 같은 답을 10번 썼는데, 가장 잘 쓴 1개만 남기고 나머지는 지우는 것
        - 친구 사진을 10장 찍었는데, 가장 잘 나온 1장만 골라서 인스타에 올리는 것
        """)

        st.info("""
        ### 🔥 2025년 현재도 사용하나요?

        **네, 여전히 핵심 기술입니다!** ✅

        **왜 규칙 기반인데도 사용하는가?**

        1️⃣ **딥러닝의 한계**
        - AI 모델은 "예측"만 담당 (여러 박스 출력)
        - **정리는 규칙**이 더 빠르고 정확함

        2️⃣ **실전에서 여전히 사용 중**
        - ✅ YOLOv8, YOLOv9, YOLOv10 (2024)
        - ✅ DETR 계열 (Facebook AI)
        - ✅ Detectron2 (Meta)
        - ✅ MMDetection (OpenMMLab)

        3️⃣ **속도가 중요**
        - NMS: 0.001초 (초고속) ⚡
        - 딥러닝 후처리: 0.1초 (100배 느림) 🐢

        **비유**: 계산기 vs AI
        - 덧셈/뺄셈: 계산기(규칙)가 더 빠르고 정확
        - 얼굴 인식: AI가 필요
        → **적재적소!**
        """)

        st.markdown("""
        ### 📝 3단계 알고리즘

        1️⃣ **점수 순으로 정렬**: 신뢰도가 높은 박스부터 나열
        2️⃣ **1등 박스 선택**: 가장 확신하는 박스를 먼저 선택
        3️⃣ **비슷한 박스 제거**: 1등과 너무 겹치는 박스들은 삭제
        4️⃣ **반복**: 남은 박스들 중에서 다시 1~3 반복
        """)

        nms_tabs = st.tabs(["기본 NMS", "Soft NMS", "DIoU NMS"])

        with nms_tabs[0]:
            st.subheader("기본 NMS 알고리즘")

            st.info("""
            **💡 실생활 예시로 이해하기**

            **상황**: 10명이 같은 문제의 답을 제출했어요
            - 점수: 95점, 92점, 90점, 88점, 85점... (모두 비슷한 답)

            **NMS 과정**:
            1. 가장 높은 점수(95점) 선택 ✅
            2. 95점 답과 너무 비슷한 답들(92, 90, 88점) 모두 삭제 ❌
            3. 남은 답 중에서 다시 1번으로 돌아가기

            **결과**: 서로 다른 영역의 객체만 남음!
            """)

            st.code("""
def non_max_suppression(detections, iou_threshold=0.5):
    # 1️⃣ 점수 높은 순으로 정렬 (1등부터 꼴등까지)
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    keep = []  # 최종 결과 저장

    while detections:  # 박스가 남아있는 동안 반복
        # 2️⃣ 1등 박스 선택
        best = detections.pop(0)  # 가장 앞에 있는 = 점수 1등
        keep.append(best)         # 최종 결과에 저장

        # 3️⃣ 1등과 겹치는 박스들 제거
        remaining = []
        for det in detections:
            iou = calculate_iou(best['bbox'], det['bbox'])

            # IoU가 낮으면 = 다른 객체 → 남김
            if iou <= iou_threshold:
                remaining.append(det)
            # IoU가 높으면 = 같은 객체 → 버림

        detections = remaining  # 남은 박스로 업데이트

    return keep  # 중복 제거된 최종 박스들
            """, language="python")

            st.markdown("""
            **🔍 IoU 임계값이란?**
            - **0.5**: 박스가 50% 이상 겹치면 "같은 객체"로 판단 → 삭제
            - **높을수록** (0.7, 0.9): 더 많이 겹쳐야 삭제 → 박스가 더 많이 남음
            - **낮을수록** (0.3): 조금만 겹쳐도 삭제 → 박스가 적게 남음
            """)

            st.markdown("#### NMS 시뮬레이션")

            iou_threshold = st.slider("IoU 임계값", 0.0, 1.0, 0.5, 0.05)

            # 샘플 검출 결과 생성
            detections = [
                {"bbox": [100, 100, 200, 200], "confidence": 0.9},
                {"bbox": [105, 95, 205, 195], "confidence": 0.85},
                {"bbox": [98, 102, 198, 202], "confidence": 0.8},
                {"bbox": [300, 150, 400, 250], "confidence": 0.95},
            ]

            st.write(f"원본 검출 개수: {len(detections)}")

            # NMS 적용
            filtered = self.apply_nms(detections, iou_threshold)
            st.write(f"NMS 후 검출 개수: {len(filtered)}")

            # 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Before NMS
            ax1.set_xlim(0, 500)
            ax1.set_ylim(0, 300)
            ax1.set_title('Before NMS')
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor='red', facecolor='none',
                                    label=f"conf={det['confidence']:.2f}")
                ax1.add_patch(rect)
            ax1.invert_yaxis()

            # After NMS
            ax2.set_xlim(0, 500)
            ax2.set_ylim(0, 300)
            ax2.set_title('After NMS')
            for det in filtered:
                x1, y1, x2, y2 = det['bbox']
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor='green', facecolor='none',
                                    label=f"conf={det['confidence']:.2f}")
                ax2.add_patch(rect)
            ax2.invert_yaxis()

            st.pyplot(fig)
            plt.close()

        with nms_tabs[1]:
            st.subheader("Soft NMS")

            st.markdown("""
            ### 😱 기본 NMS의 문제점

            **상황**: 사람들이 옹기종기 모여있는 사진
            - 사람 A와 사람 B가 겹쳐있음
            - 기본 NMS: "겹치니까 B는 삭제!" → **실제로 있는 사람도 삭제됨** ❌

            ### 💡 Soft NMS의 똑똑한 해결책

            **"바로 삭제하지 말고, 점수만 낮춰주자!"**

            **비유**:
            - 기본 NMS: "탈락!" (0점 처리)
            - Soft NMS: "음... 좀 애매하니까 감점!" (90점 → 60점으로 낮춤)

            **장점**:
            - 실제로 다른 객체인데 겹친 경우 → 점수는 낮지만 살아남음 ✅
            - 나중에 점수 순으로 다시 선택 가능
            """)

            st.code("""
# 기본 NMS
if iou > 0.5:
    박스.삭제()  # 무조건 0점 처리 → 영영 사라짐 💀

# Soft NMS
if iou > 0.5:
    박스.점수 = 박스.점수 × (1 - iou)  # 점수만 깎음 📉
    # 예: 0.9점 → 0.9 × (1 - 0.6) = 0.36점
    #     완전히 사라지진 않음!
""")

            st.code("""
def soft_nms(detections, sigma=0.5):
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            iou = calculate_iou(detections[i]['bbox'],
                              detections[j]['bbox'])

            if iou > threshold:
                # 가우시안 가중치 적용
                weight = np.exp(-(iou ** 2) / sigma)
                detections[j]['confidence'] *= weight

    return detections
            """, language="python")

        with nms_tabs[2]:
            st.subheader("DIoU NMS (고급)")

            st.markdown("""
            ### 🎯 DIoU = Distance-IoU (거리를 고려한 IoU)

            **문제 상황**:
            - 두 박스가 겹치는 정도는 같은데
            - 하나는 바로 옆에 붙어있고, 하나는 멀리 떨어져 있음
            - 기본 IoU: 둘 다 똑같이 판단 😕

            ### 💡 DIoU의 개선

            **"겹치는 정도 + 중심점 간 거리를 함께 고려하자!"**

            **비유**:
            - 기본 IoU: 두 사람이 얼마나 겹치는가만 봄
            - DIoU: 겹치는 정도 + 두 사람 사이 거리도 봄

            **실제 효과**:
            - 중심점이 가까우면 → 같은 객체일 확률 높음
            - 중심점이 멀면 → 다른 객체일 확률 높음
            """)

            st.code("""
# 기본 IoU
겹치는_면적 / 합친_면적  # 거리는 무시

# DIoU
기본_IoU - (중심점_거리² / 대각선_거리²)
           ↑ 이 값이 클수록 패널티

예시:
박스A, 박스B: IoU = 0.6, 중심점 거리 = 10픽셀  → DIoU 높음 ✅
박스A, 박스C: IoU = 0.6, 중심점 거리 = 100픽셀 → DIoU 낮음 ✅
→ A와 C는 다른 객체로 판단!
""", language="python")

            st.info("""
            **🎓 요약**
            - 기본 NMS: 단순하고 빠름
            - Soft NMS: 겹친 객체도 살림 (사람 여러명)
            - DIoU NMS: 거리까지 고려 (더 정확)
            """)

    def render_projects(self):
        """실전 프로젝트"""
        st.header("💻 실전 프로젝트")

        project_tabs = st.tabs([
            "교실 물건 탐지",
            "얼굴 감지",
            "차량 번호판 인식",
            "손동작 인식"
        ])

        with project_tabs[0]:
            self.classroom_detector_project()

        with project_tabs[1]:
            self.face_detection_project()

        with project_tabs[2]:
            self.license_plate_project()

        with project_tabs[3]:
            self.hand_gesture_project()

    def classroom_detector_project(self):
        """교실 물건 탐지 프로젝트 - 실제 YOLOv8 모델 사용"""
        st.subheader("🏫 교실 물건 탐지기")

        with st.expander("📚 이론적 배경: YOLOv8과 실시간 객체 탐지", expanded=False):
            st.markdown("""
            ### YOLOv8 아키텍처

            **YOLOv8**은 Ultralytics가 2023년 출시한 최신 YOLO 시리즈입니다.

            #### 핵심 개선 사항
            1. **Anchor-free 설계**
               - 기존 YOLO의 앵커 박스 제거
               - 객체 중심점을 직접 예측
               - 더 빠르고 정확한 탐지

            2. **CSPNet + C2f 모듈**
               - Cross Stage Partial Networks로 효율적 특징 추출
               - C2f (Coarse-to-Fine) 모듈로 다중 스케일 특징 융합

            3. **Task-aligned Head**
               - 분류와 위치 예측을 독립적으로 최적화
               - TaskAlignedAssigner로 더 정확한 타겟 할당

            #### COCO 데이터셋
            - **80개 클래스**: 일상적 객체 (사람, 동물, 교통수단, 가구 등)
            - **330K 이미지**: 대규모 학습 데이터
            - **교실 관련 클래스**: book, laptop, chair, backpack, person, cell phone, cup 등

            #### 실시간 탐지 프로세스
            ```
            입력 이미지 → 전처리 (640×640) → YOLOv8 모델
                ↓
            특징 추출 (Backbone) → 특징 융합 (Neck)
                ↓
            탐지 헤드 → [Bounding Box + Class + Confidence]
                ↓
            NMS 적용 → 최종 탐지 결과
            ```

            #### YOLOv8 모델 변형
            - **YOLOv8n (Nano)**: 3.2M 파라미터 - 실시간 처리 (사용 중)
            - **YOLOv8s (Small)**: 11.2M 파라미터 - 균형잡힌 성능
            - **YOLOv8m (Medium)**: 25.9M 파라미터 - 고정확도
            - **YOLOv8l/x**: 대규모 모델 - 최고 정확도
            """)

        st.info("""
        💡 **실제 YOLOv8 모델 사용**: Ultralytics의 사전학습된 YOLOv8 모델로 객체를 탐지합니다.
        - 모델: `yolov8n.pt` (COCO 데이터셋 학습)
        - 80개 클래스 탐지 가능 (사람, 책, 노트북, 의자, 가방 등)
        """)

        # Step 1: 모델 준비
        st.markdown("### 1️⃣ 모델 준비")
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            st.warning(f"'{model_path}' 모델 파일이 없습니다. 아래 버튼을 눌러 다운로드하세요.")
            if st.button(f"⬇️ '{model_path}' 다운로드"):
                if self._ensure_yolo_model(model_path):
                    st.rerun()
        else:
            st.success(f"✅ '{model_path}' 모델이 준비되었습니다.")
            st.caption(f"위치: {os.path.abspath(model_path)}")

        st.markdown("---")
        st.markdown("### 2️⃣ 객체 탐지")

        # 모델이 있을 때만 이미지 업로드 활성화
        if os.path.exists(model_path):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **프로젝트 목표:**
                - YOLOv8로 실시간 객체 탐지
                - COCO 데이터셋 80개 클래스 인식
                - 바운딩 박스 + 신뢰도 표시

                **탐지 가능한 물건 (COCO 클래스):**
                - 📚 책 (book)
                - 💻 노트북 (laptop)
                - 🪑 의자 (chair)
                - 🎒 가방 (backpack)
                - 👤 사람 (person)
                - 📱 휴대폰 (cell phone)
                - ☕ 컵 (cup)
                """)

                uploaded_file = st.file_uploader(
                    "이미지 업로드",
                    type=['png', 'jpg', 'jpeg'],
                    key="classroom_upload"
                )

            with col2:
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="업로드된 이미지", use_container_width=True)

                    if st.button("🎯 YOLOv8으로 객체 탐지", key="classroom_detect", type="primary"):
                        if not os.path.exists(model_path):
                            st.error("모델 파일이 없습니다. 먼저 모델을 다운로드하세요.")
                            return

                        with st.spinner("YOLOv8 모델 로딩 및 객체 탐지 중..."):
                            try:
                                from ultralytics import YOLO
                                import matplotlib.pyplot as plt
                                import matplotlib.patches as patches
                                import matplotlib.cm as cm

                                model = YOLO(model_path)

                                # PIL Image를 numpy array로 변환
                                image_array = np.array(image)

                                # 객체 탐지 실행
                                results = model.predict(
                                    source=image_array,
                                    conf=0.25,  # 신뢰도 임계값
                                    iou=0.45,   # NMS IoU 임계값
                                    verbose=False
                                )[0]

                                st.success("✅ 탐지 완료!")

                                # 탐지 결과 통계
                                if results.boxes is not None and len(results.boxes) > 0:
                                    st.markdown(f"### 📊 탐지된 객체: {len(results.boxes)}개")

                                    # 결과 시각화
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    ax.imshow(image_array)

                                    boxes = results.boxes.xyxy.cpu().numpy()
                                    confidences = results.boxes.conf.cpu().numpy()
                                    class_ids = results.boxes.cls.cpu().numpy().astype(int)
                                    class_names = results.names

                                    # 색상 팔레트
                                    cmap = cm.get_cmap('tab20')

                                    # 바운딩 박스 그리기
                                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                                        x1, y1, x2, y2 = box
                                        class_name = class_names[class_id]
                                        color = cmap(class_id % 20 / 20)

                                        # 박스 그리기
                                        rect = patches.Rectangle(
                                            (x1, y1), x2 - x1, y2 - y1,
                                            linewidth=2, edgecolor=color, facecolor='none'
                                        )
                                        ax.add_patch(rect)

                                        # 레이블 그리기
                                        label = f"{class_name}: {conf:.2f}"
                                        ax.text(
                                            x1, y1 - 5, label,
                                            color='white',
                                            fontsize=10,
                                            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
                                        )

                                    ax.axis('off')
                                    st.pyplot(fig)
                                    plt.close()

                                    # 탐지 결과 상세 정보
                                    st.markdown("### 🔍 탐지 결과 상세")

                                    # 클래스별 그룹화
                                    class_counts = {}
                                    for class_id in class_ids:
                                        class_name = class_names[class_id]
                                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

                                    col_a, col_b = st.columns(2)

                                    with col_a:
                                        st.markdown("#### 클래스별 개수")
                                        for class_name, count in sorted(class_counts.items()):
                                            st.metric(class_name, count)

                                    with col_b:
                                        st.markdown("#### 개별 객체 정보")
                                        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                                            x1, y1, x2, y2 = box
                                            class_name = class_names[class_id]
                                            st.text(f"{i+1}. {class_name} - 신뢰도: {conf:.2%}")
                                            st.caption(f"   위치: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

                                else:
                                    st.warning("⚠️ 탐지된 객체가 없습니다. 다른 이미지를 시도해보세요.")

                                # 모델 정보
                                with st.expander("📊 YOLOv8 모델 정보"):
                                    st.markdown("""
                                    **모델**: YOLOv8n (Nano)
                                    - **파라미터**: 3.2M
                                    - **학습 데이터**: COCO 데이터셋 (80 클래스)
                                    - **입력 크기**: 640×640
                                    - **속도**: ~100 FPS (GPU)
                                    - **mAP50-95**: 37.3%

                                    **COCO 80 클래스**:
                                    - 사람, 자전거, 자동차, 오토바이, 비행기, 버스, 기차, 트럭, 보트
                                    - 의자, 소파, 침대, 식탁, 화장실, TV, 노트북, 마우스, 키보드
                                    - 핸드폰, 책, 시계, 꽃병, 가위, 곰 인형, 칫솔 등
                                    """)

                            except Exception as e:
                                st.error(f"❌ 모델 로딩 실패: {str(e)}")
                                st.info("""
                                **해결 방법:**
                                1. 인터넷 연결 확인 (모델 다운로드 필요)
                                2. 필요한 패키지 설치: `pip install ultralytics`
                                3. 충분한 디스크 공간 확인
                                """)
        else:
            st.info("⬆️ 먼저 위의 '1️⃣ 모델 준비' 섹션에서 모델을 다운로드하세요.")

        st.markdown("### 학습 코드")
        st.code("""
from ultralytics import YOLO

# 모델 학습
model = YOLO('yolov8n.pt')

results = model.train(
    data='classroom.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='classroom_detector',
    name='yolov8n_classroom'
)

# 추론
model = YOLO('best.pt')
results = model.predict('classroom.jpg')
        """, language="python")

    def face_detection_project(self):
        """얼굴 감지 프로젝트"""
        st.subheader("😊 얼굴 감지 시스템")

        # 이론적 배경 추가
        with st.expander("📚 이론적 배경: 얼굴 감지 기술", expanded=False):
            st.markdown("""
            ### 얼굴 감지 (Face Detection)

            얼굴 감지는 이미지 내에서 사람의 얼굴 영역을 찾아내는 기술입니다.

            #### 주요 알고리즘 발전 과정

            **1. Viola-Jones (2001)**
            - **Haar Cascade**: 간단한 특징으로 빠른 탐지
            - **Integral Image**: 효율적인 특징 계산
            - **AdaBoost**: 약한 분류기 조합
            - 장점: 빠른 속도, 실시간 처리 가능
            - 단점: 정면 얼굴만 잘 감지, 조명에 민감

            **2. HOG + SVM (2005)**
            - **HOG (Histogram of Oriented Gradients)**: 얼굴 윤곽 특징 추출
            - **SVM (Support Vector Machine)**: 분류
            - 장점: 다양한 각도 얼굴 감지
            - 단점: Haar보다 느림

            **3. MTCNN (2016)**
            - **Multi-task CNN**: 3단계 CNN 네트워크
            - **P-Net → R-Net → O-Net**: 점진적 정제
            - **얼굴 랜드마크 동시 예측**: 눈, 코, 입 좌표
            - 장점: 높은 정확도, 다양한 포즈/크기 감지
            - 단점: 3단계 처리로 속도 저하

            **4. RetinaFace (2020)**
            - **Single-stage Detector**: YOLO 스타일의 빠른 탐지
            - **Multi-task Learning**:
              - 얼굴 박스 예측
              - 5개 랜드마크 (양 눈, 코, 양쪽 입꼬리)
              - 3D 얼굴 정보
            - **Feature Pyramid**: 다중 스케일 특징 추출
            - 장점: 속도와 정확도 균형

            #### 얼굴 랜드마크 (Facial Landmarks)

            얼굴 내 주요 지점을 찾아 좌표로 표현:
            - **68 Points (dlib)**: 얼굴 윤곽, 눈썹, 눈, 코, 입
            - **5 Points (RetinaFace)**: 양 눈, 코 끝, 양쪽 입꼬리
            - **106/478 Points**: 더 정밀한 3D 얼굴 모델링

            **활용 분야:**
            - 얼굴 정렬 (Face Alignment)
            - 얼굴 인식 전처리
            - 표정 분석
            - AR 필터/마스크 적용

            #### 나이/성별 추정

            얼굴 감지 후 추가 CNN으로 추정:
            - **나이 추정**: 회귀 문제 (0-100세)
            - **성별 추정**: 이진 분류 (남/여)
            - **모델**: AgeNet, GenderNet (Caffe 기반)

            #### 실시간 얼굴 감지 파이프라인
            ```
            입력 이미지/영상 → 얼굴 감지 (RetinaFace/MTCNN)
                ↓
            Bounding Box + Confidence
                ↓
            얼굴 랜드마크 추출 (5 or 68 points)
                ↓
            [선택] 나이/성별 추정 CNN
                ↓
            시각화 + 결과 출력
            ```

            #### MediaPipe 대안 (2025)

            **더 정확한 얼굴 탐지/분석을 원한다면:**

            - **YOLO Face**: YOLOv8 기반, 다중 얼굴 고속 탐지
            - **RetinaFace**: 5개 랜드마크 + 3D 정보, SOTA 성능
            - **SCRFD**: 경량 실시간 얼굴 탐지 (MMDetection)
            - **Face Mesh (MediaPipe)**: 468개 상세 랜드마크

            **API vs 로컬 모델**:
            - Gemini API: 나이/감정/표정 자연어 분석
            - MediaPipe: 빠른 실시간 탐지, 정확한 좌표
            """)

        st.markdown("""
        ### 프로젝트 개요

        이미지 또는 비디오에서 사람의 얼굴을 실시간으로 감지합니다.

        **기능:**
        - 다중 얼굴 감지
        - 얼굴 랜드마크 (눈, 코, 입)
        - 나이/성별 추정 (선택)
        """)

        # 코드 예시 - MediaPipe
        with st.expander("💻 MediaPipe 얼굴 감지 코드", expanded=False):
            st.code("""
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# MediaPipe Face Detection 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Face Detection 모델 (full-range: 5m)
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# 이미지 로드
image = Image.open('your_image.jpg').convert('RGB')
image_np = np.array(image)

# 얼굴 탐지
results = face_detection.process(image_np)

# 결과 시각화
if results.detections:
    for detection in results.detections:
        mp_drawing.draw_detection(image_np, detection)

        # 바운딩 박스 및 신뢰도
        bbox = detection.location_data.relative_bounding_box
        confidence = detection.score[0]
        print(f"얼굴 탐지 신뢰도: {confidence:.2%}")

face_detection.close()
""", language="python")

        # 코드 예시 - Gemini API
        with st.expander("💻 Gemini API 얼굴 분석 코드", expanded=False):
            st.code("""
import google.generativeai as genai
from PIL import Image
import os

# API 키 설정
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Gemini 2.0 Flash 모델 사용
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# 이미지 로드
image = Image.open('your_image.jpg')

# 얼굴 분석 프롬프트
prompt = \"\"\"
이 이미지에서 모든 얼굴을 감지하고 각 얼굴에 대해:
1. 위치 (왼쪽/중앙/오른쪽, 위/중간/아래)
2. 대략적인 나이대
3. 표정/감정
4. 얼굴 특징 (안경 착용, 수염 등)

을 자세히 분석해주세요.
\"\"\"

# API 호출
response = model.generate_content([prompt, image])
print(response.text)
""", language="python")

        # 입력 방식 선택
        input_mode = st.radio(
            "입력 방식 선택",
            ["이미지 업로드", "웹캠 실시간"],
            key="face_input_mode",
            horizontal=True
        )

        if input_mode == "웹캠 실시간":
            st.info("💡 **웹캠으로 실시간 얼굴 탐지** - MediaPipe Face Detection 사용")

            try:
                from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
                import cv2
                import mediapipe as mp
                import numpy as np
                import av

                class FaceDetectionProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.mp_face_detection = mp.solutions.face_detection
                        self.mp_drawing = mp.solutions.drawing_utils
                        self.face_detection = self.mp_face_detection.FaceDetection(
                            model_selection=1,
                            min_detection_confidence=0.5
                        )

                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")

                        # RGB로 변환
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # 얼굴 탐지
                        results = self.face_detection.process(img_rgb)

                        # 결과 그리기
                        if results.detections:
                            for detection in results.detections:
                                self.mp_drawing.draw_detection(img, detection)

                        return av.VideoFrame.from_ndarray(img, format="bgr24")

                webrtc_streamer(
                    key="face_detection_webcam",
                    video_processor_factory=FaceDetectionProcessor,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False}
                )

                st.markdown("""
                **사용 방법:**
                1. "START" 버튼 클릭
                2. 카메라 권한 허용
                3. 얼굴이 실시간으로 탐지됩니다
                """)

            except ImportError:
                st.error("❌ streamlit-webrtc가 설치되지 않았습니다.")
                st.code("pip install streamlit-webrtc av", language="bash")

        else:
            # 기존 이미지 업로드 모드
            col1, col2 = st.columns(2)

            with col1:
                detection_method = st.radio(
                    "탐지 방법 선택",
                    ["MediaPipe Face Detection", "Gemini API"],
                    key="face_method"
                )

            with col2:
                uploaded_file = st.file_uploader(
                    "얼굴 이미지 업로드",
                    type=['png', 'jpg', 'jpeg'],
                    key="face_upload"
                )

                if uploaded_file:
                    image = Image.open(uploaded_file).convert('RGB')

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.image(image, caption="원본 이미지", use_container_width=True)

                    if st.button("👤 얼굴 감지 실행", key="face_detect", type="primary"):

                        # MediaPipe Face Detection 사용
                        if detection_method == "MediaPipe Face Detection":
                            with st.spinner("MediaPipe로 얼굴 탐지 중..."):
                                try:
                                    import mediapipe as mp
                                    import cv2
                                    import numpy as np
                                    import matplotlib.pyplot as plt
                                    import matplotlib.patches as patches

                                    # MediaPipe Face Detection 초기화
                                    mp_face_detection = mp.solutions.face_detection
                                    mp_drawing = mp.solutions.drawing_utils

                                    # Face Detection 모델
                                    face_detection = mp_face_detection.FaceDetection(
                                        model_selection=1,  # 0: short-range (2m), 1: full-range (5m)
                                        min_detection_confidence=0.5
                                    )

                                    # PIL to OpenCV
                                    image_np = np.array(image)
                                    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                                    # 얼굴 탐지
                                    results = face_detection.process(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

                                    if results.detections:
                                        st.success(f"✅ {len(results.detections)}개의 얼굴 탐지 완료!")

                                        # 시각화
                                        annotated_image = image_np.copy()
                                        h, w, _ = annotated_image.shape

                                        face_info = []

                                        for idx, detection in enumerate(results.detections):
                                            # 바운딩 박스 그리기
                                            mp_drawing.draw_detection(annotated_image, detection)

                                            # 바운딩 박스 좌표
                                            bbox = detection.location_data.relative_bounding_box
                                            x = int(bbox.xmin * w)
                                            y = int(bbox.ymin * h)
                                            width = int(bbox.width * w)
                                            height = int(bbox.height * h)

                                            # 신뢰도
                                            confidence = detection.score[0]

                                            # 6개 키포인트 (오른쪽 눈, 왼쪽 눈, 코 끝, 입, 오른쪽 귀, 왼쪽 귀)
                                            keypoints = []
                                            for keypoint in detection.location_data.relative_keypoints:
                                                keypoints.append({
                                                    'x': int(keypoint.x * w),
                                                    'y': int(keypoint.y * h)
                                                })

                                            face_info.append({
                                                "bbox": [x, y, width, height],
                                                "confidence": confidence,
                                                "keypoints": keypoints
                                            })

                                        with col_b:
                                            st.image(annotated_image, caption="얼굴 탐지 결과", use_container_width=True)

                                        # 결과 출력
                                        st.markdown("#### 탐지 결과")
                                        for i, info in enumerate(face_info):
                                            x, y, w, h = info['bbox']
                                            st.markdown(f"""
                                            **얼굴 #{i+1}**
                                            - 신뢰도: {info['confidence']:.2%}
                                            - 위치: [{x}, {y}, {x+w}, {y+h}]
                                            - 크기: {w}×{h} px
                                            """)

                                        # 키포인트 정보
                                        with st.expander("📊 얼굴 키포인트 (6개)"):
                                            keypoint_names = [
                                                "오른쪽 눈", "왼쪽 눈", "코 끝",
                                                "입 중앙", "오른쪽 귀", "왼쪽 귀"
                                            ]

                                            for idx, info in enumerate(face_info):
                                                st.markdown(f"**얼굴 #{idx+1} 키포인트:**")
                                                for i, kp in enumerate(info['keypoints']):
                                                    if i < len(keypoint_names):
                                                        st.caption(f"{keypoint_names[i]}: x={kp['x']}, y={kp['y']}")

                                        # 모델 정보
                                        with st.expander("📊 MediaPipe Face Detection 정보"):
                                            st.markdown("""
                                            **모델**: BlazeFace
                                            - **아키텍처**: SSD 변형, 경량화 모델
                                            - **탐지 범위**: Full-range (최대 5m)
                                            - **출력**:
                                              - 얼굴 바운딩 박스
                                              - 신뢰도 점수
                                              - 6개 얼굴 키포인트
                                            - **속도**: 실시간 (~200 FPS on GPU)

                                            **6개 키포인트:**
                                            1. 오른쪽 눈 중심
                                            2. 왼쪽 눈 중심
                                            3. 코 끝
                                            4. 입 중앙
                                            5. 오른쪽 귀 (귀와 얼굴 경계)
                                            6. 왼쪽 귀 (귀와 얼굴 경계)
                                            """)

                                    else:
                                        st.warning("⚠️ 이미지에서 얼굴을 탐지하지 못했습니다. 다른 이미지를 시도해보세요.")

                                    face_detection.close()

                                except ImportError:
                                    st.error("❌ MediaPipe가 설치되지 않았습니다.")
                                    st.code("pip install mediapipe opencv-python", language="bash")
                                except Exception as e:
                                    st.error(f"❌ 오류 발생: {str(e)}")

                        # Gemini API 사용
                        else:
                            api_key = os.getenv('GOOGLE_API_KEY')
                            if api_key and api_key != 'your_api_key_here':
                                with st.spinner("Gemini API로 얼굴 분석 중..."):
                                    try:
                                        genai.configure(api_key=api_key)
                                        model = genai.GenerativeModel('gemini-2.0-flash-exp')

                                        prompt = """
이 이미지에서 모든 얼굴을 감지하고 각 얼굴에 대해:
1. 위치 (왼쪽/중앙/오른쪽, 위/중간/아래)
2. 대략적인 나이대
3. 표정/감정
4. 얼굴 특징 (안경 착용, 수염 등)

을 자세히 분석해주세요.
                                        """

                                        response = model.generate_content([prompt, image])

                                        with col_b:
                                            st.success("✅ Gemini API 분석 완료!")
                                            st.markdown(response.text)

                                    except Exception as e:
                                        st.error(f"❌ API 오류: {str(e)}")
                            else:
                                st.warning("⚠️ GOOGLE_API_KEY가 설정되지 않았습니다.")

    def license_plate_project(self):
        """차량 번호판 인식"""
        st.subheader("🚗 차량 번호판 인식")

        # 이론적 배경 추가
        with st.expander("📚 이론적 배경: 번호판 인식 시스템 (ALPR)", expanded=False):
            st.markdown("""
            ### ALPR (Automatic License Plate Recognition)

            **ALPR**은 차량 번호판을 자동으로 읽는 컴퓨터 비전 시스템입니다.

            #### 3단계 파이프라인

            **Stage 1: 차량 탐지 (Vehicle Detection)**
            - **모델**: YOLOv8, Faster R-CNN
            - **목적**: 이미지에서 차량 위치 찾기
            - **COCO 클래스**: car, truck, bus, motorcycle
            - **출력**: 차량 바운딩 박스

            **Stage 2: 번호판 탐지 (License Plate Detection)**
            - **모델**: 특화된 YOLO 또는 CNN
            - **목적**: 차량 내 번호판 영역 정확히 찾기
            - **데이터**: 각국 번호판 형태에 맞춘 학습
            - **전처리**:
              - 원근 변환 (Perspective Transform)
              - 기울기 보정 (Deskewing)
              - 크기 정규화
            - **출력**: 번호판 바운딩 박스

            **Stage 3: OCR (Optical Character Recognition)**
            - **전통적 방법**:
              - 이진화 (Binarization)
              - 문자 분할 (Character Segmentation)
              - 템플릿 매칭
            - **딥러닝 방법**:
              - **CRNN (CNN + RNN + CTC)**: 문자 시퀀스 인식
              - **EasyOCR/PaddleOCR**: 사전학습 OCR 모델
              - **TrOCR (Transformer OCR)**: Transformer 기반 최신 기술
            - **출력**: 번호판 텍스트

            #### 번호판 특화 OCR 도전과제

            **1. 다양한 번호판 포맷**
            - 한국: 12가 3456, 서울12가3456
            - 미국: ABC-1234
            - 유럽: XX-123-YY
            → 국가별 정규표현식 필요

            **2. 이미지 품질 문제**
            - 모션 블러 (Motion Blur)
            - 조명 변화 (야간, 역광)
            - 번호판 오염/손상
            - 카메라 각도 (원근 왜곡)
            → 전처리와 데이터 증강 필수

            **3. 유사 문자 구분**
            - O (알파벳) vs 0 (숫자)
            - I (알파벳) vs 1 (숫자)
            - B vs 8, D vs 0
            → 문맥 기반 후처리 필요

            #### CRNN 아키텍처 (OCR의 핵심)

            ```
            입력 번호판 이미지 (H×W×3)
                ↓
            CNN Backbone (특징 추출)
                ↓
            Feature Map (1×W'×C)
                ↓
            Bidirectional LSTM (시퀀스 모델링)
                ↓
            CTC Loss (정렬 없는 학습)
                ↓
            출력 텍스트: "12가3456"
            ```

            **CTC (Connectionist Temporal Classification)**:
            - 문자 위치를 미리 알 필요 없음
            - 가변 길이 출력 가능
            - Blank 토큰으로 중복 제거

            #### 실전 ALPR 시스템 구현

            **오픈소스 라이브러리:**
            - **EasyOCR**: 80개 언어 지원, PyTorch 기반
            - **PaddleOCR**: 중국 바이두, PP-OCR 모델
            - **Tesseract**: 전통적 OCR, 번호판엔 부적합

            **성능 최적화:**
            - **추적 (Tracking)**: 여러 프레임 결과 결합
            - **앙상블**: 다중 OCR 모델 결과 투표
            - **정규표현식 필터**: 형식에 맞는 결과만 선택
            """)

        st.markdown("""
        ### 프로젝트 개요

        차량 번호판을 탐지하고 OCR로 번호를 인식합니다.

        **파이프라인:**
        1. 차량 탐지 (Vehicle Detection)
        2. 번호판 영역 탐지 (License Plate Detection)
        3. OCR로 번호 인식 (Text Recognition)
        """)

        use_api = st.checkbox("실제 Gemini API 사용", key="plate_api")

        uploaded_file = st.file_uploader(
            "차량 이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="plate_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", width='stretch')

            if st.button("번호판 인식 실행", key="plate_detect"):
                if use_api:
                    api_key = os.getenv('GOOGLE_API_KEY')
                    if api_key and api_key != 'your_api_key_here':
                        with st.spinner("번호판 인식 중..."):
                            try:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-2.0-flash-exp')

                                prompt = """
이 이미지에서:
1. 차량을 탐지하고
2. 번호판 위치를 찾고
3. 번호판의 숫자/문자를 읽어주세요.

번호판이 명확하지 않다면 그 이유도 설명해주세요.
                                """

                                response = model.generate_content([prompt, image])

                                st.success("✅ 인식 완료!")
                                st.write(response.text)

                            except Exception as e:
                                st.error(f"API 오류: {str(e)}")
                    else:
                        st.warning("⚠️ API Key가 설정되지 않았습니다.")
                else:
                    with st.spinner("시뮬레이션 인식 중..."):
                        st.success("✅ 시뮬레이션 완료!")
                        st.info("""
**인식 결과:**

차량: 승용차 (신뢰도 0.95)
번호판 위치: 전면 중앙
번호판 번호: 12가 3456

추가 정보:
- 차량 색상: 흰색
- 차량 타입: 세단
                        """)

    def hand_gesture_project(self):
        """손동작 인식"""
        st.subheader("✋ 손동작 인식")

        # 이론적 배경 추가
        with st.expander("📚 이론적 배경: 손 탐지 및 제스처 인식", expanded=False):
            st.markdown("""
            ### Hand Detection & Gesture Recognition

            손 탐지와 제스처 인식은 Human-Computer Interaction(HCI)의 핵심 기술입니다.

            #### 1. 손 탐지 (Hand Detection)

            **객체 탐지 기반 접근**
            - **YOLO/SSD**: 일반 객체 탐지 모델로 손 탐지
            - **데이터**: EgoHands, COCO (person 키포인트)
            - **문제점**: 손은 작고 배경과 비슷해 탐지 어려움

            **특화 모델**
            - **MediaPipe Hands (Google)**:
              - Palm Detection + Hand Landmark 2단계
              - 경량 모델로 모바일에서 실시간 동작
              - 21개 손 랜드마크 제공
            - **OpenPose Hand**:
              - 전신 포즈 추정의 확장
              - 21개 손 키포인트

            #### 2. 손 랜드마크 (Hand Landmarks)

            **MediaPipe 21개 랜드마크 구조:**
            ```
            0: 손목 (Wrist)
            1-4: 엄지 (Thumb)
            5-8: 검지 (Index)
            9-12: 중지 (Middle)
            13-16: 약지 (Ring)
            17-20: 새끼 (Pinky)
            ```

            **랜드마크로 추출 가능한 정보:**
            - **손가락 펼침 여부**: 관절 각도 계산
            - **손 방향**: 손목-중지 벡터
            - **손 크기**: 손목-중지 거리
            - **손 모양**: 손가락 간 각도 관계

            #### 3. 제스처 인식 방법

            **A. Rule-based (규칙 기반)**
            - 손가락 개수 세기
              - 펼쳐진 손가락 끝이 MCP(손등 관절)보다 위에 있으면 펼침
              - 예: 5개 → "보", 0개 → "주먹", 2개 → "가위"
            - 손 형태 패턴 매칭
              - 엄지+검지만 펼침 → "총"
              - 엄지+새끼만 펼침 → "샤카(Shaka)"
            - 장점: 빠르고 정확
            - 단점: 미리 정의된 제스처만 인식

            **B. Machine Learning (딥러닝)**
            - **입력**: 21개 랜드마크 좌표 (x, y, z) × 21 = 63차원
            - **모델**:
              - MLP (Multi-Layer Perceptron): 간단한 분류
              - LSTM/GRU: 시간 순서 제스처 (동적 제스처)
              - Transformer: 복잡한 시퀀스 제스처
            - **출력**: 제스처 클래스 (Rock, Paper, Scissors, OK, Peace 등)
            - 장점: 복잡하고 다양한 제스처 학습 가능
            - 단점: 학습 데이터 필요

            **C. Sequence-based (동적 제스처)**
            - 정적 제스처: 한 프레임 (예: 엄지척)
            - 동적 제스처: 여러 프레임 (예: 손 흔들기, 스와이프)
            - **DTW (Dynamic Time Warping)**: 시계열 패턴 매칭
            - **3D CNN / LSTM**: 비디오 시퀀스 학습

            #### 4. MediaPipe Hands 파이프라인

            ```
            입력 이미지 (RGB)
                ↓
            Palm Detection Model (손바닥 탐지)
                ↓
            Hand Bounding Box (손 영역)
                ↓
            Hand Landmark Model (21개 키포인트 회귀)
                ↓
            3D Hand Landmarks (x, y, z) × 21
                ↓
            [응용] 제스처 분류 / 손가락 카운팅
            ```

            **Palm Detection Model**:
            - **입력**: 전체 이미지
            - **출력**: 손바닥 중심 + 회전각 + 크기
            - **특징**: 손바닥은 손가락보다 덜 움직여 안정적

            **Hand Landmark Model**:
            - **입력**: Crop된 손 영역 (256×256)
            - **출력**: 21개 3D 좌표 + 손 존재 여부(handedness)
            - **특징**: 왼손/오른손 구분 가능

            #### 5. 실전 응용 예시

            **손가락 개수 세기 알고리즘:**
            ```python
            def count_fingers(landmarks):
                fingers = []

                # 엄지: x 좌표 비교 (좌우 반전 주의)
                if landmarks[4].x < landmarks[3].x:  # 오른손 기준
                    fingers.append(1)

                # 나머지 손가락: y 좌표 비교 (끝 < 관절)
                for id in [8, 12, 16, 20]:  # 검지, 중지, 약지, 새끼
                    if landmarks[id].y < landmarks[id-2].y:
                        fingers.append(1)

                return sum(fingers)
            ```

            **제스처 분류 데이터셋:**
            - **Jester**: 148K 비디오, 27개 제스처
            - **ASL (American Sign Language)**: 수화 알파벳
            - **Custom**: 직접 수집한 특정 도메인 제스처

            #### 6. MediaPipe 대안 라이브러리 (2025)

            **더 높은 성능을 원한다면?**

            | 라이브러리 | 속도 (FPS) | 정확도 | 다중인물 | 난이도 | 추천 용도 |
            |-----------|-----------|--------|---------|--------|----------|
            | **MediaPipe** | 200+ | 중상 | ❌ (1명) | 쉬움 | 실시간 단일 인물, 프로토타입 |
            | **MMPose** | 430+ | ⭐최고 | ✅ | 중간 | 연구, 고정밀도 필요 |
            | **YOLOv8 Pose** | 100+ | 상 | ✅ | 쉬움 | 다중 인물, 객체+포즈 동시 |
            | **OpenPose** | 15 | 중 | ✅ | 어려움 | 연구용 (레거시) |

            **MMPose (OpenMMLab)** - 2025년 SOTA:
            - RTMPose 모델: 430+ FPS (GTX 1660 Ti)
            - COCO 75.8% AP (MediaPipe보다 우수)
            - 손, 얼굴, 전신, 3D 포즈 모두 지원
            - PyTorch 기반, 크로스 플랫폼

            **YOLOv8/v7 Pose**:
            - 다중 인물 동시 추적 (MediaPipe는 1명만)
            - 객체 탐지 + 포즈 추정 통합
            - Ultralytics 패키지로 간편 사용

            **선택 가이드**:
            - 빠른 프로토타입, 학습용 → **MediaPipe** ✅
            - 최고 정확도, 연구 → **MMPose**
            - 다중 인물, 실무 → **YOLOv8 Pose**

            #### 7. API vs 로컬 모델 비교

            **Gemini API 장점**:
            - 복잡한 추론: "이 제스처의 의미는?"
            - 자연어 출력: 인간 친화적 설명
            - 맥락 이해: 나이, 감정 분석

            **MediaPipe/로컬 모델 장점**:
            - ✅ **무료**: API 비용 $0
            - ✅ **빠름**: 10-20배 빠른 속도
            - ✅ **프라이버시**: 데이터가 외부로 나가지 않음
            - ✅ **오프라인**: 인터넷 없이 동작
            - ✅ **정확한 좌표**: 21개 랜드마크 (x, y, z)
            - ✅ **실시간**: 비디오, 웹캠, AR/VR 가능

            **실무 하이브리드 전략**:
            1. MediaPipe로 빠른 랜드마크 추출
            2. 복잡한 경우만 Gemini API 호출
            → 비용 절감 + 성능 최적화
            """)

        st.markdown("""
        ### 프로젝트 개요

        손을 탐지하고 손가락 개수를 세거나 제스처를 인식합니다.

        **응용 분야:**
        - 가상 마우스
        - 수화 번역
        - 게임 컨트롤
        - 스마트홈 제어
        """)

        # 코드 예시 - MediaPipe
        with st.expander("💻 MediaPipe 손동작 인식 코드", expanded=False):
            st.code("""
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hands 모델 (최대 2개 손 탐지)
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# 이미지 로드
image = Image.open('your_image.jpg').convert('RGB')
image_np = np.array(image)

# 손 탐지
results = hands.process(image_np)

# 결과 시각화
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # 손 랜드마크 그리기 (21개 keypoints)
        mp_drawing.draw_landmarks(
            image_np,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # 손가락 개수 세기
        landmarks = hand_landmarks.landmark
        finger_count = 0

        # 엄지 (x 좌표 비교)
        if landmarks[4].x < landmarks[3].x:
            finger_count += 1

        # 나머지 손가락 (y 좌표 비교)
        for tip_id in [8, 12, 16, 20]:
            if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                finger_count += 1

        print(f"펼친 손가락 개수: {finger_count}")

hands.close()
""", language="python")

        # 코드 예시 - Gemini API
        with st.expander("💻 Gemini API 손동작 분석 코드", expanded=False):
            st.code("""
import google.generativeai as genai
from PIL import Image
import os

# API 키 설정
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Gemini 2.0 Flash 모델 사용
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# 이미지 로드
image = Image.open('your_image.jpg')

# 손동작 분석 프롬프트
prompt = \"\"\"
이 이미지에서 손을 분석하고:
1. 손 개수
2. 펼쳐진 손가락 개수
3. 손동작/제스처 (예: 가위, 바위, 보, 엄지척, V사인 등)
4. 손의 위치
5. 왼손/오른손 구분

를 자세히 알려주세요.
\"\"\"

# API 호출
response = model.generate_content([prompt, image])
print(response.text)
""", language="python")

        # 입력 방식 선택
        input_mode = st.radio(
            "입력 방식 선택",
            ["이미지 업로드", "웹캠 실시간"],
            key="hand_input_mode",
            horizontal=True
        )

        if input_mode == "웹캠 실시간":
            st.info("💡 **웹캠으로 실시간 손동작 탐지** - MediaPipe Hands 사용")

            try:
                from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
                import cv2
                import mediapipe as mp
                import numpy as np
                import av

                class HandDetectionProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.mp_hands = mp.solutions.hands
                        self.mp_drawing = mp.solutions.drawing_utils
                        self.mp_drawing_styles = mp.solutions.drawing_styles
                        self.hands = self.mp_hands.Hands(
                            static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5
                        )

                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")

                        # RGB로 변환
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # 손 탐지
                        results = self.hands.process(img_rgb)

                        # 결과 그리기
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                self.mp_drawing.draw_landmarks(
                                    img,
                                    hand_landmarks,
                                    self.mp_hands.HAND_CONNECTIONS,
                                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                    self.mp_drawing_styles.get_default_hand_connections_style()
                                )

                                # 손가락 개수 표시
                                # 간단한 카운팅 (엄지는 x좌표, 나머지는 y좌표 비교)
                                finger_count = 0
                                landmarks = hand_landmarks.landmark

                                # 엄지
                                if landmarks[4].x < landmarks[3].x:
                                    finger_count += 1

                                # 나머지 손가락
                                for tip_id in [8, 12, 16, 20]:
                                    if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                                        finger_count += 1

                                # 화면에 표시
                                cv2.putText(img, f"Fingers: {finger_count}", (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        return av.VideoFrame.from_ndarray(img, format="bgr24")

                webrtc_streamer(
                    key="hand_detection_webcam",
                    video_processor_factory=HandDetectionProcessor,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False}
                )

                st.markdown("""
                **사용 방법:**
                1. "START" 버튼 클릭
                2. 카메라 권한 허용
                3. 손을 카메라에 비추면 실시간으로 탐지되고 손가락 개수가 표시됩니다

                **제스처 테스트:**
                - ✊ 주먹: 0개
                - ☝️ 검지: 1개
                - ✌️ 가위/V사인: 2개
                - 🖐️ 보: 5개
                """)

            except ImportError:
                st.error("❌ streamlit-webrtc가 설치되지 않았습니다.")
                st.code("pip install streamlit-webrtc av", language="bash")

        else:
            # 기존 이미지 업로드 모드
            col1, col2 = st.columns(2)

            with col1:
                detection_method = st.radio(
                    "탐지 방법 선택",
                    ["MediaPipe Hand Landmarker", "Gemini API"],
                    key="hand_method"
                )

            with col2:
                uploaded_file = st.file_uploader(
                    "손동작 이미지 업로드",
                    type=['png', 'jpg', 'jpeg'],
                    key="hand_upload"
                )

                if uploaded_file:
                    image = Image.open(uploaded_file).convert('RGB')

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.image(image, caption="원본 이미지", use_container_width=True)

                    if st.button("🤚 손동작 인식 실행", key="hand_detect", type="primary"):

                        # MediaPipe 사용
                        if detection_method == "MediaPipe Hand Landmarker":
                            st.warning("""
                            ⚠️ **MediaPipe 설치 안내**

                            MediaPipe는 Python 3.13을 아직 지원하지 않습니다.

                            **해결 방법:**
                            1. Python 3.11 또는 3.12 환경 사용
                            2. 또는 Gemini API 방식 선택

                            **설치 명령 (Python 3.11/3.12):**
                            ```bash
                            pip install mediapipe opencv-python
                            ```
                            """)

                            with st.spinner("MediaPipe로 손 랜드마크 탐지 중..."):
                                try:
                                    import mediapipe as mp
                                    import cv2
                                    import numpy as np
                                    import matplotlib.pyplot as plt
                                    import matplotlib.patches as patches

                                    # MediaPipe Hands 초기화
                                    mp_hands = mp.solutions.hands
                                    mp_drawing = mp.solutions.drawing_utils
                                    mp_drawing_styles = mp.solutions.drawing_styles

                                    # Hands 모델 (static_image_mode=True for images)
                                    hands = mp_hands.Hands(
                                        static_image_mode=True,
                                        max_num_hands=2,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5
                                    )

                                    # PIL to OpenCV
                                    image_np = np.array(image)
                                    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                                    # 손 탐지
                                    results = hands.process(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

                                    if results.multi_hand_landmarks:
                                        st.success(f"✅ {len(results.multi_hand_landmarks)}개의 손 탐지 완료!")

                                        # 시각화
                                        annotated_image = image_np.copy()

                                        hand_info = []

                                        for idx, (hand_landmarks, handedness) in enumerate(zip(
                                            results.multi_hand_landmarks,
                                            results.multi_handedness
                                        )):
                                            # 손 랜드마크 그리기
                                            mp_drawing.draw_landmarks(
                                                annotated_image,
                                                hand_landmarks,
                                                mp_hands.HAND_CONNECTIONS,
                                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                                mp_drawing_styles.get_default_hand_connections_style()
                                            )

                                            # 손가락 개수 세기
                                            def count_fingers(landmarks):
                                                fingers = []

                                                # 엄지: x 좌표 비교
                                                if handedness.classification[0].label == "Right":
                                                    if landmarks[4].x < landmarks[3].x:
                                                        fingers.append(1)
                                                else:  # Left
                                                    if landmarks[4].x > landmarks[3].x:
                                                        fingers.append(1)

                                                # 나머지 손가락: y 좌표 비교
                                                tip_ids = [8, 12, 16, 20]  # 검지, 중지, 약지, 새끼
                                                for tip_id in tip_ids:
                                                    if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                                                        fingers.append(1)

                                                return sum(fingers)

                                            finger_count = count_fingers(hand_landmarks.landmark)
                                            hand_label = handedness.classification[0].label
                                            hand_score = handedness.classification[0].score

                                            # 제스처 인식 (간단한 규칙 기반)
                                            gesture = "알 수 없음"
                                            if finger_count == 0:
                                                gesture = "주먹 (Rock)"
                                            elif finger_count == 2:
                                                gesture = "가위 (Scissors) 또는 V사인"
                                            elif finger_count == 5:
                                                gesture = "보 (Paper)"
                                            elif finger_count == 1:
                                                gesture = "포인팅 또는 엄지척"

                                            hand_info.append({
                                                "hand": hand_label,
                                                "confidence": hand_score,
                                                "fingers": finger_count,
                                                "gesture": gesture
                                            })

                                            # 이미지에 손가락 개수와 제스처 표시
                                            h, w, _ = annotated_image.shape
                                            wrist = hand_landmarks.landmark[0]
                                            text_x = int(wrist.x * w)
                                            text_y = int(wrist.y * h) - 20

                                            # 텍스트 배경
                                            cv2.rectangle(
                                                annotated_image,
                                                (text_x - 10, text_y - 30),
                                                (text_x + 200, text_y + 10),
                                                (0, 0, 0),
                                                -1
                                            )

                                            # 손가락 개수 표시
                                            cv2.putText(
                                                annotated_image,
                                                f"Fingers: {finger_count}",
                                                (text_x, text_y - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 255, 0),
                                                2
                                            )

                                            # 제스처 표시
                                            cv2.putText(
                                                annotated_image,
                                                gesture,
                                                (text_x, text_y + 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (255, 255, 0),
                                                1
                                            )

                                        with col_b:
                                            st.image(annotated_image, caption="손 랜드마크 탐지 결과", use_container_width=True)

                                        # 결과 출력
                                        st.markdown("#### 탐지 결과")
                                        for i, info in enumerate(hand_info):
                                            st.markdown(f"""
                                            **손 #{i+1}**
                                            - 손: {info['hand']} (신뢰도: {info['confidence']:.2%})
                                            - 펼쳐진 손가락: {info['fingers']}개
                                            - 추정 제스처: {info['gesture']}
                                            """)

                                        # 랜드마크 좌표 정보
                                        with st.expander("📊 21개 손 랜드마크 좌표"):
                                            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                                st.markdown(f"**손 #{idx+1} 랜드마크:**")
                                                landmark_names = [
                                                    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
                                                    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
                                                    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
                                                    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                                                    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
                                                ]

                                                for i, landmark in enumerate(hand_landmarks.landmark):
                                                    st.caption(f"{i}: {landmark_names[i]} - x:{landmark.x:.3f}, y:{landmark.y:.3f}, z:{landmark.z:.3f}")

                                    else:
                                        st.warning("⚠️ 이미지에서 손을 탐지하지 못했습니다. 다른 이미지를 시도해보세요.")

                                    hands.close()

                                except ImportError:
                                    st.error("❌ MediaPipe가 설치되지 않았습니다.")
                                    st.code("pip install mediapipe opencv-python", language="bash")
                                except Exception as e:
                                    st.error(f"❌ 오류 발생: {str(e)}")

                        # Gemini API 사용
                        else:
                            api_key = os.getenv('GOOGLE_API_KEY')
                            if api_key and api_key != 'your_api_key_here':
                                with st.spinner("Gemini API로 손동작 분석 중..."):
                                    try:
                                        genai.configure(api_key=api_key)
                                        model = genai.GenerativeModel('gemini-2.0-flash-exp')

                                        prompt = """
이 이미지에서 손을 분석하고:
1. 손 개수
2. 펼쳐진 손가락 개수
3. 손동작/제스처 (예: 가위, 바위, 보, 엄지척, V사인 등)
4. 손의 위치
5. 왼손/오른손 구분

를 자세히 알려주세요.
                                        """

                                        response = model.generate_content([prompt, image])

                                        with col_b:
                                            st.success("✅ Gemini API 분석 완료!")
                                            st.markdown(response.text)

                                    except Exception as e:
                                        st.error(f"❌ API 오류: {str(e)}")
                            else:
                                st.warning("⚠️ GOOGLE_API_KEY가 설정되지 않았습니다.")

        st.markdown("### MediaPipe Hand Tracking")
        st.code("""
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# 이미지 처리
results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )
        """, language="python")

    # Helper methods
    def calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def apply_nms(self, detections, iou_threshold):
        """NMS 적용"""
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            remaining = []
            for det in detections:
                iou = self.calculate_iou(best['bbox'], det['bbox'])
                if iou <= iou_threshold:
                    remaining.append(det)

            detections = remaining

        return keep