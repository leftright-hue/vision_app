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
    def __init__(self):
        self.name = "Week 5: Object Detection & YOLO"

    def render(self):
        st.title("🎯 Week 5: 객체 탐지와 YOLO")
        st.markdown("**객체 탐지의 이론부터 YOLO 실전 구현까지**")

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

            st.image("https://via.placeholder.com/800x300/4CAF50/FFFFFF?text=R-CNN+Architecture",
                    caption="R-CNN 구조")

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
        ### NMS의 필요성

        객체 탐지 모델은 같은 객체에 대해 여러 개의 바운딩 박스를 예측할 수 있습니다.
        NMS는 중복된 검출을 제거하여 최선의 박스만 남깁니다.
        """)

        nms_tabs = st.tabs(["기본 NMS", "Soft NMS", "DIoU NMS"])

        with nms_tabs[0]:
            st.subheader("기본 NMS 알고리즘")

            st.code("""
def non_max_suppression(detections, iou_threshold=0.5):
    # 1. 신뢰도 기준 내림차순 정렬
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    keep = []
    while detections:
        # 2. 가장 높은 신뢰도 선택
        best = detections.pop(0)
        keep.append(best)

        # 3. IoU가 임계값 이상인 박스 제거
        remaining = []
        for det in detections:
            iou = calculate_iou(best['bbox'], det['bbox'])
            if iou <= iou_threshold:
                remaining.append(det)

        detections = remaining

    return keep
            """, language="python")

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
            ### 기본 NMS의 문제점

            겹쳐있는 여러 객체를 탐지할 때, 정상적인 검출도 제거될 수 있습니다.

            ### Soft NMS의 해결책

            IoU가 높은 박스의 신뢰도를 0으로 만들지 않고,
            가우시안 함수로 **부드럽게 감소**시킵니다.
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
            st.subheader("DIoU NMS")

            st.markdown("""
            ### Distance-IoU NMS

            기본 IoU에 **중심점 간 거리**를 추가로 고려합니다.
            """)

            st.latex(r"\text{DIoU} = \text{IoU} - \frac{d^2}{c^2}")

            st.markdown("""
            - **d**: 두 박스 중심점 간 거리
            - **c**: 두 박스를 포함하는 최소 박스의 대각선 길이
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
        """교실 물건 탐지 프로젝트"""
        st.subheader("🏫 교실 물건 탐지기")

        st.markdown("""
        ### 프로젝트 개요

        교실에서 흔히 볼 수 있는 물건들을 탐지하는 커스텀 YOLO 모델을 구축합니다.

        **탐지 대상:**
        - 📚 책 (Book)
        - 💻 노트북 (Laptop)
        - 🪑 의자 (Chair)
        - 🖊️ 칠판 (Whiteboard)
        - 🎒 가방 (Bag)
        """)

        st.markdown("### API 활용 객체 탐지")

        use_api = st.checkbox("실제 Gemini API 사용", key="classroom_api")

        uploaded_file = st.file_uploader(
            "교실 이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="classroom_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_container_width=True)

            if st.button("객체 탐지 실행", key="classroom_detect"):
                if use_api:
                    api_key = os.getenv('GOOGLE_API_KEY')
                    if api_key and api_key != 'your_api_key_here':
                        with st.spinner("Gemini로 객체 탐지 중..."):
                            try:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-2.5-pro')

                                prompt = """
이 교실 이미지에서 다음 물건들을 찾아주세요:
- 책 (Book)
- 노트북 (Laptop)
- 의자 (Chair)
- 칠판/화이트보드 (Whiteboard)
- 가방 (Bag)

각 물건에 대해:
1. 물건 이름
2. 대략적인 위치 (왼쪽/중앙/오른쪽, 위/중간/아래)
3. 신뢰도 (높음/중간/낮음)

형식으로 답변해주세요.
                                """

                                response = model.generate_content([prompt, image])

                                st.success("✅ 탐지 완료!")
                                st.markdown("### 탐지 결과")
                                st.write(response.text)

                            except Exception as e:
                                st.error(f"API 오류: {str(e)}")
                    else:
                        st.warning("⚠️ API Key가 설정되지 않았습니다.")
                else:
                    # 시뮬레이션
                    with st.spinner("시뮬레이션 탐지 중..."):
                        st.success("✅ 시뮬레이션 완료!")
                        st.markdown("### 탐지 결과 (시뮬레이션)")

                        detections = [
                            {"class": "책", "confidence": 0.92, "location": "중앙-위"},
                            {"class": "노트북", "confidence": 0.88, "location": "왼쪽-중간"},
                            {"class": "의자", "confidence": 0.95, "location": "오른쪽-아래"},
                        ]

                        for det in detections:
                            st.info(f"**{det['class']}** - 신뢰도: {det['confidence']:.2f} - 위치: {det['location']}")

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

        st.markdown("""
        ### 프로젝트 개요

        이미지 또는 비디오에서 사람의 얼굴을 실시간으로 감지합니다.

        **기능:**
        - 다중 얼굴 감지
        - 얼굴 랜드마크 (눈, 코, 입)
        - 나이/성별 추정 (선택)
        """)

        use_api = st.checkbox("실제 Gemini API 사용", key="face_api")

        uploaded_file = st.file_uploader(
            "얼굴 이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="face_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_container_width=True)

            if st.button("얼굴 감지 실행", key="face_detect"):
                if use_api:
                    api_key = os.getenv('GOOGLE_API_KEY')
                    if api_key and api_key != 'your_api_key_here':
                        with st.spinner("얼굴 감지 중..."):
                            try:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-2.5-pro')

                                prompt = """
이 이미지에서 모든 얼굴을 감지하고 각 얼굴에 대해:
1. 위치 (왼쪽/중앙/오른쪽, 위/중간/아래)
2. 대략적인 나이대
3. 표정/감정

을 분석해주세요.
                                """

                                response = model.generate_content([prompt, image])

                                st.success("✅ 감지 완료!")
                                st.write(response.text)

                            except Exception as e:
                                st.error(f"API 오류: {str(e)}")
                    else:
                        st.warning("⚠️ API Key가 설정되지 않았습니다.")
                else:
                    with st.spinner("시뮬레이션 감지 중..."):
                        st.success("✅ 시뮬레이션 완료!")
                        st.info("""
**감지된 얼굴: 2개**

얼굴 1:
- 위치: 중앙-위
- 연령대: 20-30대
- 표정: 미소

얼굴 2:
- 위치: 오른쪽-중간
- 연령대: 30-40대
- 표정: 중립
                        """)

    def license_plate_project(self):
        """차량 번호판 인식"""
        st.subheader("🚗 차량 번호판 인식")

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
            st.image(image, caption="업로드된 이미지", use_container_width=True)

            if st.button("번호판 인식 실행", key="plate_detect"):
                if use_api:
                    api_key = os.getenv('GOOGLE_API_KEY')
                    if api_key and api_key != 'your_api_key_here':
                        with st.spinner("번호판 인식 중..."):
                            try:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-2.5-pro')

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

        st.markdown("""
        ### 프로젝트 개요

        손을 탐지하고 손가락 개수를 세거나 제스처를 인식합니다.

        **응용 분야:**
        - 가상 마우스
        - 수화 번역
        - 게임 컨트롤
        - 스마트홈 제어
        """)

        use_api = st.checkbox("실제 Gemini API 사용", key="hand_api")

        uploaded_file = st.file_uploader(
            "손동작 이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="hand_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_container_width=True)

            if st.button("손동작 인식 실행", key="hand_detect"):
                if use_api:
                    api_key = os.getenv('GOOGLE_API_KEY')
                    if api_key and api_key != 'your_api_key_here':
                        with st.spinner("손동작 인식 중..."):
                            try:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-2.5-pro')

                                prompt = """
이 이미지에서 손을 분석하고:
1. 손 개수
2. 펼쳐진 손가락 개수
3. 손동작/제스처 (예: 가위, 바위, 보, 엄지척, V사인 등)
4. 손의 위치

를 알려주세요.
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

손 개수: 1개
펼쳐진 손가락: 2개
제스처: V사인 (평화)
손 위치: 중앙
신뢰도: 0.94
                        """)

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