"""
Lab 5: Practical Applications
- 자동 라벨링 도구
- 객체 카운팅 시스템
- SAM 응용 사례 종합
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from modules.week06.sam_helpers import get_sam_helper


def run():
    st.title("💼 Lab 5: Practical Applications")

    st.markdown("""
    ## 학습 목표
    - 자동 라벨링 시스템 구축
    - 객체 카운팅 응용
    - 실전 프로젝트 사례 학습
    """)

    tabs = st.tabs([
        "1️⃣ 자동 라벨링 도구",
        "2️⃣ 객체 카운팅 시스템",
        "3️⃣ 프로젝트 아이디어"
    ])

    with tabs[0]:
        demo_auto_labeling()

    with tabs[1]:
        demo_counting_system()

    with tabs[2]:
        show_project_ideas()


def demo_auto_labeling():
    """자동 라벨링 도구"""
    st.header("1️⃣ 자동 라벨링 도구")

    st.markdown("""
    ### 목적: 객체 탐지 학습 데이터 생성 자동화

    **전통적 방법**:
    - 수작업 라벨링: 이미지당 5-30분
    - 비용: 이미지당 $0.5-$5
    - 오류율: 5-10%

    **SAM 활용**:
    - 반자동 라벨링: 이미지당 30초-2분
    - 비용: 인건비 대폭 절감
    - 일관성: 향상된 품질

    ### 워크플로우
    1. 자동 마스크 생성
    2. 마스크 검토 및 필터링
    3. 클래스 레이블 할당
    4. BBox 추출
    5. COCO/YOLO 포맷 내보내기
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b"], key="label_model")
    sam = get_sam_helper(model_type)

    uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], key="label_upload")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="원본 이미지", use_container_width=True)

        if st.button("🤖 자동 마스크 생성"):
            with st.spinner("처리 중..."):
                masks = sam.generate_auto_masks(image, points_per_side=24)
                st.session_state.labeling_masks = masks
                st.success(f"✅ {len(masks)}개 후보 생성")

        if 'labeling_masks' in st.session_state:
            masks = st.session_state.labeling_masks

            st.markdown("### 레이블 할당")

            # 클래스 정의
            class_input = st.text_input("클래스 목록 (쉼표 구분)", "person,car,dog,cat,bicycle")
            classes = [c.strip() for c in class_input.split(",")]

            # 초기화
            if 'assigned_labels' not in st.session_state:
                st.session_state.assigned_labels = {}

            # 레이블 할당 UI
            num_display = st.slider("표시할 마스크 수", 5, min(30, len(masks)), 10)

            for i, mask_data in enumerate(masks[:num_display]):
                with st.expander(f"Mask {i} (Area: {mask_data['area']}px²)", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        # 미리보기
                        preview = create_mask_preview(image, mask_data)
                        st.image(preview, use_container_width=True)

                    with col2:
                        label = st.selectbox(
                            "클래스",
                            ["(skip)"] + classes,
                            key=f"class_{i}"
                        )

                        if label != "(skip)":
                            st.session_state.assigned_labels[i] = {
                                'class': label,
                                'bbox': mask_data['bbox'],
                                'area': mask_data['area'],
                                'mask': mask_data['segmentation']
                            }

                    with col3:
                        st.metric("영역", f"{mask_data['area']}px²")
                        x1, y1, x2, y2 = mask_data['bbox']
                        st.metric("크기", f"{x2-x1}×{y2-y1}")

            # 할당된 레이블 수
            st.info(f"**할당된 레이블**: {len(st.session_state.assigned_labels)}개")

            # 내보내기
            if st.session_state.assigned_labels:
                export_format = st.selectbox("내보내기 포맷", ["COCO JSON", "YOLO TXT", "CSV"])

                if st.button("📦 내보내기"):
                    if export_format == "COCO JSON":
                        data = export_coco_format(
                            image,
                            st.session_state.assigned_labels,
                            classes
                        )
                        st.download_button(
                            "💾 COCO JSON 다운로드",
                            data=json.dumps(data, indent=2),
                            file_name="annotations.json",
                            mime="application/json"
                        )

                    elif export_format == "YOLO TXT":
                        data = export_yolo_format(
                            image,
                            st.session_state.assigned_labels,
                            classes
                        )
                        st.download_button(
                            "💾 YOLO TXT 다운로드",
                            data=data,
                            file_name="labels.txt",
                            mime="text/plain"
                        )

                    else:  # CSV
                        data = export_csv_format(st.session_state.assigned_labels)
                        st.download_button(
                            "💾 CSV 다운로드",
                            data=data,
                            file_name="annotations.csv",
                            mime="text/csv"
                        )


def demo_counting_system():
    """객체 카운팅 시스템"""
    st.header("2️⃣ 객체 카운팅 시스템")

    st.markdown("""
    ### 실전 응용 사례

    | 분야 | 응용 | 효과 |
    |------|------|------|
    | **소매업** | 재고 관리 | 재고 파악 자동화 |
    | **제조업** | 불량품 검수 | 품질 관리 효율화 |
    | **농업** | 작물/과일 계수 | 수확량 예측 |
    | **교통** | 차량 카운팅 | 교통 흐름 분석 |
    | **의료** | 세포 카운팅 | 진단 보조 |

    ### 고급 기능
    - 크기 필터링
    - 영역별 카운팅
    - 통계 분석
    - 시계열 추적
    """)

    model_type = st.selectbox("SAM 모델", ["vit_b"], key="count_model_v2")
    sam = get_sam_helper(model_type)

    uploaded = st.file_uploader("이미지", type=['png', 'jpg', 'jpeg'], key="count_upload_v2")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="원본 이미지", use_container_width=True)

        # 카운팅 파라미터
        col1, col2, col3 = st.columns(3)

        with col1:
            min_area = st.number_input("최소 크기", 100, 10000, 500)

        with col2:
            max_area = st.number_input("최대 크기", 1000, 100000, 20000)

        with col3:
            grid_density = st.slider("그리드 밀도", 8, 48, 24)

        # 영역별 카운팅 옵션
        enable_zones = st.checkbox("영역별 카운팅 활성화")

        if enable_zones:
            st.markdown("### 영역 설정")

            num_zones = st.number_input("영역 개수", 1, 4, 2)

            zones = []
            zone_cols = st.columns(num_zones)

            for i in range(num_zones):
                with zone_cols[i]:
                    st.markdown(f"**Zone {i+1}**")
                    z_x1 = st.number_input(f"X1", 0, image.width, 0, key=f"z{i}_x1")
                    z_y1 = st.number_input(f"Y1", 0, image.height, 0, key=f"z{i}_y1")
                    z_x2 = st.number_input(f"X2", 0, image.width, image.width//num_zones, key=f"z{i}_x2")
                    z_y2 = st.number_input(f"Y2", 0, image.height, image.height, key=f"z{i}_y2")
                    zones.append((z_x1, z_y1, z_x2, z_y2))

        if st.button("🔢 카운팅 실행", type="primary"):
            with st.spinner("처리 중..."):
                # 자동 마스크 생성
                masks = sam.generate_auto_masks(image, points_per_side=grid_density)

                # 필터링
                filtered = [
                    m for m in masks
                    if min_area <= m['area'] <= max_area
                ]

                st.success(f"### 총 {len(filtered)}개 객체 검출")

                # 영역별 카운팅
                if enable_zones:
                    zone_counts = count_by_zones(filtered, zones)

                    zone_cols = st.columns(len(zones))
                    for i, count in enumerate(zone_counts):
                        zone_cols[i].metric(f"Zone {i+1}", f"{count}개")

                # 시각화
                fig = visualize_counting_result(image, filtered, zones if enable_zones else None)
                st.pyplot(fig)
                plt.close()

                # 통계
                if filtered:
                    show_counting_statistics(filtered)


def show_project_ideas():
    """프로젝트 아이디어"""
    st.header("3️⃣ 프로젝트 아이디어")

    st.markdown("""
    ## SAM 활용 프로젝트 아이디어

    ### 🎓 교육용 프로젝트

    #### 1. 스마트 칠판 주석 도구
    - **기능**: 수업 중 칠판 내용 자동 분할 및 구조화
    - **기술**: SAM + OCR (텍스트 인식)
    - **효과**: 강의 노트 자동 생성

    #### 2. 온라인 시험 모니터링
    - **기능**: 수험자 얼굴 및 손 위치 추적
    - **기술**: SAM + 포즈 추정
    - **효과**: 부정행위 방지

    ---

    ### 🏥 의료 프로젝트

    #### 3. 의료 영상 자동 분할
    - **기능**: X-ray, CT에서 장기/병변 자동 분할
    - **기술**: SAM Fine-tuning on medical data
    - **효과**: 진단 보조, 시간 단축

    #### 4. 세포 카운팅 시스템
    - **기능**: 현미경 이미지에서 세포 자동 계수
    - **기술**: SAM + Statistical analysis
    - **효과**: 연구 효율 향상

    ---

    ### 🛒 전자상거래 프로젝트

    #### 5. 가상 의류 피팅
    - **기능**: 사용자 사진에서 몸 분할 → 의류 합성
    - **기술**: SAM + GAN (이미지 생성)
    - **효과**: 온라인 쇼핑 경험 향상

    #### 6. 상품 이미지 자동 편집
    - **기능**: 배경 제거, 표준화, 리사이징
    - **기술**: SAM + Batch processing
    - **효과**: 등록 시간 90% 단축

    ---

    ### 🏭 산업용 프로젝트

    #### 7. 제조 불량품 검수
    - **기능**: 제품 표면 결함 자동 탐지
    - **기술**: SAM + Anomaly detection
    - **효과**: 품질 관리 자동화

    #### 8. 창고 재고 관리
    - **기능**: 팔레트/박스 자동 카운팅
    - **기술**: SAM + 3D reconstruction
    - **효과**: 재고 파악 실시간화

    ---

    ### 🌾 농업 프로젝트

    #### 9. 작물 성장 모니터링
    - **기능**: 드론 촬영 → 개별 작물 분할 및 추적
    - **기술**: SAM + Time series analysis
    - **효과**: 수확량 예측, 병해충 조기 발견

    #### 10. 스마트 수확 로봇
    - **기능**: 과일 인식 및 위치 파악
    - **기술**: SAM + Robotic arm control
    - **효과**: 수확 자동화

    ---

    ### 🚗 자율주행 프로젝트

    #### 11. 도로 세그멘테이션
    - **기능**: 차선, 도로, 보행자 영역 실시간 분할
    - **기술**: SAM + Real-time optimization
    - **효과**: 자율주행 인지 성능 향상

    ---

    ### 🎨 크리에이티브 프로젝트

    #### 12. AI 영상 편집 도구
    - **기능**: 영상에서 객체 자동 추출 → 효과 적용
    - **기술**: SAM + Video tracking
    - **효과**: 편집 시간 단축

    #### 13. AR 필터 제작 도구
    - **기능**: 실시간 얼굴/객체 분할 → AR 효과
    - **기술**: SAM + AR SDK
    - **효과**: 크리에이터 지원

    ---

    ## 프로젝트 선정 가이드

    ### 난이도별 추천

    | 난이도 | 프로젝트 | 소요 시간 |
    |--------|---------|----------|
    | **초급** | 배경 제거 앱 | 1주 |
    | **중급** | 자동 라벨링 도구 | 2-3주 |
    | **고급** | 의료 영상 분할 | 4-8주 |

    ### 성공 체크리스트

    ✅ **명확한 문제 정의**
    - 해결하고자 하는 실제 문제가 무엇인가?
    - 사용자는 누구인가?

    ✅ **데이터 확보 가능성**
    - 필요한 이미지 데이터를 확보할 수 있는가?
    - 라벨링이 필요한가?

    ✅ **기술적 실현 가능성**
    - SAM만으로 충분한가?
    - 추가 기술이 필요한가?

    ✅ **성능 요구사항**
    - 실시간 처리가 필요한가?
    - 정확도 목표는?

    ✅ **배포 계획**
    - 웹앱? 모바일? 스탠드얼론?
    - Streamlit? Gradio? Custom?

    ---

    ## 다음 단계

    1. **프로젝트 선정**: 관심사와 난이도 고려
    2. **기획서 작성**: 목표, 기능, 일정
    3. **프로토타입**: 핵심 기능부터 구현
    4. **테스트 및 개선**: 사용자 피드백 반영
    5. **배포**: GitHub + Hugging Face Space

    ---

    ### 참고 자료

    - [SAM GitHub](https://github.com/facebookresearch/segment-anything)
    - [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
    - [Awesome SAM](https://github.com/Hedlen/awesome-segment-anything)
    """)


# ==================== 헬퍼 함수 ====================

def create_mask_preview(image: Image.Image, mask_data: dict) -> Image.Image:
    """마스크 미리보기 생성"""
    preview = image.copy()
    draw = ImageDraw.Draw(preview, 'RGBA')

    # 마스크 오버레이
    mask = mask_data['segmentation']
    overlay = Image.new('RGBA', image.size, (255, 0, 0, 0))
    overlay_array = np.array(overlay)
    overlay_array[mask, 3] = 128  # 반투명 빨강
    overlay = Image.fromarray(overlay_array)

    preview = Image.alpha_composite(preview.convert('RGBA'), overlay)

    # BBox 그리기
    x1, y1, x2, y2 = mask_data['bbox']
    draw_bbox = ImageDraw.Draw(preview)
    draw_bbox.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=2)

    return preview.convert('RGB')


def export_coco_format(image: Image.Image, labels: dict, classes: list) -> dict:
    """COCO 포맷으로 내보내기"""
    coco_data = {
        "images": [{
            "id": 1,
            "file_name": "image.jpg",
            "width": image.width,
            "height": image.height
        }],
        "annotations": [],
        "categories": [
            {"id": i+1, "name": cls}
            for i, cls in enumerate(classes)
        ]
    }

    for i, (mask_id, label_data) in enumerate(labels.items()):
        class_id = classes.index(label_data['class']) + 1
        x1, y1, x2, y2 = label_data['bbox']

        coco_data["annotations"].append({
            "id": i+1,
            "image_id": 1,
            "category_id": class_id,
            "bbox": [x1, y1, x2-x1, y2-y1],
            "area": label_data['area'],
            "iscrowd": 0
        })

    return coco_data


def export_yolo_format(image: Image.Image, labels: dict, classes: list) -> str:
    """YOLO 포맷으로 내보내기"""
    lines = []

    for label_data in labels.values():
        class_id = classes.index(label_data['class'])
        x1, y1, x2, y2 = label_data['bbox']

        # YOLO 형식: class_id center_x center_y width height (정규화)
        center_x = ((x1 + x2) / 2) / image.width
        center_y = ((y1 + y2) / 2) / image.height
        width = (x2 - x1) / image.width
        height = (y2 - y1) / image.height

        lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return "\n".join(lines)


def export_csv_format(labels: dict) -> str:
    """CSV 포맷으로 내보내기"""
    lines = ["mask_id,class,x1,y1,x2,y2,area"]

    for mask_id, label_data in labels.items():
        x1, y1, x2, y2 = label_data['bbox']
        lines.append(
            f"{mask_id},{label_data['class']},{x1},{y1},{x2},{y2},{label_data['area']}"
        )

    return "\n".join(lines)


def count_by_zones(masks: list, zones: list) -> list:
    """영역별 카운팅"""
    counts = [0] * len(zones)

    for mask_data in masks:
        x1, y1, x2, y2 = mask_data['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        for i, (zx1, zy1, zx2, zy2) in enumerate(zones):
            if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                counts[i] += 1
                break

    return counts


def visualize_counting_result(image: Image.Image, masks: list, zones=None):
    """카운팅 결과 시각화"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    # 영역 그리기
    if zones:
        for i, (x1, y1, x2, y2) in enumerate(zones):
            from matplotlib.patches import Rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor='blue',
                           facecolor='blue', alpha=0.1)
            ax.add_patch(rect)
            ax.text(x1+5, y1+20, f"Zone {i+1}",
                   color='blue', fontsize=12, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 마스크 및 번호
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        color = np.random.rand(3)

        # 오버레이
        colored_mask = np.zeros((*mask.shape, 3))
        colored_mask[mask] = color
        ax.imshow(colored_mask, alpha=0.3)

        # 번호
        x1, y1, x2, y2 = mask_data['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax.text(cx, cy, str(i+1), color='white',
               fontsize=10, weight='bold', ha='center', va='center',
               bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8))

    ax.set_title(f"총 {len(masks)}개 객체 검출")
    ax.axis('off')

    return fig


def show_counting_statistics(masks: list):
    """카운팅 통계"""
    areas = [m['area'] for m in masks]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균 크기", f"{np.mean(areas):.0f}px²")
    col2.metric("최소 크기", f"{np.min(areas)}px²")
    col3.metric("최대 크기", f"{np.max(areas)}px²")
    col4.metric("표준편차", f"{np.std(areas):.0f}px²")

    # 히스토그램
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(areas, bins=20, edgecolor='black')
    ax.set_xlabel("영역 크기 (px²)")
    ax.set_ylabel("빈도")
    ax.set_title("크기 분포")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    run()
