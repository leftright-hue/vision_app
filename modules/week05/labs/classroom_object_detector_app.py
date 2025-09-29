#!/usr/bin/env python3
"""
Week 5 Lab: 교실 물건 탐지기 웹 애플리케이션
Gradio를 사용한 실시간 교실 물건 탐지 웹 앱

이 애플리케이션에서는:
1. 실시간 이미지/비디오 객체 탐지
2. 사용자 친화적 웹 인터페이스
3. 탐지 결과 분석 및 시각화
4. 모델 성능 모니터링
5. HuggingFace Space 배포 준비
"""

import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import time
import json
from pathlib import Path
from collections import Counter, defaultdict
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# YOLOv8 import
try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLOv8 패키지 로드 완료")
except ImportError:
    print("❌ Ultralytics 패키지가 설치되지 않았습니다.")
    print("설치 명령어: pip install ultralytics")

# 전역 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassroomObjectDetector:
    """
    교실 물건 탐지기 클래스
    """
    
    def __init__(self, model_path=None):
        """
        탐지기 초기화
        
        Args:
            model_path: 커스텀 모델 경로 (None이면 기본 YOLO 모델 사용)
        """
        self.model_path = model_path
        self.model = None
        self.is_custom_model = model_path is not None
        
        # 교실 물건 클래스 (커스텀 모델용)
        self.classroom_classes = {
            0: 'book',
            1: 'laptop', 
            2: 'chair',
            3: 'whiteboard',
            4: 'bag'
        }
        
        # COCO 클래스에서 교실 관련 클래스들
        self.coco_classroom_classes = {
            'book': 84,
            'laptop': 73,
            'chair': 62,
            'backpack': 27,
            'handbag': 31,
            'suitcase': 33,
            'bottle': 44,
            'cup': 47,
            'cell phone': 77,
            'clock': 85,
            'mouse': 74,
            'keyboard': 76,
            'remote': 75
        }
        
        # 클래스별 색상
        self.class_colors = {
            'book': (255, 0, 0),
            'laptop': (0, 255, 0),
            'chair': (0, 0, 255),
            'whiteboard': (255, 255, 0),
            'bag': (255, 0, 255),
            'backpack': (255, 0, 255),
            'handbag': (255, 128, 0),
            'suitcase': (128, 255, 0),
            'bottle': (0, 255, 255),
            'cup': (255, 192, 203),
            'cell phone': (128, 0, 128),
            'clock': (255, 165, 0),
            'mouse': (0, 128, 255),
            'keyboard': (128, 128, 128),
            'remote': (64, 224, 208)
        }
        
        # 통계 저장
        self.detection_history = []
        self.performance_stats = {
            'total_detections': 0,
            'total_images': 0,
            'avg_inference_time': 0,
            'class_counts': Counter()
        }
        
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            if self.is_custom_model and self.model_path and Path(self.model_path).exists():
                print(f"🔄 커스텀 모델 로딩: {self.model_path}")
                self.model = YOLO(self.model_path)
                print("✅ 커스텀 교실 물건 탐지 모델 로드 완료")
            else:
                print("🔄 기본 YOLOv8 모델 로딩...")
                self.model = YOLO('yolov8n.pt')  # Nano 모델 (빠른 추론)
                print("✅ 기본 YOLOv8n 모델 로드 완료")
                self.is_custom_model = False
        
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.model = None
    
    def detect_objects(self, image, conf_threshold=0.25, iou_threshold=0.7):
        """
        객체 탐지 수행
        
        Args:
            image: 입력 이미지
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값
        
        Returns:
            results: 탐지 결과
            inference_time: 추론 시간 (ms)
        """
        if self.model is None:
            return None, 0
        
        start_time = time.time()
        
        try:
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            
            return results[0] if results else None, inference_time
            
        except Exception as e:
            print(f"❌ 탐지 실패: {e}")
            return None, 0
    
    def filter_classroom_objects(self, results):
        """
        교실 관련 객체만 필터링 (기본 YOLO 모델 사용 시)
        
        Args:
            results: YOLO 결과
        
        Returns:
            filtered_results: 필터링된 결과
        """
        if results is None or results.boxes is None:
            return results
        
        if self.is_custom_model:
            return results  # 커스텀 모델은 이미 교실 객체만 탐지
        
        # COCO 클래스에서 교실 관련 객체만 필터링
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # COCO 클래스 이름
        coco_names = results.names
        
        filtered_indices = []
        for i, class_id in enumerate(class_ids):
            class_name = coco_names[class_id]
            if class_name in self.coco_classroom_classes.values() or \
               any(classroom_name in class_name.lower() for classroom_name in self.coco_classroom_classes.keys()):
                filtered_indices.append(i)
        
        if filtered_indices:
            # 필터링된 결과로 새로운 결과 객체 생성
            filtered_boxes = boxes[filtered_indices]
            filtered_confidences = confidences[filtered_indices]
            filtered_class_ids = class_ids[filtered_indices]
            
            # 결과 업데이트 (간단한 방식)
            results.boxes.xyxy = torch.tensor(filtered_boxes)
            results.boxes.conf = torch.tensor(filtered_confidences)
            results.boxes.cls = torch.tensor(filtered_class_ids)
        else:
            # 탐지된 교실 객체가 없음
            results.boxes = None
        
        return results
    
    def visualize_results(self, image, results, inference_time):
        """
        탐지 결과 시각화
        
        Args:
            image: 원본 이미지
            results: 탐지 결과
            inference_time: 추론 시간
        
        Returns:
            annotated_image: 어노테이션된 이미지
            detection_info: 탐지 정보
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        detection_info = {
            'total_objects': 0,
            'classes': {},
            'inference_time': inference_time,
            'image_size': image.size
        }
        
        if results is None or results.boxes is None:
            return annotated_image, detection_info
        
        # 탐지 결과 추출
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        detection_info['total_objects'] = len(boxes)
        
        # 클래스별 카운트
        class_counts = Counter()
        
        try:
            # 폰트 로드 시도
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 각 탐지 결과 그리기
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            
            # 클래스 이름 결정
            if self.is_custom_model:
                class_name = self.classroom_classes.get(class_id, f'class_{class_id}')
            else:
                class_name = results.names[class_id]
            
            class_counts[class_name] += 1
            
            # 색상 선택
            color = self.class_colors.get(class_name, (128, 128, 128))
            
            # 바운딩 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 라벨 텍스트
            label = f"{class_name}: {conf:.2f}"
            
            # 라벨 배경
            bbox = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(bbox, fill=color)
            
            # 라벨 텍스트
            draw.text((x1, y1-25), label, fill='white', font=font)
        
        detection_info['classes'] = dict(class_counts)
        
        # 통계 업데이트
        self.update_statistics(detection_info)
        
        return annotated_image, detection_info
    
    def update_statistics(self, detection_info):
        """통계 정보 업데이트"""
        self.performance_stats['total_images'] += 1
        self.performance_stats['total_detections'] += detection_info['total_objects']
        
        # 평균 추론 시간 업데이트
        current_avg = self.performance_stats['avg_inference_time']
        total_images = self.performance_stats['total_images']
        new_avg = (current_avg * (total_images - 1) + detection_info['inference_time']) / total_images
        self.performance_stats['avg_inference_time'] = new_avg
        
        # 클래스별 카운트 업데이트
        for class_name, count in detection_info['classes'].items():
            self.performance_stats['class_counts'][class_name] += count
        
        # 히스토리에 추가 (최근 100개만 유지)
        self.detection_history.append({
            'timestamp': datetime.now().isoformat(),
            'objects': detection_info['total_objects'],
            'inference_time': detection_info['inference_time'],
            'classes': detection_info['classes']
        })
        
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
    
    def get_statistics_summary(self):
        """통계 요약 정보 반환"""
        stats = self.performance_stats.copy()
        
        if stats['total_images'] > 0:
            stats['avg_objects_per_image'] = stats['total_detections'] / stats['total_images']
        else:
            stats['avg_objects_per_image'] = 0
        
        # 최근 10개 이미지의 성능
        recent_history = self.detection_history[-10:] if len(self.detection_history) >= 10 else self.detection_history
        
        if recent_history:
            recent_times = [h['inference_time'] for h in recent_history]
            stats['recent_avg_time'] = sum(recent_times) / len(recent_times)
            stats['recent_fps'] = 1000 / stats['recent_avg_time'] if stats['recent_avg_time'] > 0 else 0
        else:
            stats['recent_avg_time'] = 0
            stats['recent_fps'] = 0
        
        return stats
    
    def create_statistics_plot(self):
        """통계 시각화 플롯 생성"""
        if len(self.detection_history) < 2:
            # 데이터가 부족한 경우 빈 플롯 반환
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '충분한 데이터가 없습니다\n더 많은 이미지를 처리해주세요', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 시간별 추론 시간
        times = [h['inference_time'] for h in self.detection_history]
        ax1.plot(times, 'b-', alpha=0.7, linewidth=2)
        ax1.set_title('추론 시간 변화')
        ax1.set_xlabel('이미지 순서')
        ax1.set_ylabel('시간 (ms)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 시간별 탐지 객체 수
        object_counts = [h['objects'] for h in self.detection_history]
        ax2.plot(object_counts, 'g-', alpha=0.7, linewidth=2, marker='o', markersize=4)
        ax2.set_title('탐지 객체 수 변화')
        ax2.set_xlabel('이미지 순서')
        ax2.set_ylabel('객체 수')
        ax2.grid(True, alpha=0.3)
        
        # 3. 클래스별 누적 탐지 수
        class_counts = self.performance_stats['class_counts']
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            bars = ax3.bar(classes, counts, alpha=0.7, 
                          color=[np.array(self.class_colors.get(cls, (128, 128, 128)))/255 for cls in classes])
            ax3.set_title('클래스별 누적 탐지 수')
            ax3.set_xlabel('클래스')
            ax3.set_ylabel('탐지 수')
            ax3.tick_params(axis='x', rotation=45)
            
            # 막대 위에 값 표시
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, '탐지된 객체가 없습니다', ha='center', va='center')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
        
        # 4. 성능 요약
        stats = self.get_statistics_summary()
        
        summary_text = f"""성능 요약:
        
총 처리 이미지: {stats['total_images']}개
총 탐지 객체: {stats['total_detections']}개
평균 객체/이미지: {stats['avg_objects_per_image']:.1f}개

평균 추론 시간: {stats['avg_inference_time']:.1f}ms
최근 평균 시간: {stats['recent_avg_time']:.1f}ms
예상 FPS: {stats['recent_fps']:.1f}

디바이스: {DEVICE}
모델 타입: {'커스텀' if self.is_custom_model else 'COCO'}"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig

# 전역 탐지기 인스턴스
detector = ClassroomObjectDetector()

def detect_objects_interface(image, conf_threshold, iou_threshold, show_stats):
    """
    Gradio 인터페이스용 객체 탐지 함수
    
    Args:
        image: 입력 이미지
        conf_threshold: 신뢰도 임계값
        iou_threshold: IoU 임계값
        show_stats: 통계 표시 여부
    
    Returns:
        annotated_image: 어노테이션된 이미지
        detection_summary: 탐지 요약
        stats_plot: 통계 플롯 (선택사항)
    """
    if image is None:
        return None, "이미지를 업로드해주세요.", None
    
    try:
        # 객체 탐지
        results, inference_time = detector.detect_objects(image, conf_threshold, iou_threshold)
        
        # 교실 객체만 필터링 (기본 모델 사용 시)
        results = detector.filter_classroom_objects(results)
        
        # 결과 시각화
        annotated_image, detection_info = detector.visualize_results(image, results, inference_time)
        
        # 탐지 요약 생성
        summary = f"""🔍 탐지 결과:
        
📊 기본 정보:
• 탐지된 객체: {detection_info['total_objects']}개
• 추론 시간: {detection_info['inference_time']:.1f}ms
• 이미지 크기: {detection_info['image_size'][0]}×{detection_info['image_size'][1]}

📋 클래스별 탐지:"""
        
        if detection_info['classes']:
            for class_name, count in detection_info['classes'].items():
                summary += f"\n• {class_name}: {count}개"
        else:
            summary += "\n• 탐지된 교실 물건이 없습니다"
        
        # 성능 정보 추가
        stats = detector.get_statistics_summary()
        summary += f"""

📈 누적 통계:
• 총 처리 이미지: {stats['total_images']}개
• 평균 추론 시간: {stats['avg_inference_time']:.1f}ms
• 예상 FPS: {stats['recent_fps']:.1f}"""
        
        # 통계 플롯 생성 (요청 시)
        stats_plot = None
        if show_stats and len(detector.detection_history) > 1:
            stats_plot = detector.create_statistics_plot()
        
        return annotated_image, summary, stats_plot
        
    except Exception as e:
        error_msg = f"❌ 처리 중 오류 발생: {str(e)}"
        return None, error_msg, None

def reset_statistics():
    """통계 초기화"""
    global detector
    detector.detection_history = []
    detector.performance_stats = {
        'total_detections': 0,
        'total_images': 0,
        'avg_inference_time': 0,
        'class_counts': Counter()
    }
    return "✅ 통계가 초기화되었습니다."

def load_custom_model(model_file):
    """커스텀 모델 로드"""
    global detector
    
    if model_file is None:
        return "❌ 모델 파일을 선택해주세요."
    
    try:
        # 임시로 파일 저장
        temp_path = "temp_model.pt"
        with open(temp_path, "wb") as f:
            f.write(model_file)
        
        # 새로운 탐지기 생성
        detector = ClassroomObjectDetector(temp_path)
        
        if detector.model is not None:
            return "✅ 커스텀 모델이 성공적으로 로드되었습니다."
        else:
            return "❌ 모델 로드에 실패했습니다."
            
    except Exception as e:
        return f"❌ 모델 로드 중 오류: {str(e)}"

def create_sample_images():
    """샘플 이미지 생성"""
    samples = []
    
    # 샘플 1: 간단한 교실 장면
    img1 = Image.new('RGB', (640, 480), color='lightgray')
    draw = ImageDraw.Draw(img1)
    
    # 책상과 의자
    draw.rectangle([100, 200, 200, 400], fill='brown', outline='black', width=2)
    draw.rectangle([250, 300, 350, 350], fill='blue', outline='black', width=2)
    draw.rectangle([400, 150, 500, 300], fill='gray', outline='black', width=2)
    
    samples.append(img1)
    
    # 샘플 2: 복잡한 교실 환경
    img2 = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(img2)
    
    # 여러 객체들
    objects = [
        ([50, 100, 150, 200], 'red'),      # 책
        ([200, 150, 350, 250], 'green'),   # 노트북
        ([400, 100, 550, 300], 'blue'),    # 의자
        ([100, 350, 300, 400], 'orange'),  # 가방
        ([350, 50, 600, 150], 'yellow')    # 화이트보드
    ]
    
    for (x1, y1, x2, y2), color in objects:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=2)
    
    samples.append(img2)
    
    return samples

def create_gradio_interface():
    """Gradio 웹 인터페이스 생성"""
    
    # 커스텀 CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .detect-button {
        background: linear-gradient(45deg, #FF6B6B 30%, #4ECDC4 90%);
        border: none;
        border-radius: 25px;
        color: white;
        padding: 15px 30px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .detect-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    """
    
    with gr.Blocks(title="🎯 교실 물건 탐지기", theme=gr.themes.Soft(), css=custom_css) as demo:
        
        # 헤더
        gr.HTML("""
        <div class="main-header">
            <h1>🎯 교실 물건 탐지기</h1>
            <p>YOLOv8을 활용한 실시간 교실 물건 탐지 시스템</p>
            <p><strong>탐지 가능한 물건:</strong> 책, 노트북, 의자, 화이트보드, 가방</p>
        </div>
        """)
        
        with gr.Row():
            # 입력 섹션
            with gr.Column(scale=1):
                gr.Markdown("### 📸 이미지 업로드")
                
                image_input = gr.Image(
                    type="pil",
                    label="탐지할 이미지",
                    height=300
                )
                
                # 설정 옵션
                with gr.Accordion("⚙️ 탐지 설정", open=False):
                    conf_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="신뢰도 임계값",
                        info="낮을수록 더 많은 객체를 탐지하지만 오탐지 증가"
                    )
                    
                    iou_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="IoU 임계값 (NMS)",
                        info="중복 탐지 제거 강도"
                    )
                    
                    show_stats = gr.Checkbox(
                        value=False,
                        label="통계 차트 표시",
                        info="성능 통계 시각화"
                    )
                
                # 버튼들
                with gr.Row():
                    detect_btn = gr.Button(
                        "🔍 객체 탐지",
                        variant="primary",
                        size="lg",
                        elem_classes=["detect-button"]
                    )
                    
                    reset_btn = gr.Button(
                        "🔄 통계 초기화",
                        variant="secondary"
                    )
                
                # 커스텀 모델 업로드
                with gr.Accordion("🤖 커스텀 모델", open=False):
                    model_file = gr.File(
                        label="YOLOv8 모델 파일 (.pt)",
                        file_types=[".pt"]
                    )
                    
                    load_model_btn = gr.Button("모델 로드")
                    model_status = gr.Textbox(
                        label="모델 상태",
                        value="기본 YOLOv8n 모델 사용 중",
                        interactive=False
                    )
                
                # 시스템 정보
                gr.Markdown(f"""
                ### 💻 시스템 정보
                - **디바이스**: {DEVICE}
                - **모델**: YOLOv8n (기본)
                - **지원 형식**: JPG, PNG, WebP
                """)
            
            # 결과 섹션
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("🖼️ 탐지 결과"):
                        output_image = gr.Image(
                            label="탐지 결과",
                            height=400
                        )
                        
                        detection_summary = gr.Textbox(
                            label="탐지 요약",
                            lines=10,
                            max_lines=15,
                            value="이미지를 업로드하고 '객체 탐지' 버튼을 클릭하세요."
                        )
                    
                    with gr.Tab("📊 성능 통계"):
                        stats_plot = gr.Plot(
                            label="성능 통계 차트"
                        )
                        
                        gr.Markdown("""
                        ### 📈 통계 설명
                        - **추론 시간**: 각 이미지 처리에 걸린 시간
                        - **탐지 객체 수**: 이미지별 탐지된 객체 개수
                        - **클래스별 누적**: 전체 세션에서 탐지된 클래스별 개수
                        - **성능 요약**: 전체적인 성능 지표
                        """)
        
        # 예제 이미지
        sample_images = create_sample_images()
        gr.Examples(
            examples=[[img] for img in sample_images],
            inputs=[image_input],
            label="📋 예제 이미지"
        )
        
        # 사용 가이드
        with gr.Accordion("ℹ️ 사용 가이드", open=False):
            gr.Markdown("""
            ### 🔧 사용 방법
            1. **이미지 업로드**: 교실 사진을 업로드하거나 예제 이미지를 사용하세요
            2. **설정 조정**: 필요에 따라 신뢰도 임계값을 조정하세요
            3. **탐지 실행**: '객체 탐지' 버튼을 클릭하여 분석을 시작하세요
            4. **결과 확인**: 탐지된 객체와 통계를 확인하세요
            
            ### 🎯 탐지 가능한 객체
            - **책 (Book)**: 교과서, 노트, 참고서 등
            - **노트북 (Laptop)**: 노트북 컴퓨터
            - **의자 (Chair)**: 학생용 의자, 교사용 의자
            - **화이트보드 (Whiteboard)**: 칠판, 화이트보드
            - **가방 (Bag)**: 백팩, 핸드백, 서류가방
            
            ### ⚙️ 설정 팁
            - **높은 정확도**: 신뢰도 임계값을 0.5 이상으로 설정
            - **더 많은 탐지**: 신뢰도 임계값을 0.2 이하로 설정
            - **중복 제거**: IoU 임계값을 0.5-0.7 사이로 설정
            
            ### 🚀 성능 최적화
            - GPU 사용 시 더 빠른 처리 속도
            - 이미지 크기가 클수록 처리 시간 증가
            - 배치 처리로 여러 이미지 동시 처리 가능
            """)
        
        # 이벤트 연결
        detect_btn.click(
            fn=detect_objects_interface,
            inputs=[image_input, conf_threshold, iou_threshold, show_stats],
            outputs=[output_image, detection_summary, stats_plot],
            show_progress=True
        )
        
        reset_btn.click(
            fn=reset_statistics,
            outputs=[model_status]
        )
        
        load_model_btn.click(
            fn=load_custom_model,
            inputs=[model_file],
            outputs=[model_status]
        )
    
    return demo

def main():
    """메인 함수"""
    print("🎯 교실 물건 탐지기 웹 애플리케이션 시작")
    print("=" * 60)
    
    # Gradio 인터페이스 생성
    demo = create_gradio_interface()
    
    # 앱 실행
    print("🌐 웹 인터페이스 시작 중...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # 공개 링크 생성
        debug=False,
        show_error=True,
        favicon_path=None,
        app_kwargs={"docs_url": None, "redoc_url": None}
    )

if __name__ == "__main__":
    main()
