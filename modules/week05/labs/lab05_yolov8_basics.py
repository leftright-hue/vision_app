#!/usr/bin/env python3
"""
Week 5 Lab: YOLOv8 기초 실습
YOLOv8을 사용한 객체 탐지 기본 구현 및 분석

이 실습에서는:
1. YOLOv8 모델 로드 및 기본 사용법
2. 이미지/비디오에서 객체 탐지
3. 결과 시각화 및 분석
4. 성능 벤치마킹
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import time
from pathlib import Path
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# YOLOv8 설치 확인 및 import
try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLOv8 패키지가 설치되어 있습니다.")
except ImportError:
    print("❌ Ultralytics 패키지가 설치되지 않았습니다.")
    print("다음 명령어로 설치하세요: pip install ultralytics")
    exit(1)

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {DEVICE}")

class YOLOv8Analyzer:
    """
    YOLOv8 모델 분석 및 시각화 클래스
    """
    
    def __init__(self, model_size='n'):
        """
        YOLOv8 분석기 초기화
        
        Args:
            model_size: 모델 크기 ('n', 's', 'm', 'l', 'x')
        """
        self.model_size = model_size
        self.model = None
        self.model_info = {
            'n': {'params': '3.2M', 'size': '6MB', 'description': 'Nano - 가장 빠름'},
            's': {'params': '11.2M', 'size': '22MB', 'description': 'Small - 속도와 정확도 균형'},
            'm': {'params': '25.9M', 'size': '50MB', 'description': 'Medium - 높은 정확도'},
            'l': {'params': '43.7M', 'size': '87MB', 'description': 'Large - 매우 높은 정확도'},
            'x': {'params': '68.2M', 'size': '136MB', 'description': 'Extra Large - 최고 정확도'}
        }
        
        # COCO 클래스 이름
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        self.load_model()
    
    def load_model(self):
        """YOLOv8 모델 로드"""
        try:
            model_name = f'yolov8{self.model_size}.pt'
            print(f"🔄 {model_name} 모델 로딩 중...")
            
            self.model = YOLO(model_name)
            
            info = self.model_info[self.model_size]
            print(f"✅ YOLOv8{self.model_size.upper()} 모델 로드 완료")
            print(f"   - 파라미터: {info['params']}")
            print(f"   - 모델 크기: {info['size']}")
            print(f"   - 설명: {info['description']}")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("💡 인터넷 연결을 확인하거나 다른 모델 크기를 시도해보세요.")
    
    def detect_objects(self, image_source, conf_threshold=0.25, iou_threshold=0.7):
        """
        객체 탐지 수행
        
        Args:
            image_source: 이미지 경로, PIL Image, 또는 numpy array
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값 (NMS용)
        
        Returns:
            results: YOLO 결과 객체
        """
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        try:
            results = self.model.predict(
                image_source,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            return results[0] if results else None
            
        except Exception as e:
            print(f"❌ 객체 탐지 실패: {e}")
            return None
    
    def visualize_results(self, image, results, save_path=None):
        """
        탐지 결과 시각화
        
        Args:
            image: 원본 이미지
            results: YOLO 결과
            save_path: 저장 경로 (선택사항)
        """
        if results is None or results.boxes is None:
            print("⚠️ 탐지된 객체가 없습니다.")
            return image
        
        # 이미지 복사
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        annotated_image = image.copy()
        
        # 탐지 결과 추출
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # 색상 팔레트
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (255, 165, 0), (255, 20, 147)
        ]
        
        # 각 탐지 결과 그리기
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            
            # 바운딩 박스 그리기
            color = colors[class_id % len(colors)]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트
            class_name = self.coco_classes[class_id]
            label = f"{class_name}: {conf:.2f}"
            
            # 라벨 배경 그리기
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 라벨 텍스트 그리기
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 결과 표시
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('원본 이미지')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(annotated_image)
        plt.title(f'탐지 결과 ({len(boxes)}개 객체)')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 결과가 {save_path}에 저장되었습니다.")
        
        plt.show()
        
        return annotated_image
    
    def analyze_detection_results(self, results):
        """
        탐지 결과 상세 분석
        
        Args:
            results: YOLO 결과
        
        Returns:
            analysis: 분석 결과 딕셔너리
        """
        if results is None or results.boxes is None:
            return {"total_objects": 0, "classes": {}, "confidence_stats": {}}
        
        # 기본 정보 추출
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # 클래스별 통계
        class_counts = Counter(class_ids)
        class_stats = {}
        
        for class_id, count in class_counts.items():
            class_name = self.coco_classes[class_id]
            class_confidences = confidences[class_ids == class_id]
            
            class_stats[class_name] = {
                'count': count,
                'avg_confidence': float(np.mean(class_confidences)),
                'max_confidence': float(np.max(class_confidences)),
                'min_confidence': float(np.min(class_confidences))
            }
        
        # 신뢰도 통계
        confidence_stats = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences))
        }
        
        # 바운딩 박스 크기 분석
        box_areas = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            box_areas.append(area)
        
        box_stats = {
            'mean_area': float(np.mean(box_areas)) if box_areas else 0,
            'std_area': float(np.std(box_areas)) if box_areas else 0,
            'min_area': float(np.min(box_areas)) if box_areas else 0,
            'max_area': float(np.max(box_areas)) if box_areas else 0
        }
        
        analysis = {
            'total_objects': len(boxes),
            'unique_classes': len(class_counts),
            'classes': class_stats,
            'confidence_stats': confidence_stats,
            'box_stats': box_stats
        }
        
        return analysis
    
    def benchmark_performance(self, test_images, num_runs=5):
        """
        모델 성능 벤치마크
        
        Args:
            test_images: 테스트 이미지 리스트
            num_runs: 각 이미지당 실행 횟수
        
        Returns:
            benchmark_results: 벤치마크 결과
        """
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        print(f"🔄 성능 벤치마크 시작 ({len(test_images)}개 이미지, {num_runs}회 반복)")
        
        all_times = []
        all_detections = []
        
        for i, image_path in enumerate(test_images):
            print(f"   이미지 {i+1}/{len(test_images)} 처리 중...")
            
            image_times = []
            
            # Warm-up
            _ = self.model.predict(image_path, verbose=False)
            
            # 실제 측정
            for run in range(num_runs):
                start_time = time.time()
                
                results = self.model.predict(image_path, verbose=False)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                inference_time = (end_time - start_time) * 1000  # ms
                image_times.append(inference_time)
                
                # 첫 번째 실행에서만 탐지 결과 저장
                if run == 0:
                    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                    all_detections.append(num_detections)
            
            all_times.extend(image_times)
        
        # 통계 계산
        benchmark_results = {
            'model_size': self.model_size,
            'device': str(DEVICE),
            'num_images': len(test_images),
            'num_runs_per_image': num_runs,
            'inference_times': {
                'mean': np.mean(all_times),
                'std': np.std(all_times),
                'min': np.min(all_times),
                'max': np.max(all_times),
                'median': np.median(all_times)
            },
            'fps': 1000 / np.mean(all_times),
            'total_detections': sum(all_detections),
            'avg_detections_per_image': np.mean(all_detections)
        }
        
        return benchmark_results
    
    def compare_model_sizes(self, test_image):
        """
        다양한 YOLOv8 모델 크기 비교
        
        Args:
            test_image: 테스트 이미지 경로
        
        Returns:
            comparison_results: 비교 결과
        """
        model_sizes = ['n', 's', 'm', 'l']  # 'x'는 시간이 오래 걸려서 제외
        comparison_results = {}
        
        print("🔄 다양한 YOLOv8 모델 크기 비교 중...")
        
        for size in model_sizes:
            print(f"   YOLOv8{size.upper()} 테스트 중...")
            
            try:
                # 임시 모델 생성
                temp_model = YOLO(f'yolov8{size}.pt')
                
                # 성능 측정
                times = []
                for _ in range(5):  # 5회 반복
                    start_time = time.time()
                    results = temp_model.predict(test_image, verbose=False)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                
                # 탐지 결과
                num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                comparison_results[size] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'fps': 1000 / np.mean(times),
                    'num_detections': num_detections,
                    'model_info': self.model_info[size]
                }
                
            except Exception as e:
                print(f"   ❌ YOLOv8{size.upper()} 테스트 실패: {e}")
                comparison_results[size] = None
        
        return comparison_results
    
    def visualize_benchmark_results(self, benchmark_results, comparison_results=None):
        """
        벤치마크 결과 시각화
        
        Args:
            benchmark_results: 벤치마크 결과
            comparison_results: 모델 크기 비교 결과 (선택사항)
        """
        if comparison_results:
            # 모델 크기 비교 시각화
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 유효한 결과만 필터링
            valid_results = {k: v for k, v in comparison_results.items() if v is not None}
            
            if valid_results:
                models = list(valid_results.keys())
                avg_times = [valid_results[m]['avg_time'] for m in models]
                fps_values = [valid_results[m]['fps'] for m in models]
                detections = [valid_results[m]['num_detections'] for m in models]
                
                # 추론 시간 비교
                bars1 = axes[0, 0].bar(models, avg_times, alpha=0.7, color='skyblue')
                axes[0, 0].set_title('모델별 평균 추론 시간')
                axes[0, 0].set_ylabel('시간 (ms)')
                axes[0, 0].set_xlabel('모델 크기')
                
                for bar, time_val in zip(bars1, avg_times):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                   f'{time_val:.1f}ms', ha='center', va='bottom')
                
                # FPS 비교
                bars2 = axes[0, 1].bar(models, fps_values, alpha=0.7, color='lightcoral')
                axes[0, 1].set_title('모델별 FPS')
                axes[0, 1].set_ylabel('FPS')
                axes[0, 1].set_xlabel('모델 크기')
                
                for bar, fps_val in zip(bars2, fps_values):
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                   f'{fps_val:.1f}', ha='center', va='bottom')
                
                # 탐지 개수 비교
                bars3 = axes[1, 0].bar(models, detections, alpha=0.7, color='lightgreen')
                axes[1, 0].set_title('모델별 탐지 객체 수')
                axes[1, 0].set_ylabel('객체 수')
                axes[1, 0].set_xlabel('모델 크기')
                
                for bar, det_val in zip(bars3, detections):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   f'{det_val}', ha='center', va='bottom')
                
                # 성능 대비 효율성 (FPS/Parameters)
                param_counts = []
                for model in models:
                    param_str = valid_results[model]['model_info']['params']
                    param_count = float(param_str.replace('M', ''))
                    param_counts.append(param_count)
                
                efficiency = [fps / params for fps, params in zip(fps_values, param_counts)]
                
                bars4 = axes[1, 1].bar(models, efficiency, alpha=0.7, color='gold')
                axes[1, 1].set_title('효율성 (FPS/파라미터 수)')
                axes[1, 1].set_ylabel('FPS per Million Parameters')
                axes[1, 1].set_xlabel('모델 크기')
                
                for bar, eff_val in zip(bars4, efficiency):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{eff_val:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
        
        # 단일 모델 벤치마크 결과 출력
        print("\n📊 벤치마크 결과 요약")
        print("=" * 50)
        print(f"모델: YOLOv8{benchmark_results['model_size'].upper()}")
        print(f"디바이스: {benchmark_results['device']}")
        print(f"테스트 이미지: {benchmark_results['num_images']}개")
        print(f"반복 횟수: {benchmark_results['num_runs_per_image']}회")
        print(f"평균 추론 시간: {benchmark_results['inference_times']['mean']:.2f}ms")
        print(f"표준 편차: {benchmark_results['inference_times']['std']:.2f}ms")
        print(f"최소 시간: {benchmark_results['inference_times']['min']:.2f}ms")
        print(f"최대 시간: {benchmark_results['inference_times']['max']:.2f}ms")
        print(f"평균 FPS: {benchmark_results['fps']:.1f}")
        print(f"총 탐지 객체: {benchmark_results['total_detections']}개")
        print(f"이미지당 평균 객체: {benchmark_results['avg_detections_per_image']:.1f}개")

def create_test_images():
    """테스트용 이미지 생성"""
    test_images = []
    
    # 1. 간단한 기하학적 도형 이미지
    img1 = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(img1)
    
    # 여러 도형 그리기 (의자, 책 등을 연상시키는 형태)
    draw.rectangle([100, 200, 200, 400], fill='brown', outline='black', width=3)  # 의자 등받이
    draw.rectangle([100, 350, 250, 380], fill='brown', outline='black', width=3)  # 의자 좌석
    draw.rectangle([300, 250, 400, 300], fill='blue', outline='black', width=2)   # 책
    draw.rectangle([450, 200, 550, 350], fill='gray', outline='black', width=2)   # 노트북
    
    test_images.append(('geometric_shapes.jpg', img1))
    
    # 2. 복잡한 실내 장면 시뮬레이션
    img2 = Image.new('RGB', (640, 480), color='lightgray')
    draw = ImageDraw.Draw(img2)
    
    # 교실 환경 시뮬레이션
    draw.rectangle([50, 100, 150, 300], fill='brown', outline='black', width=2)   # 책상
    draw.rectangle([200, 150, 300, 250], fill='red', outline='black', width=2)    # 책
    draw.rectangle([350, 120, 450, 280], fill='black', outline='gray', width=2)   # 칠판
    draw.ellipse([500, 200, 600, 300], fill='green', outline='black', width=2)    # 가방
    
    test_images.append(('classroom_scene.jpg', img2))
    
    # 3. 다양한 객체가 있는 복합 장면
    img3 = Image.new('RGB', (640, 480), color='skyblue')
    draw = ImageDraw.Draw(img3)
    
    # 여러 객체 배치
    objects = [
        ([100, 100, 180, 200], 'orange'),   # 사람 형태
        ([250, 150, 350, 250], 'red'),      # 자동차 형태
        ([400, 200, 500, 300], 'blue'),     # 의자 형태
        ([150, 300, 250, 350], 'green'),    # 책 형태
        ([350, 350, 450, 400], 'purple'),   # 가방 형태
    ]
    
    for bbox, color in objects:
        draw.rectangle(bbox, fill=color, outline='black', width=2)
    
    test_images.append(('mixed_objects.jpg', img3))
    
    return test_images

def demonstrate_yolo_features():
    """YOLOv8 주요 기능 시연"""
    print("🚀 YOLOv8 기능 시연 시작")
    print("=" * 50)
    
    # 1. YOLOv8 분석기 생성
    print("\n1️⃣ YOLOv8 모델 초기화")
    analyzer = YOLOv8Analyzer(model_size='n')  # Nano 버전 사용 (빠른 테스트)
    
    # 2. 테스트 이미지 생성
    print("\n2️⃣ 테스트 이미지 생성")
    test_images = create_test_images()
    
    # 이미지 저장 및 표시
    saved_paths = []
    for name, img in test_images:
        img.save(name)
        saved_paths.append(name)
        print(f"   💾 {name} 저장 완료")
    
    # 생성된 이미지 표시
    fig, axes = plt.subplots(1, len(test_images), figsize=(15, 5))
    for i, (name, img) in enumerate(test_images):
        axes[i].imshow(img)
        axes[i].set_title(name.replace('.jpg', ''))
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
    # 3. 객체 탐지 수행
    print("\n3️⃣ 객체 탐지 수행")
    
    for i, (name, img) in enumerate(test_images):
        print(f"\n📸 {name} 분석 중...")
        
        # 탐지 수행
        results = analyzer.detect_objects(name, conf_threshold=0.1)  # 낮은 임계값으로 더 많은 탐지
        
        # 결과 분석
        analysis = analyzer.analyze_detection_results(results)
        
        print(f"   탐지된 객체: {analysis['total_objects']}개")
        print(f"   고유 클래스: {analysis['unique_classes']}개")
        
        if analysis['classes']:
            print("   클래스별 탐지 결과:")
            for class_name, stats in analysis['classes'].items():
                print(f"     - {class_name}: {stats['count']}개 (평균 신뢰도: {stats['avg_confidence']:.2f})")
        
        # 시각화
        annotated_img = analyzer.visualize_results(img, results, save_path=f"result_{name}")
    
    # 4. 성능 벤치마크
    print("\n4️⃣ 성능 벤치마크")
    benchmark_results = analyzer.benchmark_performance(saved_paths, num_runs=3)
    
    # 5. 모델 크기 비교
    print("\n5️⃣ 모델 크기 비교")
    comparison_results = analyzer.compare_model_sizes(saved_paths[0])
    
    # 6. 결과 시각화
    print("\n6️⃣ 결과 시각화")
    analyzer.visualize_benchmark_results(benchmark_results, comparison_results)
    
    # 7. 정리
    print("\n🧹 임시 파일 정리")
    for path in saved_paths:
        try:
            Path(path).unlink()
            print(f"   🗑️ {path} 삭제")
        except:
            pass
    
    print("\n🎉 YOLOv8 기능 시연 완료!")
    
    return analyzer, benchmark_results, comparison_results

def advanced_detection_demo():
    """고급 탐지 기능 데모"""
    print("\n🔬 고급 YOLOv8 기능 데모")
    print("=" * 50)
    
    analyzer = YOLOv8Analyzer('s')  # Small 모델 사용
    
    # 1. 다양한 임계값 테스트
    print("\n1️⃣ 신뢰도 임계값 영향 분석")
    
    # 테스트 이미지 생성
    test_img = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(test_img)
    
    # 복잡한 장면 생성
    for i in range(10):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 400)
        w = np.random.randint(30, 100)
        h = np.random.randint(30, 100)
        color = tuple(np.random.randint(0, 256, 3))
        draw.rectangle([x, y, x+w, y+h], fill=color, outline='black')
    
    test_img.save('complex_scene.jpg')
    
    # 다양한 임계값으로 테스트
    thresholds = [0.1, 0.25, 0.5, 0.7, 0.9]
    threshold_results = {}
    
    for threshold in thresholds:
        results = analyzer.detect_objects('complex_scene.jpg', conf_threshold=threshold)
        analysis = analyzer.analyze_detection_results(results)
        threshold_results[threshold] = analysis['total_objects']
        print(f"   임계값 {threshold}: {analysis['total_objects']}개 객체 탐지")
    
    # 임계값 영향 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [threshold_results[t] for t in thresholds], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('신뢰도 임계값')
    plt.ylabel('탐지된 객체 수')
    plt.title('신뢰도 임계값이 탐지 결과에 미치는 영향')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 2. 실시간 처리 시뮬레이션
    print("\n2️⃣ 실시간 처리 성능 시뮬레이션")
    
    # 연속 이미지 처리 시뮬레이션
    processing_times = []
    
    for i in range(20):  # 20프레임 시뮬레이션
        # 랜덤 이미지 생성
        random_img = Image.new('RGB', (640, 480), 
                              color=tuple(np.random.randint(100, 256, 3)))
        draw = ImageDraw.Draw(random_img)
        
        # 랜덤 객체 추가
        for _ in range(np.random.randint(1, 5)):
            x = np.random.randint(0, 500)
            y = np.random.randint(0, 400)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            color = tuple(np.random.randint(0, 256, 3))
            draw.rectangle([x, y, x+w, y+h], fill=color)
        
        # 처리 시간 측정
        start_time = time.time()
        results = analyzer.detect_objects(random_img)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000
        processing_times.append(processing_time)
        
        if i % 5 == 0:
            print(f"   프레임 {i+1}: {processing_time:.1f}ms")
    
    # 실시간 성능 분석
    avg_time = np.mean(processing_times)
    fps = 1000 / avg_time
    
    print(f"\n📊 실시간 처리 성능:")
    print(f"   평균 처리 시간: {avg_time:.1f}ms")
    print(f"   예상 FPS: {fps:.1f}")
    print(f"   실시간 처리 가능: {'✅' if fps >= 30 else '❌'}")
    
    # 처리 시간 분포 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(processing_times, 'b-', alpha=0.7)
    plt.axhline(y=avg_time, color='r', linestyle='--', label=f'평균: {avg_time:.1f}ms')
    plt.xlabel('프레임 번호')
    plt.ylabel('처리 시간 (ms)')
    plt.title('프레임별 처리 시간')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(processing_times, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=avg_time, color='r', linestyle='--', label=f'평균: {avg_time:.1f}ms')
    plt.xlabel('처리 시간 (ms)')
    plt.ylabel('빈도')
    plt.title('처리 시간 분포')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 정리
    try:
        Path('complex_scene.jpg').unlink()
    except:
        pass
    
    print("\n🎉 고급 기능 데모 완료!")

def main():
    """메인 실습 함수"""
    print("🎯 Week 5: YOLOv8 기초 실습")
    print("=" * 60)
    
    try:
        # 1. 기본 기능 시연
        analyzer, benchmark_results, comparison_results = demonstrate_yolo_features()
        
        # 2. 고급 기능 데모
        advanced_detection_demo()
        
        print("\n📚 추가 실험 아이디어:")
        print("   - 실제 이미지 데이터셋으로 테스트")
        print("   - 비디오 파일에서 객체 탐지")
        print("   - 웹캠을 사용한 실시간 탐지")
        print("   - 커스텀 클래스로 모델 파인튜닝")
        print("   - 다른 YOLO 버전과 성능 비교")
        
    except Exception as e:
        print(f"❌ 실습 중 오류 발생: {e}")
        print("💡 패키지 설치 상태를 확인하고 다시 시도해보세요.")

if __name__ == "__main__":
    main()
