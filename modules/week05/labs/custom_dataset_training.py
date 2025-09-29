#!/usr/bin/env python3
"""
Week 5 Lab: 커스텀 데이터셋 준비 및 YOLOv8 학습
교실 물건 탐지를 위한 커스텀 데이터셋 생성 및 모델 학습

이 실습에서는:
1. 교실 물건 데이터셋 시뮬레이션 생성
2. YOLO 형식 라벨링
3. 데이터 증강 및 전처리
4. YOLOv8 커스텀 학습
5. 모델 평가 및 최적화
"""

import os
import json
import yaml
import shutil
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# YOLOv8 import
try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLOv8 패키지 로드 완료")
except ImportError:
    print("❌ Ultralytics 패키지가 설치되지 않았습니다.")
    print("설치 명령어: pip install ultralytics")
    exit(1)

class ClassroomDatasetGenerator:
    """
    교실 물건 탐지를 위한 데이터셋 생성기
    """
    
    def __init__(self, output_dir="classroom_dataset"):
        """
        데이터셋 생성기 초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.classes = {
            0: 'book',
            1: 'laptop', 
            2: 'chair',
            3: 'whiteboard',
            4: 'bag'
        }
        
        # 클래스별 색상 (시각화용)
        self.class_colors = {
            'book': (255, 0, 0),      # 빨강
            'laptop': (0, 255, 0),    # 초록
            'chair': (0, 0, 255),     # 파랑
            'whiteboard': (255, 255, 0), # 노랑
            'bag': (255, 0, 255)      # 마젠타
        }
        
        # 데이터셋 구조 생성
        self.setup_directory_structure()
    
    def setup_directory_structure(self):
        """데이터셋 디렉토리 구조 생성"""
        directories = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'images' / 'test',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val',
            self.output_dir / 'labels' / 'test'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ 데이터셋 디렉토리 구조 생성: {self.output_dir}")
    
    def generate_synthetic_image(self, image_id, image_size=(640, 480)):
        """
        합성 교실 이미지 생성
        
        Args:
            image_id: 이미지 ID
            image_size: 이미지 크기 (width, height)
        
        Returns:
            image: PIL Image
            annotations: 바운딩 박스 정보 리스트
        """
        width, height = image_size
        
        # 배경 색상 랜덤 선택
        background_colors = [
            (240, 240, 240),  # 밝은 회색
            (255, 255, 255),  # 흰색
            (230, 230, 250),  # 연한 보라
            (245, 245, 220),  # 베이지
            (248, 248, 255)   # 고스트 화이트
        ]
        
        bg_color = random.choice(background_colors)
        image = Image.new('RGB', image_size, color=bg_color)
        draw = ImageDraw.Draw(image)
        
        annotations = []
        
        # 객체 개수 랜덤 결정 (1-8개)
        num_objects = random.randint(1, 8)
        
        for _ in range(num_objects):
            # 클래스 랜덤 선택
            class_id = random.randint(0, 4)
            class_name = self.classes[class_id]
            
            # 객체 크기와 위치 결정
            obj_width, obj_height, obj_x, obj_y = self.generate_object_bbox(
                class_name, width, height
            )
            
            # 겹침 검사 (간단한 버전)
            bbox = [obj_x, obj_y, obj_x + obj_width, obj_y + obj_height]
            if self.check_overlap(bbox, annotations):
                continue
            
            # 객체 그리기
            self.draw_object(draw, class_name, obj_x, obj_y, obj_width, obj_height)
            
            # YOLO 형식으로 변환 (중심점, 정규화된 좌표)
            center_x = (obj_x + obj_width / 2) / width
            center_y = (obj_y + obj_height / 2) / height
            norm_width = obj_width / width
            norm_height = obj_height / height
            
            annotations.append({
                'class_id': class_id,
                'class_name': class_name,
                'bbox': [center_x, center_y, norm_width, norm_height],
                'bbox_xyxy': bbox
            })
        
        return image, annotations
    
    def generate_object_bbox(self, class_name, img_width, img_height):
        """
        클래스별 적절한 바운딩 박스 크기와 위치 생성
        
        Args:
            class_name: 객체 클래스 이름
            img_width: 이미지 너비
            img_height: 이미지 높이
        
        Returns:
            width, height, x, y: 객체의 크기와 위치
        """
        # 클래스별 크기 범위 정의
        size_ranges = {
            'book': {'w': (40, 120), 'h': (60, 150)},
            'laptop': {'w': (100, 200), 'h': (80, 150)},
            'chair': {'w': (80, 150), 'h': (120, 200)},
            'whiteboard': {'w': (200, 400), 'h': (150, 250)},
            'bag': {'w': (60, 120), 'h': (80, 140)}
        }
        
        size_range = size_ranges[class_name]
        
        # 크기 랜덤 결정
        obj_width = random.randint(*size_range['w'])
        obj_height = random.randint(*size_range['h'])
        
        # 이미지 경계 내 위치 결정
        max_x = max(0, img_width - obj_width)
        max_y = max(0, img_height - obj_height)
        
        obj_x = random.randint(0, max_x) if max_x > 0 else 0
        obj_y = random.randint(0, max_y) if max_y > 0 else 0
        
        return obj_width, obj_height, obj_x, obj_y
    
    def check_overlap(self, new_bbox, existing_annotations, overlap_threshold=0.3):
        """
        새로운 바운딩 박스가 기존 박스들과 겹치는지 확인
        
        Args:
            new_bbox: 새로운 바운딩 박스 [x1, y1, x2, y2]
            existing_annotations: 기존 어노테이션 리스트
            overlap_threshold: 겹침 임계값
        
        Returns:
            bool: 겹침 여부
        """
        for ann in existing_annotations:
            existing_bbox = ann['bbox_xyxy']
            
            # IoU 계산
            x1 = max(new_bbox[0], existing_bbox[0])
            y1 = max(new_bbox[1], existing_bbox[1])
            x2 = min(new_bbox[2], existing_bbox[2])
            y2 = min(new_bbox[3], existing_bbox[3])
            
            if x2 <= x1 or y2 <= y1:
                continue  # 겹치지 않음
            
            intersection = (x2 - x1) * (y2 - y1)
            
            area1 = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
            area2 = (existing_bbox[2] - existing_bbox[0]) * (existing_bbox[3] - existing_bbox[1])
            
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > overlap_threshold:
                return True
        
        return False
    
    def draw_object(self, draw, class_name, x, y, width, height):
        """
        특정 클래스의 객체를 이미지에 그리기
        
        Args:
            draw: ImageDraw 객체
            class_name: 클래스 이름
            x, y: 좌상단 좌표
            width, height: 객체 크기
        """
        color = self.class_colors[class_name]
        
        if class_name == 'book':
            # 책: 사각형 + 선들
            draw.rectangle([x, y, x + width, y + height], fill=color, outline='black', width=2)
            # 페이지 선들
            for i in range(3):
                line_x = x + (i + 1) * width // 4
                draw.line([line_x, y, line_x, y + height], fill='white', width=1)
        
        elif class_name == 'laptop':
            # 노트북: 두 개의 사각형 (화면 + 키보드)
            screen_height = height * 0.6
            keyboard_height = height * 0.4
            
            # 화면
            draw.rectangle([x, y, x + width, y + screen_height], 
                          fill=color, outline='black', width=2)
            # 키보드
            draw.rectangle([x, y + screen_height, x + width, y + height], 
                          fill=(color[0]//2, color[1]//2, color[2]//2), outline='black', width=2)
        
        elif class_name == 'chair':
            # 의자: 등받이 + 좌석 + 다리
            back_width = width * 0.8
            seat_height = height * 0.3
            
            # 등받이
            draw.rectangle([x + width * 0.1, y, x + width * 0.9, y + height * 0.7], 
                          fill=color, outline='black', width=2)
            # 좌석
            draw.rectangle([x, y + height * 0.5, x + width, y + height * 0.8], 
                          fill=color, outline='black', width=2)
            # 다리들
            leg_positions = [(x + 5, y + height * 0.8), (x + width - 15, y + height * 0.8)]
            for leg_x, leg_y in leg_positions:
                draw.rectangle([leg_x, leg_y, leg_x + 10, y + height], 
                              fill=(color[0]//2, color[1]//2, color[2]//2), outline='black')
        
        elif class_name == 'whiteboard':
            # 화이트보드: 큰 사각형 + 프레임
            draw.rectangle([x, y, x + width, y + height], fill='white', outline='black', width=3)
            # 프레임
            draw.rectangle([x + 5, y + 5, x + width - 5, y + height - 5], 
                          fill=None, outline=color, width=2)
            # 간단한 내용 (선들)
            for i in range(3):
                line_y = y + (i + 1) * height // 4
                draw.line([x + 20, line_y, x + width - 20, line_y], fill='blue', width=2)
        
        elif class_name == 'bag':
            # 가방: 타원형 + 손잡이
            draw.ellipse([x, y + height * 0.2, x + width, y + height], 
                        fill=color, outline='black', width=2)
            # 손잡이
            handle_y = y + height * 0.1
            draw.arc([x + width * 0.2, handle_y, x + width * 0.8, y + height * 0.4], 
                    start=0, end=180, fill='black', width=3)
    
    def save_yolo_annotation(self, annotations, label_path):
        """
        YOLO 형식으로 어노테이션 저장
        
        Args:
            annotations: 어노테이션 리스트
            label_path: 라벨 파일 경로
        """
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id = ann['class_id']
                bbox = ann['bbox']
                # YOLO 형식: class_id center_x center_y width height
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def generate_dataset(self, num_images=1000, train_ratio=0.7, val_ratio=0.2):
        """
        전체 데이터셋 생성
        
        Args:
            num_images: 생성할 이미지 수
            train_ratio: 훈련 데이터 비율
            val_ratio: 검증 데이터 비율
        """
        print(f"🔄 {num_images}개 이미지 데이터셋 생성 중...")
        
        # 데이터 분할 계산
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        num_test = num_images - num_train - num_val
        
        print(f"   훈련: {num_train}개, 검증: {num_val}개, 테스트: {num_test}개")
        
        # 통계 수집
        class_counts = Counter()
        total_objects = 0
        
        # 데이터 생성
        splits = [
            ('train', num_train),
            ('val', num_val),
            ('test', num_test)
        ]
        
        image_id = 0
        
        for split_name, split_count in splits:
            print(f"\n📁 {split_name} 데이터 생성 중...")
            
            for i in range(split_count):
                # 이미지 생성
                image, annotations = self.generate_synthetic_image(image_id)
                
                # 파일 경로
                image_filename = f"image_{image_id:06d}.jpg"
                label_filename = f"image_{image_id:06d}.txt"
                
                image_path = self.output_dir / 'images' / split_name / image_filename
                label_path = self.output_dir / 'labels' / split_name / label_filename
                
                # 이미지 저장
                image.save(image_path, quality=95)
                
                # 라벨 저장
                self.save_yolo_annotation(annotations, label_path)
                
                # 통계 업데이트
                for ann in annotations:
                    class_counts[ann['class_name']] += 1
                    total_objects += 1
                
                image_id += 1
                
                if (i + 1) % 100 == 0:
                    print(f"   진행률: {i + 1}/{split_count}")
        
        # 데이터셋 설정 파일 생성
        self.create_dataset_config()
        
        # 통계 출력
        print(f"\n📊 데이터셋 생성 완료!")
        print(f"   총 이미지: {num_images}개")
        print(f"   총 객체: {total_objects}개")
        print(f"   클래스별 분포:")
        for class_name, count in class_counts.items():
            print(f"     {class_name}: {count}개 ({count/total_objects*100:.1f}%)")
        
        return class_counts
    
    def create_dataset_config(self):
        """YOLO 데이터셋 설정 파일 생성"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.classes),
            'names': list(self.classes.values())
        }
        
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ 데이터셋 설정 파일 생성: {config_path}")
    
    def visualize_samples(self, num_samples=6):
        """
        생성된 데이터셋 샘플 시각화
        
        Args:
            num_samples: 시각화할 샘플 수
        """
        print(f"🎨 데이터셋 샘플 {num_samples}개 시각화")
        
        # 훈련 데이터에서 샘플 선택
        train_images_dir = self.output_dir / 'images' / 'train'
        train_labels_dir = self.output_dir / 'labels' / 'train'
        
        image_files = list(train_images_dir.glob('*.jpg'))[:num_samples]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, image_path in enumerate(image_files):
            # 이미지 로드
            image = Image.open(image_path)
            
            # 라벨 로드
            label_path = train_labels_dir / (image_path.stem + '.txt')
            
            # 바운딩 박스 그리기
            fig_ax = axes[i]
            fig_ax.imshow(image)
            fig_ax.set_title(f'Sample {i+1}: {image_path.name}')
            fig_ax.axis('off')
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                img_width, img_height = image.size
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        
                        # YOLO 형식을 픽셀 좌표로 변환
                        x = (cx - w/2) * img_width
                        y = (cy - h/2) * img_height
                        width = w * img_width
                        height = h * img_height
                        
                        # 바운딩 박스 그리기
                        rect = patches.Rectangle(
                            (x, y), width, height,
                            linewidth=2, 
                            edgecolor=np.array(self.class_colors[self.classes[int(class_id)]])/255,
                            facecolor='none'
                        )
                        fig_ax.add_patch(rect)
                        
                        # 클래스 라벨
                        fig_ax.text(x, y-5, self.classes[int(class_id)], 
                                  fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

class DataAugmentation:
    """
    데이터 증강 클래스
    """
    
    def __init__(self):
        """데이터 증강 파이프라인 초기화"""
        self.transform = A.Compose([
            # 기하학적 변환
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.5
            ),
            
            # 색상 변환
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.3
            ),
            
            # 노이즈 및 블러
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=7, p=0.2),
            
            # 날씨 효과
            A.RandomRain(p=0.1),
            A.RandomShadow(p=0.2),
            A.RandomSunFlare(p=0.1),
            
            # 컷아웃
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=0.3
            ),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def augment_dataset(self, dataset_dir, output_dir, multiplier=2):
        """
        데이터셋에 증강 적용
        
        Args:
            dataset_dir: 원본 데이터셋 디렉토리
            output_dir: 증강된 데이터셋 출력 디렉토리
            multiplier: 증강 배수
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        
        print(f"🔄 데이터 증강 시작 (배수: {multiplier})")
        
        # 출력 디렉토리 생성
        for split in ['train', 'val', 'test']:
            (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            print(f"\n📁 {split} 데이터 증강 중...")
            
            images_dir = dataset_dir / 'images' / split
            labels_dir = dataset_dir / 'labels' / split
            
            output_images_dir = output_dir / 'images' / split
            output_labels_dir = output_dir / 'labels' / split
            
            image_files = list(images_dir.glob('*.jpg'))
            
            for i, image_path in enumerate(image_files):
                # 원본 복사
                shutil.copy2(image_path, output_images_dir)
                
                label_path = labels_dir / (image_path.stem + '.txt')
                if label_path.exists():
                    shutil.copy2(label_path, output_labels_dir)
                
                # 증강 버전 생성
                if split == 'train':  # 훈련 데이터만 증강
                    for aug_idx in range(multiplier - 1):
                        self.create_augmented_sample(
                            image_path, label_path,
                            output_images_dir, output_labels_dir,
                            aug_idx + 1
                        )
                
                if (i + 1) % 100 == 0:
                    print(f"   진행률: {i + 1}/{len(image_files)}")
        
        # 설정 파일 복사
        shutil.copy2(dataset_dir / 'dataset.yaml', output_dir / 'dataset.yaml')
        
        print("✅ 데이터 증강 완료!")
    
    def create_augmented_sample(self, image_path, label_path, 
                              output_images_dir, output_labels_dir, aug_idx):
        """
        단일 샘플에 대한 증강 수행
        
        Args:
            image_path: 원본 이미지 경로
            label_path: 원본 라벨 경로
            output_images_dir: 출력 이미지 디렉토리
            output_labels_dir: 출력 라벨 디렉토리
            aug_idx: 증강 인덱스
        """
        # 이미지 로드
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 라벨 로드
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        bboxes.append([cx, cy, w, h])
                        class_labels.append(int(class_id))
        
        try:
            # 증강 적용
            augmented = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            # 증강된 이미지 저장
            aug_image_name = f"{image_path.stem}_aug{aug_idx}.jpg"
            aug_image_path = output_images_dir / aug_image_name
            
            aug_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_image_path), aug_image)
            
            # 증강된 라벨 저장
            aug_label_name = f"{image_path.stem}_aug{aug_idx}.txt"
            aug_label_path = output_labels_dir / aug_label_name
            
            with open(aug_label_path, 'w') as f:
                for bbox, class_label in zip(augmented['bboxes'], augmented['class_labels']):
                    f.write(f"{class_label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        except Exception as e:
            print(f"   ⚠️ 증강 실패: {image_path.name} - {e}")

class YOLOv8Trainer:
    """
    YOLOv8 커스텀 학습 클래스
    """
    
    def __init__(self, dataset_config_path):
        """
        학습기 초기화
        
        Args:
            dataset_config_path: 데이터셋 설정 파일 경로
        """
        self.dataset_config_path = dataset_config_path
        self.model = None
        self.training_results = None
    
    def train_model(self, model_size='n', epochs=100, batch_size=16, 
                   learning_rate=0.01, patience=20, project_name='classroom_detector'):
        """
        모델 학습
        
        Args:
            model_size: 모델 크기 ('n', 's', 'm', 'l', 'x')
            epochs: 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            patience: 조기 종료 patience
            project_name: 프로젝트 이름
        
        Returns:
            model: 학습된 모델
            results: 학습 결과
        """
        print(f"🚀 YOLOv8{model_size.upper()} 모델 학습 시작")
        print(f"   에포크: {epochs}, 배치 크기: {batch_size}, 학습률: {learning_rate}")
        
        # 모델 초기화
        model_name = f'yolov8{model_size}.pt'
        self.model = YOLO(model_name)
        
        # 학습 설정
        training_args = {
            'data': self.dataset_config_path,
            'epochs': epochs,
            'batch': batch_size,
            'lr0': learning_rate,
            'patience': patience,
            'project': project_name,
            'name': f'yolov8{model_size}_classroom',
            'save_period': 10,
            'plots': True,
            'val': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }
        
        try:
            # 학습 실행
            self.training_results = self.model.train(**training_args)
            
            print("✅ 모델 학습 완료!")
            
            # 최고 성능 모델 로드
            best_model_path = Path(project_name) / f'yolov8{model_size}_classroom' / 'weights' / 'best.pt'
            if best_model_path.exists():
                self.model = YOLO(str(best_model_path))
                print(f"📁 최고 성능 모델 로드: {best_model_path}")
            
            return self.model, self.training_results
            
        except Exception as e:
            print(f"❌ 학습 중 오류 발생: {e}")
            return None, None
    
    def evaluate_model(self, test_data_path=None):
        """
        모델 평가
        
        Args:
            test_data_path: 테스트 데이터 경로 (선택사항)
        
        Returns:
            evaluation_results: 평가 결과
        """
        if self.model is None:
            print("❌ 학습된 모델이 없습니다.")
            return None
        
        print("📊 모델 평가 중...")
        
        try:
            # 검증 데이터로 평가
            if test_data_path:
                results = self.model.val(data=test_data_path, split='test')
            else:
                results = self.model.val(data=self.dataset_config_path, split='val')
            
            # 평가 결과 정리
            evaluation_results = {
                'mAP50': results.box.map50,
                'mAP50-95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0
            }
            
            # 클래스별 성능
            class_names = ['book', 'laptop', 'chair', 'whiteboard', 'bag']
            evaluation_results['per_class'] = {}
            
            for i, class_name in enumerate(class_names):
                if i < len(results.box.ap50):
                    evaluation_results['per_class'][class_name] = {
                        'AP50': results.box.ap50[i],
                        'AP50-95': results.box.ap[i] if i < len(results.box.ap) else 0,
                        'precision': results.box.p[i] if i < len(results.box.p) else 0,
                        'recall': results.box.r[i] if i < len(results.box.r) else 0
                    }
            
            print("✅ 모델 평가 완료!")
            self.print_evaluation_results(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ 평가 중 오류 발생: {e}")
            return None
    
    def print_evaluation_results(self, results):
        """평가 결과 출력"""
        print("\n📈 평가 결과:")
        print(f"   mAP@0.5: {results['mAP50']:.3f}")
        print(f"   mAP@0.5:0.95: {results['mAP50-95']:.3f}")
        print(f"   Precision: {results['precision']:.3f}")
        print(f"   Recall: {results['recall']:.3f}")
        print(f"   F1-Score: {results['f1_score']:.3f}")
        
        print("\n📋 클래스별 성능:")
        for class_name, metrics in results['per_class'].items():
            print(f"   {class_name}:")
            print(f"     AP@0.5: {metrics['AP50']:.3f}")
            print(f"     Precision: {metrics['precision']:.3f}")
            print(f"     Recall: {metrics['recall']:.3f}")

def main():
    """메인 실습 함수"""
    print("🎯 Week 5: 커스텀 데이터셋 준비 및 YOLOv8 학습")
    print("=" * 60)
    
    # 1. 데이터셋 생성
    print("\n1️⃣ 교실 물건 데이터셋 생성")
    
    dataset_generator = ClassroomDatasetGenerator("classroom_dataset")
    class_counts = dataset_generator.generate_dataset(
        num_images=500,  # 빠른 테스트를 위해 적은 수
        train_ratio=0.7,
        val_ratio=0.2
    )
    
    # 2. 데이터셋 시각화
    print("\n2️⃣ 데이터셋 샘플 시각화")
    dataset_generator.visualize_samples(num_samples=6)
    
    # 3. 데이터 증강 (선택사항)
    print("\n3️⃣ 데이터 증강")
    augmentation = DataAugmentation()
    
    # 증강된 데이터셋 생성 (시간이 오래 걸릴 수 있음)
    create_augmented = input("데이터 증강을 수행하시겠습니까? (y/n): ").lower() == 'y'
    
    if create_augmented:
        augmentation.augment_dataset(
            "classroom_dataset",
            "classroom_dataset_augmented",
            multiplier=2
        )
        dataset_path = "classroom_dataset_augmented/dataset.yaml"
    else:
        dataset_path = "classroom_dataset/dataset.yaml"
    
    # 4. 모델 학습
    print("\n4️⃣ YOLOv8 모델 학습")
    
    train_model = input("모델 학습을 수행하시겠습니까? (y/n): ").lower() == 'y'
    
    if train_model:
        trainer = YOLOv8Trainer(dataset_path)
        
        model, results = trainer.train_model(
            model_size='n',  # Nano 모델 (빠른 학습)
            epochs=50,       # 적은 에포크 (테스트용)
            batch_size=8,    # 작은 배치 크기
            learning_rate=0.01,
            patience=10
        )
        
        if model is not None:
            # 5. 모델 평가
            print("\n5️⃣ 모델 평가")
            evaluation_results = trainer.evaluate_model()
            
            # 6. 테스트 이미지로 추론
            print("\n6️⃣ 테스트 추론")
            test_image_path = "classroom_dataset/images/test"
            test_images = list(Path(test_image_path).glob("*.jpg"))[:3]
            
            for test_img in test_images:
                print(f"\n🔍 {test_img.name} 추론 중...")
                results = model.predict(str(test_img), conf=0.25, save=True)
                
                if results and results[0].boxes is not None:
                    num_detections = len(results[0].boxes)
                    print(f"   탐지된 객체: {num_detections}개")
    
    print("\n🎉 커스텀 데이터셋 학습 실습 완료!")
    print("\n📚 추가 실험 아이디어:")
    print("   - 실제 교실 사진으로 데이터셋 구성")
    print("   - 더 많은 클래스 추가")
    print("   - 하이퍼파라미터 튜닝")
    print("   - 모델 크기별 성능 비교")
    print("   - 실시간 웹캠 탐지 구현")

if __name__ == "__main__":
    main()
