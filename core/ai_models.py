"""
AI 모델 관리 클래스
HuggingFace와 다른 AI 모델들을 관리합니다.
"""

import torch
from transformers import pipeline
from typing import Dict, List, Any, Optional
import streamlit as st
from PIL import Image
import numpy as np

class AIModelManager:
    """AI 모델 매니저"""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.models = {}
        self.model_configs = self._get_model_configs()

    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """모델 설정 정보"""
        return {
            'image-classification': {
                'default': 'google/vit-base-patch16-224',
                'alternatives': [
                    'microsoft/resnet-50',
                    'facebook/deit-base-distilled-patch16-224'
                ]
            },
            'object-detection': {
                'default': 'facebook/detr-resnet-50',
                'alternatives': [
                    'hustvl/yolos-tiny',
                    'facebook/detr-resnet-101'
                ]
            },
            'image-segmentation': {
                'default': 'facebook/detr-resnet-50-panoptic',
                'alternatives': [
                    'nvidia/segformer-b0-finetuned-ade-512-512'
                ]
            },
            'zero-shot-image-classification': {
                'default': 'openai/clip-vit-base-patch32',
                'alternatives': [
                    'openai/clip-vit-large-patch14'
                ]
            }
        }

    @st.cache_resource
    def load_model(_self, task: str, model_name: Optional[str] = None) -> pipeline:
        """모델 로드 (캐싱 적용)"""
        if model_name is None:
            model_name = _self.model_configs[task]['default']

        key = f"{task}:{model_name}"
        if key not in _self.models:
            try:
                _self.models[key] = pipeline(
                    task,
                    model=model_name,
                    device=_self.device
                )
                print(f"✅ 모델 로드 완료: {model_name}")
            except Exception as e:
                print(f"❌ 모델 로드 실패: {model_name} - {e}")
                # 대체 모델 시도
                if model_name != _self.model_configs[task]['default']:
                    return _self.load_model(task, _self.model_configs[task]['default'])
                raise e

        return _self.models[key]

    def classify_image(self, image: Image.Image, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """이미지 분류"""
        classifier = self.load_model('image-classification', model_name)
        return classifier(image)

    def detect_objects(self, image: Image.Image, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """객체 검출"""
        detector = self.load_model('object-detection', model_name)
        return detector(image)

    def segment_image(self, image: Image.Image, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """이미지 세그멘테이션"""
        segmenter = self.load_model('image-segmentation', model_name)
        return segmenter(image)

    def zero_shot_classify(self, image: Image.Image, labels: List[str],
                          model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """제로샷 이미지 분류"""
        classifier = self.load_model('zero-shot-image-classification', model_name)
        return classifier(image, candidate_labels=labels)

    def draw_detection_boxes(self, image: Image.Image, detections: List[Dict[str, Any]],
                           color: tuple = (255, 0, 0), thickness: int = 2) -> np.ndarray:
        """검출 박스 그리기"""
        import cv2
        img_array = np.array(image)

        for detection in detections:
            box = detection['box']
            xmin = int(box['xmin'])
            ymin = int(box['ymin'])
            xmax = int(box['xmax'])
            ymax = int(box['ymax'])

            # 박스 그리기
            cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), color, thickness)

            # 레이블 표시
            label = f"{detection['label']}: {detection['score']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 10)

            # 레이블 배경
            cv2.rectangle(img_array,
                        (xmin, label_ymin - label_size[1] - 10),
                        (xmin + label_size[0], label_ymin),
                        color, -1)

            # 레이블 텍스트
            cv2.putText(img_array, label,
                      (xmin, label_ymin - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img_array

    def get_available_models(self, task: str) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        if task in self.model_configs:
            models = [self.model_configs[task]['default']]
            models.extend(self.model_configs[task]['alternatives'])
            return models
        return []

    def get_model_info(self, task: str, model_name: str) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'task': task,
            'model': model_name,
            'device': 'GPU' if self.device == 0 else 'CPU',
            'loaded': f"{task}:{model_name}" in self.models
        }