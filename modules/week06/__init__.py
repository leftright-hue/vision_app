"""
Week 6: 이미지 세그멘테이션과 SAM (Segment Anything Model)

이 모듈은 이미지 세그멘테이션의 이론과 실습을 다룹니다:
- U-Net 아키텍처
- Instance vs Panoptic Segmentation
- Segment Anything Model (SAM) 활용
- 실전 응용: 배경 제거, 자동 라벨링 등
"""

from .segmentation_module import SegmentationModule

__all__ = ['SegmentationModule']
