"""
Multi-modal Helper Functions
Multi-modal API 관련 유틸리티 함수들
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any


class MultiModalHelper:
    """Multi-modal 헬퍼 클래스"""

    def visualize_clip_embeddings(self):
        """CLIP 임베딩 공간 시각화"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # 시뮬레이션: 텍스트와 이미지 임베딩
        n_images = 50
        n_texts = 10

        # 이미지 임베딩 (클러스터)
        for i in range(3):  # 3개의 이미지 클러스터
            center = np.random.randn(2) * 3
            image_embeddings = np.random.randn(n_images // 3, 2) * 0.5 + center
            ax.scatter(image_embeddings[:, 0], image_embeddings[:, 1],
                      c=f'C{i}', alpha=0.5, s=50, marker='o',
                      label=f'Image Cluster {i+1}')

        # 텍스트 임베딩
        text_embeddings = np.random.randn(n_texts, 2) * 2
        ax.scatter(text_embeddings[:, 0], text_embeddings[:, 1],
                  c='red', alpha=0.8, s=100, marker='*',
                  label='Text Queries', edgecolors='black', linewidth=1)

        # 연결선 그리기 (유사도 표시)
        for i in range(min(3, n_texts)):
            # 가장 가까운 이미지 찾기
            closest_img_idx = np.random.randint(0, n_images // 3)
            ax.plot([text_embeddings[i, 0], image_embeddings[closest_img_idx, 0]],
                   [text_embeddings[i, 1], image_embeddings[closest_img_idx, 1]],
                   'k--', alpha=0.3, linewidth=1)

        ax.set_title('CLIP 임베딩 공간 시각화')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def get_api_comparison_data(self, selected_apis: List[str]) -> pd.DataFrame:
        """API 비교 데이터 생성"""
        all_features = {
            "OpenAI CLIP": {
                "텍스트-이미지 검색": "✅",
                "이미지-텍스트 생성": "❌",
                "Zero-shot 분류": "✅",
                "다국어 지원": "⚠️",
                "실시간 처리": "✅",
                "오프라인 사용": "✅",
                "가격": "무료(오픈소스)",
                "API 제한": "없음"
            },
            "Google Vision API": {
                "텍스트-이미지 검색": "⚠️",
                "이미지-텍스트 생성": "✅",
                "Zero-shot 분류": "❌",
                "다국어 지원": "✅",
                "실시간 처리": "✅",
                "오프라인 사용": "❌",
                "가격": "$1.5/1000 requests",
                "API 제한": "60 req/min"
            },
            "Azure Computer Vision": {
                "텍스트-이미지 검색": "⚠️",
                "이미지-텍스트 생성": "✅",
                "Zero-shot 분류": "❌",
                "다국어 지원": "✅",
                "실시간 처리": "✅",
                "오프라인 사용": "❌",
                "가격": "$1/1000 requests",
                "API 제한": "20 req/sec"
            },
            "AWS Rekognition": {
                "텍스트-이미지 검색": "❌",
                "이미지-텍스트 생성": "⚠️",
                "Zero-shot 분류": "❌",
                "다국어 지원": "✅",
                "실시간 처리": "✅",
                "오프라인 사용": "❌",
                "가격": "$1/1000 images",
                "API 제한": "50 req/sec"
            },
            "Hugging Face": {
                "텍스트-이미지 검색": "✅",
                "이미지-텍스트 생성": "✅",
                "Zero-shot 분류": "✅",
                "다국어 지원": "✅",
                "실시간 처리": "⚠️",
                "오프라인 사용": "✅",
                "가격": "무료/Pro($9/월)",
                "API 제한": "변동"
            },
            "OpenAI GPT-4V": {
                "텍스트-이미지 검색": "✅",
                "이미지-텍스트 생성": "✅",
                "Zero-shot 분류": "✅",
                "다국어 지원": "✅",
                "실시간 처리": "✅",
                "오프라인 사용": "❌",
                "가격": "$0.03/1K tokens",
                "API 제한": "500 req/min"
            }
        }

        # 선택된 API만 필터링
        filtered_data = {api: all_features[api] for api in selected_apis if api in all_features}
        return pd.DataFrame(filtered_data).T

    def create_speed_comparison_chart(self, selected_apis: List[str]):
        """API 속도 비교 차트"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 시뮬레이션 데이터
        speed_data = {
            "OpenAI CLIP": 45,
            "Google Vision API": 120,
            "Azure Computer Vision": 110,
            "AWS Rekognition": 95,
            "Hugging Face": 200,
            "OpenAI GPT-4V": 150
        }

        apis = [api for api in selected_apis if api in speed_data]
        speeds = [speed_data[api] for api in apis]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

        bars = ax.bar(apis, speeds, color=colors[:len(apis)])
        ax.set_ylabel('Response Time (ms)')
        ax.set_title('API 응답 속도 비교')
        ax.set_ylim(0, max(speeds) * 1.2)

        # 값 표시
        for bar, speed in zip(bars, speeds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speed}ms', ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def create_accuracy_comparison_chart(self, selected_apis: List[str]):
        """API 정확도 비교 차트"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 시뮬레이션 데이터
        accuracy_data = {
            "OpenAI CLIP": 92,
            "Google Vision API": 88,
            "Azure Computer Vision": 87,
            "AWS Rekognition": 85,
            "Hugging Face": 90,
            "OpenAI GPT-4V": 95
        }

        apis = [api for api in selected_apis if api in accuracy_data]
        accuracies = [accuracy_data[api] for api in apis]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

        bars = ax.bar(apis, accuracies, color=colors[:len(apis)])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('API 정확도 비교 (ImageNet 벤치마크)')
        ax.set_ylim(80, 100)

        # 값 표시
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc}%', ha='center', va='bottom')

        # 기준선 표시
        ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='Industry Standard (90%)')
        ax.legend()

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def get_api_recommendation(self, use_case: str, selected_apis: List[str]) -> str:
        """사용 사례별 API 추천"""
        recommendations = {
            "이미지 검색": {
                "best": "OpenAI CLIP",
                "reason": "텍스트-이미지 검색에 최적화되어 있고, Zero-shot 성능이 뛰어남"
            },
            "콘텐츠 모더레이션": {
                "best": "Google Vision API",
                "reason": "안전하지 않은 콘텐츠 감지 기능이 내장되어 있고, 다국어 지원이 우수함"
            },
            "의료 이미지 분석": {
                "best": "Azure Computer Vision",
                "reason": "의료 이미지 전용 모델 제공, HIPAA 준수 옵션 available"
            },
            "제품 추천": {
                "best": "AWS Rekognition",
                "reason": "제품 카탈로그 통합이 쉽고, AWS 생태계와 연동 우수"
            },
            "자동 태깅": {
                "best": "Hugging Face",
                "reason": "다양한 사전 학습 모델 선택 가능, 커스터마이징 용이"
            },
            "시각적 질의응답": {
                "best": "OpenAI GPT-4V",
                "reason": "복잡한 시각적 추론 능력이 뛰어나고, 자연어 이해력이 우수"
            }
        }

        if use_case in recommendations:
            rec = recommendations[use_case]
            if rec["best"] in selected_apis:
                return f"**추천 API: {rec['best']}**\n\n이유: {rec['reason']}"
            else:
                return f"**추천 API: {rec['best']}** (선택된 API 중에는 없음)\n\n이유: {rec['reason']}\n\n선택된 API 중에서는 기능 비교표를 참고하여 선택하세요."
        else:
            return "해당 사용 사례에 대한 추천 정보가 없습니다."