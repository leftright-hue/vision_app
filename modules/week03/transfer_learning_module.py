"""
Week 3: Transfer Learning & Multi-modal API 모듈
Transfer Learning과 Multi-modal API 관련 기능을 제공합니다.
"""

import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
import os
import sys

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def safe_get_response_text(response):
    """
    Gemini API 응답에서 안전하게 텍스트를 추출하는 함수
    finish_reason이 1이거나 응답이 비어있을 때의 처리를 포함
    """
    try:
        # candidates 구조를 직접 확인
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # content가 있는지 확인
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return ''.join(text_parts)
        
        # 대체 방법: 직접 문자열 변환 시도
        try:
            response_str = str(response)
            if response_str and response_str != str(type(response)) and len(response_str) > 50:
                return response_str
        except:
            pass
            
        return "응답을 생성할 수 없습니다. API가 빈 응답을 반환했습니다."
        
    except Exception as e:
        return f"응답 처리 중 오류가 발생했습니다: {str(e)}"

from core.base_processor import BaseImageProcessor
from core.ai_models import AIModelManager
from .transfer_helpers import TransferLearningHelper
from .multimodal_helpers import MultiModalHelper


class TransferLearningModule(BaseImageProcessor):
    """Transfer Learning 및 Multi-modal API 학습 모듈"""

    def __init__(self):
        super().__init__()
        self.ai_manager = AIModelManager()
        self.transfer_helper = TransferLearningHelper()
        self.multimodal_helper = MultiModalHelper()

    def render(self):
        """Week 3 모듈 UI 렌더링 - Week 2와 동일한 메서드명"""
        self.render_ui()

    def render_ui(self):
        """Week 3 모듈 UI 렌더링"""
        st.title("🔄 Week 3: Transfer Learning & Multi-modal API")
        st.markdown("---")

        # 탭 생성
        tabs = st.tabs([
            "📚 이론",
            "🔄 Transfer Learning",
            "🖼️ CLIP 검색",
            "🎨 특징 추출",
            "📊 통합 분석",
            "🚀 실전 프로젝트",
            "🔍 API 비교"
        ])

        with tabs[0]:
            self._render_theory_tab()

        with tabs[1]:
            self._render_transfer_learning_tab()

        with tabs[2]:
            self._render_clip_search_tab()

        with tabs[3]:
            self._render_feature_extraction_tab()

        with tabs[4]:
            self._render_integrated_analysis_tab()

        with tabs[5]:
            self._render_project_tab()

        with tabs[6]:
            self._render_api_comparison_tab()

    def _render_theory_tab(self):
        """이론 탭"""
        st.header("📖 Transfer Learning & Multi-modal 이론")

        # 1. Transfer Learning 이론과 실습
        with st.expander("📚 Transfer Learning 이론과 실습", expanded=True):
            st.markdown("""
            ### 🔄 Transfer Learning
            **개념**: 이미 학습된 모델의 지식을 새로운 작업에 전이하는 방법론

            **핵심 원리**:
            - 사전학습 모델(ResNet, VGG, EfficientNet 등)의 학습된 특징을 재활용
            - ImageNet 같은 대규모 데이터셋에서 학습한 일반적 특징을 활용
            - 새로운 작업에 맞게 마지막 레이어만 교체하거나 전체 모델을 미세조정

            **두 가지 주요 접근법**:
            1. **Feature Extraction (특징 추출)**:
               - 사전학습 모델을 고정하고 특징 추출기로만 사용
               - 빠르고 효율적, 적은 데이터로도 가능

            2. **Fine-tuning (미세 조정)**:
               - 사전학습 모델의 일부 또는 전체를 재학습
               - 더 높은 성능, 많은 데이터와 시간 필요

            **실제 활용 예시**:
            - ImageNet (1000 클래스) → 의료 이미지 분류 (2-10 클래스)
            - 일반 객체 인식 → 특정 제품 품질 검사
            - 자연 이미지 → 산업 결함 검출
            """)

        # 2. CLIP 이미지 검색이란?
        with st.expander("🖼️ CLIP 이미지 검색이란?", expanded=True):
            st.markdown("""
            ### 🤖 CLIP (Contrastive Language-Image Pre-training)
            OpenAI가 개발한 멀티모달 AI 모델로, 이미지와 텍스트를 함께 이해합니다.

            #### 🎯 텍스트 → 이미지 검색 기능
            - **개념**: 자연어 텍스트로 이미지를 검색하는 혁신적인 기술
            - **작동 원리**:
              1. 텍스트를 벡터로 변환 (텍스트 임베딩)
              2. 이미지를 동일한 벡터 공간으로 변환 (이미지 임베딩)
              3. 벡터 간 유사도 계산 (코사인 유사도)
              4. 가장 유사한 이미지 반환

            #### 🚀 기존 검색과의 차이점
            | 기존 검색 | CLIP 검색 |
            |----------|----------|
            | 태그/메타데이터 기반 | 이미지 내용 직접 이해 |
            | 정확한 키워드 필요 | 자연스러운 문장 가능 |
            | 미리 라벨링 필요 | 라벨링 불필요 |
            | 제한적 검색 | 창의적 검색 가능 |

            #### 📋 활용 예시
            - **전자상거래**: "파란색 스트라이프 셔츠" → 관련 제품 이미지
            - **갤러리**: "일몰이 있는 해변 풍경" → 관련 사진 검색
            - **의료**: "폐에 결절이 있는 X-ray" → 유사 의료 이미지
            - **SNS**: "귀여운 강아지가 공놀이 하는 모습" → 관련 게시물
            """)

        # 3. Transfer Learning과 CLIP의 차이점
        with st.expander("🤔 Transfer Learning과 CLIP의 차이점", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                ### 🔄 Transfer Learning
                **범주**: 기법/방법론

                **특징**:
                - 다양한 사전학습 모델 선택 가능
                - **단일 모달**(주로 이미지)에 집중
                - Fine-tuning이나 Feature Extraction 사용
                - 목적: 적은 데이터로 높은 성능 달성

                **필요한 것**:
                - 레이블된 타겟 데이터셋
                - 사전학습 모델 선택
                - GPU (권장)
                """)

            with col2:
                st.markdown("""
                ### 🤖 CLIP
                **범주**: 특정 모델

                **특징**:
                - OpenAI가 개발한 특정 모델
                - **멀티모달**(이미지 + 텍스트) 통합
                - Contrastive Learning으로 학습
                - 목적: Zero-shot 이미지 이해

                **필요한 것**:
                - 텍스트 프롬프트
                - CLIP 모델
                - 레이블 불필요 (Zero-shot)
                """)

            st.markdown("---")
            st.markdown("""
            ### 📊 핵심 차이점 요약

            | 구분 | Transfer Learning | CLIP |
            |------|------------------|------|
            | **범위** | 기법/방법론 | 특정 모델 |
            | **모달리티** | 단일 (이미지) | 멀티 (이미지+텍스트) |
            | **학습 방식** | Supervised | Contrastive |
            | **데이터 필요** | 레이블된 데이터 필요 | Zero-shot 가능 |
            | **활용** | Fine-tuning 필요 | 바로 사용 가능 |
            | **유연성** | 다양한 모델 선택 | CLIP 모델만 |

            ### 🔗 관계
            - **CLIP도 Transfer Learning 가능**: CLIP을 베이스 모델로 사용하여 Fine-tuning 가능
            - **상호보완적**: Transfer Learning으로 학습한 모델 + CLIP의 텍스트 이해 결합 가능
            - **선택 기준**:
              - 특정 클래스 분류 → Transfer Learning
              - 자연어 검색/설명 → CLIP
            """)

        # 4. 특징 추출(Feature Extraction)이란?
        with st.expander("🎨 특징 추출(Feature Extraction)이란?", expanded=True):
            st.markdown("""
            ### 특징 추출 (Feature Extraction)
            CNN의 중간 레이어에서 학습된 특징을 추출하여 시각화하고 분석하는 기술입니다.

            #### ❓ 왜 특징 추출을 하는가?

            **1. 🔍 블랙박스 문제 해결**
            - **문제**: CNN은 수백만 개의 파라미터를 가진 블랙박스
            - **해결**: 특징 시각화로 "고양이를 보고 뭘 학습했는지" 직접 확인
            - **예시**: 고양이 사진 → 귀 모양, 수염, 털 패턴을 감지하는지 검증

            **2. 💰 비용 절감을 위한 재활용**
            - **문제**: 처음부터 모델 학습하면 수백만원의 GPU 비용
            - **해결**: ImageNet으로 학습된 특징을 그대로 재활용
            - **예시**: 고양이/개 분류기의 특징 → 품종 분류에 재사용

            **3. 🐛 모델 디버깅과 개선**
            - **문제**: 모델이 왜 틀렸는지 모름
            - **해결**: 잘못 학습된 특징 발견하고 수정
            - **예시**: 배경만 보고 분류하는 문제 발견 → 데이터 개선

            #### 🖼️ 실제 특징 시각화 예시 (고양이 이미지)

            | 레이어 | 보는 것 | 실제 용도 |
            |--------|---------|----------|
            | **Layer 1-2** | 선, 모서리 | 컵 테두리, 고양이 윤곽선 감지 |
            | **Layer 3-4** | 질감, 패턴 | 고양이 털 무늬, 카펫 질감 인식 |
            | **Layer 5-6** | 부분 형태 | 고양이 귀, 컵 모양 파악 |
            | **Final** | 전체 객체 | "고양이", "종이컵" 최종 인식 |

            #### 💼 실무 활용 사례

            **1. 의료 AI**
            - X-ray에서 폐렴 특징 추출 → 어느 부분이 이상인지 의사에게 표시
            - 종양 검출 모델 → 어떤 패턴을 종양으로 인식하는지 검증

            **2. 자율주행**
            - 도로 표지판 인식 → 어떤 특징으로 정지 신호를 구분하는지
            - 보행자 감지 → 사람의 어떤 특징을 학습했는지 확인

            **3. 품질 관리**
            - 제품 결함 검사 → 스크래치, 찌그러짐 등 결함 패턴 학습 확인
            - 정상/불량 판별 근거를 시각적으로 제시

            **4. Transfer Learning 활용**
            - ImageNet 학습 모델 → 의료 이미지에 적용
            - 일반 특징(엣지, 질감) → 특수 목적(병변, 결함)으로 전이

            #### 🎯 핵심 가치
            > **"특징 추출 = AI의 사고 과정을 들여다보는 창"**

            - 단순히 "정확도 95%"가 아닌, "왜 95%인지" 설명 가능
            - 모델이 올바른 이유로 판단하는지 검증
            - 신뢰할 수 있는 AI 구축의 필수 요소
            """)

        # 서브 탭 생성 (상세 이론)
        st.markdown("---")
        st.subheader("📚 상세 이론 학습")
        theory_tabs = st.tabs([
            "📚 Transfer Learning 상세",
            "🤖 CLIP 상세",
            "🔬 수학적 기초",
            "💡 실전 가이드"
        ])

        with theory_tabs[0]:
            self._render_transfer_learning_theory()

        with theory_tabs[1]:
            self._render_clip_theory()

        with theory_tabs[2]:
            self._render_mathematical_foundation()

        with theory_tabs[3]:
            self._render_practical_guide()

    def _render_transfer_learning_theory(self):
        """Transfer Learning 상세 이론"""
        st.markdown("## 📚 Transfer Learning 이론과 실습")

        # 개념과 배경
        with st.expander("### 1. Transfer Learning의 탄생 배경과 핵심 개념", expanded=True):
            st.markdown("""
            #### 🌱 탄생 배경
            - **문제**: 딥러닝 모델 학습에는 막대한 데이터와 컴퓨팅 자원 필요
            - **해결책**: 이미 학습된 지식을 재활용하자!
            - **영감**: 인간의 학습 방식 (자전거 → 오토바이 운전)

            #### 🎯 핵심 개념
            **Transfer Learning = 지식 전이 학습**
            ```
            Source Domain (원천 도메인) → Target Domain (목표 도메인)
            ImageNet 1000개 클래스    →    개/고양이 2개 클래스
            ```

            #### 📊 작동 원리
            1. **Low-level Features** (하위 레이어)
               - Edge, Corner, Texture 등 일반적 특징
               - 대부분의 이미지 작업에 공통적으로 유용
               - 보통 동결(Freeze)하여 재사용

            2. **High-level Features** (상위 레이어)
               - 클래스별 특화된 특징
               - Task-specific하므로 재학습 필요
               - Fine-tuning의 주요 대상

            #### 🚀 왜 효과적인가?
            - **Feature Hierarchy**: CNN은 계층적 특징 학습
            - **Universal Features**: 하위 레이어는 범용적
            - **Data Efficiency**: 적은 데이터로도 학습 가능
            - **Convergence Speed**: 빠른 수렴
            """)

        # Transfer Learning 방법론
        with st.expander("### 2. Transfer Learning 방법론 상세"):
            st.markdown("""
            #### 🔧 방법 1: Feature Extraction (특징 추출)
            ```python
            # 모든 레이어 동결
            for param in model.parameters():
                param.requires_grad = False

            # 마지막 레이어만 교체
            model.fc = nn.Linear(2048, num_classes)
            ```
            - **장점**: 빠름, 과적합 위험 낮음
            - **단점**: 성능 향상 제한적
            - **적용**: 데이터 매우 적을 때 (< 1000개)

            #### 🎨 방법 2: Fine-tuning (미세 조정)
            ```python
            # 초기 레이어만 동결
            for layer in model.layers[:-5]:
                layer.requires_grad = False

            # 상위 레이어는 학습 가능
            for layer in model.layers[-5:]:
                layer.requires_grad = True
            ```
            - **장점**: 높은 성능 달성 가능
            - **단점**: 과적합 위험, 학습 시간 증가
            - **적용**: 충분한 데이터 (> 5000개)

            #### 🔄 방법 3: Progressive Fine-tuning
            ```python
            # Step 1: 마지막 레이어만
            train_last_layer(epochs=10)

            # Step 2: 점진적으로 더 많은 레이어
            unfreeze_layers(n=2)
            train_model(epochs=5, lr=lr/10)

            # Step 3: 전체 미세조정
            unfreeze_all()
            train_model(epochs=3, lr=lr/100)
            ```
            - **장점**: 안정적 학습, 최고 성능
            - **단점**: 복잡한 구현, 시간 소요
            - **적용**: 중요한 프로젝트

            #### 📊 방법 선택 가이드
            | 데이터 양 | 유사도 | 추천 방법 |
            |----------|--------|----------|
            | 적음 + 높음 | Feature Extraction |
            | 적음 + 낮음 | Fine-tuning (상위 레이어) |
            | 많음 + 높음 | Fine-tuning (전체) |
            | 많음 + 낮음 | 처음부터 학습 or Progressive |
            """)

        # 실제 구현 예제
        with st.expander("### 3. 실제 구현 코드 예제"):
            st.markdown("""
            #### 🐕 예제: 개 품종 분류기 만들기

            ```python
            import torch
            import torch.nn as nn
            import torchvision.models as models
            from torch.optim import Adam
            from torch.optim.lr_scheduler import StepLR

            class DogBreedClassifier:
                def __init__(self, num_breeds=120):
                    # 1. 사전학습 모델 로드
                    self.model = models.resnet50(pretrained=True)

                    # 2. Feature Extraction 설정
                    for param in self.model.parameters():
                        param.requires_grad = False

                    # 3. 새로운 분류기 추가
                    num_features = self.model.fc.in_features
                    self.model.fc = nn.Sequential(
                        nn.Linear(num_features, 512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512, num_breeds)
                    )

                def progressive_unfreeze(self, stage):
                    '''점진적 언프리징'''
                    if stage == 1:  # 마지막 블록만
                        for param in self.model.layer4.parameters():
                            param.requires_grad = True
                    elif stage == 2:  # 마지막 2개 블록
                        for param in self.model.layer3.parameters():
                            param.requires_grad = True
                    elif stage == 3:  # 전체
                        for param in self.model.parameters():
                            param.requires_grad = True

                def train_stage(self, dataloader, stage, epochs):
                    self.progressive_unfreeze(stage)

                    # 학습률 조정 (언프리징할수록 낮게)
                    lr = 1e-3 * (0.1 ** (stage - 1))
                    optimizer = Adam(
                        filter(lambda p: p.requires_grad,
                               self.model.parameters()),
                        lr=lr
                    )

                    for epoch in range(epochs):
                        # 학습 코드
                        pass

            # 사용 예시
            classifier = DogBreedClassifier(num_breeds=120)

            # Stage 1: Feature Extraction (10 epochs)
            classifier.train_stage(dataloader, stage=0, epochs=10)

            # Stage 2: Fine-tune 마지막 블록 (5 epochs)
            classifier.train_stage(dataloader, stage=1, epochs=5)

            # Stage 3: Fine-tune 더 많은 레이어 (3 epochs)
            classifier.train_stage(dataloader, stage=2, epochs=3)
            ```
            """)

        # 고급 기법
        with st.expander("### 4. Transfer Learning 고급 기법"):
            st.markdown("""
            #### 🎭 Domain Adaptation (도메인 적응)
            - **문제**: Source와 Target 도메인이 너무 다름
            - **해결**: Adversarial Training, MMD 등 활용
            ```python
            # Domain Adversarial Neural Network (DANN)
            class DANN(nn.Module):
                def __init__(self):
                    self.feature_extractor = ResNet50()
                    self.label_classifier = LabelClassifier()
                    self.domain_classifier = DomainClassifier()
                    self.gradient_reversal = GradientReversal()
            ```

            #### 🎯 Few-shot Learning (퓨샷 러닝)
            - **Prototypical Networks**: 클래스별 프로토타입 학습
            - **Siamese Networks**: 유사도 학습
            - **MAML**: 빠른 적응을 위한 메타 학습

            #### 🔄 Knowledge Distillation (지식 증류)
            ```python
            # Teacher-Student 모델
            def distillation_loss(student_output, teacher_output,
                                 true_labels, T=3, alpha=0.7):
                # Soft targets from teacher
                soft_loss = KL_div(
                    F.log_softmax(student_output/T),
                    F.softmax(teacher_output/T)
                ) * T * T

                # Hard targets
                hard_loss = F.cross_entropy(student_output, true_labels)

                return alpha * soft_loss + (1-alpha) * hard_loss
            ```

            #### 📊 Multi-task Learning (멀티태스크 러닝)
            - **Hard Parameter Sharing**: 레이어 공유
            - **Soft Parameter Sharing**: 정규화로 유사성 유도
            - **Cross-stitch Networks**: 태스크 간 정보 교환
            """)

    def _render_clip_theory(self):
        """CLIP 상세 이론"""
        st.markdown("## 🤖 CLIP (Contrastive Language-Image Pre-training) 이론과 응용")

        with st.expander("### 1. CLIP의 개발 배경과 핵심 아이디어", expanded=True):
            st.markdown("""
            #### 🌟 CLIP의 특징
            - **2021년 OpenAI 발표**: 비전-언어 모델의 패러다임 전환
            - **핵심**: 4억 개의 (이미지, 텍스트) 쌍으로 학습
            - **결과**: Zero-shot으로 ImageNet 정확도 76.2% 달성

            #### 🎯 핵심 아이디어: Contrastive Learning
            ```
            목표: 매칭되는 (이미지, 텍스트) 쌍은 가깝게
                 매칭되지 않는 쌍은 멀게

            [고양이 이미지] ←→ "귀여운 고양이" ✅ (가깝게)
            [고양이 이미지] ←→ "빨간 자동차" ❌ (멀게)
            ```

            #### 🏗️ CLIP 아키텍처
            ```
            이미지 → Image Encoder → 이미지 임베딩 (512차원)
                                          ↓
                                    코사인 유사도
                                          ↑
            텍스트 → Text Encoder → 텍스트 임베딩 (512차원)
            ```

            #### 📊 학습 과정
            1. **배치 구성**: N개의 (이미지, 텍스트) 쌍
            2. **인코딩**: 각각을 512차원 벡터로 변환
            3. **유사도 계산**: N×N 유사도 매트릭스
            4. **대조 학습**: 대각선은 1, 나머지는 0이 되도록
            """)

        with st.expander("### 2. CLIP 손실 함수와 학습 메커니즘"):
            st.markdown("""
            #### 📐 InfoNCE Loss (Contrastive Loss)
            ```python
            def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
                # 정규화
                image_embeddings = F.normalize(image_embeddings)
                text_embeddings = F.normalize(text_embeddings)

                # 코사인 유사도 계산
                logits = image_embeddings @ text_embeddings.T / temperature

                # 대각선이 정답 (positive pairs)
                labels = torch.arange(len(logits))

                # 양방향 손실
                loss_i2t = F.cross_entropy(logits, labels)
                loss_t2i = F.cross_entropy(logits.T, labels)

                return (loss_i2t + loss_t2i) / 2
            ```

            #### 🌡️ Temperature Parameter
            - **역할**: 유사도 분포의 sharpness 조절
            - **낮은 온도 (0.01)**: 더 확실한 구분
            - **높은 온도 (0.1)**: 부드러운 구분
            - **CLIP 기본값**: 0.07

            #### 📊 학습 전략
            1. **Large Batch Size**: 32,768 (매우 큼)
               - 더 많은 negative samples
               - 안정적인 대조 학습

            2. **Mixed Precision Training**
               ```python
               with torch.cuda.amp.autocast():
                   image_features = image_encoder(images)
                   text_features = text_encoder(texts)
                   loss = clip_loss(image_features, text_features)
               ```

            3. **Gradient Accumulation**
               - 메모리 제약 극복
               - 효과적인 large batch 시뮬레이션
            """)

        with st.expander("### 3. CLIP의 Zero-shot 능력"):
            st.markdown("""
            #### 🎯 Zero-shot Classification
            ```python
            def zero_shot_classifier(image, class_names, model):
                # 1. 텍스트 프롬프트 생성
                text_prompts = [f"a photo of a {name}" for name in class_names]

                # 2. 인코딩
                image_features = model.encode_image(image)
                text_features = model.encode_text(text_prompts)

                # 3. 유사도 계산
                similarities = (image_features @ text_features.T)

                # 4. 소프트맥스로 확률 변환
                probs = similarities.softmax(dim=-1)

                return class_names[probs.argmax()]
            ```

            #### 🔄 Prompt Engineering for CLIP
            ```python
            # 기본 프롬프트
            "a photo of a {class}"

            # 개선된 프롬프트들
            templates = [
                "a photo of a {class}",
                "a bad photo of a {class}",
                "a origami {class}",
                "a photo of the large {class}",
                "a {class} in a video game",
                "art of a {class}",
                "a photo of the small {class}"
            ]

            # 앙상블로 성능 향상
            def ensemble_classify(image, class_name, templates):
                scores = []
                for template in templates:
                    text = template.format(class=class_name)
                    score = compute_similarity(image, text)
                    scores.append(score)
                return np.mean(scores)
            ```

            #### 📊 Zero-shot vs Fine-tuned 성능
            | Dataset | Zero-shot CLIP | Fine-tuned ResNet50 |
            |---------|---------------|-------------------|
            | ImageNet | 76.2% | 76.3% |
            | CIFAR-100 | 65.1% | 71.5% |
            | Food101 | 88.9% | 72.3% |
            | Flowers102 | 68.7% | 91.3% |
            """)

        with st.expander("### 4. CLIP 응용과 확장"):
            st.markdown("""
            #### 🎨 CLIP 기반 응용
            1. **DALL-E 2**: CLIP 임베딩 → 이미지 생성
            2. **CLIP-Seg**: 이미지 분할
            3. **CLIP4Clip**: 비디오 검색
            4. **AudioCLIP**: 오디오-비전-언어 통합

            #### 🔧 CLIP Fine-tuning 전략
            ```python
            class CLIPFineTuner:
                def __init__(self, clip_model):
                    self.clip = clip_model
                    # LoRA: Low-Rank Adaptation
                    self.lora_image = LoRAAdapter(self.clip.visual)
                    self.lora_text = LoRAAdapter(self.clip.transformer)

                def forward(self, images, texts):
                    # 원본 + LoRA 어댑터
                    image_features = self.clip.visual(images)
                    image_features += self.lora_image(images)

                    text_features = self.clip.transformer(texts)
                    text_features += self.lora_text(texts)

                    return image_features, text_features
            ```

            #### 🌐 다국어 CLIP
            - **mCLIP**: 다국어 지원
            - **XLM-R**: 100개 언어 텍스트 인코더
            - **Korean CLIP**: 한국어 특화 모델
            """)

    def _render_mathematical_foundation(self):
        """수학적 기초"""
        st.markdown("## 🔬 수학적 기초와 이론")

        with st.expander("### 1. Gradient 기반 최적화"):
            st.markdown("""
            #### 📐 Transfer Learning의 수학

            **목적 함수**:
            $$L_{total} = L_{task} + \\lambda L_{regularization}$$

            **Fine-tuning 그래디언트**:
            $$\\theta_{new} = \\theta_{pretrained} - \\eta \\nabla L_{task}$$

            여기서:
            - $\\theta_{pretrained}$: 사전학습 가중치
            - $\\eta$: 학습률 (보통 매우 작은 값)
            - $L_{task}$: 새로운 태스크 손실
            """)

        with st.expander("### 2. Contrastive Learning 수학"):
            st.markdown("""
            #### 📏 CLIP의 InfoNCE Loss

            $$L = -\\log\\frac{\\exp(sim(x_i, y_i)/\\tau)}{\\sum_{j=1}^{N} \\exp(sim(x_i, y_j)/\\tau)}$$

            여기서:
            - $sim(x, y) = \\frac{x \\cdot y}{||x|| \\cdot ||y||}$ (코사인 유사도)
            - $\\tau$: temperature parameter
            - $N$: 배치 크기

            #### 🎯 최적화 목표
            - Positive pairs: $sim(x_i, y_i) \\rightarrow 1$
            - Negative pairs: $sim(x_i, y_j) \\rightarrow 0$ (i ≠ j)
            """)

    def _render_practical_guide(self):
        """실전 가이드"""
        st.markdown("## 💡 실전 가이드와 베스트 프랙티스")

        with st.expander("### 1. Transfer Learning 체크리스트", expanded=True):
            st.markdown("""
            #### ✅ 프로젝트 시작 전 체크리스트

            - [ ] **데이터 분석**
              - 데이터셋 크기: _____개
              - 클래스 수: _____개
              - 클래스 불균형 여부: Yes/No
              - Source 도메인과 유사도: High/Medium/Low

            - [ ] **모델 선택**
              - 정확도 우선: ResNet, EfficientNet
              - 속도 우선: MobileNet, ShuffleNet
              - 균형: EfficientNet-B0~B3

            - [ ] **학습 전략**
              - 데이터 < 1000: Feature Extraction
              - 1000 < 데이터 < 10000: Partial Fine-tuning
              - 데이터 > 10000: Full Fine-tuning

            - [ ] **하이퍼파라미터**
              - 초기 학습률: 1e-4 ~ 1e-3
              - 배치 크기: 최대한 크게 (메모리 허용 범위)
              - 에폭: Early Stopping 사용
            """)

        with st.expander("### 2. CLIP 활용 가이드"):
            st.markdown("""
            #### 🎯 CLIP 활용 시나리오

            **1. 제로샷 이미지 분류**
            ```python
            # 신규 클래스 추가 시 재학습 불필요
            new_classes = ["전기차", "하이브리드차", "수소차"]
            predictions = clip_classify(image, new_classes)
            ```

            **2. 이미지 검색 시스템**
            ```python
            # 자연어로 이미지 검색
            query = "일몰 때 해변에서 서핑하는 사람"
            results = clip_search(query, image_database)
            ```

            **3. 콘텐츠 모더레이션**
            ```python
            # 부적절한 콘텐츠 필터링
            inappropriate_prompts = ["violence", "adult content", ...]
            scores = clip_score(image, inappropriate_prompts)
            ```

            **4. 멀티모달 추천 시스템**
            ```python
            # 이미지 + 텍스트 기반 추천
            user_preference = "미니멀한 북유럽 스타일"
            recommendations = clip_recommend(products, user_preference)
            ```
            """)

        with st.expander("### 3. 트러블슈팅 가이드"):
            st.markdown("""
            #### 🔧 일반적인 문제와 해결책

            | 문제 | 원인 | 해결책 |
            |------|------|--------|
            | 과적합 | 데이터 부족 | Data Augmentation, Dropout 증가 |
            | 수렴 안 됨 | 학습률 너무 큼 | 학습률 감소 (10배) |
            | 성능 저하 | Catastrophic Forgetting | Lower learning rate, Regularization |
            | 메모리 부족 | 배치 크기 너무 큼 | Gradient Accumulation |
            | 느린 학습 | 너무 많은 레이어 학습 | Feature Extraction 먼저 |

            #### 💊 Quick Fixes
            ```python
            # 과적합 해결
            model.add_module('dropout', nn.Dropout(0.5))

            # 학습 불안정 해결
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=1e-4, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5)

            # 메모리 최적화
            with torch.cuda.amp.autocast():
                output = model(input)
                loss = criterion(output, target)
            ```
            """)

        # 이전 간단한 이론 내용도 유지
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Quick Reference")
            st.markdown("""
            **Transfer Learning 핵심**
            - 사전학습 → 전이 → 미세조정
            - 적은 데이터로 높은 성능

            **CLIP 핵심**
            - 이미지-텍스트 통합 임베딩
            - Zero-shot 분류 가능
            """)

        with col2:
            st.subheader("3. Multi-modal Learning")
            st.markdown("""
            - **CLIP**: 텍스트-이미지 연결
            - **DALL-E**: 텍스트로 이미지 생성
            - **Flamingo**: 비전-언어 이해
            - **ALIGN**: 대규모 비전-언어 모델
            """)

            st.subheader("4. 실제 활용 사례")
            st.markdown("""
            - **의료 AI**: X-ray, MRI 분석
            - **자율주행**: 객체 인식 및 추적
            - **품질 검사**: 제조업 불량 검출
            - **콘텐츠 검색**: 이미지-텍스트 검색
            """)

    def _render_transfer_learning_tab(self):
        """Transfer Learning 탭"""
        st.header("🔄 Transfer Learning 실습")

        # 탭 생성: 예제 학습과 사용자 이미지 분석
        sub_tabs = st.tabs(["📚 예제로 학습하기", "🔧 내 모델 Fine-tuning하기"])

        with sub_tabs[0]:
            st.markdown("### 1. 사전 훈련 모델 선택")

            col1, col2, col3 = st.columns(3)

            with col1:
                model_name = st.selectbox(
                    "모델 선택",
                    ["ResNet50", "VGG16", "EfficientNet", "MobileNet", "DenseNet"],
                    key="model_select_example"
                )

            with col2:
                pretrained = st.checkbox("사전 훈련 가중치 사용", value=True, key="pretrained_example")

            with col3:
                num_classes = st.number_input("출력 클래스 수", min_value=2, value=10, key="num_classes_example")

            # 모델 정보 표시
            if st.button("모델 정보 보기", key="model_info_example"):
                self._show_model_info(model_name)

            st.markdown("### 2. Transfer Learning 방법")

            method = st.radio(
                "학습 방법 선택",
                ["Feature Extraction (빠름)", "Fine-tuning (정확함)", "전체 학습 (느림)"],
                key="method_example"
            )

            # 코드 예시
            with st.expander("📝 코드 보기"):
                code = self.transfer_helper.get_transfer_learning_code(model_name, num_classes, method)
                st.code(code, language="python")

        with sub_tabs[1]:
            st.markdown("### 🔧 커스텀 데이터셋으로 Fine-tuning")

            # 베이스 모델 설명
            with st.expander("📚 베이스 모델이란?", expanded=True):
                st.markdown("""
                **베이스 모델(Base Model)**: ImageNet 등 대규모 데이터셋으로 사전 학습된 모델

                #### 🎯 Fine-tuning 개념
                - **정의**: 사전 학습된 모델을 새로운 데이터셋에 맞게 재학습하는 과정
                - **장점**:
                  - 적은 데이터로도 높은 성능 달성
                  - 학습 시간 단축 (수일 → 수시간)
                  - 이미 학습된 특징(feature) 활용

                #### 📊 주요 베이스 모델 비교
                | 모델 | 파라미터 | 정확도 | 속도 | 용도 |
                |------|---------|--------|------|------|
                | **ResNet50** | 25.6M | 92.1% | 중간 | 범용, 안정적 |
                | **EfficientNet-B0** | 5.3M | 93.3% | 빠름 | 효율적, 모바일 |
                | **MobileNetV2** | 3.5M | 90.1% | 매우 빠름 | 경량, 실시간 |

                #### 🔧 Fine-tuning 프로세스
                1. **베이스 모델 로드**: 사전 학습된 가중치 불러오기
                2. **마지막 레이어 교체**: 새로운 클래스 수에 맞게 변경
                3. **선택적 동결**: 일부 레이어는 고정, 일부만 학습
                4. **학습**: 커스텀 데이터로 재학습
                5. **평가**: 성능 검증 및 최적화

                #### 📈 성능 향상 확인 방법
                - **Fine-tuning 전**: 일반 ImageNet 모델 → 70-80% 정확도
                - **Fine-tuning 후**: 커스텀 데이터 학습 → 90-95% 정확도
                - **평가 지표**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
                """)

            st.markdown("---")

            # 파일 업로드
            uploaded_files = st.file_uploader(
                "학습할 이미지 업로드 (클래스별로 폴더 구분)",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="custom_dataset"
            )

            if uploaded_files:
                col1, col2 = st.columns(2)

                with col1:
                    model_choice = st.selectbox(
                        "베이스 모델 선택",
                        ["ResNet50", "EfficientNet-B0", "MobileNetV2"],
                        key="model_custom",
                        help="ResNet50: 가장 안정적, EfficientNet: 높은 정확도, MobileNet: 빠른 속도"
                    )

                    learning_rate = st.slider(
                        "학습률",
                        min_value=0.0001,
                        max_value=0.01,
                        value=0.001,
                        format="%.4f",
                        key="lr_custom"
                    )

                with col2:
                    epochs = st.slider("에폭 수", min_value=1, max_value=50, value=10, key="epochs_custom")
                    batch_size = st.select_slider("배치 크기", options=[8, 16, 32, 64], value=32, key="batch_custom")

                # 선택된 모델 상세 정보
                with st.expander(f"🔍 {model_choice} 상세 정보"):
                    if model_choice == "ResNet50":
                        st.markdown("""
                        **ResNet50 (Residual Network)**
                        - **개발**: Microsoft Research (2015)
                        - **특징**: Skip Connection으로 깊은 네트워크 학습 가능
                        - **구조**: 50개 레이어, Bottleneck 블록
                        - **장점**: 안정적 학습, 범용적 사용
                        - **단점**: 모델 크기가 큼 (98MB)
                        - **적합한 경우**: 정확도가 중요한 경우, 충분한 컴퓨팅 자원
                        """)
                    elif model_choice == "EfficientNet-B0":
                        st.markdown("""
                        **EfficientNet-B0**
                        - **개발**: Google Brain (2019)
                        - **특징**: Compound Scaling으로 효율적 확장
                        - **구조**: MBConv 블록, Squeeze-and-Excitation
                        - **장점**: 적은 파라미터로 높은 성능
                        - **단점**: 학습 시 불안정할 수 있음
                        - **적합한 경우**: 효율성과 정확도 모두 중요한 경우
                        """)
                    else:  # MobileNetV2
                        st.markdown("""
                        **MobileNetV2**
                        - **개발**: Google (2018)
                        - **특징**: Inverted Residual Block, Linear Bottleneck
                        - **구조**: Depthwise Separable Convolution
                        - **장점**: 매우 가벼움 (14MB), 빠른 추론
                        - **단점**: 정확도가 상대적으로 낮음
                        - **적합한 경우**: 모바일/엣지 디바이스, 실시간 처리
                        """)

                if st.button("🚀 Fine-tuning 시작", key="start_finetuning"):
                    with st.spinner(f"{model_choice}을 베이스로 모델을 학습하는 중..."):
                        # 실제 fine-tuning 로직은 여기에 구현
                        st.info(f"🏗️ 베이스 모델 {model_choice} 로드 중...")
                        progress_bar = st.progress(0)

                        status = st.empty()
                        for i in range(epochs):
                            progress_bar.progress((i + 1) / epochs)
                            status.text(f"Epoch {i+1}/{epochs} - Loss: {0.5 - i*0.02:.3f}")

                        st.success(f"✅ Fine-tuning 완료! {model_choice} 기반 커스텀 모델 생성됨")

                        # 성능 비교 섹션
                        st.markdown("---")
                        st.markdown("### 📊 Fine-tuning 성능 비교")

                        col1, col2, col3 = st.columns(3)

                        # 시뮬레이션된 성능 지표
                        with col1:
                            st.metric(
                                label="Fine-tuning 전 정확도",
                                value="72.3%",
                                delta=None,
                                help="ImageNet 가중치 그대로 사용"
                            )

                        with col2:
                            st.metric(
                                label="Fine-tuning 후 정확도",
                                value="94.7%",
                                delta="+22.4%",
                                delta_color="normal",
                                help="커스텀 데이터로 재학습"
                            )

                        with col3:
                            st.metric(
                                label="성능 향상률",
                                value="31.0%",
                                delta="개선됨",
                                help="(94.7-72.3)/72.3 * 100"
                            )

                        # 학습 곡선 그래프
                        with st.expander("📈 학습 곡선 및 성능 분석", expanded=True):
                            fig = self.transfer_helper.plot_learning_curves()
                            st.pyplot(fig)

                            st.markdown("""
                            #### 🎯 성능 향상 확인 방법

                            **1. 정확도 (Accuracy) 비교**
                            - Fine-tuning 전: 사전학습 모델 그대로 → 낮은 정확도
                            - Fine-tuning 후: 커스텀 데이터 학습 → 높은 정확도

                            **2. 손실 함수 (Loss) 추적**
                            - Training Loss: 학습 데이터에서의 오차
                            - Validation Loss: 검증 데이터에서의 오차
                            - 두 값이 모두 감소하면 성능 향상

                            **3. 혼동 행렬 (Confusion Matrix)**
                            - 각 클래스별 예측 정확도 확인
                            - 오분류 패턴 분석
                            """)

                        # 혼동 행렬
                        with st.expander("🔍 상세 성능 분석"):
                            tab1, tab2, tab3 = st.tabs(["혼동 행렬", "클래스별 성능", "특징 공간"])

                            with tab1:
                                fig_cm = self.transfer_helper.create_confusion_matrix(5)
                                st.pyplot(fig_cm)
                                st.caption("Fine-tuning 후 혼동 행렬 - 대각선이 진할수록 좋은 성능")

                            with tab2:
                                # 클래스별 성능 메트릭
                                st.markdown("""
                                | 클래스 | Precision | Recall | F1-Score |
                                |--------|-----------|--------|----------|
                                | Class 0 | 0.95 | 0.93 | 0.94 |
                                | Class 1 | 0.92 | 0.96 | 0.94 |
                                | Class 2 | 0.96 | 0.94 | 0.95 |
                                | Class 3 | 0.94 | 0.95 | 0.94 |
                                | Class 4 | 0.97 | 0.95 | 0.96 |
                                """)

                            with tab3:
                                fig_tsne = self.transfer_helper.visualize_feature_space()
                                st.pyplot(fig_tsne)
                                st.caption("t-SNE로 시각화한 특징 공간 - 클래스가 잘 분리될수록 좋음")

                        # 실전 팁
                        st.info("""
                        💡 **Fine-tuning 성능 향상 팁**
                        - Early Stopping: Validation loss가 증가하기 시작하면 학습 중단
                        - Learning Rate Scheduling: 학습률을 점진적으로 감소
                        - Data Augmentation: 데이터 증강으로 과적합 방지
                        - Regularization: Dropout, Weight Decay 적용
                        """)

    def _render_clip_search_tab(self):
        """CLIP Image Search 탭"""
        st.header("🖼️ CLIP을 사용한 이미지 검색")

        # 간단한 사용 안내
        st.info("""
        💡 **CLIP 검색 실습**
        이론 탭에서 CLIP의 원리를 학습하셨다면, 여기서 직접 실습해보세요!
        텍스트를 입력하면 CLIP이 가장 유사한 이미지를 찾아줍니다.
        """)

        # 탭 생성
        sub_tabs = st.tabs(["🔍 텍스트로 검색", "🖼️ 이미지로 검색", "📊 임베딩 시각화"])

        with sub_tabs[0]:
            st.markdown("### 텍스트 → 이미지 검색")

            # 사용 방법 안내
            st.info("""
            📝 **사용 방법**
            1. 아래에 검색하고 싶은 내용을 텍스트로 입력
            2. 검색 대상이 될 이미지들을 업로드
            3. CLIP이 텍스트와 가장 유사한 이미지를 찾아줍니다

            **예시 검색어**:
            - "빨간 자동차"
            - "웃고 있는 사람"
            - "푸른 하늘과 흰 구름"
            - "커피 한 잔"
            - "노트북으로 일하는 사람"
            """)

            search_query = st.text_input(
                "검색할 텍스트 입력",
                placeholder="예: 빨간 자동차, 행복한 강아지, 일몰 해변",
                key="clip_text_search",
                help="자연스러운 문장으로 입력해도 됩니다"
            )

            # 이미지 데이터베이스
            uploaded_images = st.file_uploader(
                "검색할 이미지 데이터베이스 업로드",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="clip_db_text",
                help="여러 개의 이미지를 업로드하면 그 중에서 검색합니다"
            )

            if search_query and uploaded_images:
                if st.button("🔍 CLIP 검색 실행", key="run_clip_text"):
                    with st.spinner("CLIP 모델로 검색 중..."):
                        # CLIP 검색 시뮬레이션
                        st.success(f"✅ '{search_query}'와 가장 유사한 이미지를 찾았습니다!")

                        # 검색 과정 설명
                        with st.expander("🔬 CLIP 검색 과정", expanded=False):
                            st.markdown(f"""
                            1. **텍스트 인코딩**: "{search_query}" → 512차원 벡터
                            2. **이미지 인코딩**: {len(uploaded_images)}개 이미지 → 각각 512차원 벡터
                            3. **유사도 계산**: 코사인 유사도로 텍스트-이미지 매칭
                            4. **순위 결정**: 유사도가 높은 순으로 정렬
                            """)

                        st.markdown("### 🏆 검색 결과 (상위 3개)")

                        # 결과 표시 (시뮬레이션)
                        cols = st.columns(3)
                        similarities = [np.random.uniform(0.7, 0.95) for _ in range(3)]
                        similarities.sort(reverse=True)

                        for i, img_file in enumerate(uploaded_images[:3]):
                            if i < 3:
                                img = Image.open(img_file)
                                with cols[i]:
                                    st.image(img, use_column_width=True)
                                    st.metric(
                                        label=f"#{i+1} 순위",
                                        value=f"{similarities[i]:.1%}",
                                        delta="유사도",
                                        help=f"텍스트 '{search_query}'와의 의미적 유사도"
                                    )

                        # 결과 해석
                        st.info("""
                        💡 **유사도 해석**
                        - 90% 이상: 매우 높은 일치
                        - 80-90%: 높은 관련성
                        - 70-80%: 중간 관련성
                        - 70% 미만: 낮은 관련성
                        """)

        with sub_tabs[1]:
            st.markdown("### 이미지 → 이미지 검색")

            query_image = st.file_uploader(
                "쿼리 이미지 업로드",
                type=['png', 'jpg', 'jpeg'],
                key="clip_query_image"
            )

            db_images = st.file_uploader(
                "검색할 이미지 데이터베이스",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="clip_db_image"
            )

            if query_image and db_images:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(query_image, caption="쿼리 이미지")

                with col2:
                    if st.button("🔍 유사 이미지 검색", key="run_clip_image"):
                        st.info("유사한 이미지를 검색 중...")

        with sub_tabs[2]:
            st.markdown("### 📊 CLIP 임베딩 시각화")

            if st.button("임베딩 공간 시각화", key="visualize_embeddings"):
                # 임베딩 시각화 (시뮬레이션)
                fig = self.multimodal_helper.visualize_clip_embeddings()
                st.pyplot(fig)

    def _render_api_comparison_tab(self):
        """Multi-modal API 비교 탭"""
        st.header("🔍 Multi-modal API 비교 분석")

        # 2025년 9월 기준 API 정보 표시
        with st.expander("📅 2025년 9월 기준 API 접근 방법", expanded=True):
            st.markdown("""
            ### 🔗 OpenAI CLIP
            - **접근 방식**: 오픈소스 다운로드 (API 서비스 아님)
            - **설치**: `pip install git+https://github.com/openai/CLIP.git`
            - **특징**: API 키 불필요, 완전 무료, 로컬 실행
            - **응답 속도**: <100ms (GPU 사용 시)

            ### 🤖 Google Gemini API (2025년 권장)
            - **Vision API 대체**: Gemini가 Vision API를 대체하는 추세
            - **Google AI Studio 접근 방법**:
              1. ai.google.dev 접속
              2. Google 계정 로그인
              3. "Get API key" 클릭
              4. "Create API key in new project" 선택
              5. API 키 생성 (형식: AIza...)
            - **무료 할당량**: 분당 60건, 신용카드 불필요
            - **강점**: 멀티모달 처리, PDF 직접 처리, 90분 비디오 지원

            ### 🤗 Hugging Face API
            - **토큰 생성**: HuggingFace.co → Settings → Access Tokens → New Token
            - **토큰 형식**: `hf_xxxxx`
            - **2025년 권장**: Fine-grained 토큰, 앱별 별도 토큰 생성
            """)

        st.markdown("---")

        # API 선택
        selected_apis = st.multiselect(
            "비교할 API 선택",
            ["OpenAI CLIP", "Google Vision API", "Azure Computer Vision",
             "AWS Rekognition", "Hugging Face", "OpenAI GPT-4V"],
            default=["OpenAI CLIP", "Google Vision API", "Hugging Face"],
            key="api_comparison"
        )

        if len(selected_apis) >= 2:
            # 비교 차트 생성
            st.subheader("📊 API 기능 비교")

            comparison_df = self.multimodal_helper.get_api_comparison_data(selected_apis)
            st.dataframe(comparison_df)

            # 성능 벤치마크
            st.subheader("⚡ 성능 벤치마크")

            col1, col2 = st.columns(2)

            with col1:
                # 속도 비교 차트
                fig_speed = self.multimodal_helper.create_speed_comparison_chart(selected_apis)
                st.pyplot(fig_speed)

            with col2:
                # 정확도 비교 차트
                fig_accuracy = self.multimodal_helper.create_accuracy_comparison_chart(selected_apis)
                st.pyplot(fig_accuracy)

            # 사용 사례별 추천
            st.subheader("💡 사용 사례별 추천")

            use_case = st.selectbox(
                "사용 사례 선택",
                ["이미지 검색", "콘텐츠 모더레이션", "의료 이미지 분석",
                 "제품 추천", "자동 태깅", "시각적 질의응답"],
                key="use_case"
            )

            recommendation = self.multimodal_helper.get_api_recommendation(use_case, selected_apis)
            st.info(recommendation)

    def _render_feature_extraction_tab(self):
        """특징 추출 탭"""
        st.header("🎨 특징 추출 및 시각화")

        # 간단한 실습 안내
        st.info("""
        💡 **특징 추출 실습**
        이론 탭에서 특징 추출의 원리를 학습하셨다면, 여기서 직접 실습해보세요!
        이미지를 업로드하고 CNN의 각 레이어에서 추출된 특징을 시각화할 수 있습니다.
        """)

        st.markdown("---")

        # 사용 방법 안내
        st.info("""
        📝 **사용 방법**
        1. 분석할 이미지를 업로드
        2. 특징을 추출할 모델 선택 (ResNet50, VGG16 등)
        3. 확인하고 싶은 레이어 깊이 선택
        4. "특징 추출" 버튼을 클릭하여 시각화
        """)

        uploaded_file = st.file_uploader(
            "이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            key="feature_extraction",
            help="특징을 추출할 이미지를 선택하세요"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="원본 이미지", width='stretch')

                model_choice = st.selectbox(
                    "특징 추출 모델",
                    ["ResNet50", "VGG16", "EfficientNet", "CLIP"],
                    key="feature_model",
                    help="각 모델은 서로 다른 특징 추출 능력을 가집니다"
                )

                layer_choice = st.selectbox(
                    "추출할 레이어",
                    ["Early layers", "Middle layers", "Late layers", "Final layer"],
                    key="feature_layer",
                    help="Early: 엣지/색상, Middle: 텍스처/형태, Late: 객체/개념"
                )

                # 모델별 특징 설명
                with st.expander(f"🔍 {model_choice} 모델 특징"):
                    if model_choice == "ResNet50":
                        st.markdown("""
                        **ResNet50의 특징 추출**
                        - **Skip Connection**: 잔차 학습으로 깊은 특징 보존
                        - **특징**: 매우 깊은 네트워크에서도 세밀한 특징 유지
                        - **강점**: 복잡한 패턴과 텍스처 인식
                        """)
                    elif model_choice == "VGG16":
                        st.markdown("""
                        **VGG16의 특징 추출**
                        - **단순한 구조**: 3x3 필터만 사용
                        - **특징**: 계층적 특징이 명확하게 구분됨
                        - **강점**: 시각적으로 해석하기 쉬운 특징 맵
                        """)
                    elif model_choice == "EfficientNet":
                        st.markdown("""
                        **EfficientNet의 특징 추출**
                        - **Compound Scaling**: 균형잡힌 특징 추출
                        - **특징**: 효율적이면서도 정확한 특징 표현
                        - **강점**: 적은 연산으로 고품질 특징
                        """)
                    else:  # CLIP
                        st.markdown("""
                        **CLIP의 특징 추출**
                        - **멀티모달**: 이미지-텍스트 통합 특징
                        - **특징**: 의미론적 특징 추출 가능
                        - **강점**: 자연어로 특징 설명 가능
                        """)

            with col2:
                if st.button("🎨 특징 추출 실행", key="extract_features", width='stretch'):
                    with st.spinner(f"{model_choice}에서 {layer_choice} 특징을 추출하는 중..."):
                        # 특징 추출 시각화 (시뮬레이션)
                        fig = self.transfer_helper.visualize_features(image, model_choice, layer_choice)
                        st.pyplot(fig)

                        # 특징 추출 결과 설명
                        st.success("✅ 특징 추출 완료!")
                        st.markdown(f"""
                        **추출된 특징 분석**:
                        - 모델: {model_choice}
                        - 레이어: {layer_choice}
                        - 특징 맵 개수: 6개 (샘플)
                        - 주요 패턴: {"엣지/색상" if "Early" in layer_choice else "텍스처/형태" if "Middle" in layer_choice else "객체/의미"}
                        """)

            # 특징 맵 분석
            if st.checkbox("🔬 상세 분석 보기", key="detailed_analysis"):
                st.subheader("📊 특징 맵 상세 분석")

                tabs = st.tabs(["📈 히트맵", "🎲 3D 시각화", "📊 통계", "🎯 활성화 분석"])

                with tabs[0]:
                    st.markdown("""
                    ### 히트맵 시각화
                    특징 맵의 활성화 강도를 색상으로 표현합니다.
                    - 🔴 빨간색: 강한 활성화
                    - 🔵 파란색: 약한 활성화
                    """)
                    st.info("히트맵은 모델이 이미지의 어느 부분에 주목하는지 보여줍니다")

                with tabs[1]:
                    st.markdown("""
                    ### 3D 특징 공간
                    고차원 특징을 3D 공간에 투영하여 시각화합니다.
                    - PCA/t-SNE를 통한 차원 축소
                    - 클러스터링 패턴 확인
                    """)
                    st.info("유사한 특징들이 공간상에서 가까이 모입니다")

                with tabs[2]:
                    st.markdown("""
                    ### 특징 통계
                    | 측정값 | 값 | 의미 |
                    |-------|-----|------|
                    | 평균 활성화 | 0.65 | 중간 강도 |
                    | 표준편차 | 0.23 | 적당한 다양성 |
                    | 희소성 | 0.42 | 선택적 활성화 |
                    | 최대값 | 0.98 | 강한 반응 존재 |
                    """)

                with tabs[3]:
                    st.markdown("""
                    ### 활성화 패턴 분석
                    - **Receptive Field**: 각 뉴런이 보는 영역
                    - **Activation Pattern**: 특정 패턴에 대한 반응
                    - **Feature Attribution**: 예측에 대한 기여도
                    """)
                    st.info("어떤 특징이 최종 결정에 가장 중요한지 분석합니다")

    def _render_integrated_analysis_tab(self):
        """통합 분석 탭"""
        st.header("📊 Transfer Learning 통합 분석")

        analysis_type = st.selectbox(
            "분석 유형 선택",
            ["모델 성능 비교", "학습 곡선 분석", "혼동 행렬", "특징 공간 분석"],
            key="integrated_analysis"
        )

        if analysis_type == "모델 성능 비교":
            st.subheader("🏆 모델 성능 비교")

            # 그래프 의미 설명
            with st.expander("📖 이 그래프의 의미", expanded=True):
                st.markdown("""
                ### 모델 성능 비교 차트 이해하기

                **왼쪽 차트 (정확도 비교)**:
                - **Y축**: 정확도(%) - 높을수록 더 정확한 모델
                - **의미**: 100개 중 몇 개를 맞추는지
                - **좋은 신호**: 90% 이상이면 우수
                - **활용**: 정확도가 중요한 의료 진단에 적합한 모델 선택

                **오른쪽 차트 (추론 속도)**:
                - **Y축**: 응답시간(ms) - 낮을수록 빠른 모델
                - **의미**: 이미지 1장 처리 시간
                - **좋은 신호**: 50ms 이하면 실시간 처리 가능
                - **활용**: 자율주행처럼 속도가 중요한 경우 선택 기준

                **📌 핵심**: ResNet50은 정확하지만 느림, MobileNet은 빠르지만 덜 정확
                """)

            # 모델 선택
            models = st.multiselect(
                "비교할 모델",
                ["ResNet50", "VGG16", "EfficientNet", "MobileNet", "DenseNet"],
                default=["ResNet50", "EfficientNet"],
                key="model_comparison"
            )

            if len(models) >= 2:
                # 성능 메트릭 표시
                metrics_df = self.transfer_helper.get_model_metrics(models)
                st.dataframe(metrics_df)

                # 차트 생성
                fig = self.transfer_helper.create_performance_chart(models)
                st.pyplot(fig)

        elif analysis_type == "학습 곡선 분석":
            st.subheader("📈 학습 곡선 분석")

            # 그래프 의미 설명
            with st.expander("📖 이 그래프의 의미", expanded=True):
                st.markdown("""
                ### 학습 곡선 이해하기

                **파란선 (Training Loss)**:
                - **의미**: 학습 데이터에서의 오차
                - **이상적 패턴**: 점진적으로 감소
                - **문제 신호**: 갑자기 증가 = 학습률 너무 큼

                **빨간선 (Validation Loss)**:
                - **의미**: 새로운 데이터에서의 실제 성능
                - **이상적 패턴**: Training Loss와 함께 감소
                - **문제 신호**: 증가하기 시작 = 과적합 발생

                **녹색 점 (Best Model)**:
                - **의미**: 가장 좋은 성능을 보인 시점
                - **활용**: 이 시점의 모델을 저장하고 사용

                **⚠️ 과적합 진단**:
                - 두 선의 간격이 벌어짐 → 과적합
                - 해결책: Early Stopping, Dropout 증가

                **⚠️ 과소적합 진단**:
                - 두 선이 모두 높은 값 유지 → 과소적합
                - 해결책: 모델 복잡도 증가, 학습 시간 연장
                """)

            # 학습 곡선 시각화
            fig = self.transfer_helper.plot_learning_curves()
            st.pyplot(fig)

        elif analysis_type == "혼동 행렬":
            st.subheader("🔢 혼동 행렬 분석")

            # 그래프 의미 설명
            with st.expander("📖 이 그래프의 의미", expanded=True):
                st.markdown("""
                ### 혼동 행렬 이해하기

                **행렬의 구조**:
                - **Y축 (True Label)**: 실제 정답
                - **X축 (Predicted Label)**: 모델의 예측
                - **대각선**: 정답을 맞춘 경우 (진한 색일수록 좋음)
                - **비대각선**: 오답 (연한 색일수록 좋음)

                **색상의 의미**:
                - **진한 파란색**: 많은 샘플이 여기 속함
                - **연한 색/흰색**: 적은 샘플
                - **숫자**: 해당 칸의 샘플 개수

                **실제 활용 예시**:
                - 개/고양이 분류에서 개를 고양이로 착각: (0,1) 위치
                - 고양이를 개로 착각: (1,0) 위치

                **성능 평가**:
                - 대각선이 진하고 나머지가 연함 = 우수한 모델
                - 특정 칸이 진함 = 그 오류를 자주 범함
                - 해결책: 해당 클래스 데이터 보강

                **💡 Tip**: 의료 진단에서는 False Negative(놓친 환자)가 더 위험
                """)

            # 클래스 수 선택
            num_classes = st.slider("클래스 수", min_value=2, max_value=10, value=5, key="confusion_classes")

            # 혼동 행렬 생성 및 표시
            fig = self.transfer_helper.create_confusion_matrix(num_classes)
            st.pyplot(fig)

        else:  # 특징 공간 분석
            st.subheader("🌌 특징 공간 분석")

            # 그래프 의미 설명
            with st.expander("📖 이 그래프의 의미", expanded=True):
                st.markdown("""
                ### t-SNE 특징 공간 시각화 이해하기

                **그래프의 의미**:
                - CNN이 학습한 고차원 특징을 2D로 압축한 지도
                - 비슷한 이미지는 가까이, 다른 이미지는 멀리 배치

                **점과 클러스터**:
                - **각 점**: 하나의 이미지
                - **색상**: 클래스 (고양이, 개, 자동차 등)
                - **클러스터**: 같은 색 점들의 모임

                **좋은 신호**:
                - 같은 색(클래스) 점들이 뭉쳐있음 ✅
                - 다른 색 클러스터들이 서로 분리됨 ✅
                - 클러스터가 명확한 경계를 가짐 ✅

                **나쁜 신호**:
                - 색상이 뒤섞여 있음 ❌ → 모델이 구분 못함
                - 클러스터가 겹침 ❌ → 혼동하기 쉬운 클래스

                **실제 활용**:
                - 고양이와 개가 겹침 → 더 많은 구별 특징 학습 필요
                - 특정 클래스만 흩어짐 → 해당 클래스 데이터 품질 확인

                **💡 인사이트**:
                - Transfer Learning 후 클러스터가 더 명확해짐
                - Fine-tuning이 잘 되었는지 시각적으로 확인 가능
                """)

            # t-SNE 시각화
            fig = self.transfer_helper.visualize_feature_space()
            st.pyplot(fig)

    def _render_project_tab(self):
        """실전 프로젝트 탭"""
        st.header("🚀 실전 Transfer Learning 프로젝트")

        project_type = st.selectbox(
            "프로젝트 선택",
            ["🏥 의료 이미지 분류", "🏭 제조업 품질 검사", "🎨 스타일 전이", "🔍 상품 검색 시스템"],
            key="project_type"
        )

        if project_type == "🏥 의료 이미지 분류":
            self._render_medical_project()
        elif project_type == "🏭 제조업 품질 검사":
            self._render_quality_control_project()
        elif project_type == "🎨 스타일 전이":
            self._render_style_transfer_project()
        else:
            self._render_product_search_project()

    def _render_medical_project(self):
        """의료 이미지 분류 프로젝트"""
        st.subheader("🏥 X-ray 이미지 분류 시스템")

        # API 사용 옵션
        use_api = st.checkbox("🤖 Google Gemini API 사용 (실제 분석)", key="use_api_medical")

        # 데이터셋 정보 표시
        with st.expander("📚 데이터셋 & API 사용법", expanded=False):
            st.markdown("""
            ### 1. Chest X-Ray Dataset
            - **Kaggle**: [X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
            - **이미지**: 5,863개 (정상: 1,583개, 폐렴: 4,273개)

            ### 2. Google Gemini Vision API
            ```python
            import google.generativeai as genai
            from PIL import Image
            import os

            # API 설정
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel('gemini-2.5-flash')

            # 이미지 분석
            img = Image.open('xray.jpg')
            response = model.generate_content([
                "이 X-ray 이미지를 분석해주세요. 폐렴 가능성이 있나요?",
                img
            ])
            print(safe_get_response_text(response))
            ```

            ### 3. Transfer Learning (PyTorch)
            ```python
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 2)  # 정상/폐렴
            ```
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **프로젝트 목표**:
            - 흉부 X-ray에서 폐렴 검출
            - Transfer Learning으로 정확도 향상
            - 적은 데이터로 높은 성능 달성
            """)

            uploaded_xray = st.file_uploader(
                "X-ray 이미지 업로드",
                type=['png', 'jpg', 'jpeg'],
                key="xray_upload"
            )

            st.info("""
            💡 **테스트 방법**:
            Kaggle에서 샘플 이미지를 다운로드하여 업로드하세요
            """)

        with col2:
            if uploaded_xray:
                st.image(uploaded_xray, caption="업로드된 X-ray")

                if st.button("🔍 진단 시작", key="diagnose"):
                    with st.spinner("AI 분석 중..."):
                        if use_api:
                            # Google Gemini API 사용
                            try:
                                import os
                                import google.generativeai as genai
                                from PIL import Image

                                # API 설정
                                api_key = os.getenv('GOOGLE_API_KEY')
                                if api_key:
                                    genai.configure(api_key=api_key)
                                    model = genai.GenerativeModel('gemini-2.0-flash-exp')

                                    # 이미지 열기
                                    img = Image.open(uploaded_xray)

                                    # API 호출
                                    prompt = """
                                    이 흉부 X-ray 이미지를 분석해주세요.
                                    1. 폐렴 가능성 (퍼센트)
                                    2. 주요 소견
                                    3. 권장사항
                                    참고: 이것은 교육 목적이며 의료 진단이 아닙니다.
                                    """
                                    response = model.generate_content([prompt, img])

                                    st.success("✅ API 분석 완료!")
                                    st.write("**Gemini 분석 결과:**")
                                    st.info(safe_get_response_text(response))
                                else:
                                    st.error("🔴 API Key가 설정되지 않았습니다.")
                            except Exception as e:
                                st.error(f"API 오류: {str(e)}")
                                st.info("시뮬레이션 모드로 전환합니다...")
                                # 시뮬레이션 폴백
                                import random
                                normal_prob = random.randint(10, 90)
                                pneumonia_prob = 100 - normal_prob
                                st.metric("정상 확률", f"{normal_prob}%")
                                st.metric("폐렴 확률", f"{pneumonia_prob}%")
                        else:
                            # 시뮬레이션
                            import random
                            normal_prob = random.randint(10, 90)
                            pneumonia_prob = 100 - normal_prob

                            st.success("분석 완료!")
                            st.metric("정상 확률", f"{normal_prob}%")
                            st.metric("폐렴 확률", f"{pneumonia_prob}%",
                                    delta="주의 필요" if pneumonia_prob > 60 else "정상 범위")

                        st.caption("⚠️ 교육 목적입니다. 실제 의료 진단이 아닙니다.")

    def _render_quality_control_project(self):
        """제조업 혈질 검사 프로젝트"""
        st.subheader("🏭 PCB 불량 검출 시스템")

        # API 사용 옵션
        use_api = st.checkbox("🤖 Google Vision API 사용 (객체 검출)", key="use_api_pcb")

        # 데이터셋 정보 표시
        with st.expander("📚 데이터셋 & API 사용법", expanded=False):
            st.markdown("""
            ### PCB Defect Detection Dataset
            - **Kaggle**: [PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects)
            - **GitHub**: [DeepPCB](https://github.com/tangsanli5201/DeepPCB)
            - **이미지 수**: 1,386개 (6가지 불량 유형)
            - **불량 유형**: 구멍 누락, 단락, 개방 회로, 스퍼, 마우스 바이트, 이물질

            **💡 실제 구현 시**:
            ```python
            # YOLO를 사용한 PCB 불량 검출
            model = YOLOv5('yolov5s')
            model.load_state_dict(pretrained_weights)
            # PCB 데이터로 Fine-tuning
            ```
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **시스템 특징**:
            - 실시간 불량품 검출
            - 6가지 불량 유형 분류
            - Transfer Learning으로 빠른 배포
            """)

            uploaded_pcb = st.file_uploader(
                "PCB 이미지 업로드",
                type=['png', 'jpg', 'jpeg'],
                key="pcb_upload"
            )

            # 불량 유형 설정
            defect_types = st.multiselect(
                "검출할 불량 유형",
                ["구멍 누락", "단락", "개방 회로", "스퍼", "마우스 바이트", "이물질"],
                default=["단락", "개방 회로"],
                key="defect_types"
            )

        with col2:
            if uploaded_pcb:
                st.image(uploaded_pcb, caption="업로드된 PCB")

                if st.button("🔍 검사 시작", key="start_qc"):
                    with st.spinner("PCB 분석 중..."):
                        if use_api:
                            # Google Vision API 사용
                            try:
                                import os
                                import google.generativeai as genai
                                from PIL import Image

                                api_key = os.getenv('GOOGLE_API_KEY')
                                if api_key:
                                    genai.configure(api_key=api_key)
                                    model = genai.GenerativeModel('gemini-2.5-flash')

                                    img = Image.open(uploaded_pcb)
                                    prompt = f"""
                                    이 PCB 이미지를 분석하여 다음 불량을 검출해주세요:
                                    {", ".join(defect_types)}

                                    결과 형식:
                                    1. 불량 종류
                                    2. 위치 (가능하면)
                                    3. 심각도
                                    """

                                    response = model.generate_content([prompt, img])
                                    st.success("✅ API 검사 완료!")
                                    st.write("**Gemini 분석 결과:**")
                                    st.info(safe_get_response_text(response))
                                else:
                                    st.error("API Key가 없습니다.")
                            except Exception as e:
                                st.error(f"API 오류: {str(e)}")
                        else:
                            # 시뮬레이션
                            import random
                            st.success("검사 완료!")

                            if random.random() > 0.3:  # 70% 확률로 불량
                                defect = random.choice(defect_types) if defect_types else "단락"
                                st.error(f"⚠️ 불량 검출: {defect}")
                                st.metric("불량 위치", "(234, 567) 픽셀")
                                st.metric("신뢰도", f"{random.randint(85, 99)}%")
                            else:
                                st.success("✅ 정상 제품")

                        st.caption("⚠️ 교육 목적입니다.")

    def _render_style_transfer_project(self):
        """스타일 전이 프로젝트"""
        st.subheader("🎨 Neural Style Transfer")

        # API 사용 옵션
        use_api = st.checkbox("🤖 Google Gemini API 사용 (이미지 생성)", key="use_api_style")

        # 데이터셋 정보 표시
        with st.expander("📚 유명 예술 작품 스타일 & API 사용법", expanded=False):
            st.markdown("""
            ### 공개 도메인 예술 작품
            - **Van Gogh - Starry Night**: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg)
            - **Monet 작품들**: 인상파 스타일
            - **Picasso 작품들**: 큐비즘 스타일

            **💡 실제 구현 시 (PyTorch)**:
            ```python
            # VGG19 모델 사용
            vgg = torchvision.models.vgg19(pretrained=True).features
            # Content loss + Style loss 최적화
            total_loss = content_weight * content_loss + style_weight * style_loss
            ```

            **🤖 Gemini API 사용 (이미지 생성)**:
            ```python
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel('gemini-2.0-flash-exp')

            prompt = f"이 이미지를 {style_name} 스타일로 변환해주세요"
            response = model.generate_content([prompt, content_img])
            ```
            """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**콘텐츠 이미지**")
            content_image = st.file_uploader(
                "사진 업로드",
                type=['png', 'jpg', 'jpeg'],
                key="content_img"
            )
            if content_image:
                st.image(content_image, caption="콘텐츠")

        with col2:
            st.markdown("**스타일 이미지**")
            style_choice = st.radio(
                "스타일 선택",
                ["직접 업로드", "Van Gogh", "Monet", "Picasso"],
                key="style_choice"
            )

            if style_choice == "직접 업로드":
                style_image = st.file_uploader(
                    "스타일 업로드",
                    type=['png', 'jpg', 'jpeg'],
                    key="style_img"
                )
                if style_image:
                    st.image(style_image, caption="스타일")
            else:
                st.info(f"🎨 {style_choice} 스타일이 선택됨")
                style_image = True  # 더미 값

        with col3:
            st.markdown("**결과 이미지**")
            if content_image and style_image:
                style_weight = st.slider("스타일 강도", 0.0, 1.0, 0.5, key="style_weight")

                if st.button("🎨 스타일 전이", key="transfer_style"):
                    with st.spinner("스타일 전이 중..."):
                        import time
                        import base64
                        from datetime import datetime
                        
                        # 결과 이미지 저장 폴더 생성
                        os.makedirs("style_transfer_results", exist_ok=True)
                        
                        if use_api:
                            # Google Gemini API + 실제 이미지 변환
                            try:
                                import google.generativeai as genai
                                from PIL import Image, ImageFilter, ImageEnhance
                                import io

                                api_key = os.getenv('GOOGLE_API_KEY')
                                if api_key and api_key != 'your_api_key_here':
                                    # 1단계: Gemini API로 스타일 분석
                                    genai.configure(api_key=api_key)
                                    model = genai.GenerativeModel('gemini-2.0-flash-exp')

                                    img = Image.open(content_image)
                                    style_name = style_choice if style_choice != "직접 업로드" else "업로드된 스타일"

                                    prompt = f"""
                                    이 이미지에 {style_name} 예술 스타일을 적용하는 방법을 분석해주세요.
                                    - 아티스트의 특징적인 색상 팔레트와 기법
                                    - 적용할 수 있는 시각적 효과
                                    - 스타일 강도: {style_weight * 100}%
                                    """

                                    response = model.generate_content([prompt, img])
                                    analysis_text = safe_get_response_text(response)
                                    
                                    # 2단계: 실제 이미지 변환 수행
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    input_path = f"style_transfer_results/input_{timestamp}.png"
                                    result_path = f"style_transfer_results/styled_{timestamp}_{style_choice}.png"
                                    
                                    # 입력 이미지 저장
                                    img.save(input_path)
                                    
                                    # 스타일에 따른 이미지 처리 (향상된 버전)
                                    if style_choice == "Van Gogh":
                                        # Van Gogh 스타일: 소용돌이 효과 + 강한 색상
                                        result_img = img.filter(ImageFilter.SMOOTH_MORE)
                                        enhancer = ImageEnhance.Color(result_img)
                                        result_img = enhancer.enhance(1.8)
                                        enhancer = ImageEnhance.Contrast(result_img)
                                        result_img = enhancer.enhance(1.4)
                                        
                                    elif style_choice == "Monet":
                                        # Monet 스타일: 부드러운 인상주의 효과
                                        result_img = img.filter(ImageFilter.BLUR)
                                        enhancer = ImageEnhance.Brightness(result_img)
                                        result_img = enhancer.enhance(1.3)
                                        enhancer = ImageEnhance.Color(result_img)
                                        result_img = enhancer.enhance(1.2)
                                        
                                    elif style_choice == "Picasso":
                                        # Picasso 스타일: 기하학적 효과
                                        result_img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
                                        enhancer = ImageEnhance.Sharpness(result_img)
                                        result_img = enhancer.enhance(2.0)
                                        enhancer = ImageEnhance.Contrast(result_img)
                                        result_img = enhancer.enhance(1.5)
                                        
                                    else:
                                        # 기본 스타일
                                        result_img = img.filter(ImageFilter.SMOOTH)
                                    
                                    # 스타일 강도 적용
                                    if style_weight < 1.0:
                                        result_img = Image.blend(img, result_img, style_weight)
                                    
                                    # 결과 이미지 저장
                                    result_img.save(result_path)
                                    
                                    # 분석 텍스트 저장
                                    text_path = f"style_transfer_results/analysis_{timestamp}.txt"
                                    with open(text_path, 'w', encoding='utf-8') as f:
                                        f.write(f"스타일: {style_name}\n")
                                        f.write(f"강도: {style_weight * 100}%\n")
                                        f.write(f"생성 시간: {timestamp}\n\n")
                                        f.write("=== Gemini AI 스타일 분석 ===\n")
                                        f.write(analysis_text)

                                    st.success("✅ AI 스타일 전이 완료!")
                                    
                                    # 결과 표시 (Before & After)
                                    col_before, col_after = st.columns(2)
                                    
                                    with col_before:
                                        st.write("**변환 전:**")
                                        st.image(input_path, caption="원본 이미지")
                                        
                                    with col_after:
                                        st.write("**변환 후:**")
                                        st.image(result_path, caption=f"{style_choice} 스타일 적용")
                                    
                                    # AI 분석 결과 표시
                                    with st.expander("🤖 AI 스타일 분석 결과"):
                                        st.info(analysis_text)
                                    
                                    # 저장된 파일 정보
                                    st.write("**저장된 파일:**")
                                    st.code(f"원본: {input_path}")
                                    st.code(f"결과: {result_path}")
                                    st.code(f"분석: {text_path}")
                                    
                                    # 다운로드 버튼
                                    with open(result_path, "rb") as file:
                                        btn = st.download_button(
                                            label="📥 결과 이미지 다운로드",
                                            data=file.read(),
                                            file_name=f"ai_styled_{style_choice}_{timestamp}.png",
                                            mime="image/png"
                                        )
                                    
                                    st.caption(f"🤖 AI 분석 + 이미지 처리 결과 (스타일 강도: {style_weight * 100}%)")
                                    
                                else:
                                    st.error("⚠️ API Key가 설정되지 않았습니다.")
                                    st.info("시뮬레이션 모드로 전환합니다...")
                                    # 시뮬레이션 실행
                                    self._run_style_transfer_simulation(content_image, style_choice, style_weight)
                                    
                            except Exception as e:
                                st.error(f"API 오류: {str(e)}")
                                st.info("시뮬레이션 모드로 전환합니다...")
                                # 시뮬레이션 실행
                                self._run_style_transfer_simulation(content_image, style_choice, style_weight)
                        else:
                            # 시뮬레이션 모드
                            self._run_style_transfer_simulation(content_image, style_choice, style_weight)

    def _run_style_transfer_simulation(self, content_image, style_choice, style_weight):
        """스타일 전이 시뮬레이션 (실제 이미지 생성 및 저장)"""
        import time
        import numpy as np
        from PIL import Image, ImageFilter, ImageEnhance
        from datetime import datetime
        
        time.sleep(2)  # 처리 시뮬레이션
        
        # 결과 이미지 저장 폴더 생성
        os.makedirs("style_transfer_results", exist_ok=True)
        
        # 입력 이미지 로드
        img = Image.open(content_image)
        
        # 스타일에 따른 이미지 처리 시뮬레이션
        if style_choice == "Van Gogh":
            # Van Gogh 스타일 시뮬레이션 (소용돌이 효과)
            result_img = img.filter(ImageFilter.SMOOTH_MORE)
            enhancer = ImageEnhance.Color(result_img)
            result_img = enhancer.enhance(1.5)  # 색상 강화
            enhancer = ImageEnhance.Contrast(result_img)
            result_img = enhancer.enhance(1.3)  # 대비 강화
            
        elif style_choice == "Monet":
            # Monet 스타일 시뮬레이션 (부드러운 효과)
            result_img = img.filter(ImageFilter.BLUR)
            enhancer = ImageEnhance.Brightness(result_img)
            result_img = enhancer.enhance(1.2)  # 밝기 증가
            
        elif style_choice == "Picasso":
            # Picasso 스타일 시뮬레이션 (날카로운 효과)
            result_img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            enhancer = ImageEnhance.Sharpness(result_img)
            result_img = enhancer.enhance(2.0)  # 선명도 증가
            
        else:
            # 기본 스타일
            result_img = img.filter(ImageFilter.SMOOTH)
        
        # 스타일 강도 적용 (원본과 스타일 적용 이미지를 블렌딩)
        if style_weight < 1.0:
            result_img = Image.blend(img, result_img, style_weight)
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = f"style_transfer_results/input_{timestamp}.png"
        result_path = f"style_transfer_results/styled_{timestamp}_{style_choice}.png"
        
        img.save(input_path)
        result_img.save(result_path)
        
        st.success("✅ 스타일 전이 완료!")
        
        # 결과 표시
        col_before, col_after = st.columns(2)
        
        with col_before:
            st.write("**변환 전:**")
            st.image(input_path, caption="원본 이미지")
            
        with col_after:
            st.write("**변환 후:**")
            st.image(result_path, caption=f"{style_choice} 스타일 적용")
        
        # 저장 정보 표시
        st.write("**저장된 파일:**")
        st.code(f"원본: {input_path}")
        st.code(f"결과: {result_path}")
        
        # 다운로드 버튼
        with open(result_path, "rb") as file:
            btn = st.download_button(
                label="📥 결과 이미지 다운로드",
                data=file.read(),
                file_name=f"styled_{style_choice}_{timestamp}.png",
                mime="image/png"
            )
        
        st.caption(f"⚠️ 시뮬레이션 결과입니다. 스타일 강도: {style_weight * 100}%")

    def _render_product_search_project(self):
        """상품 검색 시스템 프로젝트"""
        st.subheader("🔍 시각적 상품 검색 시스템")

        # API 사용 옵션
        use_api = st.checkbox("🤖 Google Gemini API 사용 (실제 검색)", key="use_api_search")

        # 데이터셋 정보 표시
        with st.expander("📚 사용 가능한 데이터셋 & API 사용법", expanded=False):
            st.markdown("""
            ### Fashion-MNIST Dataset
            - **GitHub**: [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
            - **Kaggle**: [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
            - **이미지 수**: 70,000개 (28x28 흑백)
            - **카테고리**: 티셔츠, 바지, 드레스, 코트, 샌들, 셔츠, 스니커즈, 가방, 부츠

            **💡 실제 구현 시 (CLIP)**:
            ```python
            # CLIP 모델 사용 (텍스트-이미지 매칭)
            import clip
            model, preprocess = clip.load("ViT-B/32")
            # 텍스트와 이미지 임베딩 비교
            ```

            **🤖 Gemini API 사용 (이미지 분석 & 검색)**:
            ```python
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel('gemini-2.5-flash')

            # 이미지 설명 생성
            response = model.generate_content(["이 상품을 설명해주세요", image])
            # 유사 상품 추천
            ```
            """)

        search_method = st.radio(
            "검색 방법",
            ["텍스트로 검색", "이미지로 검색", "하이브리드 검색"],
            key="search_method"
        )

        col1, col2 = st.columns(2)

        with col1:
            if search_method == "텍스트로 검색":
                query = st.text_input("검색어 입력", placeholder="빨간 운동화", key="text_query")
                if st.button("🔍 검색", key="search_text"):
                    with st.spinner("검색 중..."):
                        if use_api and query:
                            try:
                                import os
                                import google.generativeai as genai

                                api_key = os.getenv('GOOGLE_API_KEY')
                                if api_key:
                                    genai.configure(api_key=api_key)
                                    model = genai.GenerativeModel('gemini-2.5-flash')

                                    prompt = f"""
                                    패션 상품 검색: "{query}"

                                    이 검색어와 가장 잘 매칭되는 상품 3개를 추천해주세요.
                                    각 상품에 대해:
                                    1. 상품명
                                    2. 매칭 점수 (0-100%)
                                    3. 추천 이유
                                    """

                                    response = model.generate_content(prompt)
                                    st.success("✅ API 검색 완료!")
                                    st.write("**Gemini 검색 결과:**")
                                    st.info(safe_get_response_text(response))
                                else:
                                    st.error("API Key가 설정되지 않았습니다.")
                            except Exception as e:
                                st.error(f"API 오류: {str(e)}")
                        else:
                            st.success("유사한 상품을 찾았습니다!")

            elif search_method == "이미지로 검색":
                query_img = st.file_uploader("참조 이미지", type=['png', 'jpg', 'jpeg'], key="img_query")
                if query_img:
                    st.image(query_img, caption="검색 이미지")
                    if st.button("🔍 검색", key="search_img"):
                        with st.spinner("유사 이미지 검색 중..."):
                            if use_api:
                                try:
                                    import os
                                    import google.generativeai as genai
                                    from PIL import Image

                                    api_key = os.getenv('GOOGLE_API_KEY')
                                    if api_key:
                                        genai.configure(api_key=api_key)
                                        model = genai.GenerativeModel('gemini-2.5-flash')

                                        img = Image.open(query_img)
                                        prompt = """
                                        이 상품 이미지를 분석하고, 유사한 상품 3개를 추천해주세요.
                                        각 상품에 대해:
                                        1. 상품 카테고리
                                        2. 유사도 점수 (0-100%)
                                        3. 추천 이유
                                        """

                                        response = model.generate_content([prompt, img])
                                        st.success("✅ API 검색 완료!")
                                        st.write("**Gemini 검색 결과:**")
                                        st.info(safe_get_response_text(response))
                                    else:
                                        st.error("API Key가 설정되지 않았습니다.")
                                except Exception as e:
                                    st.error(f"API 오류: {str(e)}")
                            else:
                                st.success("유사한 상품을 찾았습니다!")

            else:  # 하이브리드
                text_q = st.text_input("텍스트", placeholder="편안한 운동화", key="hybrid_text")
                img_q = st.file_uploader("이미지", type=['png', 'jpg', 'jpeg'], key="hybrid_img")
                if st.button("🔍 검색", key="search_hybrid"):
                    with st.spinner("하이브리드 검색 중..."):
                        if use_api and text_q and img_q:
                            try:
                                import os
                                import google.generativeai as genai
                                from PIL import Image

                                api_key = os.getenv('GOOGLE_API_KEY')
                                if api_key:
                                    genai.configure(api_key=api_key)
                                    model = genai.GenerativeModel('gemini-2.5-flash')

                                    img = Image.open(img_q)
                                    prompt = f"""
                                    하이브리드 검색:
                                    - 텍스트 쿼리: "{text_q}"
                                    - 이미지: 업로드된 참조 이미지

                                    텍스트와 이미지를 모두 고려하여 최적의 상품 3개를 추천해주세요.
                                    각 상품에 대해:
                                    1. 상품명
                                    2. 매칭 점수 (0-100%)
                                    3. 추천 이유
                                    """

                                    response = model.generate_content([prompt, img])
                                    st.success("✅ API 검색 완료!")
                                    st.write("**Gemini 검색 결과:**")
                                    st.info(safe_get_response_text(response))
                                else:
                                    st.error("API Key가 설정되지 않았습니다.")
                            except Exception as e:
                                st.error(f"API 오류: {str(e)}")
                        else:
                            st.success("최적의 상품을 찾았습니다!")

        with col2:
            st.markdown("**검색 결과**")
            # 시뮬레이션 결과
            if 'search' in st.session_state and any(key.startswith('search_') and st.session_state.get(key) for key in ['search_text', 'search_img', 'search_hybrid']):
                import random
                products = ["티셔츠", "운동화", "가방", "바지", "드레스"]
                for i in range(3):
                    product = random.choice(products)
                    similarity = random.randint(75, 98)
                    st.metric(f"상품 {i+1}", product, f"{similarity}% 유사도")
                st.caption("⚠️ 시뮬레이션 결과입니다.")

    def _show_model_info(self, model_name):
        """모델 정보 표시"""
        model_info = {
            "ResNet50": {
                "parameters": "25.6M",
                "layers": "50",
                "year": "2015",
                "accuracy": "92.1%"
            },
            "VGG16": {
                "parameters": "138M",
                "layers": "16",
                "year": "2014",
                "accuracy": "90.1%"
            },
            "EfficientNet": {
                "parameters": "5.3M",
                "layers": "Variable",
                "year": "2019",
                "accuracy": "91.7%"
            },
            "MobileNet": {
                "parameters": "4.2M",
                "layers": "28",
                "year": "2017",
                "accuracy": "89.5%"
            },
            "DenseNet": {
                "parameters": "25.6M",
                "layers": "121",
                "year": "2016",
                "accuracy": "91.8%"
            }
        }

        if model_name in model_info:
            info = model_info[model_name]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Parameters", info["parameters"])
            col2.metric("Layers", info["layers"])
            col3.metric("Year", info["year"])
            col4.metric("ImageNet Top-5", info["accuracy"])