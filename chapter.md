# 딥러닝 영상처리 강의계획서 (수정판)
**2025학년도 2학기 - 무료 AI 서비스 활용 실전 교육**

---

## 📋 과목 개요

### 기본 정보
- **과목명**: 딥러닝 영상처리 (Deep Learning for Computer Vision)
- **학점**: 3학점 (주 3시간: 이론-실습 통합)
- **담당교수**: 이석인
- **연구실**: 소통관 8303호
- **이메일**: newmind68@hs.ac.kr
- **상담시간**: 수요일 14:00-16:00 (사전 예약 권장)

### 과목 특징
"최신 AI를 무료로 마스터한다" - 2025년 공개된 무료 AI 서비스를 최대한 활용하여, 비용 부담 없이 최첨단 기술을 학습합니다. Google Gemini, Hugging Face, Meta Llama 등 대기업이 제공하는 무료 API를 활용하여 실무급 프로젝트를 완성합니다.

---

## 🎯 학습 목표

### 핵심 역량
1. **이론적 기초**: CNN부터 Transformer까지 원리 이해
2. **무료 도구 활용**: Google AI Studio, Hugging Face API 마스터
3. **실전 프로젝트**: 비용 $0로 실제 서비스 구축
4. **포트폴리오**: GitHub + Hugging Face Space 배포

---

## 🆓 100% 무료 개발 환경

### 핵심 무료 서비스 (2025년 기준)
- **Google AI Studio**: Gemini 2.5 Pro/Flash 완전 무료
- **Hugging Face**: Serverless Inference API + 월간 무료 크레딧
- **Together AI**: Llama 3.2 Vision 3개월 무료
- **Google Colab**: T4 GPU 주 30시간
- **Kaggle Notebooks**: P100 GPU 주 30시간
- **GitHub Copilot**: 학생 라이선스 무료

### 추가 무료 도구
- **Gradio/Streamlit**: 웹 데모 즉시 배포
- **Weights & Biases**: 실험 추적 (학생 무료)
- **Roboflow**: 데이터셋 라벨링 (무료 티어)

---

## 📚 주차별 학습 내용

### 제1부: 강의 및 실습 (1-10주차)

#### **1주차: Google AI Studio 입문 (워밍업)**
- **이론 파트 (90분)**:
  - Google AI Studio 소개 및 생태계 이해
  - Gemini Vision API vs Gemini 2.5 Flash Image (nano-banana) 차이점
  - 이미지 분석(이해) vs 이미지 생성(창작)의 개념
  - 모듈화 설계 원리와 프로젝트 구조
- **무료 서비스 실습 (90분)**:
  - Google AI Studio 계정 생성 (완전 무료)
  - Gemini Vision API로 이미지 분석 (10줄 코드)
  - Gemini 2.5 Flash Image로 이미지 생성/편집
  - Smart Vision App 모듈 구조 이해
  - API 키 관리 및 보안 설정
- **실습 결과물**: 
  - 이미지 분석기 (analyzer.py)
  - 이미지 생성기 (generator.py)
  - CLI 통합 애플리케이션 (main.py)

#### **2주차: 디지털 이미지 기초 + CNN 원리 + Hugging Face 입문**
- **이론 파트 (90분)**:
  - 디지털 이미지의 구조 (픽셀, RGB, 색상 공간)
  - Convolution 연산의 수학적 원리
  - CNN 핵심 구성요소 (Conv Layer, Pooling, Activation)
  - 주요 아키텍처 발전사: LeNet → AlexNet → VGG → ResNet
- **무료 서비스 실습 (90분)**:
  - Hugging Face 계정 생성 및 생태계 탐색
  - Serverless Inference API 활용법
  - 사전훈련 모델 5개 테스트 (ResNet, EfficientNet, ViT, ConvNeXt, Swin)
  - 이미지 처리 기본 연산 실습
  - **Gradio Space 배포 실습** (중점)
    - Gradio 인터페이스 디자인
    - HF Space에 실제 배포 및 공유
- **실습 결과물**: HF Space에 작동하는 이미지 분류기 공개 배포

#### **3주차: CNN 심화 + 전이학습 + CLIP 멀티모달**
- **이론 파트 (90분)**:
  - 2주차 CNN 아키텍처 심화 복습
  - Transfer Learning 원리 (사전훈련 모델 활용)
  - Fine-tuning vs Feature Extraction
  - CLIP: CNN과 Transformer의 결합
- **무료 서비스 실습 (90분)**:
  - **CLIP 상세 실습** (2주차 CNN 연계)
    - 이미지 인코더(CNN/ViT) 이해
    - 텍스트 인코더(Transformer) 이해
    - 텍스트-이미지 유사도 계산
  - CLIP으로 제로샷 분류기 구현
  - Gemini Vision과 CLIP 비교 실습
  - 멀티모달 검색 시스템 구축
- **실습 결과물**: CLIP 기반 자연어 이미지 검색 앱 (HF Space 배포)

#### **4주차: Vision Transformer + 최신 모델 비교**
- **이론 파트 (90분)**:
  - Self-Attention 메커니즘
  - Vision Transformer (ViT) 구조
  - DINO와 자기지도학습
- **무료 서비스 실습 (90분)**:
  - DINOv2 활용 (HF Serverless)
  - SAM 데모 체험 (Meta 제공)
  - Gemini vs GPT-4V vs Llama Vision 비교
  - 최적 모델 선택 가이드
- **실습 결과물**: 멀티모달 벤치마크 앱

#### **5주차: 객체 탐지 이론 + YOLO 실습**
- **이론 파트 (90분)**:
  - R-CNN 계열 발전사
  - One-stage vs Two-stage Detectors
  - NMS와 IoU 개념
- **무료 실습 (90분)**:
  - Google Colab에서 YOLOv8 실행
  - Roboflow로 데이터셋 준비 (무료 티어)
  - 커스텀 객체 5종 학습
  - Hugging Face에 모델 업로드
- **실습 결과물**: 교실 물건 탐지기

#### **6주차: 세그멘테이션 이론 + SAM 활용**
- **이론 파트 (90분)**:
  - U-Net 아키텍처
  - Instance vs Panoptic Segmentation
  - Segment Anything Model 원리
- **무료 실습 (90분)**:
  - SAM API 활용 (HF Serverless)
  - 1-클릭 배경 제거
  - Grounded-SAM으로 텍스트 기반 마스킹
  - 자동 라벨링 도구 제작
- **실습 결과물**: 증명사진 자동 편집기

#### **7주차 행동인식: Action Recognition **


#### **8주차: API 통합 + 프로젝트 기획 + 팀 구성**
- **이론 파트 (60분)**:
  - 멀티 API 조합 전략
  - 서비스 아키텍처 설계
  - 사용자 경험(UX) 고려사항
- **프로젝트 브레인스토밍 (60분)**:
  - 성공적인 AI 서비스 사례 분석
  - 창의적 아이디어 발굴 기법
  - 기술 스택 선택 가이드
  - **팀 구성** (3-4명/팀)
- **무료 실습 (60분)**:
  - 3개 이상 API 조합 실습
  - 통합 서비스 프로토타입 제작
  - 오류 처리 및 예외 상황 대응
  - 팀별 프로젝트 아이디어 초안 작성
- **실습 결과물**: 통합 AI 서비스 프로토타입 + 팀 프로젝트 제안서

#### **9주차: 생성 모델 이론 + Stable Diffusion**
- **이론 파트 (90분)**:
  - VAE와 Latent Space
  - Diffusion Process 이해
  - LoRA와 Fine-tuning
- **무료 실습 (90분)**:
  - Stable Diffusion WebUI (Colab)
  - HF Diffusers 라이브러리
  - 프롬프트 엔지니어링
  - ComfyUI 워크플로우 (무료)
- **실습 결과물**: AI 프로필 생성기

#### **10주차: 이미지 생성 심화 + 통합 서비스 완성**
- **이론 파트 (90분)**:
  - ControlNet 원리
  - Inpainting과 Outpainting
  - Multi-modal Conditioning
  - 지금까지 학습한 모든 기술 통합 전략
- **무료 실습 (90분)**:
  - Stable Diffusion API (HF)
  - DALL-E 3 대체 오픈소스
  - Gemini Imagen API 체험
  - 멀티모달 파이프라인 구축
  - **팀 프로젝트 최종 준비**
- **실습 결과물**: 상품 이미지 자동 생성 서비스 + 프로젝트 기술 스택 확정

### 제2부: 팀 프로젝트 (11-15주차)

#### **11주차: 팀 프로젝트 기획 + 역할 분담**
- **프로젝트 주제 선정 (90분)**:
  - 무료 AI API 3개 이상 조합 필수
  - Gradio/Streamlit 기반 웹 데모 구축
  - 영상처리 중심의 실용적 서비스
  - 주제 예시: 이미지 분석기, 스타일 변환기, 객체 탐지 앱 등
- **개발 계획 수립 (90분)**:
  - 기능 명세서 작성
  - 역할 분담 (AI 모델 통합, UI 디자인, 테스트, 문서화)
  - 개발 일정 수립
  - 기술 스택: Gradio/Streamlit + HF Space (백엔드 불필요)

#### **12주차: 프로젝트 개발 1차**
- **핵심 기능 구현 (180분)**:
  - AI API 연동 및 테스트
  - Gradio/Streamlit UI 구현
  - 이미지 처리 파이프라인 구축
  - 중간 결과물 테스트
- **진행 상황 점검**:
  - 팀별 5분 진행 보고
  - 기술적 어려움 공유 및 해결
  - 교수 및 다른 팀과의 상호 피드백

#### **13주차: 프로젝트 개발 2차**
- **고도화 및 최적화 (180분)**:
  - 추가 기능 구현
  - 성능 최적화 및 안정성 개선
  - 사용자 테스트 및 피드백 반영
  - 배포 준비 (HF Space, Vercel 등)
- **베타 테스트**:
  - 다른 팀이 서비스 테스트
  - 사용성 개선 사항 수집

#### **14주차: 프로젝트 완성 + 배포**
- **최종 완성 (90분)**:
  - 버그 수정 및 마무리
  - 사용자 가이드 작성
  - 프레젠테이션 자료 준비
- **배포 및 홍보 (90분)**:
  - 실제 서비스 배포
  - SNS 홍보 페이지 제작
  - 사용 통계 모니터링 설정
  - 발표 리허설

#### **15주차: 최종 발표회 + 시연**
- **팀별 최종 발표 (180분)**:
  - 팀당 15분 발표 + 5분 질의응답
  - 실시간 서비스 시연
  - 기술적 구현 내용 설명
  - 사용자 반응 및 통계 공유
  - 향후 개선 계획 제시
- **우수작 선정 및 시상**:
  - 창의성, 기술력, 완성도, 실용성 종합 평가
  - 실제 사용자 수 및 만족도 반영

---
