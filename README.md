# 🎯 Smart Vision App

AI 비전 학습 통합 플랫폼 - 이미지 처리부터 최신 AI 모델까지

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow)

## 📚 프로젝트 소개

Smart Vision App은 컴퓨터 비전과 딥러닝을 체계적으로 학습할 수 있는 통합 플랫폼입니다.
이론 학습부터 실습, 그리고 최신 AI 모델 활용까지 한 곳에서 경험할 수 있습니다.

### ✨ 주요 특징

- **📖 체계적인 커리큘럼**: 주차별로 구성된 학습 모듈
- **🔬 실습 중심**: 각 개념을 직접 코드로 구현하고 시각화
- **🤖 최신 AI 모델**: HuggingFace의 사전훈련 모델 활용
- **🎨 인터랙티브 UI**: Streamlit 기반의 직관적인 웹 인터페이스

## 🗂️ 프로젝트 구조

```
smart_vision_app/
├── app.py                      # 메인 Streamlit 애플리케이션
├── run.sh                      # 실행 스크립트
├── requirements.txt            # 의존성 패키지
│
├── core/                       # 핵심 공통 모듈
│   ├── __init__.py
│   ├── base_processor.py      # 기본 이미지 처리 클래스
│   ├── ai_models.py           # AI 모델 관리자
│   └── utils.py               # 유틸리티 함수
│
└── modules/                    # 학습 모듈
    └── week02_cnn/            # Week 2: CNN과 이미지 처리
        ├── __init__.py
        ├── cnn_module.py      # CNN 메인 모듈
        ├── filters.py         # 이미지 필터
        │
        ├── labs/              # 실습 코드
        │   ├── 01_digital_image_basics.py
        │   ├── 02_image_filtering_convolution.py
        │   ├── 03_cnn_basics.py
        │   ├── 04_cnn_visualization.py
        │   ├── 05_huggingface_models.py
        │   └── 06_integrated_project.py
        │
        └── lectures/          # 강의 자료
            └── slides.md      # 강의 슬라이드
```

## 🚀 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/yourusername/smart_vision_app.git
cd smart_vision_app

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 애플리케이션 실행

```bash
# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Mac/Linux)
source venv/bin/activate

# Streamlit 앱 실행
streamlit run app.py

# 또는 실행 스크립트 사용 (Mac/Linux)
./run.sh
```

브라우저에서 `http://localhost:8501` 접속

## 📖 학습 모듈

### ✅ Week 2: CNN과 디지털 이미지

#### 학습 내용
- **디지털 이미지 기초**: 픽셀, 색상 공간, 이미지 표현
- **이미지 필터링**: Convolution 연산, 다양한 필터 적용
- **CNN 이론**: 합성곱층, 풀링층, 활성화 함수
- **CNN 시각화**: 특징 맵, 필터 시각화
- **HuggingFace 활용**: 사전훈련 모델로 이미지 분석

#### 실습 파일
1. `01_digital_image_basics.py` - 디지털 이미지의 구조와 표현
2. `02_image_filtering_convolution.py` - 이미지 필터링과 Convolution
3. `03_cnn_basics.py` - CNN 수동 구현
4. `04_cnn_visualization.py` - CNN 시각화
5. `05_huggingface_models.py` - HuggingFace 모델 활용
6. `06_integrated_project.py` - 통합 프로젝트

### 🔜 Week 3: Transfer Learning (예정)
- 사전훈련 모델 활용
- Fine-tuning 기법
- 도메인 적응

### 🔜 Week 4: Multimodal AI (예정)
- 이미지-텍스트 통합
- CLIP 모델
- 비전-언어 태스크

## 💡 주요 기능

### 1. 이미지 분석
- 기본 속성 분석 (크기, 채널, 통계)
- 히스토그램 분석
- 색상 공간 변환

### 2. 이미지 필터링
- 블러, 샤프닝, 엣지 검출
- 커스텀 커널 적용
- 실시간 필터 비교

### 3. AI 모델 활용
- 이미지 분류
- 객체 검출
- 제로샷 분류
- 특징 추출

### 4. CNN 학습
- CNN 구조 시각화
- 특징 맵 관찰
- 필터 효과 분석

## 🛠️ 기술 스택

- **Backend**: Python 3.8+
- **Web Framework**: Streamlit
- **Deep Learning**: PyTorch, Transformers
- **Image Processing**: OpenCV, Pillow
- **Visualization**: Matplotlib, Plotly
- **AI Models**: HuggingFace Hub

## 📦 필요 패키지

```txt
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
```

## 🎯 학습 로드맵

### Phase 1: 기초 (Week 2)
- [x] 디지털 이미지 이해
- [x] Convolution 연산 학습
- [x] CNN 구조 파악
- [x] HuggingFace 모델 사용

### Phase 2: 심화 (Week 3)
- [ ] Transfer Learning
- [ ] Model Fine-tuning
- [ ] Custom Dataset 처리

### Phase 3: 고급 (Week 4)
- [ ] Multimodal Learning
- [ ] CLIP 모델 활용
- [ ] 실제 프로젝트 적용

## 📚 실습 가이드

### 개별 실습 파일 실행

```bash
# 디지털 이미지 기초 실습
python modules/week02_cnn/labs/01_digital_image_basics.py

# 이미지 필터링 실습
python modules/week02_cnn/labs/02_image_filtering_convolution.py

# CNN 시각화 실습
python modules/week02_cnn/labs/04_cnn_visualization.py
```

### 통합 앱 실행 (VIBE 코딩)

**AI에게 요청하는 방법:**

```
"통합 앱을 실행해줘"
```

또는

```
"streamlit으로 app.py 실행해줘"
```

**구체적 요청:**
```
"Smart Vision App을 streamlit으로 실행하고 브라우저에서 열어줘"
```

**AI가 자동으로 수행하는 작업:**
1. 가상환경 활성화 확인
2. `streamlit run app.py` 실행
3. 브라우저에서 `http://localhost:8501` 자동 열기
4. 사이드바에서 "Week 2: CNN" 선택 안내
5. 각 탭에서 실습 진행 방법 설명

**문제 해결 요청:**
```
"streamlit이 실행 안 돼. 해결해줘"
```

```
"ModuleNotFoundError: No module named 'streamlit' 에러가 나와"
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 🙏 감사의 말

- HuggingFace 팀의 훌륭한 모델과 라이브러리
- PyTorch 커뮤니티
- Streamlit 팀의 직관적인 프레임워크


---

**Smart Vision App** - AI 비전의 세계로 떠나는 여정 🚀