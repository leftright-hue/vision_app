# 🤖 Smart Vision App

AI 모델을 활용한 이미지 분석 및 생성 웹 애플리케이션

## 📋 프로젝트 개요

이 프로젝트는 Google AI Studio의 Gemini 모델을 활용하여 이미지 분석 및 생성을 수행하는 **Flask 기반 웹 애플리케이션**입니다.

### 🎯 핵심 학습 목표
- **AI 모델 활용 방법론**: Google AI Studio API 사용법
- **모듈화 설계 원리**: 단일 책임 원칙, 의존성 주입, 확장성
- **웹 애플리케이션 설계**: Flask 프레임워크와 RESTful API
- **사용자 인터페이스**: 직관적인 웹 UI 설계

## 🏗️ 프로젝트 구조

```
smart_vision_app/
├── app.py            # Flask 웹 애플리케이션
├── analyzer.py       # 이미지 분석 모듈 (Gemini Vision API)
├── generator.py      # 이미지 생성 모듈 (Gemini 2.5 Flash Image)
├── utils.py          # 유틸리티 함수 (이미지 처리, 보안)
├── config.py         # 설정 관리 (API 키, 환경 변수)
├── templates/        # HTML 템플릿
│   ├── index.html    # 메인 페이지
│   ├── ai_studio.html    # AI 사진 탐정 페이지
│   └── nano_banana.html  # 이미지 생성/편집 페이지
├── static/           # 정적 파일
│   └── generated/    # 생성된 이미지
├── requirements.txt  # 의존성 패키지
└── README.md         # 프로젝트 설명서
```

## 💡 핵심 파일들

### `app.py` - Flask 웹 애플리케이션
- **역할**: 웹 서버 및 라우팅 관리
- **핵심 기능**: RESTful API 엔드포인트, 템플릿 렌더링
- **설계 원리**: MVC 패턴, 모듈 통합

### `analyzer.py` - 이미지 분석 모듈
- **역할**: Gemini Vision API를 활용한 이미지 분석
- **핵심 기능**: 객체 검출, 감정 분석, 기술적 분석 등
- **설계 원리**: 단일 책임 원칙, 의존성 주입

### `generator.py` - 이미지 생성 모듈
- **역할**: Gemini 2.5 Flash Image를 활용한 이미지 생성/편집
- **핵심 기능**: 텍스트 프롬프트 기반 생성, 스타일 적용
- **설계 원리**: 확장 가능한 모듈 설계

### `config.py` - 설정 관리
- **역할**: 환경 변수 및 애플리케이션 설정 관리
- **핵심 기능**: API 키 관리, 파일 경로 설정
- **설계 원리**: 중앙화된 설정 관리

### `utils.py` - 유틸리티 함수
- **역할**: 공통 기능 제공 (이미지 처리, 보안, 응답 형식화)
- **핵심 기능**: 파일 검증, 이미지 최적화, 에러 처리
- **설계 원리**: 재사용 가능한 컴포넌트

## 🚀 시작하기

### 1. 환경 설정

```bash
# 1. 프로젝트 디렉터리로 이동
cd smart_vision_app

# 2. 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
# .env 파일 생성 (방법 1: .env.example 복사)
cp .env.example .env

# 또는 직접 생성 (방법 2)
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

**중요**: 
- Google AI Studio에서 API 키 발급: https://makersuite.google.com/app/apikey
- `.env` 파일의 `your_api_key_here`를 실제 API 키로 교체
- `.env` 파일은 절대 git에 커밋하지 마세요 (.gitignore에 포함됨)

### 3. Flask 서버 실행

```bash
python app.py
```

### 4. 웹 브라우저로 접속

```
http://localhost:5001
```

## 📖 사용법

### 웹 인터페이스 기능

#### 🏠 메인 페이지 (/)
- 모든 기능에 대한 통합 메뉴
- 각 기능 페이지로 쉽게 이동

#### 🕵️ AI 사진 탐정 (/ai_studio)
- 이미지 업로드 및 분석
- 커스텀 질문으로 AI와 대화
- 실시간 분석 결과 확인

#### 🍌 Nano Banana (/nano_banana)
- 텍스트 프롬프트로 이미지 생성
- 기존 이미지 편집 및 스타일 변환
- 다양한 스타일 템플릿 제공

### API 엔드포인트

#### 이미지 분석
```javascript
POST /analyze
Content-Type: multipart/form-data

FormData:
- image: 이미지 파일
- prompt: 분석 프롬프트 (선택)
```

#### 이미지 생성
```javascript
POST /generate
Content-Type: application/json

Body:
{
  "prompt": "생성할 이미지 설명"
}
```

#### 연결 테스트
```javascript
GET /test

Response:
{
  "success": true,
  "analyzer": true,
  "generator": true,
  "gemini": true
}
```

## 🔧 분석 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| `general` | 일반적인 이미지 분석 | "이 이미지를 자세히 분석해주세요" |
| `objects` | 객체 검출 | "이미지에서 보이는 모든 객체를 찾아주세요" |
| `emotions` | 감정 분석 | "이 이미지에서 느껴지는 감정을 분석해주세요" |
| `technical` | 기술적 분석 | "이미지의 기술적 특징을 분석해주세요" |
| `artistic` | 예술적 분석 | "이 이미지의 예술적 스타일을 분석해주세요" |

## 🎨 생성 스타일

| 스타일 | 설명 | 특징 |
|--------|------|------|
| `realistic` | 사실적 | 고품질, 사실적인 이미지 |
| `cartoon` | 만화 | 컬러풀하고 만화적인 스타일 |
| `artistic` | 예술적 | 화가의 작품 같은 스타일 |
| `minimalist` | 미니멀 | 단순하고 깔끔한 스타일 |
| `vintage` | 빈티지 | 레트로한 느낌의 스타일 |
| `futuristic` | 미래적 | SF적인 미래 스타일 |

## 📚 학습 체크리스트

### ✅ 기본 설정
- [ ] Python 환경 설정
- [ ] 의존성 패키지 설치
- [ ] Google API 키 발급 및 설정
- [ ] Flask 서버 실행 테스트

### ✅ 웹 인터페이스 사용
- [ ] 메인 페이지 접속
- [ ] AI 사진 탐정 기능 테스트
- [ ] 이미지 생성 기능 테스트
- [ ] API 연결 테스트

### ✅ 모듈 이해
- [ ] `app.py`의 라우팅 구조 이해
- [ ] `config.py`의 설정 관리 방식 이해
- [ ] `utils.py`의 유틸리티 함수 활용
- [ ] `analyzer.py`의 이미지 분석 로직 이해
- [ ] `generator.py`의 이미지 생성 로직 이해

### ✅ 설계 원리
- [ ] Flask 프레임워크 구조 이해
- [ ] RESTful API 설계 원칙 이해
- [ ] 모듈 간 결합도와 응집도 이해
- [ ] 웹 보안 및 에러 처리 이해

## 🔍 문제 해결

### 일반적인 문제들

**Q: API 키 오류가 발생합니다.**
A: `.env` 파일에 올바른 API 키가 설정되어 있는지 확인하세요.

**Q: 포트 5001이 이미 사용 중입니다.**
A: `app.py`에서 포트 번호를 변경하거나, 기존 프로세스를 종료하세요.

**Q: 이미지 업로드가 안 됩니다.**
A: 파일 크기가 16MB를 초과하지 않는지, 지원되는 형식인지 확인하세요.

**Q: 분석 결과가 나오지 않습니다.**
A: 인터넷 연결과 API 키 상태를 확인하고, `/test` 페이지로 연결을 테스트하세요.

## 🎯 다음 단계

이 프로젝트를 완료한 후에는:
1. **Week 2**: YOLO 객체 탐지 모듈 추가
2. **Week 3**: 실시간 처리 기능 확장
3. **Week 4**: 고급 이미지 처리 알고리즘 적용

## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

**💡 핵심 메시지**: Google AI Studio는 단지 도구일 뿐입니다. 중요한 것은 **AI 모델 활용 방법론**과 **웹 애플리케이션 설계 원리**를 이해하는 것입니다!