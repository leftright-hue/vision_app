# Smart Vision App 환경 설정 가이드

## VIBE 코딩으로 환경 설정하기

### 1. 프로젝트 클론 후 AI에게 요청하기

**간단한 요청 방법:**
```
"이 프로젝트를 실행하기 위한 환경을 설정해줘"
```

**구체적인 요청 방법:**
```
"smart_vision_app 프로젝트를 위한 Python 가상환경을 만들고 requirements.txt에 있는 패키지들을 설치해줘"
```

### 2. AI가 자동으로 수행하는 작업들

AI에게 위와 같이 요청하면 다음 작업들을 자동으로 수행합니다:

1. **Python 가상환경 생성**
   - Windows: `python -m venv venv`
   - Mac/Linux: `python3 -m venv venv`

2. **가상환경 활성화**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **패키지 설치**
   - `pip install -r requirements.txt`

4. **환경 변수 설정**
   - `.env` 파일 생성
   - Google API Key 설정 안내

### 3. VS Code에서 가상환경 활성화

**가상환경이 활성화되지 않았을 때:**
```
"가상환경을 활성화해줘"
```

또는
```
"venv를 활성화해서 패키지를 실행할 수 있게 해줘"
```

**VS Code에서 Python 인터프리터 설정:**
```
"VS Code에서 가상환경을 Python 인터프리터로 설정해줘"
```

AI가 자동으로:
1. 터미널에서 `venv\Scripts\activate` (Windows) 또는 `source venv/bin/activate` (Mac/Linux) 실행
2. VS Code 하단의 Python 인터프리터를 venv로 변경하도록 안내
3. Ctrl+Shift+P → "Python: Select Interpreter" → venv 선택 안내

### 4. 다양한 요청 예시

**기본 설정:**
```
"파이썬 가상환경 만들고 패키지 설치해줘"
```

**가상환경 활성화:**
```
"가상환경 활성화해줘"
```

**문제 해결:**
```
"pip install이 안 돼. 해결해줘"
```

**패키지 업데이트:**
```
"requirements.txt의 패키지들을 최신 버전으로 업데이트해줘"
```

**환경 확인:**
```
"현재 설치된 패키지들 확인하고 requirements.txt와 비교해줘"
```

**Google API 키 설정:**
```
"Google AI Studio API 키 설정하는 방법 알려줘"
```

### 4. 주요 패키지 설명

AI가 설치하는 주요 패키지들:
- `google-generativeai`: Google AI Studio 연동
- `opencv-python`: 이미지 처리
- `numpy`: 수치 연산
- `pillow`: 이미지 조작
- `matplotlib`: 데이터 시각화
- `scikit-image`: 고급 이미지 처리
- `torch`, `torchvision`: 딥러닝 (CNN)
- `transformers`: Hugging Face 모델

### 5. 환경 설정 완료 확인

**AI에게 확인 요청:**
```
"환경이 제대로 설정됐는지 확인해줘"
```

AI가 다음을 확인합니다:
- Python 버전
- 가상환경 활성화 상태
- 필수 패키지 설치 여부
- .env 파일 존재 여부
- Google API Key 설정 상태

### 6. 실습 시작하기

환경 설정이 완료되면:
```
"Week 02 CNN 실습을 시작하고 싶어"
```

또는:
```
"modules/week02_cnn/labs에 있는 실습 파일들 실행해줘"
```

### 7. 트러블슈팅

**일반적인 문제들:**

**Python이 없을 때:**
```
"Python이 설치 안 되어 있어. 설치 방법 알려줘"
```

**가상환경 활성화 안 될 때:**
```
"가상환경 활성화가 안 돼. 도와줘"
```

**VS Code에서 모듈을 찾을 수 없을 때:**
```
"ModuleNotFoundError가 나와. 가상환경이 제대로 설정됐는지 확인해줘"
```

**터미널에 (venv)가 안 보일 때:**
```
"터미널에 (venv) 표시가 없어. 가상환경 활성화해줘"
```

**패키지 충돌:**
```
"패키지 설치 중 에러가 났어. 해결해줘"
```

## 핵심 포인트

✅ **복잡한 명령어 외울 필요 없음** - AI가 알아서 처리
✅ **에러 발생시 바로 해결** - 에러 메시지 복사해서 AI에게 전달
✅ **OS별 차이 자동 처리** - Windows/Mac/Linux 자동 감지
✅ **단계별 설명 제공** - AI가 각 단계를 설명하며 진행

## 빠른 시작 (One-liner)

```
"git clone한 smart_vision_app 프로젝트 환경 설정하고 실행 준비해줘"
```

이 한 줄이면 AI가 모든 설정을 자동으로 완료합니다!