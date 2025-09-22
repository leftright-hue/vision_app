# Assignment 4: 멀티모달 벤치마크 앱 구축

## 📋 과제 개요

**목표**: Vision Transformer와 최신 비전 모델들을 활용하여 종합적인 멀티모달 벤치마크 시스템을 구축합니다.

**제출 기한**: 4주차 수업 종료 후 1주일

**제출 방법**: 
1. GitHub Repository (소스 코드 및 문서)
2. Hugging Face Space URL (배포된 벤치마크 앱)
3. 벤치마크 리포트 (PDF, 5-7페이지)
4. 프레젠테이션 자료 (5분 발표용)

---

## 🎯 과제 요구사항

### Part A: Vision Transformer 구현 (25점)

#### 1. ViT 모델 구현 및 분석 (15점)
- [ ] Vision Transformer 아키텍처 구현 (처음부터 또는 라이브러리 활용)
- [ ] 어텐션 메커니즘 시각화 기능
- [ ] 다양한 모델 크기 지원 (Tiny, Small, Base, Large)
- [ ] 성능 프로파일링 및 최적화

**평가 기준**:
- 구현 정확성 (5점)
- 어텐션 시각화 품질 (5점)
- 코드 품질 및 문서화 (5점)

#### 2. ViT 활용 실험 (10점)
- [ ] 이미지 분류 태스크 수행
- [ ] Transfer Learning 실험
- [ ] 성능 메트릭 측정 (정확도, 속도, 메모리)
- [ ] 다른 아키텍처와 비교 (ResNet, EfficientNet 등)

**구현 예시**:
```python
# ViT 모델 생성 및 학습
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=10,
    embed_dim=768,
    depth=12,
    n_heads=12
)

# 어텐션 시각화
attention_maps = model.get_attention_maps(image)
visualize_attention(attention_maps)

# 성능 측정
metrics = {
    'accuracy': evaluate_accuracy(model, test_loader),
    'inference_time': measure_inference_time(model),
    'memory_usage': get_memory_usage(model)
}
```

### Part B: DINOv2 활용 (25점)

#### 3. DINOv2 특징 추출 시스템 (15점)
- [ ] DINOv2 모델 통합 (Hugging Face 또는 공식 구현)
- [ ] 다양한 특징 추출 방법 구현 (CLS token, 평균 풀링, 패치별)
- [ ] 특징 시각화 (PCA, t-SNE, UMAP)
- [ ] 특징 캐싱 시스템

**평가 기준**:
- 특징 추출 정확성 (5점)
- 시각화 품질 (5점)
- 효율성 및 최적화 (5점)

#### 4. DINOv2 응용 구현 (10점)
- [ ] 이미지 검색 시스템
- [ ] 비지도 클러스터링
- [ ] Few-shot 학습 데모
- [ ] 이상 탐지 시스템

### Part C: SAM 통합 (20점)

#### 5. SAM 세그멘테이션 인터페이스 (10점)
- [ ] SAM 모델 통합
- [ ] 다양한 프롬프트 지원 (포인트, 박스, 마스크)
- [ ] 인터랙티브 세그멘테이션 UI
- [ ] 배치 처리 지원

#### 6. SAM 고급 기능 (10점)
- [ ] 자동 세그멘테이션 (everything mode)
- [ ] 세그먼트 후처리 (refinement, merging)
- [ ] 다른 모델과 통합 (CLIP으로 세그먼트 라벨링)
- [ ] 실시간 비디오 세그멘테이션

### Part D: 멀티모달 벤치마크 시스템 (25점)

#### 7. 벤치마크 프레임워크 구축 (15점)
- [ ] 최소 4개 모델 비교 (Gemini, GPT-4V, Llama Vision, Claude 등)
- [ ] 다양한 태스크 지원 (캡션, VQA, OCR, 객체 탐지 등)
- [ ] 성능 메트릭 수집 (속도, 정확도, 비용)
- [ ] 자동화된 테스트 스위트

**벤치마크 태스크 예시**:
```python
benchmark_tasks = {
    'image_captioning': {
        'prompt': "Generate a detailed caption",
        'metric': 'BLEU/METEOR score'
    },
    'visual_qa': {
        'prompt': "Answer: What is the main object?",
        'metric': 'Accuracy'
    },
    'ocr': {
        'prompt': "Extract all text from image",
        'metric': 'Character accuracy'
    },
    'object_detection': {
        'prompt': "List all objects with locations",
        'metric': 'mAP'
    },
    'reasoning': {
        'prompt': "Explain what's happening",
        'metric': 'Human evaluation'
    }
}
```

#### 8. 결과 분석 및 시각화 (10점)
- [ ] 종합 대시보드 구현
- [ ] 상세 비교 차트 (radar, heatmap, bar)
- [ ] 비용-성능 분석
- [ ] 추천 시스템 (태스크별 최적 모델)

### Part E: 통합 애플리케이션 (5점)

#### 9. Gradio/Streamlit 앱 (5점)
- [ ] 직관적인 UI/UX
- [ ] 모든 기능 통합
- [ ] 실시간 처리
- [ ] 결과 다운로드 기능

---

## 💻 구현 가이드

### Step 1: 프로젝트 구조

```
assignment4/
├── models/
│   ├── vit.py           # Vision Transformer
│   ├── dino.py          # DINOv2 wrapper
│   ├── sam.py           # SAM wrapper
│   └── multimodal.py    # API wrappers
├── benchmarks/
│   ├── tasks.py         # 벤치마크 태스크
│   ├── metrics.py       # 평가 메트릭
│   └── runner.py        # 벤치마크 실행기
├── utils/
│   ├── visualization.py # 시각화 도구
│   ├── caching.py      # 캐싱 시스템
│   └── data.py         # 데이터 로더
├── app/
│   ├── main.py         # Gradio 앱
│   └── components/     # UI 컴포넌트
├── tests/
│   └── test_*.py       # 단위 테스트
├── requirements.txt
├── README.md
└── benchmark_report.pdf
```

### Step 2: 핵심 구현 사항

```python
class MultimodalBenchmarkSystem:
    def __init__(self):
        # 모델 초기화
        self.vit_model = VisionTransformer()
        self.dino_model = DINOv2Extractor()
        self.sam_model = SAMSegmenter()
        self.api_models = {
            'gemini': GeminiAPI(),
            'gpt4v': GPT4VAPI(),
            'llama': LlamaVisionAPI(),
            'claude': ClaudeAPI()
        }
        
        # 벤치마크 태스크
        self.tasks = BenchmarkTasks()
        self.metrics = MetricCalculator()
    
    def run_comprehensive_benchmark(self, image, task_type):
        """종합 벤치마크 실행"""
        results = {}
        
        # 오픈소스 모델 테스트
        results['vit'] = self.benchmark_vit(image, task_type)
        results['dino'] = self.benchmark_dino(image, task_type)
        results['sam'] = self.benchmark_sam(image, task_type)
        
        # API 모델 테스트
        for name, model in self.api_models.items():
            results[name] = self.benchmark_api(model, image, task_type)
        
        # 결과 분석
        analysis = self.analyze_results(results)
        
        return results, analysis
    
    def generate_report(self, results):
        """벤치마크 리포트 생성"""
        report = BenchmarkReport()
        report.add_summary(results)
        report.add_visualizations(self.create_charts(results))
        report.add_recommendations(self.get_recommendations(results))
        return report
```

### Step 3: 평가 메트릭

```python
class EvaluationMetrics:
    """평가 메트릭 계산"""
    
    @staticmethod
    def calculate_accuracy(predictions, ground_truth):
        """정확도 계산"""
        pass
    
    @staticmethod
    def calculate_speed_metrics(timings):
        """속도 메트릭 계산"""
        return {
            'mean': np.mean(timings),
            'std': np.std(timings),
            'p95': np.percentile(timings, 95),
            'throughput': 1.0 / np.mean(timings)
        }
    
    @staticmethod
    def calculate_efficiency_score(accuracy, speed, cost):
        """효율성 점수 계산"""
        # Normalize metrics
        norm_accuracy = accuracy
        norm_speed = 1.0 / (speed + 0.001)
        norm_cost = 1.0 / (cost + 0.00001)
        
        # Weighted combination
        weights = {'accuracy': 0.5, 'speed': 0.3, 'cost': 0.2}
        score = (weights['accuracy'] * norm_accuracy + 
                weights['speed'] * norm_speed + 
                weights['cost'] * norm_cost)
        
        return score
```

---

## 📊 평가 기준

### 채점 루브릭 (100점)

| 카테고리 | 세부 항목 | 배점 |
|---------|-----------|------|
| **ViT 구현** | 모델 구현 및 어텐션 시각화 | 15점 |
| | 실험 및 분석 | 10점 |
| **DINOv2 활용** | 특징 추출 시스템 | 15점 |
| | 응용 구현 | 10점 |
| **SAM 통합** | 세그멘테이션 인터페이스 | 10점 |
| | 고급 기능 | 10점 |
| **벤치마크** | 프레임워크 구축 | 15점 |
| | 결과 분석 | 10점 |
| **통합 앱** | UI/UX 및 통합 | 5점 |
| **보너스** | 창의성 및 추가 기능 | +10점 |

### 보너스 포인트 기회
- 실시간 비디오 처리 구현 (+3점)
- 모바일 최적화 (+2점)
- 새로운 평가 메트릭 제안 (+2점)
- 벤치마크 데이터셋 공개 (+3점)

---

## 📝 보고서 작성 가이드

### 필수 포함 내용 (5-7페이지)

1. **Executive Summary** (1페이지)
   - 주요 발견사항
   - 모델별 강점/약점
   - 사용 추천사항

2. **기술 구현** (2페이지)
   - 시스템 아키텍처
   - 핵심 알고리즘
   - 최적화 기법

3. **벤치마크 결과** (2-3페이지)
   - 정량적 결과 (표, 차트)
   - 정성적 분석
   - 통계적 유의성

4. **응용 사례** (1페이지)
   - 실제 적용 시나리오
   - 비용-효익 분석
   - 확장 가능성

5. **결론 및 향후 과제** (1페이지)

### 보고서 템플릿

```markdown
# 멀티모달 벤치마크 시스템 리포트

## 1. Executive Summary
[핵심 발견사항 3-5개]

## 2. 시스템 설계 및 구현

### 2.1 아키텍처
[시스템 다이어그램]

### 2.2 모델 통합
- Vision Transformer: [구현 상세]
- DINOv2: [통합 방법]
- SAM: [API 사용]
- 멀티모달 APIs: [비교 대상]

## 3. 벤치마크 방법론

### 3.1 테스트 데이터셋
[데이터셋 구성 및 선택 이유]

### 3.2 평가 메트릭
[사용된 메트릭 및 정당성]

## 4. 실험 결과

### 4.1 정량적 결과
[표와 차트로 제시]

### 4.2 정성적 분석
[모델별 장단점 분석]

### 4.3 통계 분석
[유의성 검정 결과]

## 5. 실용적 권장사항

### 5.1 사용 사례별 추천
[시나리오별 최적 모델]

### 5.2 비용 최적화 전략
[효율성 개선 방안]

## 6. 결론
[주요 기여 및 향후 연구 방향]

## 참고문헌
```

---

## 🚀 제출 체크리스트

### 코드 제출
- [ ] GitHub Repository 생성 및 코드 업로드
- [ ] README.md (설치, 실행 방법 포함)
- [ ] requirements.txt (의존성 명시)
- [ ] 테스트 코드 및 커버리지 리포트
- [ ] API 키 관리 (.env 파일 사용)

### 앱 배포
- [ ] Hugging Face Space 배포
- [ ] 공개 URL 및 데모 영상
- [ ] 사용자 가이드
- [ ] 성능 모니터링 설정

### 문서
- [ ] 벤치마크 리포트 (PDF, 5-7페이지)
- [ ] 프레젠테이션 자료 (5분 발표용)
- [ ] API 문서 (선택사항)
- [ ] 시스템 설계 문서

---

## 💡 도움말 및 리소스

### 필수 읽기 자료
- [Vision Transformer 논문](https://arxiv.org/abs/2010.11929)
- [DINOv2 논문](https://arxiv.org/abs/2304.07193)
- [SAM 논문](https://arxiv.org/abs/2304.02643)
- [멀티모달 모델 서베이](https://arxiv.org/abs/2309.10020)

### 유용한 라이브러리
- `transformers`: Hugging Face 모델 라이브러리
- `timm`: PyTorch Image Models
- `segment-anything`: SAM 공식 구현
- `gradio`: 웹 인터페이스 구축

### 구현 팁

1. **성능 최적화**
   - Mixed precision training 사용
   - 모델 양자화 고려
   - 배치 처리 최적화
   - GPU 메모리 관리

2. **벤치마크 설계**
   - 다양한 난이도의 테스트 케이스
   - 통계적 유의성 확보 (반복 실험)
   - 공정한 비교 조건 설정

3. **UI/UX 개선**
   - 응답성 있는 디자인
   - 프로그레스 인디케이터
   - 에러 핸들링
   - 결과 캐싱

### 자주 묻는 질문

**Q: 모든 API를 사용해야 하나요?**
A: 최소 3개 이상의 API 비교가 필요합니다. 무료 크레딧 활용을 권장합니다.

**Q: 사전훈련 모델을 사용해도 되나요?**
A: 네, 하지만 직접 구현한 부분을 명확히 구분해주세요.

**Q: 벤치마크 데이터셋은 어떻게 구성하나요?**
A: 다양한 카테고리에서 최소 100개 이상의 이미지를 사용하세요.

**Q: 실시간 처리가 필수인가요?**
A: 필수는 아니지만 보너스 점수를 받을 수 있습니다.

---

## 📧 문의사항

- 이메일: newmind68@hs.ac.kr
- 오피스 아워: 수요일 14:00-16:00
- 조교 연락처: [조교 이메일]
- 디스코드 채널: #week4-assignment

**Good Luck! 🚀**