"""
멀티모달 API 성능 비교 시스템
Week 3: 딥러닝 영상처리

Google Gemini, Together AI Llama Vision, Hugging Face CLIP의 성능을 비교합니다.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from datetime import datetime
import logging

# API 클라이언트 임포트 (설치 필요)
try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai not installed. Gemini features will be disabled.")
    genai = None

try:
    import together
except ImportError:
    print("Warning: together not installed. Together AI features will be disabled.")
    together = None

from transformers import CLIPProcessor, CLIPModel
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """벤치마크 결과를 저장하는 데이터 클래스"""
    model: str
    task: str
    image_path: str
    response_time: float
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class MultimodalAPIBenchmark:
    """
    멀티모달 API 벤치마크 시스템
    
    지원 모델:
    - Google Gemini Vision
    - Together AI Llama Vision  
    - Hugging Face CLIP
    """
    
    def __init__(self):
        """API 클라이언트 초기화"""
        self.results: List[BenchmarkResult] = []
        self.models_available = {}
        
        # Gemini 초기화
        self.gemini_model = None
        if genai and os.getenv('GEMINI_API_KEY'):
            try:
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.models_available['Gemini'] = True
                logger.info("Gemini API initialized")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
                self.models_available['Gemini'] = False
        else:
            self.models_available['Gemini'] = False
            
        # Together AI 초기화
        self.together_available = False
        if together and os.getenv('TOGETHER_API_KEY'):
            try:
                together.api_key = os.getenv('TOGETHER_API_KEY')
                self.together_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
                self.together_available = True
                self.models_available['Together'] = True
                logger.info("Together AI initialized")
            except Exception as e:
                logger.warning(f"Together AI initialization failed: {e}")
                self.models_available['Together'] = False
        else:
            self.models_available['Together'] = False
            
        # CLIP 초기화
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.clip_model.to(self.device)
            self.models_available['CLIP'] = True
            logger.info(f"CLIP initialized on {self.device}")
        except Exception as e:
            logger.warning(f"CLIP initialization failed: {e}")
            self.models_available['CLIP'] = False
    
    def encode_image_base64(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def test_gemini(self, image_path: str, task: str, prompt: str) -> BenchmarkResult:
        """Gemini API 테스트"""
        start_time = time.time()
        
        if not self.models_available.get('Gemini'):
            return BenchmarkResult(
                model="Gemini",
                task=task,
                image_path=image_path,
                response_time=0,
                success=False,
                error="Gemini API not available"
            )
        
        try:
            image = Image.open(image_path)
            response = self.gemini_model.generate_content([prompt, image])
            
            response_time = time.time() - start_time
            
            return BenchmarkResult(
                model="Gemini",
                task=task,
                image_path=image_path,
                response_time=response_time,
                success=True,
                result=response.text
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return BenchmarkResult(
                model="Gemini",
                task=task,
                image_path=image_path,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    def test_together(self, image_path: str, task: str, prompt: str) -> BenchmarkResult:
        """Together AI 테스트"""
        start_time = time.time()
        
        if not self.models_available.get('Together'):
            return BenchmarkResult(
                model="Together",
                task=task,
                image_path=image_path,
                response_time=0,
                success=False,
                error="Together AI not available"
            )
        
        try:
            # 이미지를 base64로 인코딩
            image_base64 = self.encode_image_base64(image_path)
            
            # API 호출
            response = together.Complete.create(
                model=self.together_model,
                prompt=f"<image>{image_base64}</image>\n{prompt}",
                max_tokens=512,
                temperature=0.7
            )
            
            response_time = time.time() - start_time
            
            return BenchmarkResult(
                model="Together",
                task=task,
                image_path=image_path,
                response_time=response_time,
                success=True,
                result=response['output']['choices'][0]['text']
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return BenchmarkResult(
                model="Together",
                task=task,
                image_path=image_path,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    def test_clip(self, image_path: str, task: str, text_queries: List[str]) -> BenchmarkResult:
        """CLIP 테스트 (이미지-텍스트 매칭)"""
        start_time = time.time()
        
        if not self.models_available.get('CLIP'):
            return BenchmarkResult(
                model="CLIP",
                task=task,
                image_path=image_path,
                response_time=0,
                success=False,
                error="CLIP not available"
            )
        
        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_path)
            
            # 이미지와 텍스트 인코딩
            inputs = self.clip_processor(
                text=text_queries,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 유사도 계산
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # 가장 높은 확률의 텍스트 선택
            best_idx = probs.argmax().item()
            best_query = text_queries[best_idx]
            best_prob = probs[0, best_idx].item()
            
            response_time = time.time() - start_time
            
            return BenchmarkResult(
                model="CLIP",
                task=task,
                image_path=image_path,
                response_time=response_time,
                success=True,
                result={
                    'best_match': best_query,
                    'confidence': best_prob,
                    'all_scores': {q: p.item() for q, p in zip(text_queries, probs[0])}
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return BenchmarkResult(
                model="CLIP",
                task=task,
                image_path=image_path,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    def run_benchmark_suite(self, image_path: str) -> Dict[str, List[BenchmarkResult]]:
        """
        전체 벤치마크 스위트 실행
        
        테스트 태스크:
        1. Image Captioning
        2. Visual Q&A
        3. Object Detection
        4. Image Classification
        5. Text-Image Matching
        """
        results = {}
        
        # Task 1: Image Captioning
        task_name = "Image Captioning"
        results[task_name] = []
        
        caption_prompt = "Generate a detailed caption for this image in one sentence."
        
        # Gemini
        result = self.test_gemini(image_path, task_name, caption_prompt)
        results[task_name].append(result)
        self.results.append(result)
        
        # Together
        result = self.test_together(image_path, task_name, caption_prompt)
        results[task_name].append(result)
        self.results.append(result)
        
        # Task 2: Visual Q&A
        task_name = "Visual Q&A"
        results[task_name] = []
        
        qa_prompt = "What is the main subject of this image and what is it doing?"
        
        # Gemini
        result = self.test_gemini(image_path, task_name, qa_prompt)
        results[task_name].append(result)
        self.results.append(result)
        
        # Together
        result = self.test_together(image_path, task_name, qa_prompt)
        results[task_name].append(result)
        self.results.append(result)
        
        # Task 3: Object Detection
        task_name = "Object Detection"
        results[task_name] = []
        
        detection_prompt = "List all objects visible in this image."
        
        # Gemini
        result = self.test_gemini(image_path, task_name, detection_prompt)
        results[task_name].append(result)
        self.results.append(result)
        
        # Together
        result = self.test_together(image_path, task_name, detection_prompt)
        results[task_name].append(result)
        self.results.append(result)
        
        # Task 4: Image Classification (CLIP)
        task_name = "Image Classification"
        results[task_name] = []
        
        classification_labels = [
            "a photo of a person",
            "a photo of an animal",
            "a photo of a building",
            "a photo of a vehicle",
            "a photo of nature",
            "a photo of food",
            "a photo of an object"
        ]
        
        result = self.test_clip(image_path, task_name, classification_labels)
        results[task_name].append(result)
        self.results.append(result)
        
        # Task 5: Text-Image Matching (CLIP)
        task_name = "Text-Image Matching"
        results[task_name] = []
        
        matching_queries = [
            "something red",
            "something blue",
            "indoor scene",
            "outdoor scene",
            "daytime",
            "nighttime"
        ]
        
        result = self.test_clip(image_path, task_name, matching_queries)
        results[task_name].append(result)
        self.results.append(result)
        
        return results
    
    def generate_report(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        벤치마크 결과 리포트 생성
        
        Args:
            save_path: 리포트 저장 경로 (선택)
            
        Returns:
            결과 DataFrame
        """
        # DataFrame 생성
        data = []
        for result in self.results:
            data.append({
                'Model': result.model,
                'Task': result.task,
                'Response Time (s)': result.response_time,
                'Success': result.success,
                'Error': result.error
            })
        
        df = pd.DataFrame(data)
        
        # 통계 계산
        summary = df.groupby(['Model', 'Task']).agg({
            'Response Time (s)': 'mean',
            'Success': lambda x: x.mean() * 100
        }).round(3)
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(summary)
        
        # 모델별 평균 성능
        model_summary = df.groupby('Model').agg({
            'Response Time (s)': 'mean',
            'Success': lambda x: x.mean() * 100
        }).round(3)
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        print(model_summary)
        
        # 저장
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nDetailed results saved to: {save_path}")
        
        return df
    
    def visualize_results(self):
        """벤치마크 결과 시각화"""
        if not self.results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Response time by model and task
        pivot_time = df.pivot_table(
            index='task',
            columns='model',
            values='response_time',
            aggfunc='mean'
        )
        pivot_time.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Average Response Time by Task')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_xlabel('Task')
        axes[0, 0].legend(title='Model')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Success rate by model
        success_by_model = df.groupby('model')['success'].mean() * 100
        success_by_model.plot(kind='bar', ax=axes[0, 1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('Success Rate by Model')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylim([0, 105])
        
        # 3. Response time distribution
        for model in df['model'].unique():
            model_data = df[df['model'] == model]['response_time']
            axes[1, 0].hist(model_data, alpha=0.5, label=model, bins=10)
        axes[1, 0].set_title('Response Time Distribution')
        axes[1, 0].set_xlabel('Response Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Task complexity (average time across models)
        task_complexity = df.groupby('task')['response_time'].mean().sort_values()
        task_complexity.plot(kind='barh', ax=axes[1, 1], color='skyblue')
        axes[1, 1].set_title('Task Complexity (Avg Response Time)')
        axes[1, 1].set_xlabel('Average Response Time (seconds)')
        axes[1, 1].set_ylabel('Task')
        
        plt.tight_layout()
        plt.show()
    
    def export_detailed_results(self, filepath: str):
        """
        상세 결과를 JSON으로 내보내기
        
        Args:
            filepath: 저장할 파일 경로
        """
        detailed_results = []
        
        for result in self.results:
            detailed = asdict(result)
            # result 필드가 복잡한 경우 문자열로 변환
            if detailed['result'] and not isinstance(detailed['result'], str):
                detailed['result'] = json.dumps(detailed['result'], indent=2)
            detailed_results.append(detailed)
        
        with open(filepath, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Detailed results exported to: {filepath}")


def run_comprehensive_benchmark():
    """
    종합적인 벤치마크 실행
    """
    print("="*60)
    print("MULTIMODAL API BENCHMARK")
    print("="*60)
    
    # 벤치마크 시스템 초기화
    benchmark = MultimodalAPIBenchmark()
    
    # 사용 가능한 모델 출력
    print("\nAvailable Models:")
    for model, available in benchmark.models_available.items():
        status = "✓" if available else "✗"
        print(f"  {status} {model}")
    
    # 테스트 이미지 준비
    test_images = []
    
    # 샘플 이미지 생성 (실제로는 실제 이미지 사용)
    sample_dir = Path("test_images")
    sample_dir.mkdir(exist_ok=True)
    
    # 테스트용 샘플 이미지 생성
    for i in range(3):
        img = Image.new('RGB', (512, 512), 
                       color=(np.random.randint(100, 255),
                             np.random.randint(100, 255),
                             np.random.randint(100, 255)))
        img_path = sample_dir / f"test_{i}.jpg"
        img.save(img_path)
        test_images.append(str(img_path))
    
    # 벤치마크 실행
    print("\nRunning benchmarks...")
    all_results = {}
    
    for idx, image_path in enumerate(test_images, 1):
        print(f"\nTesting image {idx}/{len(test_images)}: {Path(image_path).name}")
        results = benchmark.run_benchmark_suite(image_path)
        all_results[image_path] = results
        
        # 각 태스크별 결과 출력
        for task, task_results in results.items():
            print(f"\n  {task}:")
            for result in task_results:
                if result.success:
                    print(f"    {result.model}: {result.response_time:.3f}s ✓")
                else:
                    print(f"    {result.model}: Failed - {result.error}")
    
    # 리포트 생성
    print("\n" + "="*60)
    df = benchmark.generate_report(save_path="benchmark_results.csv")
    
    # 시각화
    if len(benchmark.results) > 0:
        benchmark.visualize_results()
    
    # 상세 결과 내보내기
    benchmark.export_detailed_results("detailed_results.json")
    
    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60)


if __name__ == "__main__":
    # API 키 설정 확인
    api_keys_needed = []
    
    if not os.getenv('GEMINI_API_KEY'):
        api_keys_needed.append("GEMINI_API_KEY")
    
    if not os.getenv('TOGETHER_API_KEY'):
        api_keys_needed.append("TOGETHER_API_KEY")
    
    if api_keys_needed:
        print("Please set the following environment variables:")
        for key in api_keys_needed:
            print(f"  export {key}='your_api_key_here'")
        print("\nOr create a .env file with these keys.")
        print("Some features will be disabled without these API keys.")
    
    # 벤치마크 실행
    run_comprehensive_benchmark()