"""
멀티모달 모델 벤치마크
Gemini vs GPT-4V vs Llama Vision 성능 비교
"""

import time
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import gradio as gr
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp


@dataclass
class BenchmarkResult:
    """벤치마크 결과 클래스"""
    model_name: str
    task_type: str
    response_time: float
    accuracy_score: float
    cost: float
    response_text: str
    metadata: Dict[str, Any]


class MultimodalModelBenchmark:
    """멀티모달 모델 벤치마크 시스템"""
    
    def __init__(self):
        """벤치마크 시스템 초기화"""
        self.models = {
            'gemini': GeminiVisionAPI(),
            'gpt4v': GPT4VisionAPI(),
            'llama': LlamaVisionAPI(),
            'claude': ClaudeVisionAPI()
        }
        
        self.tasks = {
            'caption': "Generate a detailed caption for this image",
            'vqa': "Answer the following question about the image: {}",
            'ocr': "Extract all text from this image",
            'object_detection': "List all objects visible in this image",
            'scene_understanding': "Describe the scene, including context and relationships",
            'reasoning': "What is unusual or noteworthy about this image?"
        }
        
        self.results = []
        
    async def benchmark_single_model(
        self,
        model_name: str,
        image: Image.Image,
        task_type: str,
        task_prompt: str = None
    ) -> BenchmarkResult:
        """
        단일 모델 벤치마크
        
        Args:
            model_name: 모델 이름
            image: 입력 이미지
            task_type: 태스크 타입
            task_prompt: 커스텀 프롬프트
        
        Returns:
            벤치마크 결과
        """
        model = self.models[model_name]
        
        # 프롬프트 준비
        if task_prompt is None:
            task_prompt = self.tasks.get(task_type, "Describe this image")
        
        # 시간 측정 시작
        start_time = time.time()
        
        try:
            # 모델 호출
            response = await model.process_image(image, task_prompt)
            response_time = time.time() - start_time
            
            # 정확도 평가 (실제 환경에서는 ground truth와 비교)
            accuracy_score = self.evaluate_response(response, task_type)
            
            # 비용 계산
            cost = model.calculate_cost(len(task_prompt), len(response))
            
            result = BenchmarkResult(
                model_name=model_name,
                task_type=task_type,
                response_time=response_time,
                accuracy_score=accuracy_score,
                cost=cost,
                response_text=response,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'image_size': image.size,
                    'prompt_length': len(task_prompt)
                }
            )
            
        except Exception as e:
            result = BenchmarkResult(
                model_name=model_name,
                task_type=task_type,
                response_time=-1,
                accuracy_score=0,
                cost=0,
                response_text=f"Error: {str(e)}",
                metadata={'error': str(e)}
            )
        
        return result
    
    async def benchmark_all_models(
        self,
        image: Image.Image,
        task_type: str,
        task_prompt: str = None
    ) -> List[BenchmarkResult]:
        """
        모든 모델 벤치마크
        
        Args:
            image: 입력 이미지
            task_type: 태스크 타입
            task_prompt: 커스텀 프롬프트
        
        Returns:
            모든 모델의 벤치마크 결과
        """
        tasks = []
        for model_name in self.models.keys():
            task = self.benchmark_single_model(model_name, image, task_type, task_prompt)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        return results
    
    def evaluate_response(self, response: str, task_type: str) -> float:
        """
        응답 평가 (간단한 휴리스틱)
        
        Args:
            response: 모델 응답
            task_type: 태스크 타입
        
        Returns:
            정확도 점수 (0-1)
        """
        # 실제 환경에서는 ground truth와 비교
        # 여기서는 간단한 휴리스틱 사용
        
        score = 0.5  # 기본 점수
        
        # 응답 길이 체크
        if len(response) > 50:
            score += 0.1
        
        # 태스크별 평가
        if task_type == 'caption':
            # 캡션 품질 체크
            if any(word in response.lower() for word in ['image', 'shows', 'contains']):
                score += 0.2
        elif task_type == 'ocr':
            # OCR 결과 체크
            if len(response.split()) > 5:
                score += 0.2
        elif task_type == 'object_detection':
            # 객체 리스트 체크
            if ',' in response or '\n' in response:
                score += 0.2
        
        # 에러 체크
        if 'error' in response.lower() or 'unable' in response.lower():
            score -= 0.3
        
        return min(max(score, 0), 1)  # 0-1 범위로 클리핑
    
    def generate_report(self) -> pd.DataFrame:
        """
        벤치마크 리포트 생성
        
        Returns:
            결과 DataFrame
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Model': result.model_name,
                'Task': result.task_type,
                'Response Time (s)': result.response_time,
                'Accuracy': result.accuracy_score,
                'Cost ($)': result.cost,
                'Timestamp': result.metadata.get('timestamp', '')
            })
        
        df = pd.DataFrame(data)
        return df
    
    def visualize_results(self) -> plt.Figure:
        """
        결과 시각화
        
        Returns:
            matplotlib Figure
        """
        df = self.generate_report()
        
        if df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No results to display', 
                   ha='center', va='center', fontsize=16)
            return fig
        
        # 4개 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 응답 시간 비교
        ax1 = axes[0, 0]
        df_pivot = df.pivot_table(values='Response Time (s)', 
                                  index='Task', columns='Model', aggfunc='mean')
        df_pivot.plot(kind='bar', ax=ax1)
        ax1.set_title('Average Response Time by Task')
        ax1.set_ylabel('Time (seconds)')
        ax1.legend(title='Model')
        ax1.grid(True, alpha=0.3)
        
        # 2. 정확도 비교
        ax2 = axes[0, 1]
        df_pivot = df.pivot_table(values='Accuracy', 
                                  index='Task', columns='Model', aggfunc='mean')
        df_pivot.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Accuracy by Task')
        ax2.set_ylabel('Accuracy Score')
        ax2.legend(title='Model')
        ax2.grid(True, alpha=0.3)
        
        # 3. 비용 비교
        ax3 = axes[1, 0]
        df_cost = df.groupby('Model')['Cost ($)'].sum()
        df_cost.plot(kind='pie', ax=ax3, autopct='%1.2f%%')
        ax3.set_title('Total Cost Distribution')
        ax3.set_ylabel('')
        
        # 4. 종합 점수 (속도와 정확도 조합)
        ax4 = axes[1, 1]
        df['Combined Score'] = df['Accuracy'] / (df['Response Time (s)'] + 0.1)
        df_pivot = df.pivot_table(values='Combined Score', 
                                  index='Model', columns='Task', aggfunc='mean')
        sns.heatmap(df_pivot, annot=True, fmt='.2f', ax=ax4, cmap='YlOrRd')
        ax4.set_title('Combined Performance Score (Accuracy/Time)')
        
        plt.tight_layout()
        return fig


# API 구현 (데모용)
class GeminiVisionAPI:
    """Google Gemini Vision API"""
    
    async def process_image(self, image: Image.Image, prompt: str) -> str:
        """이미지 처리"""
        # 실제 환경에서는 google.generativeai 사용
        await asyncio.sleep(0.5)  # API 호출 시뮬레이션
        return f"Gemini response for: {prompt[:50]}... The image shows various objects and scenes."
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """비용 계산"""
        # Gemini pricing (예시)
        return (input_tokens * 0.00001 + output_tokens * 0.00003)


class GPT4VisionAPI:
    """OpenAI GPT-4 Vision API"""
    
    async def process_image(self, image: Image.Image, prompt: str) -> str:
        """이미지 처리"""
        # 실제 환경에서는 openai 라이브러리 사용
        await asyncio.sleep(0.8)  # API 호출 시뮬레이션
        return f"GPT-4V analysis: {prompt[:50]}... This image contains detailed visual information."
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """비용 계산"""
        # GPT-4V pricing (예시)
        return (input_tokens * 0.00003 + output_tokens * 0.00006)


class LlamaVisionAPI:
    """Meta Llama Vision API (Together AI)"""
    
    async def process_image(self, image: Image.Image, prompt: str) -> str:
        """이미지 처리"""
        # 실제 환경에서는 together 라이브러리 사용
        await asyncio.sleep(0.6)  # API 호출 시뮬레이션
        return f"Llama Vision output: {prompt[:50]}... The visual elements suggest multiple interpretations."
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """비용 계산"""
        # Together AI pricing (예시)
        return (input_tokens * 0.000008 + output_tokens * 0.000016)


class ClaudeVisionAPI:
    """Anthropic Claude Vision API"""
    
    async def process_image(self, image: Image.Image, prompt: str) -> str:
        """이미지 처리"""
        # 실제 환경에서는 anthropic 라이브러리 사용
        await asyncio.sleep(0.7)  # API 호출 시뮬레이션
        return f"Claude analysis: {prompt[:50]}... I can see several interesting aspects in this image."
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """비용 계산"""
        # Claude pricing (예시)
        return (input_tokens * 0.00002 + output_tokens * 0.00004)


def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    benchmark = MultimodalModelBenchmark()
    
    async def run_benchmark(image, task_type, custom_prompt, models_to_test):
        """벤치마크 실행"""
        if image is None:
            return None, None, "Please upload an image first"
        
        # 선택된 모델만 테스트
        original_models = benchmark.models.copy()
        benchmark.models = {k: v for k, v in original_models.items() if k in models_to_test}
        
        # 벤치마크 실행
        prompt = custom_prompt if custom_prompt else None
        results = await benchmark.benchmark_all_models(image, task_type, prompt)
        
        # 원래 모델 복원
        benchmark.models = original_models
        
        # 리포트 생성
        df = benchmark.generate_report()
        
        # 시각화
        fig = benchmark.visualize_results()
        
        # 텍스트 결과
        result_text = "## Benchmark Results\n\n"
        for result in results:
            result_text += f"### {result.model_name.upper()}\n"
            result_text += f"- Response Time: {result.response_time:.2f}s\n"
            result_text += f"- Accuracy Score: {result.accuracy_score:.2%}\n"
            result_text += f"- Cost: ${result.cost:.6f}\n"
            result_text += f"- Response: {result.response_text[:200]}...\n\n"
        
        return fig, df, result_text
    
    with gr.Blocks(title="Multimodal Model Benchmark") as app:
        gr.Markdown("# 🏆 Multimodal Model Benchmark")
        gr.Markdown("Compare Gemini, GPT-4V, Llama Vision, and Claude on various vision tasks")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")
                task_type = gr.Radio(
                    choices=['caption', 'vqa', 'ocr', 'object_detection', 
                            'scene_understanding', 'reasoning'],
                    value='caption',
                    label="Task Type"
                )
                custom_prompt = gr.Textbox(
                    label="Custom Prompt (Optional)",
                    placeholder="Leave empty to use default prompt for selected task"
                )
                models_to_test = gr.CheckboxGroup(
                    choices=['gemini', 'gpt4v', 'llama', 'claude'],
                    value=['gemini', 'gpt4v', 'llama', 'claude'],
                    label="Models to Test"
                )
                run_button = gr.Button("Run Benchmark", variant="primary")
            
            with gr.Column(scale=2):
                output_plot = gr.Plot(label="Benchmark Visualization")
                output_table = gr.Dataframe(label="Results Table")
                output_text = gr.Markdown(label="Detailed Results")
        
        run_button.click(
            lambda *args: asyncio.run(run_benchmark(*args)),
            inputs=[input_image, task_type, custom_prompt, models_to_test],
            outputs=[output_plot, output_table, output_text]
        )
        
        gr.Markdown("""
        ## 📊 Benchmark Metrics
        
        ### Performance Metrics
        - **Response Time**: API call latency (lower is better)
        - **Accuracy Score**: Task-specific quality metric (higher is better)
        - **Cost**: API usage cost in USD (lower is better)
        - **Combined Score**: Accuracy / Response Time (higher is better)
        
        ### Task Types
        - **Caption**: Generate image descriptions
        - **VQA**: Visual Question Answering
        - **OCR**: Optical Character Recognition
        - **Object Detection**: Identify objects in image
        - **Scene Understanding**: Comprehensive scene analysis
        - **Reasoning**: Logical reasoning about visual content
        
        ### Model Comparison
        | Model | Strengths | Best For |
        |-------|-----------|----------|
        | **Gemini** | Fast, multilingual | General purpose, high volume |
        | **GPT-4V** | High accuracy, reasoning | Complex analysis, creative tasks |
        | **Llama** | Open source, customizable | Research, custom deployments |
        | **Claude** | Safety, detailed analysis | Content moderation, careful analysis |
        
        ### Tips
        - Test with diverse images for comprehensive comparison
        - Run multiple times to account for API variability
        - Consider both performance and cost for production use
        """)
    
    return app


if __name__ == "__main__":
    print("Multimodal Model Benchmark System")
    print("=" * 50)
    
    # 벤치마크 시스템 초기화
    benchmark = MultimodalModelBenchmark()
    
    # 테스트 이미지 생성
    test_image = Image.new('RGB', (512, 512), color='white')
    
    # 동기 테스트를 위한 래퍼
    async def test_benchmark():
        # 단일 모델 테스트
        result = await benchmark.benchmark_single_model(
            'gemini',
            test_image,
            'caption'
        )
        print(f"Single model test - {result.model_name}:")
        print(f"  Response time: {result.response_time:.2f}s")
        print(f"  Accuracy: {result.accuracy_score:.2%}")
        print(f"  Cost: ${result.cost:.6f}")
        
        # 모든 모델 테스트
        print("\nRunning full benchmark...")
        results = await benchmark.benchmark_all_models(
            test_image,
            'caption'
        )
        
        print(f"\nBenchmarked {len(results)} models")
        
        # 리포트 생성
        df = benchmark.generate_report()
        print("\nBenchmark Report:")
        print(df)
    
    # 테스트 실행
    asyncio.run(test_benchmark())
    
    # Gradio 앱 실행 (실제 환경에서)
    # app = create_gradio_interface()
    # app.launch()
    
    print("\nMultimodal benchmark system ready!")