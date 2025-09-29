#!/usr/bin/env python3
"""
Week 4 Lab: Vision Model 통합 벤치마크 앱
Gradio를 사용한 실시간 Vision 모델 성능 비교 웹 애플리케이션

이 앱에서는:
1. 다양한 Vision 모델 실시간 비교
2. 성능 메트릭 시각화
3. 사용자 친화적 인터페이스
4. HuggingFace Space 배포 준비
"""

import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import io
import base64
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 전역 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 벤치마크 앱 시작 - 디바이스: {DEVICE}")

class ModelBenchmark:
    """
    Vision 모델 벤치마크 클래스
    """
    
    def __init__(self):
        """벤치마크 초기화"""
        self.models = {}
        self.model_info = {
            "resnet50": {
                "name": "ResNet-50",
                "type": "CNN",
                "params": "25.6M",
                "description": "깊은 잔차 네트워크, 이미지 분류의 기준점"
            },
            "efficientnet_b4": {
                "name": "EfficientNet-B4", 
                "type": "CNN",
                "params": "19.3M",
                "description": "효율적인 CNN 아키텍처, 모바일 최적화"
            },
            "vit_base": {
                "name": "ViT-Base/16",
                "type": "Transformer",
                "params": "86.6M", 
                "description": "Vision Transformer, 패치 기반 어텐션"
            },
            "dinov2": {
                "name": "DINOv2-Base",
                "type": "Self-Supervised",
                "params": "86.6M",
                "description": "자기지도학습 Vision Transformer"
            }
        }
        self.load_models()
    
    def load_models(self):
        """모델 로드"""
        print("🔄 모델 로딩 중...")
        
        try:
            # ResNet-50
            self.models["resnet50"] = models.resnet50(pretrained=True).to(DEVICE).eval()
            print("✅ ResNet-50 로드 완료")
            
            # EfficientNet-B4
            self.models["efficientnet_b4"] = models.efficientnet_b4(pretrained=True).to(DEVICE).eval()
            print("✅ EfficientNet-B4 로드 완료")
            
            # ViT (HuggingFace)
            try:
                from transformers import ViTModel, ViTImageProcessor
                self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
                self.models["vit_base"] = ViTModel.from_pretrained('google/vit-base-patch16-224').to(DEVICE).eval()
                print("✅ ViT-Base 로드 완료")
            except Exception as e:
                print(f"⚠️ ViT 로드 실패: {e}")
            
            # DINOv2
            try:
                self.models["dinov2"] = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(DEVICE).eval()
                print("✅ DINOv2 로드 완료")
            except Exception as e:
                print(f"⚠️ DINOv2 로드 실패: {e}")
                
        except Exception as e:
            print(f"❌ 모델 로드 중 오류: {e}")
    
    def preprocess_image(self, image, model_name):
        """모델별 이미지 전처리"""
        if model_name == "vit_base" and hasattr(self, 'vit_processor'):
            # ViT 전용 전처리
            inputs = self.vit_processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].to(DEVICE)
        else:
            # 표준 ImageNet 전처리
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            if isinstance(image, Image.Image):
                tensor = transform(image).unsqueeze(0).to(DEVICE)
            else:
                tensor = image.to(DEVICE)
            
            return tensor
    
    def benchmark_single_model(self, model_name, image, num_runs=10):
        """단일 모델 벤치마크"""
        if model_name not in self.models:
            return {"error": f"모델 {model_name}을 찾을 수 없습니다."}
        
        model = self.models[model_name]
        
        try:
            # 이미지 전처리
            input_tensor = self.preprocess_image(image, model_name)
            
            # 추론 시간 측정
            times = []
            memory_usage = []
            
            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_tensor)
            
            # 실제 측정
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # ms
                
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated() / (1024 * 1024))  # MB
            
            # 결과 정리
            result = {
                "model": model_name,
                "model_name": self.model_info[model_name]["name"],
                "avg_time": round(np.mean(times), 2),
                "std_time": round(np.std(times), 2),
                "min_time": round(np.min(times), 2),
                "max_time": round(np.max(times), 2),
                "avg_memory": round(np.mean(memory_usage), 2) if memory_usage else "N/A",
                "output_shape": str(output.shape) if hasattr(output, 'shape') else "N/A",
                "success": True
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def run_comprehensive_benchmark(self, image):
        """종합 벤치마크 실행"""
        results = []
        
        for model_name in self.models.keys():
            print(f"🔄 {self.model_info[model_name]['name']} 벤치마킹...")
            result = self.benchmark_single_model(model_name, image)
            
            if result.get("success", False):
                # 모델 정보 추가
                result.update(self.model_info[model_name])
                results.append(result)
                print(f"✅ {result['model_name']}: {result['avg_time']}ms")
            else:
                print(f"❌ {self.model_info[model_name]['name']}: {result.get('error', 'Unknown error')}")
        
        return results
    
    def create_comparison_chart(self, results):
        """비교 차트 생성"""
        if not results:
            return None
        
        df = pd.DataFrame(results)
        
        # 차트 생성
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 추론 시간 비교
        bars1 = axes[0, 0].bar(df['model_name'], df['avg_time'], 
                              yerr=df['std_time'], capsize=5, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('평균 추론 시간', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('시간 (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        for bar, time_val in zip(bars1, df['avg_time']):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{time_val}ms', ha='center', va='bottom', fontweight='bold')
        
        # 2. 메모리 사용량 (CUDA 사용 시)
        if df['avg_memory'].dtype != 'object':
            bars2 = axes[0, 1].bar(df['model_name'], df['avg_memory'], 
                                  alpha=0.7, color='lightcoral')
            axes[0, 1].set_title('메모리 사용량', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('메모리 (MB)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, mem_val in zip(bars2, df['avg_memory']):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               f'{mem_val}MB', ha='center', va='bottom', fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'GPU 메모리 정보 없음\n(CPU 모드)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('메모리 사용량')
        
        # 3. 모델 파라미터 수
        param_counts = []
        for _, row in df.iterrows():
            param_str = row['params'].replace('M', '')
            param_counts.append(float(param_str))
        
        bars3 = axes[0, 2].bar(df['model_name'], param_counts, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title('모델 파라미터 수', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('파라미터 (M)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        for bar, param_val in zip(bars3, param_counts):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{param_val}M', ha='center', va='bottom', fontweight='bold')
        
        # 4. 처리량 (FPS)
        fps_values = [1000 / time_val for time_val in df['avg_time']]
        bars4 = axes[1, 0].bar(df['model_name'], fps_values, alpha=0.7, color='gold')
        axes[1, 0].set_title('처리량 (FPS)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('FPS')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, fps_val in zip(bars4, fps_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{fps_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. 모델 타입별 분포
        type_counts = df['type'].value_counts()
        axes[1, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
                      colors=['lightblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 1].set_title('모델 타입 분포', fontsize=14, fontweight='bold')
        
        # 6. 종합 성능 점수
        # 점수 = (FPS * 0.4) + (1/Memory * 0.3) + (1/Params * 0.3)
        composite_scores = []
        for i, row in df.iterrows():
            fps = fps_values[i]
            memory = row['avg_memory'] if isinstance(row['avg_memory'], (int, float)) else 100
            params = param_counts[i]
            
            score = (fps * 0.4) + (100/memory * 0.3) + (100/params * 0.3)
            composite_scores.append(score)
        
        bars6 = axes[1, 2].bar(df['model_name'], composite_scores, alpha=0.7, color='purple')
        axes[1, 2].set_title('종합 성능 점수', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('점수')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        for bar, score_val in zip(bars6, composite_scores):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{score_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 이미지로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return Image.open(buf)
    
    def generate_report(self, results):
        """벤치마크 리포트 생성"""
        if not results:
            return "❌ 벤치마크 결과가 없습니다."
        
        report = "# 🚀 Vision Model 벤치마크 리포트\n\n"
        report += f"**테스트 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**테스트 디바이스**: {DEVICE}\n"
        report += f"**테스트된 모델 수**: {len(results)}\n\n"
        
        report += "## 📊 성능 요약\n\n"
        report += "| 모델 | 타입 | 파라미터 | 평균 시간 | 메모리 | FPS |\n"
        report += "|------|------|----------|-----------|--------|-----|\n"
        
        for result in results:
            fps = round(1000 / result['avg_time'], 1)
            memory = result['avg_memory'] if isinstance(result['avg_memory'], (int, float)) else "N/A"
            
            report += f"| {result['model_name']} | {result['type']} | {result['params']} | "
            report += f"{result['avg_time']}ms | {memory} | {fps} |\n"
        
        # 최고 성능 모델들
        fastest_model = min(results, key=lambda x: x['avg_time'])
        most_efficient = min([r for r in results if isinstance(r['avg_memory'], (int, float))], 
                           key=lambda x: x['avg_memory'], default=fastest_model)
        
        report += f"\n## 🏆 성능 하이라이트\n\n"
        report += f"**⚡ 가장 빠른 모델**: {fastest_model['model_name']} ({fastest_model['avg_time']}ms)\n"
        report += f"**💾 메모리 효율적**: {most_efficient['model_name']} ({most_efficient['avg_memory']}MB)\n"
        
        # 추천 사항
        report += f"\n## 💡 추천 사항\n\n"
        report += f"- **실시간 처리**: {fastest_model['model_name']} (가장 빠른 추론)\n"
        report += f"- **모바일/엣지**: EfficientNet-B4 (효율성과 성능의 균형)\n"
        report += f"- **고품질 특징**: DINOv2 (자기지도학습으로 학습된 범용 특징)\n"
        report += f"- **전이학습**: ViT-Base (Transformer 기반 우수한 전이 성능)\n"
        
        return report

# 전역 벤치마크 인스턴스
benchmark = ModelBenchmark()

def run_benchmark(image):
    """벤치마크 실행 함수 (Gradio 인터페이스용)"""
    if image is None:
        return "❌ 이미지를 업로드해주세요.", None, "결과 없음"
    
    try:
        # 벤치마크 실행
        results = benchmark.run_comprehensive_benchmark(image)
        
        if not results:
            return "❌ 벤치마크 실행 실패", None, "결과 없음"
        
        # 차트 생성
        chart = benchmark.create_comparison_chart(results)
        
        # 리포트 생성
        report = benchmark.generate_report(results)
        
        # 결과 테이블 HTML 생성
        df = pd.DataFrame(results)
        table_html = df[['model_name', 'type', 'params', 'avg_time', 'avg_memory']].to_html(
            index=False, classes='benchmark-table', escape=False,
            table_id='benchmark-results'
        )
        
        # CSS 스타일 추가
        styled_table = f"""
        <style>
        .benchmark-table {{
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            min-width: 400px;
            border-radius: 5px 5px 0 0;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }}
        .benchmark-table thead tr {{
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }}
        .benchmark-table th,
        .benchmark-table td {{
            padding: 12px 15px;
        }}
        .benchmark-table tbody tr {{
            border-bottom: 1px solid #dddddd;
        }}
        .benchmark-table tbody tr:nth-of-type(even) {{
            background-color: #f3f3f3;
        }}
        .benchmark-table tbody tr:last-of-type {{
            border-bottom: 2px solid #009879;
        }}
        </style>
        {table_html}
        """
        
        return styled_table, chart, report
        
    except Exception as e:
        error_msg = f"❌ 벤치마크 실행 중 오류 발생: {str(e)}"
        return error_msg, None, error_msg

def create_sample_image():
    """샘플 이미지 생성"""
    # 간단한 테스트 이미지 생성
    img = Image.new('RGB', (224, 224), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # 다양한 패턴 그리기
    draw.ellipse([50, 50, 174, 174], fill='red', outline='black', width=2)
    draw.rectangle([100, 100, 150, 150], fill='blue', outline='black', width=2)
    draw.polygon([(112, 180), (137, 140), (162, 180)], fill='green', outline='black')
    
    return img

def create_gradio_interface():
    """Gradio 웹 인터페이스 생성"""
    
    # 커스텀 CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .benchmark-button {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        border: none;
        border-radius: 25px;
        color: white;
        padding: 15px 30px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .benchmark-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    """
    
    with gr.Blocks(title="🚀 Vision Model Benchmark", theme=gr.themes.Soft(), css=custom_css) as demo:
        
        # 헤더
        gr.HTML("""
        <div class="main-header">
            <h1>🚀 Vision Model Benchmark</h1>
            <p>실시간으로 다양한 Vision 모델의 성능을 비교해보세요!</p>
        </div>
        """)
        
        # 모델 정보 표시
        with gr.Row():
            gr.Markdown(f"""
            ## 📋 지원 모델
            
            | 모델 | 타입 | 파라미터 | 설명 |
            |------|------|----------|------|
            | **ResNet-50** | CNN | 25.6M | 깊은 잔차 네트워크, 이미지 분류의 기준점 |
            | **EfficientNet-B4** | CNN | 19.3M | 효율적인 CNN 아키텍처, 모바일 최적화 |
            | **ViT-Base/16** | Transformer | 86.6M | Vision Transformer, 패치 기반 어텐션 |
            | **DINOv2-Base** | Self-Supervised | 86.6M | 자기지도학습 Vision Transformer |
            
            **측정 지표**: 추론 시간, 메모리 사용량, 처리량(FPS), 종합 성능 점수
            """)
        
        with gr.Row():
            # 입력 섹션
            with gr.Column(scale=1):
                gr.Markdown("### 📸 이미지 업로드")
                
                image_input = gr.Image(
                    type="pil",
                    label="테스트 이미지",
                    height=300
                )
                
                with gr.Row():
                    benchmark_btn = gr.Button(
                        "🔥 벤치마크 실행",
                        variant="primary",
                        size="lg",
                        elem_classes=["benchmark-button"]
                    )
                    
                    sample_btn = gr.Button(
                        "🎨 샘플 이미지",
                        variant="secondary"
                    )
                
                # 시스템 정보
                gr.Markdown(f"""
                ### 💻 시스템 정보
                - **디바이스**: {DEVICE}
                - **PyTorch**: {torch.__version__}
                - **로드된 모델**: {len(benchmark.models)}개
                """)
            
            # 결과 섹션
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 성능 차트"):
                        performance_chart = gr.Image(
                            label="성능 비교 차트",
                            height=600
                        )
                    
                    with gr.Tab("📋 결과 테이블"):
                        results_table = gr.HTML(
                            label="벤치마크 결과",
                            value="<p>벤치마크를 실행하면 결과가 여기에 표시됩니다.</p>"
                        )
                    
                    with gr.Tab("📝 상세 리포트"):
                        detailed_report = gr.Markdown(
                            value="벤치마크를 실행하면 상세 리포트가 여기에 표시됩니다."
                        )
        
        # 예제 이미지
        gr.Examples(
            examples=[
                [create_sample_image()],
            ],
            inputs=[image_input],
            label="예제 이미지"
        )
        
        # 추가 정보
        with gr.Accordion("ℹ️ 사용 가이드", open=False):
            gr.Markdown("""
            ### 🔧 사용 방법
            1. **이미지 업로드**: 테스트할 이미지를 업로드하거나 샘플 이미지를 사용하세요.
            2. **벤치마크 실행**: '벤치마크 실행' 버튼을 클릭하여 성능 측정을 시작하세요.
            3. **결과 확인**: 차트, 테이블, 리포트 탭에서 결과를 확인하세요.
            
            ### 📈 측정 지표 설명
            - **추론 시간**: 단일 이미지 처리에 걸리는 평균 시간 (ms)
            - **메모리 사용량**: GPU 메모리 사용량 (MB, CUDA 사용 시)
            - **처리량(FPS)**: 초당 처리 가능한 이미지 수
            - **종합 점수**: 속도, 메모리 효율성, 모델 크기를 종합한 점수
            
            ### 💡 모델 선택 가이드
            - **실시간 처리**: ResNet-50 (빠른 추론)
            - **모바일/엣지**: EfficientNet-B4 (효율성)
            - **고품질 특징**: DINOv2 (자기지도학습)
            - **전이학습**: ViT-Base (Transformer)
            """)
        
        # 이벤트 연결
        benchmark_btn.click(
            fn=run_benchmark,
            inputs=[image_input],
            outputs=[results_table, performance_chart, detailed_report],
            show_progress=True
        )
        
        sample_btn.click(
            fn=lambda: create_sample_image(),
            outputs=[image_input]
        )
    
    return demo

def main():
    """메인 함수"""
    print("🚀 Vision Model Benchmark App 시작")
    print("=" * 50)
    
    # Gradio 인터페이스 생성
    demo = create_gradio_interface()
    
    # 앱 실행
    print("🌐 웹 인터페이스 시작 중...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # 공개 링크 생성
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
