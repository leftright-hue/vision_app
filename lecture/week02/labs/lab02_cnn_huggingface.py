"""
Week 2: CNN 원리 + Hugging Face 생태계 실습
딥러닝 영상처리 강의 - 2주차 실습 코드

실습 목표:
1. CNN 아키텍처의 수동 구현 및 이해
2. Hugging Face Serverless Inference API 활용
3. 사전훈련 모델 테스트 (CLIP, BERT, ViT)
4. Gradio를 통한 웹 인터페이스 구축
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import json
import io
import os
from dotenv import load_dotenv
from transformers import (
    CLIPProcessor, CLIPModel,
    BertTokenizer, BertModel,
    ViTImageProcessor, ViTForImageClassification
)
import gradio as gr

# 환경 변수 로드
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

class CNNLab:
    """CNN 실습을 위한 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {self.device}")
        
    def visualize_convolution(self):
        """Convolution 연산 과정 시각화"""
        print("=== Convolution 연산 시각화 ===")
        
        # 입력 이미지 생성 (5x5)
        input_img = torch.tensor([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 엣지 검출 커널
        edge_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Convolution 연산
        output = F.conv2d(input_img, edge_kernel, padding=1)
        
        # 결과 출력
        print("입력 이미지:")
        print(input_img.squeeze())
        print("\n엣지 검출 커널:")
        print(edge_kernel.squeeze())
        print("\nConvolution 결과:")
        print(output.squeeze())
        
        return input_img, edge_kernel, output
    
    def build_simple_cnn(self):
        """간단한 CNN 모델 구축"""
        print("\n=== 간단한 CNN 모델 구축 ===")
        
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 6, 5)  # 1채널 → 6채널, 5x5 커널
                self.conv2 = nn.Conv2d(6, 16, 5)  # 6채널 → 16채널, 5x5 커널
                self.pool = nn.MaxPool2d(2, 2)    # 2x2 MaxPooling
                self.fc1 = nn.Linear(16 * 4 * 4, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
                
            def forward(self, x):
                # 첫 번째 Convolution + Pooling
                x = self.pool(F.relu(self.conv1(x)))
                
                # 두 번째 Convolution + Pooling
                x = self.pool(F.relu(self.conv2(x)))
                
                # Flatten
                x = x.view(-1, 16 * 4 * 4)
                
                # Fully Connected Layers
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                
                return x
        
        model = SimpleCNN().to(self.device)
        
        # 모델 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        print(f"모델 파라미터 수: {total_params:,}")
        
        # 테스트
        dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
        output = model(dummy_input)
        print(f"입력 크기: {dummy_input.shape}")
        print(f"출력 크기: {output.shape}")
        
        return model
    
    def demonstrate_backpropagation(self, model):
        """역전파 과정 시연"""
        print("\n=== 역전파 과정 시연 ===")
        
        # 가상의 입력 데이터
        dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
        target = torch.randint(0, 10, (1,)).to(self.device)
        
        # Forward pass
        output = model(dummy_input)
        
        # Loss 계산
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        print(f"초기 Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Gradient 확인
        print(f"Conv1 weight gradient 크기: {model.conv1.weight.grad.shape}")
        print(f"Conv2 weight gradient 크기: {model.conv2.weight.grad.shape}")
        print(f"FC1 weight gradient 크기: {model.fc1.weight.grad.shape}")
        
        return loss

class HuggingFaceLab:
    """Hugging Face API 실습을 위한 클래스"""
    
    def __init__(self):
        if not HF_TOKEN:
            print("⚠️ HF_TOKEN이 설정되지 않았습니다.")
            print("Hugging Face에서 토큰을 생성하고 .env 파일에 추가하세요.")
        else:
            print("✅ Hugging Face 토큰이 설정되었습니다.")
    
    def test_serverless_inference(self):
        """Serverless Inference API 테스트"""
        if not HF_TOKEN:
            print("토큰이 없어 API 테스트를 건너뜁니다.")
            return None
        
        print("\n=== Hugging Face Serverless Inference API 테스트 ===")
        
        # 이미지 분류 모델 API
        API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        # 샘플 이미지 생성
        sample_image = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        sample_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        try:
            # API 호출
            response = requests.post(API_URL, headers=headers, data=img_byte_arr)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 이미지 분류 결과:")
                for i, item in enumerate(result[:3]):
                    print(f"  {i+1}. {item['label']}: {item['score']:.4f}")
                return result
            else:
                print(f"❌ API 호출 실패: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ API 호출 중 오류 발생: {e}")
            return None
    
    def test_clip_model(self):
        """CLIP 모델 테스트"""
        print("\n=== CLIP 모델 테스트 ===")
        
        try:
            # 모델 로드
            model_name = "openai/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            
            # 샘플 이미지와 텍스트
            image = Image.new('RGB', (224, 224), color='blue')
            texts = ["a red car", "a blue car", "a dog", "a cat", "a building"]
            
            # 전처리
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
            
            # 추론
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # 결과 출력
            print("CLIP 텍스트-이미지 매칭 결과:")
            for text, prob in zip(texts, probs[0]):
                print(f"  {text}: {prob:.4f}")
            
            return model, processor
            
        except Exception as e:
            print(f"❌ CLIP 모델 로드 중 오류: {e}")
            return None, None
    
    def test_bert_model(self):
        """BERT 모델 테스트"""
        print("\n=== BERT 모델 테스트 ===")
        
        try:
            # 모델 로드
            model_name = "bert-base-uncased"
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            
            # 샘플 텍스트
            text = "I love computer vision and deep learning!"
            
            # 토큰화
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # 추론
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
            
            print(f"BERT 입력 텍스트: {text}")
            print(f"출력 텐서 크기: {last_hidden_states.shape}")
            print(f"임베딩 차원: {last_hidden_states.shape[-1]}")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ BERT 모델 로드 중 오류: {e}")
            return None, None
    
    def test_vit_model(self):
        """Vision Transformer 모델 테스트"""
        print("\n=== Vision Transformer 모델 테스트 ===")
        
        try:
            # 모델 로드
            model_name = "google/vit-base-patch16-224"
            processor = ViTImageProcessor.from_pretrained(model_name)
            model = ViTForImageClassification.from_pretrained(model_name)
            
            # 샘플 이미지
            image = Image.new('RGB', (224, 224), color='green')
            
            # 전처리
            inputs = processor(images=image, return_tensors="pt")
            
            # 추론
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = logits.argmax(-1).item()
            
            # 결과
            predicted_label = model.config.id2label[predicted_class]
            print(f"ViT 예측 결과: {predicted_label}")
            print(f"클래스 ID: {predicted_class}")
            
            return model, processor
            
        except Exception as e:
            print(f"❌ ViT 모델 로드 중 오류: {e}")
            return None, None

class GradioLab:
    """Gradio 웹 인터페이스 실습을 위한 클래스"""
    
    def __init__(self):
        self.vit_model = None
        self.vit_processor = None
    
    def load_models(self):
        """필요한 모델들을 로드"""
        print("\n=== Gradio 앱을 위한 모델 로드 ===")
        
        try:
            model_name = "google/vit-base-patch16-224"
            self.vit_processor = ViTImageProcessor.from_pretrained(model_name)
            self.vit_model = ViTForImageClassification.from_pretrained(model_name)
            print("✅ ViT 모델이 로드되었습니다.")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
    
    def classify_image(self, image):
        """이미지 분류 함수"""
        if image is None:
            return "이미지를 업로드해주세요."
        
        if self.vit_model is None or self.vit_processor is None:
            return "모델이 로드되지 않았습니다."
        
        try:
            # 이미지 전처리
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # ViT 모델로 분류
            inputs = self.vit_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # 상위 5개 결과
                top5_probs, top5_indices = torch.topk(probs, 5)
            
            results = []
            for prob, idx in zip(top5_probs[0], top5_indices[0]):
                label = self.vit_model.config.id2label[idx.item()]
                results.append(f"{label}: {prob:.4f}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"분류 중 오류 발생: {e}"
    
    def create_gradio_app(self):
        """Gradio 앱 생성"""
        print("\n=== Gradio 앱 생성 ===")
        
        if self.vit_model is None:
            self.load_models()
        
        # Gradio 인터페이스 생성
        iface = gr.Interface(
            fn=self.classify_image,
            inputs=gr.Image(type="pil"),
            outputs=gr.Textbox(label="분류 결과", lines=5),
            title="🖼️ AI 이미지 분류기",
            description="이미지를 업로드하면 AI가 무엇인지 분류해드립니다.",
            examples=[
                ["sample1.jpg"],
                ["sample2.jpg"],
                ["sample3.jpg"]
            ],
            theme=gr.themes.Soft()
        )
        
        print("✅ Gradio 앱이 생성되었습니다.")
        return iface

def main():
    """메인 실습 함수"""
    print("🚀 Week 2: CNN 원리 + Hugging Face 생태계 실습 시작")
    print("=" * 60)
    
    # 1. CNN 실습
    print("\n📚 1단계: CNN 원리 실습")
    cnn_lab = CNNLab()
    
    # Convolution 시각화
    input_img, kernel, output = cnn_lab.visualize_convolution()
    
    # CNN 모델 구축
    model = cnn_lab.build_simple_cnn()
    
    # 역전파 시연
    loss = cnn_lab.demonstrate_backpropagation(model)
    
    # 2. Hugging Face 실습
    print("\n🔧 2단계: Hugging Face API 실습")
    hf_lab = HuggingFaceLab()
    
    # Serverless Inference API 테스트
    api_result = hf_lab.test_serverless_inference()
    
    # 사전훈련 모델 테스트
    clip_model, clip_processor = hf_lab.test_clip_model()
    bert_model, bert_tokenizer = hf_lab.test_bert_model()
    vit_model, vit_processor = hf_lab.test_vit_model()
    
    # 3. Gradio 실습
    print("\n🌐 3단계: Gradio 웹 인터페이스 실습")
    gradio_lab = GradioLab()
    app = gradio_lab.create_gradio_app()
    
    print("\n" + "=" * 60)
    print("✅ Week 2 실습 완료!")
    print("\n📋 다음 단계:")
    print("1. app.launch()로 Gradio 앱 실행")
    print("2. Hugging Face Space에 배포")
    print("3. 추가 모델 테스트 및 성능 비교")
    
    return app

if __name__ == "__main__":
    app = main()
    
    # Gradio 앱 실행 (선택사항)
    # app.launch(share=True)
