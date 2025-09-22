"""
Hugging Face 모델 테스트
Week 2: CNN 원리 + Hugging Face 생태계

이 파일은 Hugging Face의 다양한 사전훈련 모델들을 테스트하고
성능을 비교하는 예제입니다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
import json
import io
import os
import time
from dotenv import load_dotenv
from transformers import (
    CLIPProcessor, CLIPModel,
    BertTokenizer, BertModel,
    ViTImageProcessor, ViTForImageClassification,
    AutoTokenizer, AutoModel,
    pipeline
)
import gradio as gr

# 환경 변수 로드
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

class HuggingFaceModelTester:
    """Hugging Face 모델 테스트 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {self.device}")
        
        # 모델 캐시
        self.models = {}
        self.processors = {}
        
    def create_test_images(self):
        """테스트용 이미지 생성"""
        images = {}
        
        # 1. 빨간색 이미지
        red_img = Image.new('RGB', (224, 224), color='red')
        images['red'] = red_img
        
        # 2. 파란색 이미지
        blue_img = Image.new('RGB', (224, 224), color='blue')
        images['blue'] = blue_img
        
        # 3. 녹색 이미지
        green_img = Image.new('RGB', (224, 224), color='green')
        images['green'] = green_img
        
        # 4. 그라데이션 이미지
        gradient_img = Image.new('RGB', (224, 224))
        draw = ImageDraw.Draw(gradient_img)
        for i in range(224):
            color = int(255 * i / 224)
            draw.line([(i, 0), (i, 224)], fill=(color, color, color))
        images['gradient'] = gradient_img
        
        # 5. 체크무늬 이미지
        checker_img = Image.new('RGB', (224, 224))
        draw = ImageDraw.Draw(checker_img)
        for i in range(0, 224, 28):
            for j in range(0, 224, 28):
                color = (255, 255, 255) if (i + j) % 56 == 0 else (0, 0, 0)
                draw.rectangle([i, j, i+28, j+28], fill=color)
        images['checker'] = checker_img
        
        return images
    
    def test_clip_model(self, images):
        """CLIP 모델 테스트"""
        print("\n=== CLIP 모델 테스트 ===")
        
        try:
            # 모델 로드
            model_name = "openai/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            
            # 테스트 텍스트들
            test_texts = [
                "a red object",
                "a blue object", 
                "a green object",
                "a black and white pattern",
                "a gradient image",
                "a colorful image",
                "a simple shape",
                "a complex pattern"
            ]
            
            results = {}
            
            for img_name, image in images.items():
                print(f"\n이미지: {img_name}")
                
                # 전처리
                inputs = processor(text=test_texts, images=image, return_tensors="pt", padding=True)
                
                # 추론
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # 결과 저장
                results[img_name] = {}
                for text, prob in zip(test_texts, probs[0]):
                    results[img_name][text] = prob.item()
                    print(f"  {text}: {prob:.4f}")
            
            # 모델 캐시에 저장
            self.models['clip'] = model
            self.processors['clip'] = processor
            
            return results
            
        except Exception as e:
            print(f"❌ CLIP 모델 테스트 실패: {e}")
            return None
    
    def test_bert_model(self):
        """BERT 모델 테스트"""
        print("\n=== BERT 모델 테스트 ===")
        
        try:
            # 모델 로드
            model_name = "bert-base-uncased"
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            
            # 테스트 텍스트들
            test_texts = [
                "I love computer vision and deep learning!",
                "The image shows a beautiful landscape.",
                "This is a red car parked on the street.",
                "The cat is sitting on the windowsill.",
                "Artificial intelligence is transforming our world."
            ]
            
            results = {}
            
            for text in test_texts:
                print(f"\n텍스트: {text}")
                
                # 토큰화
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                
                # 추론
                with torch.no_grad():
                    outputs = model(**inputs)
                    last_hidden_states = outputs.last_hidden_state
                    pooled_output = outputs.pooler_output
                
                # 결과 저장
                results[text] = {
                    'hidden_states_shape': last_hidden_states.shape,
                    'pooled_output_shape': pooled_output.shape,
                    'embedding_dim': last_hidden_states.shape[-1],
                    'sequence_length': last_hidden_states.shape[1]
                }
                
                print(f"  Hidden states shape: {last_hidden_states.shape}")
                print(f"  Pooled output shape: {pooled_output.shape}")
                print(f"  Embedding dimension: {last_hidden_states.shape[-1]}")
            
            # 모델 캐시에 저장
            self.models['bert'] = model
            self.processors['bert'] = tokenizer
            
            return results
            
        except Exception as e:
            print(f"❌ BERT 모델 테스트 실패: {e}")
            return None
    
    def test_vit_model(self, images):
        """Vision Transformer 모델 테스트"""
        print("\n=== Vision Transformer 모델 테스트 ===")
        
        try:
            # 모델 로드
            model_name = "google/vit-base-patch16-224"
            processor = ViTImageProcessor.from_pretrained(model_name)
            model = ViTForImageClassification.from_pretrained(model_name)
            
            results = {}
            
            for img_name, image in images.items():
                print(f"\n이미지: {img_name}")
                
                # 전처리
                inputs = processor(images=image, return_tensors="pt")
                
                # 추론
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # 상위 5개 결과
                    top5_probs, top5_indices = torch.topk(probs, 5)
                
                # 결과 저장
                results[img_name] = {}
                for prob, idx in zip(top5_probs[0], top5_indices[0]):
                    label = model.config.id2label[idx.item()]
                    results[img_name][label] = prob.item()
                    print(f"  {label}: {prob:.4f}")
            
            # 모델 캐시에 저장
            self.models['vit'] = model
            self.processors['vit'] = processor
            
            return results
            
        except Exception as e:
            print(f"❌ ViT 모델 테스트 실패: {e}")
            return None
    
    def test_serverless_inference(self, images):
        """Serverless Inference API 테스트"""
        if not HF_TOKEN:
            print("⚠️ HF_TOKEN이 없어 Serverless Inference 테스트를 건너뜁니다.")
            return None
        
        print("\n=== Serverless Inference API 테스트 ===")
        
        # 이미지 분류 모델 API
        API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        results = {}
        
        for img_name, image in images.items():
            print(f"\n이미지: {img_name}")
            
            # 이미지를 바이트로 변환
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            try:
                # API 호출
                start_time = time.time()
                response = requests.post(API_URL, headers=headers, data=img_byte_arr)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    results[img_name] = {
                        'predictions': result[:3],  # 상위 3개 결과
                        'response_time': end_time - start_time
                    }
                    
                    print(f"  응답 시간: {end_time - start_time:.3f}초")
                    for i, item in enumerate(result[:3]):
                        print(f"  {i+1}. {item['label']}: {item['score']:.4f}")
                else:
                    print(f"  ❌ API 호출 실패: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ API 호출 중 오류: {e}")
        
        return results
    
    def compare_model_performance(self, clip_results, vit_results, serverless_results):
        """모델 성능 비교"""
        print("\n=== 모델 성능 비교 ===")
        
        if not clip_results or not vit_results:
            print("모델 결과가 없어 비교를 건너뜁니다.")
            return
        
        # 공통 이미지에 대한 성능 비교
        common_images = set(clip_results.keys()) & set(vit_results.keys())
        
        for img_name in common_images:
            print(f"\n이미지: {img_name}")
            
            # CLIP 결과 (가장 높은 확률)
            if img_name in clip_results:
                clip_max_prob = max(clip_results[img_name].values())
                print(f"  CLIP 최고 확률: {clip_max_prob:.4f}")
            
            # ViT 결과 (가장 높은 확률)
            if img_name in vit_results:
                vit_max_prob = max(vit_results[img_name].values())
                print(f"  ViT 최고 확률: {vit_max_prob:.4f}")
            
            # Serverless API 결과
            if serverless_results and img_name in serverless_results:
                serverless_max_prob = max(item['score'] for item in serverless_results[img_name]['predictions'])
                response_time = serverless_results[img_name]['response_time']
                print(f"  Serverless 최고 확률: {serverless_max_prob:.4f} (응답시간: {response_time:.3f}초)")
    
    def create_model_comparison_visualization(self, clip_results, vit_results):
        """모델 비교 시각화"""
        if not clip_results or not vit_results:
            return
        
        # 공통 이미지 찾기
        common_images = list(set(clip_results.keys()) & set(vit_results.keys()))
        
        if len(common_images) == 0:
            return
        
        # 시각화 데이터 준비
        fig, axes = plt.subplots(2, len(common_images), figsize=(15, 8))
        
        for i, img_name in enumerate(common_images):
            # CLIP 결과
            if img_name in clip_results:
                clip_probs = list(clip_results[img_name].values())
                clip_labels = list(clip_results[img_name].keys())
                
                axes[0, i].barh(range(len(clip_probs)), clip_probs)
                axes[0, i].set_yticks(range(len(clip_labels)))
                axes[0, i].set_yticklabels([label[:15] + '...' if len(label) > 15 else label for label in clip_labels])
                axes[0, i].set_title(f'CLIP - {img_name}')
                axes[0, i].set_xlim(0, 1)
            
            # ViT 결과
            if img_name in vit_results:
                vit_probs = list(vit_results[img_name].values())
                vit_labels = list(vit_results[img_name].keys())
                
                axes[1, i].barh(range(len(vit_probs)), vit_probs)
                axes[1, i].set_yticks(range(len(vit_labels)))
                axes[1, i].set_yticklabels([label[:15] + '...' if len(label) > 15 else label for label in vit_labels])
                axes[1, i].set_title(f'ViT - {img_name}')
                axes[1, i].set_xlim(0, 1)
        
        plt.tight_layout()
        plt.show()

def main():
    """메인 실행 함수"""
    print("🔧 Hugging Face 모델 테스트")
    print("=" * 50)
    
    # 테스터 초기화
    tester = HuggingFaceModelTester()
    
    # 테스트 이미지 생성
    print("\n1. 테스트 이미지 생성")
    images = tester.create_test_images()
    print(f"생성된 이미지: {list(images.keys())}")
    
    # 2. CLIP 모델 테스트
    print("\n2. CLIP 모델 테스트")
    clip_results = tester.test_clip_model(images)
    
    # 3. BERT 모델 테스트
    print("\n3. BERT 모델 테스트")
    bert_results = tester.test_bert_model()
    
    # 4. ViT 모델 테스트
    print("\n4. ViT 모델 테스트")
    vit_results = tester.test_vit_model(images)
    
    # 5. Serverless Inference API 테스트
    print("\n5. Serverless Inference API 테스트")
    serverless_results = tester.test_serverless_inference(images)
    
    # 6. 모델 성능 비교
    print("\n6. 모델 성능 비교")
    tester.compare_model_performance(clip_results, vit_results, serverless_results)
    
    # 7. 시각화
    print("\n7. 모델 비교 시각화")
    tester.create_model_comparison_visualization(clip_results, vit_results)
    
    print("\n✅ Hugging Face 모델 테스트 완료!")
    print("\n📋 테스트 결과 요약:")
    print(f"- CLIP 모델: {'성공' if clip_results else '실패'}")
    print(f"- BERT 모델: {'성공' if bert_results else '실패'}")
    print(f"- ViT 모델: {'성공' if vit_results else '실패'}")
    print(f"- Serverless API: {'성공' if serverless_results else '실패'}")
    
    return tester

if __name__ == "__main__":
    tester = main()
