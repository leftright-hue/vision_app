#!/usr/bin/env python3
"""
Week 4 Lab: 멀티모달 API 비교 및 최적 모델 선택 가이드
Gemini vs GPT-4V vs Llama Vision 성능 비교 및 실제 테스트

이 실습에서는:
1. 주요 멀티모달 API 성능 비교
2. 실제 API 호출 및 응답 분석
3. 태스크별 최적 모델 선택 가이드
4. 비용 효율성 분석
"""

import asyncio
import aiohttp
import time
import json
import base64
import io
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MultimodalAPITester:
    """
    멀티모달 API 테스트 및 비교 클래스
    """
    
    def __init__(self):
        """API 테스터 초기화"""
        self.api_configs = {
            "gemini": {
                "name": "Google Gemini Vision",
                "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                "api_key_env": "GEMINI_API_KEY",
                "free_tier": "매우 관대함",
                "rate_limit": "60 RPM",
                "max_image_size": "20MB",
                "supported_formats": ["JPEG", "PNG", "WebP", "HEIC", "HEIF"]
            },
            "gpt4v": {
                "name": "OpenAI GPT-4V",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "api_key_env": "OPENAI_API_KEY", 
                "free_tier": "제한적",
                "rate_limit": "100 RPM",
                "max_image_size": "20MB",
                "supported_formats": ["JPEG", "PNG", "GIF", "WebP"]
            },
            "llama_vision": {
                "name": "Together AI Llama Vision",
                "endpoint": "https://api.together.xyz/inference",
                "api_key_env": "TOGETHER_API_KEY",
                "free_tier": "3개월 무료",
                "rate_limit": "50 RPM",
                "max_image_size": "10MB",
                "supported_formats": ["JPEG", "PNG"]
            },
            "claude_vision": {
                "name": "Anthropic Claude Vision",
                "endpoint": "https://api.anthropic.com/v1/messages",
                "api_key_env": "ANTHROPIC_API_KEY",
                "free_tier": "제한적",
                "rate_limit": "50 RPM",
                "max_image_size": "5MB",
                "supported_formats": ["JPEG", "PNG", "GIF", "WebP"]
            }
        }
        
        self.test_results = []
        self.load_api_keys()
    
    def load_api_keys(self):
        """환경변수에서 API 키 로드"""
        self.api_keys = {}
        
        for api_id, config in self.api_configs.items():
            key = os.getenv(config["api_key_env"])
            if key:
                self.api_keys[api_id] = key
                print(f"✅ {config['name']} API 키 로드됨")
            else:
                print(f"⚠️ {config['name']} API 키 없음 (환경변수: {config['api_key_env']})")
    
    def encode_image_base64(self, image_path_or_pil):
        """이미지를 base64로 인코딩"""
        if isinstance(image_path_or_pil, str):
            with open(image_path_or_pil, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image_path_or_pil, Image.Image):
            buffer = io.BytesIO()
            image_path_or_pil.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise ValueError("이미지는 파일 경로 또는 PIL Image 객체여야 합니다.")
    
    async def call_gemini_api(self, image, prompt):
        """Gemini API 호출"""
        if "gemini" not in self.api_keys:
            return {"error": "API 키 없음", "response_time": 0}
        
        try:
            image_base64 = self.encode_image_base64(image)
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }]
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_configs['gemini']['endpoint']}?key={self.api_keys['gemini']}"
                async with session.post(url, json=payload, headers=headers) as response:
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        text_response = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        return {
                            "response": text_response,
                            "response_time": response_time,
                            "success": True,
                            "tokens_used": len(text_response.split())
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "error": f"HTTP {response.status}: {error_text}",
                            "response_time": response_time,
                            "success": False
                        }
        
        except Exception as e:
            return {"error": str(e), "response_time": 0, "success": False}
    
    async def call_gpt4v_api(self, image, prompt):
        """GPT-4V API 호출"""
        if "gpt4v" not in self.api_keys:
            return {"error": "API 키 없음", "response_time": 0}
        
        try:
            image_base64 = self.encode_image_base64(image)
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_keys['gpt4v']}"
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_configs['gpt4v']['endpoint'], 
                                      json=payload, headers=headers) as response:
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        text_response = result["choices"][0]["message"]["content"]
                        return {
                            "response": text_response,
                            "response_time": response_time,
                            "success": True,
                            "tokens_used": result.get("usage", {}).get("total_tokens", 0)
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "error": f"HTTP {response.status}: {error_text}",
                            "response_time": response_time,
                            "success": False
                        }
        
        except Exception as e:
            return {"error": str(e), "response_time": 0, "success": False}
    
    async def call_llama_vision_api(self, image, prompt):
        """Llama Vision API 호출"""
        if "llama_vision" not in self.api_keys:
            return {"error": "API 키 없음", "response_time": 0}
        
        try:
            image_base64 = self.encode_image_base64(image)
            
            payload = {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_keys['llama_vision']}"
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_configs['llama_vision']['endpoint'], 
                                      json=payload, headers=headers) as response:
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        text_response = result["choices"][0]["message"]["content"]
                        return {
                            "response": text_response,
                            "response_time": response_time,
                            "success": True,
                            "tokens_used": len(text_response.split())
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "error": f"HTTP {response.status}: {error_text}",
                            "response_time": response_time,
                            "success": False
                        }
        
        except Exception as e:
            return {"error": str(e), "response_time": 0, "success": False}
    
    async def test_single_api(self, api_name, image, prompt):
        """단일 API 테스트"""
        print(f"🔄 {self.api_configs[api_name]['name']} 테스트 중...")
        
        if api_name == "gemini":
            result = await self.call_gemini_api(image, prompt)
        elif api_name == "gpt4v":
            result = await self.call_gpt4v_api(image, prompt)
        elif api_name == "llama_vision":
            result = await self.call_llama_vision_api(image, prompt)
        else:
            result = {"error": "지원하지 않는 API", "response_time": 0, "success": False}
        
        result["api"] = api_name
        result["api_name"] = self.api_configs[api_name]["name"]
        result["timestamp"] = datetime.now().isoformat()
        
        if result.get("success", False):
            print(f"✅ {self.api_configs[api_name]['name']}: {result['response_time']:.0f}ms")
        else:
            print(f"❌ {self.api_configs[api_name]['name']}: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def run_comprehensive_test(self, test_images, prompts):
        """종합 API 테스트"""
        print("🚀 멀티모달 API 종합 테스트 시작")
        print("=" * 60)
        
        all_results = []
        
        for i, (image, image_name) in enumerate(test_images):
            print(f"\n📸 이미지 {i+1}/{len(test_images)}: {image_name}")
            print("-" * 40)
            
            for j, prompt in enumerate(prompts):
                print(f"\n💬 프롬프트 {j+1}: {prompt[:50]}...")
                
                # 모든 API에 대해 병렬 테스트
                tasks = []
                for api_name in self.api_keys.keys():
                    task = self.test_single_api(api_name, image, prompt)
                    tasks.append(task)
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, dict):
                            result["image_name"] = image_name
                            result["prompt"] = prompt
                            all_results.append(result)
                
                # API 호출 간격 (Rate Limit 고려)
                await asyncio.sleep(1)
        
        self.test_results = all_results
        return all_results
    
    def analyze_results(self):
        """테스트 결과 분석"""
        if not self.test_results:
            print("❌ 분석할 결과가 없습니다.")
            return
        
        print("\n📊 테스트 결과 분석")
        print("=" * 60)
        
        # 성공한 결과만 필터링
        successful_results = [r for r in self.test_results if r.get("success", False)]
        
        if not successful_results:
            print("❌ 성공한 API 호출이 없습니다.")
            return
        
        # DataFrame 생성
        df = pd.DataFrame(successful_results)
        
        # API별 통계
        api_stats = df.groupby('api_name').agg({
            'response_time': ['mean', 'std', 'min', 'max'],
            'tokens_used': ['mean', 'sum'],
            'success': 'count'
        }).round(2)
        
        print("\n📈 API별 성능 통계")
        print(api_stats)
        
        # 시각화
        self.visualize_results(df)
        
        return api_stats
    
    def visualize_results(self, df):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 응답 시간 비교
        sns.boxplot(data=df, x='api_name', y='response_time', ax=axes[0, 0])
        axes[0, 0].set_title('API별 응답 시간 분포')
        axes[0, 0].set_ylabel('응답 시간 (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 평균 응답 시간
        avg_times = df.groupby('api_name')['response_time'].mean()
        bars = axes[0, 1].bar(avg_times.index, avg_times.values, alpha=0.7)
        axes[0, 1].set_title('평균 응답 시간')
        axes[0, 1].set_ylabel('응답 시간 (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, avg_times.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                           f'{val:.0f}ms', ha='center', va='bottom')
        
        # 3. 토큰 사용량
        if 'tokens_used' in df.columns:
            avg_tokens = df.groupby('api_name')['tokens_used'].mean()
            bars = axes[0, 2].bar(avg_tokens.index, avg_tokens.values, alpha=0.7, color='lightcoral')
            axes[0, 2].set_title('평균 토큰 사용량')
            axes[0, 2].set_ylabel('토큰 수')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 성공률
        success_rate = df.groupby('api_name').size()
        total_attempts = len(df['api_name'].unique()) * len(df) // len(df['api_name'].unique())
        success_rates = (success_rate / total_attempts * 100) if total_attempts > 0 else success_rate
        
        bars = axes[1, 0].bar(success_rates.index, success_rates.values, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('성공률')
        axes[1, 0].set_ylabel('성공률 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. 응답 길이 분포
        df['response_length'] = df['response'].str.len()
        sns.boxplot(data=df, x='api_name', y='response_length', ax=axes[1, 1])
        axes[1, 1].set_title('응답 길이 분포')
        axes[1, 1].set_ylabel('문자 수')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. 종합 점수 (응답시간 역수 + 응답품질 가중치)
        # 간단한 점수 계산: (1000/응답시간) + (응답길이/100)
        df['composite_score'] = (1000 / df['response_time']) + (df['response_length'] / 100)
        avg_scores = df.groupby('api_name')['composite_score'].mean()
        
        bars = axes[1, 2].bar(avg_scores.index, avg_scores.values, alpha=0.7, color='gold')
        axes[1, 2].set_title('종합 성능 점수')
        axes[1, 2].set_ylabel('점수')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_selection_guide(self):
        """모델 선택 가이드 생성"""
        print("\n🎯 멀티모달 API 선택 가이드")
        print("=" * 60)
        
        guide = {
            "비용 최우선": {
                "추천": "Google Gemini Vision",
                "이유": "완전 무료, 관대한 할당량",
                "적합한 용도": "프로토타입, 개인 프로젝트, 교육용"
            },
            "정확도 최우선": {
                "추천": "GPT-4V 또는 Gemini Vision",
                "이유": "높은 이해력과 정확한 분석",
                "적합한 용도": "상업용 서비스, 중요한 분석 작업"
            },
            "속도 최우선": {
                "추천": "Gemini Vision",
                "이유": "빠른 응답 시간",
                "적합한 용도": "실시간 애플리케이션, 대화형 서비스"
            },
            "커스터마이징": {
                "추천": "Llama Vision",
                "이유": "오픈소스, 로컬 실행 가능",
                "적합한 용도": "특수 도메인, 프라이버시 중요 서비스"
            },
            "안전성/윤리": {
                "추천": "Claude Vision",
                "이유": "강화된 안전성 필터",
                "적합한 용도": "교육, 의료, 법률 분야"
            }
        }
        
        for category, info in guide.items():
            print(f"\n🔹 {category}")
            print(f"   추천: {info['추천']}")
            print(f"   이유: {info['이유']}")
            print(f"   적합한 용도: {info['적합한 용도']}")
        
        # 태스크별 추천
        print(f"\n📋 태스크별 추천")
        print("-" * 30)
        
        task_recommendations = {
            "이미지 캡션 생성": "Gemini Vision (무료, 고품질)",
            "OCR/텍스트 추출": "GPT-4V (정확도 높음)",
            "객체 인식": "Gemini Vision (빠르고 정확)",
            "의료 이미지 분석": "Claude Vision (안전성)",
            "창작/스토리텔링": "GPT-4V (창의성)",
            "실시간 분석": "Gemini Vision (속도)",
            "배치 처리": "Llama Vision (비용 효율)",
            "프라이버시 중요": "Llama Vision (로컬 실행)"
        }
        
        for task, recommendation in task_recommendations.items():
            print(f"• {task}: {recommendation}")

def create_test_images():
    """테스트용 이미지 생성"""
    test_images = []
    
    # 1. 텍스트가 포함된 이미지 (OCR 테스트용)
    img1 = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img1)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 50), "Hello World!", fill='black', font=font)
    draw.text((50, 100), "API Test 2024", fill='blue', font=font)
    test_images.append((img1, "텍스트_이미지"))
    
    # 2. 기하학적 도형 (객체 인식 테스트용)
    img2 = Image.new('RGB', (400, 400), color='lightgray')
    draw = ImageDraw.Draw(img2)
    draw.ellipse([50, 50, 150, 150], fill='red')
    draw.rectangle([200, 50, 350, 200], fill='blue')
    draw.polygon([(100, 250), (200, 200), (300, 300)], fill='green')
    test_images.append((img2, "기하학적_도형"))
    
    # 3. 복잡한 장면 (종합 분석 테스트용)
    img3 = Image.new('RGB', (400, 400), color='skyblue')
    draw = ImageDraw.Draw(img3)
    
    # 집 그리기
    draw.rectangle([150, 200, 250, 300], fill='brown')  # 집 몸체
    draw.polygon([(130, 200), (200, 150), (270, 200)], fill='red')  # 지붕
    draw.rectangle([170, 240, 190, 300], fill='yellow')  # 문
    draw.rectangle([210, 220, 230, 250], fill='lightblue')  # 창문
    
    # 나무 그리기
    draw.rectangle([80, 250, 100, 350], fill='brown')  # 나무 줄기
    draw.ellipse([60, 200, 120, 260], fill='green')  # 나무 잎
    
    # 태양 그리기
    draw.ellipse([320, 50, 370, 100], fill='yellow')
    
    test_images.append((img3, "복합_장면"))
    
    return test_images

def simulate_api_responses():
    """API 응답 시뮬레이션 (실제 API 키가 없을 때)"""
    print("🎭 API 응답 시뮬레이션 모드")
    print("=" * 50)
    
    # 시뮬레이션 데이터
    simulation_data = {
        "Google Gemini Vision": {
            "response_times": np.random.normal(1200, 200, 10),
            "accuracy_score": 0.95,
            "sample_responses": [
                "이 이미지는 'Hello World!'와 'API Test 2024'라는 텍스트가 포함된 흰색 배경의 이미지입니다.",
                "빨간색 원, 파란색 사각형, 녹색 삼각형이 있는 기하학적 도형들의 이미지입니다.",
                "집, 나무, 태양이 있는 간단한 풍경화입니다. 파란 하늘 배경에 빨간 지붕의 집과 녹색 나무가 보입니다."
            ]
        },
        "OpenAI GPT-4V": {
            "response_times": np.random.normal(2100, 300, 10),
            "accuracy_score": 0.92,
            "sample_responses": [
                "The image contains text reading 'Hello World!' and 'API Test 2024' on a white background. The text appears to be in a standard font.",
                "This image shows geometric shapes: a red circle in the upper left, a blue rectangle in the upper right, and a green triangle at the bottom.",
                "A simple drawing depicting a house with a red roof, a tree, and a sun. The scene has a childlike, cartoon-style appearance with basic shapes and bright colors."
            ]
        },
        "Together AI Llama Vision": {
            "response_times": np.random.normal(1800, 250, 10),
            "accuracy_score": 0.88,
            "sample_responses": [
                "I can see text in this image that says 'Hello World!' and 'API Test 2024' written on what appears to be a white background.",
                "The image contains several geometric shapes including a red circular shape, a blue rectangular shape, and a green triangular shape arranged on a gray background.",
                "This appears to be a simple illustration of a house scene with a brown house that has a red roof, a green tree, and a yellow sun in a blue sky."
            ]
        }
    }
    
    # 시뮬레이션 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 응답 시간 비교
    apis = list(simulation_data.keys())
    avg_times = [np.mean(data['response_times']) for data in simulation_data.values()]
    std_times = [np.std(data['response_times']) for data in simulation_data.values()]
    
    bars1 = axes[0, 0].bar(apis, avg_times, yerr=std_times, capsize=5, alpha=0.7)
    axes[0, 0].set_title('평균 응답 시간 (시뮬레이션)')
    axes[0, 0].set_ylabel('시간 (ms)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    for bar, time_val in zip(bars1, avg_times):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                       f'{time_val:.0f}ms', ha='center', va='bottom')
    
    # 정확도 점수
    accuracy_scores = [data['accuracy_score'] for data in simulation_data.values()]
    bars2 = axes[0, 1].bar(apis, accuracy_scores, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('정확도 점수 (시뮬레이션)')
    axes[0, 1].set_ylabel('정확도')
    axes[0, 1].set_ylim(0.8, 1.0)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar, acc_val in zip(bars2, accuracy_scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc_val:.2f}', ha='center', va='bottom')
    
    # 응답 길이 분포
    response_lengths = []
    api_labels = []
    for api, data in simulation_data.items():
        for response in data['sample_responses']:
            response_lengths.append(len(response))
            api_labels.append(api)
    
    df_sim = pd.DataFrame({'API': api_labels, 'Response_Length': response_lengths})
    sns.boxplot(data=df_sim, x='API', y='Response_Length', ax=axes[1, 0])
    axes[1, 0].set_title('응답 길이 분포')
    axes[1, 0].set_ylabel('문자 수')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 종합 성능 점수
    composite_scores = []
    for api, data in simulation_data.items():
        time_score = 1000 / np.mean(data['response_times'])  # 속도 점수
        acc_score = data['accuracy_score'] * 10  # 정확도 점수
        composite = time_score + acc_score
        composite_scores.append(composite)
    
    bars4 = axes[1, 1].bar(apis, composite_scores, alpha=0.7, color='gold')
    axes[1, 1].set_title('종합 성능 점수')
    axes[1, 1].set_ylabel('점수')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for bar, score_val in zip(bars4, composite_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{score_val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 샘플 응답 출력
    print("\n📝 샘플 응답 비교")
    print("=" * 60)
    
    test_prompts = [
        "이 이미지에 있는 텍스트를 읽어주세요.",
        "이 이미지에 있는 도형들을 설명해주세요.",
        "이 이미지의 장면을 자세히 설명해주세요."
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🔸 프롬프트: {prompt}")
        print("-" * 40)
        
        for api, data in simulation_data.items():
            print(f"\n{api}:")
            print(f"응답: {data['sample_responses'][i]}")
            print(f"예상 응답시간: {np.mean(data['response_times']):.0f}ms")

async def main():
    """메인 실습 함수"""
    print("🤖 Week 4: 멀티모달 API 비교 실습")
    print("=" * 60)
    
    # 1. 테스트 이미지 생성
    print("\n1️⃣ 테스트 이미지 생성")
    test_images = create_test_images()
    print(f"✅ {len(test_images)}개 테스트 이미지 생성 완료")
    
    # 테스트 이미지 표시
    fig, axes = plt.subplots(1, len(test_images), figsize=(15, 5))
    for i, (img, name) in enumerate(test_images):
        axes[i].imshow(img)
        axes[i].set_title(name)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
    # 2. API 테스터 초기화
    print("\n2️⃣ API 테스터 초기화")
    tester = MultimodalAPITester()
    
    # 3. 테스트 프롬프트 정의
    test_prompts = [
        "이 이미지를 자세히 설명해주세요.",
        "이 이미지에서 볼 수 있는 객체들을 나열해주세요.",
        "이 이미지의 주요 특징은 무엇인가요?"
    ]
    
    # 4. API 키 확인 및 테스트 실행
    if tester.api_keys:
        print("\n3️⃣ 실제 API 테스트 실행")
        try:
            results = await tester.run_comprehensive_test(test_images, test_prompts)
            
            # 결과 분석
            print("\n4️⃣ 결과 분석")
            stats = tester.analyze_results()
            
        except Exception as e:
            print(f"❌ API 테스트 실패: {e}")
            print("🎭 시뮬레이션 모드로 전환합니다.")
            simulate_api_responses()
    else:
        print("\n⚠️ API 키가 설정되지 않았습니다.")
        print("🎭 시뮬레이션 모드로 실행합니다.")
        simulate_api_responses()
    
    # 5. 선택 가이드 생성
    print("\n5️⃣ 모델 선택 가이드")
    tester.generate_selection_guide()
    
    print("\n🎉 멀티모달 API 비교 실습 완료!")
    print("\n📚 추가 실험 아이디어:")
    print("   - 실제 API 키로 성능 테스트")
    print("   - 다양한 이미지 타입으로 정확도 비교")
    print("   - 비용 효율성 분석")
    print("   - 응답 품질 평가 시스템 구축")

if __name__ == "__main__":
    # 비동기 함수 실행
    asyncio.run(main())
