"""
이미지 생성 모듈
AI 모델을 활용한 이미지 생성 및 편집 기능을 제공합니다.

핵심 학습 목표:
- AI 모델을 활용한 이미지 생성 방법론
- 프롬프트 기반 이미지 생성 기술
- 이미지 편집 및 변환 기법
- 생성 결과 품질 관리
"""

import os
import time
from typing import Dict, Optional, List
from PIL import Image

# Google AI Studio API (도구일 뿐, 핵심은 방법론)
try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai 패키지가 설치되지 않았습니다.")
    print("💡 설치 방법: pip install google-generativeai")
    raise

from config import config
from utils import image_processor, security_utils, response_formatter, file_manager

class ImageGenerator:
    """
    이미지 생성 클래스
    
    핵심 설계 원리:
    1. 단일 책임 원칙: 이미지 생성만 담당
    2. 의존성 주입: config, utils 모듈 활용
    3. 에러 처리: 안정적인 생성 결과 제공
    4. 확장성: 다양한 AI 모델로 교체 가능
    """
    
    def __init__(self):
        """이미지 생성기 초기화"""
        try:
            # AI 모델 클라이언트 초기화 (도구 설정)
            genai.configure(api_key=config.google_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ 이미지 생성기가 초기화되었습니다.")
        except Exception as e:
            print(f"❌ 이미지 생성기 초기화 실패: {e}")
            raise
    
    def generate_image(self, prompt: str, output_path: Optional[str] = None) -> Dict:
        """
        텍스트 프롬프트로 이미지 생성
        
        핵심 방법론:
        1. 프롬프트 검증 및 최적화
        2. AI 모델 호출 (생성 로직)
        3. 결과 검증 및 저장
        4. 메타데이터 관리
        5. 에러 처리
        
        Args:
            prompt: 이미지 생성 프롬프트
            output_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            생성 결과 딕셔너리
        """
        try:
            # 1. 프롬프트 검증
            if not prompt or len(prompt.strip()) < 5:
                return response_formatter.error_response(
                    "프롬프트가 너무 짧습니다. 더 구체적으로 작성해주세요.",
                    code="INVALID_PROMPT"
                )
            
            # 2. 출력 경로 설정
            if output_path is None:
                timestamp = int(time.time())
                filename = f"generated_{timestamp}.png"
                output_path = config.get_output_path(filename)
            
            # 3. AI 모델 호출 (핵심 생성 로직)
            print(f"🎨 이미지 생성 중: {prompt[:50]}...")
            
            response = self.model.generate_content([prompt])
            
            # 4. 결과 처리 및 저장
            for part in response.parts:
                if image := part.as_image():
                    # 이미지 저장
                    image.save(output_path)
                    
                    # 메타데이터 수집
                    generation_info = {
                        'output_path': output_path,
                        'prompt': prompt,
                        'generation_time': time.time(),
                        'model_used': config.image_model,
                        'image_size': image.size,
                        'image_mode': image.mode
                    }
                    
                    print(f"✅ 이미지가 생성되었습니다: {output_path}")
                    return response_formatter.success_response(generation_info)
            
            return response_formatter.error_response(
                "이미지 생성에 실패했습니다.",
                code="GENERATION_FAILED"
            )
            
        except Exception as e:
            return response_formatter.error_response(
                f"이미지 생성 중 오류 발생: {str(e)}",
                code="GENERATION_ERROR"
            )
    
    def edit_image(self, image_path: str, edit_prompt: str, output_path: Optional[str] = None) -> Dict:
        """
        기존 이미지 편집
        
        핵심 방법론:
        1. 원본 이미지 검증
        2. 편집 프롬프트 최적화
        3. AI 모델 호출 (편집 로직)
        4. 결과 검증 및 저장
        5. 원본-편집본 관계 관리
        
        Args:
            image_path: 편집할 이미지 경로
            edit_prompt: 편집 지시사항
            output_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            편집 결과 딕셔너리
        """
        try:
            # 1. 원본 이미지 검증
            if not security_utils.validate_file_type(image_path):
                return response_formatter.error_response(
                    "지원하지 않는 파일 형식입니다.",
                    code="INVALID_FILE_TYPE"
                )
            
            if not os.path.exists(image_path):
                return response_formatter.error_response(
                    "원본 이미지를 찾을 수 없습니다.",
                    code="FILE_NOT_FOUND"
                )
            
            # 2. 출력 경로 설정
            if output_path is None:
                timestamp = int(time.time())
                filename = f"edited_{timestamp}.png"
                output_path = config.get_output_path(filename)
            
            # 3. 이미지 로드 및 전처리
            image = Image.open(image_path)
            optimized_path = image_processor.optimize_image(image_path)
            optimized_image = Image.open(optimized_path)
            
            # 4. AI 모델 호출 (핵심 편집 로직)
            print(f"✏️ 이미지 편집 중: {edit_prompt[:50]}...")
            
            response = self.model.generate_content(
                [edit_prompt, optimized_image]
            )
            
            # 5. 결과 처리 및 저장
            for part in response.parts:
                if image := part.as_image():
                    # 편집된 이미지 저장
                    image.save(output_path)
                    
                    # 메타데이터 수집
                    edit_info = {
                        'output_path': output_path,
                        'original_path': image_path,
                        'edit_prompt': edit_prompt,
                        'edit_time': time.time(),
                        'model_used': config.image_model,
                        'image_size': image.size,
                        'image_mode': image.mode
                    }
                    
                    print(f"✅ 이미지 편집이 완료되었습니다: {output_path}")
                    return response_formatter.success_response(edit_info)
            
            return response_formatter.error_response(
                "이미지 편집에 실패했습니다.",
                code="EDIT_FAILED"
            )
            
        except Exception as e:
            return response_formatter.error_response(
                f"이미지 편집 중 오류 발생: {str(e)}",
                code="EDIT_ERROR"
            )
    
    def batch_generate_images(self, prompts: List[str]) -> List[Dict]:
        """
        여러 이미지 배치 생성
        
        핵심 방법론:
        1. 병렬 처리 고려사항
        2. 리소스 관리
        3. 부분 실패 처리
        4. 진행 상황 추적
        
        Args:
            prompts: 이미지 생성 프롬프트 리스트
            
        Returns:
            생성 결과 리스트
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"🎨 이미지 생성 중... ({i+1}/{len(prompts)})")
            
            result = self.generate_image(prompt)
            results.append({
                'prompt': prompt,
                'result': result,
                'index': i
            })
            
            # API 호출 간격 조절 (Rate Limiting)
            if i < len(prompts) - 1:
                time.sleep(config.rate_limit_delay)
        
        return results
    
    def generate_with_style_template(self, prompt: str, style: str) -> Dict:
        """
        스타일 템플릿을 활용한 이미지 생성
        
        핵심 방법론:
        1. 스타일 템플릿 설계
        2. 프롬프트 조합 기법
        3. 일관된 스타일 적용
        
        Args:
            prompt: 기본 이미지 설명
            style: 적용할 스타일
            
        Returns:
            생성 결과
        """
        # 스타일 템플릿 (도메인별 최적화)
        style_templates = {
            'realistic': f"Create a realistic, high-quality image of {prompt}",
            'cartoon': f"Create a cartoon-style, colorful image of {prompt}",
            'artistic': f"Create an artistic, painterly image of {prompt}",
            'minimalist': f"Create a minimalist, simple image of {prompt}",
            'vintage': f"Create a vintage, retro-style image of {prompt}",
            'futuristic': f"Create a futuristic, sci-fi style image of {prompt}"
        }
        
        enhanced_prompt = style_templates.get(style, prompt)
        
        return self.generate_image(enhanced_prompt)
    
    def test_connection(self) -> bool:
        """
        AI 모델 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            response = self.model.generate_content(
                ["Hello, this is a connection test."]
            )
            return bool(response.text)
        except Exception as e:
            print(f"❌ 연결 테스트 실패: {e}")
            return False

# 전역 인스턴스
generator = ImageGenerator()
