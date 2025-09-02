"""
이미지 분석 모듈
Gemini Vision API를 활용한 이미지 분석 기능을 제공합니다.

핵심 학습 목표:
- AI 모델을 활용한 이미지 분석 방법론
- 프롬프트 엔지니어링 기법
- 이미지 전처리 및 후처리 기술
- 에러 처리 및 안정성 확보
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
from utils import image_processor, security_utils, response_formatter

class ImageAnalyzer:
    """
    이미지 분석 클래스
    
    핵심 설계 원리:
    1. 단일 책임 원칙: 이미지 분석만 담당
    2. 의존성 주입: config, utils 모듈 활용
    3. 에러 처리: 안정적인 분석 결과 제공
    4. 확장성: 다양한 AI 모델로 교체 가능
    """
    
    def __init__(self):
        """이미지 분석기 초기화"""
        try:
            # AI 모델 클라이언트 초기화 (도구 설정)
            genai.configure(api_key=config.google_api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash-image-preview')
            print("✅ 이미지 분석기가 초기화되었습니다.")
        except Exception as e:
            print(f"❌ 이미지 분석기 초기화 실패: {e}")
            raise
    
    def analyze_image(self, image_path: str, prompt: str = "이 이미지를 자세히 분석해주세요.") -> Dict:
        """
        이미지 분석 수행
        
        핵심 방법론:
        1. 이미지 전처리 (크기 조정, 형식 변환)
        2. 보안 검증 (파일 타입, 크기 확인)
        3. AI 모델 호출 (프롬프트 엔지니어링)
        4. 결과 후처리 (응답 형식화)
        5. 에러 처리 (예외 상황 대응)
        
        Args:
            image_path: 분석할 이미지 경로
            prompt: 분석 요청 프롬프트
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            # 1. 보안 검증
            if not security_utils.validate_file_type(image_path):
                return response_formatter.error_response(
                    "지원하지 않는 파일 형식입니다.",
                    code="INVALID_FILE_TYPE"
                )
            
            # 2. 이미지 전처리
            optimized_path = image_processor.optimize_image(image_path)
            
            # 3. 이미지 로드
            image = Image.open(optimized_path)
            
            # 4. AI 모델 호출 (핵심 분석 로직)
            response = self.model.generate_content([prompt, image])
            
            # 5. 결과 후처리
            analysis_result = {
                'analysis': response.text,
                'image_path': image_path,
                'prompt': prompt,
                'execution_time': time.time(),
                'model_used': config.vision_model
            }
            
            return response_formatter.success_response(analysis_result)
            
        except Exception as e:
            return response_formatter.error_response(
                f"이미지 분석 중 오류 발생: {str(e)}",
                code="ANALYSIS_ERROR"
            )
    
    def analyze_multiple_images(self, image_paths: List[str], prompt: str) -> List[Dict]:
        """
        여러 이미지 배치 분석
        
        핵심 방법론:
        1. 병렬 처리 고려사항
        2. 리소스 관리
        3. 부분 실패 처리
        4. 진행 상황 추적
        
        Args:
            image_paths: 분석할 이미지 경로 리스트
            prompt: 분석 요청 프롬프트
            
        Returns:
            분석 결과 리스트
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"📊 이미지 분석 중... ({i+1}/{len(image_paths)})")
            
            result = self.analyze_image(image_path, prompt)
            results.append({
                'image_path': image_path,
                'result': result,
                'index': i
            })
            
            # API 호출 간격 조절 (Rate Limiting)
            if i < len(image_paths) - 1:
                time.sleep(config.rate_limit_delay)
        
        return results
    
    def analyze_with_custom_prompt(self, image_path: str, analysis_type: str) -> Dict:
        """
        분석 유형별 맞춤 프롬프트 사용
        
        핵심 방법론:
        1. 프롬프트 템플릿 설계
        2. 도메인별 최적화
        3. 결과 일관성 확보
        
        Args:
            image_path: 분석할 이미지 경로
            analysis_type: 분석 유형
            
        Returns:
            분석 결과
        """
        # 프롬프트 템플릿 (도메인별 최적화)
        prompt_templates = {
            'general': "이 이미지를 자세히 분석해주세요.",
            'objects': "이미지에서 보이는 모든 객체를 찾아 나열해주세요.",
            'emotions': "이 이미지에서 느껴지는 감정과 분위기를 설명해주세요.",
            'technical': "이 이미지의 기술적 특징(해상도, 구도, 조명 등)을 분석해주세요.",
            'artistic': "이 이미지의 예술적 스타일과 구성을 분석해주세요.",
            'safety': "이 이미지에 안전상 위험한 요소가 있는지 확인해주세요."
        }
        
        prompt = prompt_templates.get(analysis_type, prompt_templates['general'])
        
        return self.analyze_image(image_path, prompt)
    
    def test_connection(self) -> bool:
        """
        AI 모델 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            response = self.model.generate_content(
                contents=["Hello, this is a connection test."]
            )
            return bool(response.text)
        except Exception as e:
            print(f"❌ 연결 테스트 실패: {e}")
            return False

# 전역 인스턴스
analyzer = ImageAnalyzer()
