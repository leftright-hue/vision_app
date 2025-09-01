"""
유틸리티 함수 모듈
이미지 처리, 파일 관리, 보안 검증 등의 공통 기능을 제공합니다.
"""

import os
import time
import hashlib
from PIL import Image
from typing import Optional, Tuple
from config import config

class ImageProcessor:
    """이미지 처리 유틸리티 클래스"""
    
    @staticmethod
    def validate_image(file_path: str) -> bool:
        """이미지 파일의 유효성을 검증합니다."""
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def optimize_image(image_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> str:
        """이미지를 최적화하여 저장합니다."""
        try:
            with Image.open(image_path) as img:
                # RGB 변환
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 크기 조정
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 최적화된 이미지 저장
                optimized_path = f"{image_path}_optimized.jpg"
                img.save(optimized_path, 'JPEG', quality=85, optimize=True)
                
                return optimized_path
        except Exception as e:
            print(f"이미지 최적화 오류: {e}")
            return image_path
    
    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """이미지 정보를 반환합니다."""
        try:
            with Image.open(image_path) as img:
                return {
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': os.path.getsize(image_path)
                }
        except Exception as e:
            return {'error': str(e)}

class FileManager:
    """파일 관리 유틸리티 클래스"""
    
    @staticmethod
    def generate_unique_filename(original_filename: str) -> str:
        """고유한 파일명을 생성합니다."""
        timestamp = int(time.time())
        name, ext = os.path.splitext(original_filename)
        return f"{name}_{timestamp}{ext}"
    
    @staticmethod
    def secure_filename(filename: str) -> str:
        """안전한 파일명으로 변환합니다."""
        # 특수문자 제거 및 공백을 언더스코어로 변경
        import re
        filename = re.sub(r'[^\w\s-]', '', filename)
        filename = re.sub(r'[-\s]+', '_', filename)
        return filename.strip('_')
    
    @staticmethod
    def cleanup_old_files(directory: str, max_age_hours: int = 24):
        """오래된 파일들을 정리합니다."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    try:
                        os.remove(file_path)
                        print(f"삭제된 파일: {filename}")
                    except Exception as e:
                        print(f"파일 삭제 실패: {filename}, 오류: {e}")

class SecurityUtils:
    """보안 관련 유틸리티 클래스"""
    
    @staticmethod
    def validate_file_type(filename: str) -> bool:
        """파일 타입을 검증합니다."""
        return config.is_allowed_file(filename)
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """파일 크기를 검증합니다."""
        return file_size <= config.max_file_size
    
    @staticmethod
    def generate_file_hash(file_path: str) -> str:
        """파일의 해시값을 생성합니다."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class ResponseFormatter:
    """응답 형식화 유틸리티 클래스"""
    
    @staticmethod
    def success_response(data: dict, message: str = "성공") -> dict:
        """성공 응답을 형식화합니다."""
        return {
            'success': True,
            'message': message,
            'data': data,
            'timestamp': time.time()
        }
    
    @staticmethod
    def error_response(error: str, code: str = "UNKNOWN_ERROR") -> dict:
        """오류 응답을 형식화합니다."""
        return {
            'success': False,
            'error': error,
            'code': code,
            'timestamp': time.time()
        }
    
    @staticmethod
    def format_analysis_result(result: dict) -> dict:
        """분석 결과를 형식화합니다."""
        return {
            'analysis': result.get('response', ''),
            'execution_time': result.get('execution_time', 0),
            'image_path': result.get('image_path', ''),
            'prompt': result.get('prompt', '')
        }

# 전역 유틸리티 인스턴스들
image_processor = ImageProcessor()
file_manager = FileManager()
security_utils = SecurityUtils()
response_formatter = ResponseFormatter()
