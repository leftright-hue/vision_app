"""
설정 관리 모듈
API 키, 환경 변수, 애플리케이션 설정을 관리합니다.
"""

import os
from dotenv import load_dotenv

class Config:
    """애플리케이션 설정 클래스"""
    
    def __init__(self):
        """환경 변수 로드 및 설정 초기화"""
        # .env 파일 로드
        load_dotenv()
        
        # API 설정
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 애플리케이션 설정
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.host = os.getenv('HOST', '127.0.0.1')
        self.port = int(os.getenv('PORT', 5000))
        
        # 파일 업로드 설정
        self.upload_folder = os.getenv('UPLOAD_FOLDER', 'uploads')
        self.max_file_size = int(os.getenv('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        
        # AI 모델 설정
        self.vision_model = os.getenv('VISION_MODEL', 'gemini-2.5-flash')
        self.image_model = os.getenv('IMAGE_MODEL', 'gemini-2.5-flash-image-preview')
        
        # 성능 설정
        self.rate_limit_delay = float(os.getenv('RATE_LIMIT_DELAY', 1.0))
        self.max_retries = int(os.getenv('MAX_RETRIES', 3))
        
        # 출력 설정
        self.output_folder = os.getenv('OUTPUT_FOLDER', 'results')
        
        # 디렉터리 생성
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉터리들을 생성합니다."""
        directories = [
            self.upload_folder,
            self.output_folder,
            'static/css',
            'static/js',
            'templates'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate_api_key(self):
        """API 키 유효성을 검증합니다."""
        if not self.google_api_key or len(self.google_api_key) < 10:
            return False
        return True
    
    def get_upload_path(self, filename):
        """업로드 파일의 전체 경로를 반환합니다."""
        return os.path.join(self.upload_folder, filename)
    
    def get_output_path(self, filename):
        """출력 파일의 전체 경로를 반환합니다."""
        return os.path.join(self.output_folder, filename)
    
    def is_allowed_file(self, filename):
        """파일 확장자가 허용되는지 확인합니다."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

# 전역 설정 인스턴스
config = Config()
