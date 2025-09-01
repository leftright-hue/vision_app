"""
Smart Vision Flask App - 통합 웹 인터페이스
"""

from flask import Flask, render_template, request, jsonify
import os
import time
import io
import base64
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from analyzer import analyzer
from generator import generator

# 환경 설정
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
else:
    model = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 출력 디렉터리
os.makedirs('static/generated', exist_ok=True)

@app.route('/')
def index():
    """메인 페이지 - 모든 기능 통합"""
    return render_template('index.html')

@app.route('/ai_studio')
def ai_studio():
    """AI Studio 페이지"""
    return render_template('ai_studio.html')

@app.route('/nano_banana')
def nano_banana():
    """Nano Banana 페이지"""
    return render_template('nano_banana.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """이미지 분석"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '이미지가 없습니다'})
        
        image = request.files['image']
        if image.filename == '':
            return jsonify({'success': False, 'error': '파일을 선택해주세요'})
        
        # 임시 저장
        temp_path = f'temp_{image.filename}'
        image.save(temp_path)
        
        # 분석 수행
        prompt = request.form.get('prompt', '이 이미지를 분석해주세요')
        result = analyzer.analyze_image(temp_path, prompt)
        
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate', methods=['POST'])
def generate():
    """이미지 생성"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'success': False, 'error': '프롬프트를 입력해주세요'})
        
        result = generator.generate_image(prompt)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test')
def test():
    """연결 테스트"""
    try:
        analyzer_status = analyzer.test_connection()
        generator_status = generator.test_connection()
        model_status = model is not None
        
        return jsonify({
            'success': True,
            'analyzer': analyzer_status,
            'generator': generator_status,
            'gemini': model_status
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ai_analyze', methods=['POST'])
def ai_analyze():
    """AI Studio 이미지 분석"""
    try:
        if not model:
            return jsonify({'success': False, 'error': 'API 키가 설정되지 않았습니다'})
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '이미지가 없습니다'})
        
        image_file = request.files['image']
        image = Image.open(image_file)
        prompt = request.form.get('prompt', '이 이미지를 분석해주세요')
        
        response = model.generate_content([prompt, image])
        
        return jsonify({
            'success': True,
            'result': response.text
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)