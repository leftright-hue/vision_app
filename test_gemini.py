"""
Gemini 2.5 Flash Image (Nano Banana) 테스트
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io

# 환경 변수 로드
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

if api_key:
    genai.configure(api_key=api_key)
    
    # 모델 초기화
    model = genai.GenerativeModel('models/gemini-2.5-flash-image-preview')
    
    # 간단한 프롬프트로 테스트
    prompt = "A cute cat eating a banana in a fancy restaurant"
    
    print(f"🎨 프롬프트: {prompt}")
    print("=" * 50)
    
    try:
        # 이미지 생성 요청
        response = model.generate_content([prompt])
        
        print(f"✅ 응답 받음!")
        print(f"Candidates 수: {len(response.candidates) if response.candidates else 0}")
        
        if response.candidates:
            for idx, candidate in enumerate(response.candidates):
                print(f"\n📦 Candidate {idx}:")
                print(f"  Parts 수: {len(candidate.content.parts)}")
                
                for part_idx, part in enumerate(candidate.content.parts):
                    print(f"\n  📄 Part {part_idx}:")
                    
                    # Part의 모든 속성 확인
                    attrs = dir(part)
                    print(f"    속성들: {[a for a in attrs if not a.startswith('_')]}")
                    
                    # 텍스트 확인
                    if hasattr(part, 'text') and part.text:
                        print(f"    📝 텍스트: {part.text[:100]}...")
                    
                    # inline_data 확인
                    if hasattr(part, 'inline_data'):
                        print(f"    🖼️ inline_data 존재: {part.inline_data is not None}")
                        if part.inline_data:
                            print(f"       데이터 크기: {len(part.inline_data.data) if hasattr(part.inline_data, 'data') and part.inline_data.data else 'N/A'}")
                            print(f"       MIME 타입: {part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else 'N/A'}")
                            
                            # 이미지 저장 테스트
                            if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                try:
                                    image = Image.open(io.BytesIO(part.inline_data.data))
                                    test_path = "test_generated.png"
                                    image.save(test_path)
                                    print(f"    ✅ 이미지 저장 성공: {test_path}")
                                    print(f"       크기: {image.size}")
                                except Exception as e:
                                    print(f"    ❌ 이미지 저장 실패: {e}")
                    
                    # blob 확인
                    if hasattr(part, 'blob'):
                        print(f"    💾 blob 존재: {part.blob is not None}")
                    
                    # 기타 이미지 관련 속성 확인
                    if hasattr(part, 'image'):
                        print(f"    🎨 image 속성 존재!")
                    
                    if hasattr(part, 'file_data'):
                        print(f"    📁 file_data 존재: {part.file_data is not None}")
        
        # Response 객체의 속성도 확인
        print(f"\n📊 Response 속성들:")
        resp_attrs = [a for a in dir(response) if not a.startswith('_')]
        print(f"  {resp_attrs}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print(f"오류 타입: {type(e).__name__}")
        import traceback
        traceback.print_exc()
else:
    print("❌ API 키가 설정되지 않았습니다.")