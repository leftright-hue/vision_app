#!/usr/bin/env python3
"""
Week 1 Resources: 샘플 데이터 생성기
딥러닝 영상처리 강의 - 1주차 샘플 이미지 생성

이 스크립트는 실습용 샘플 이미지를 생성합니다.
실제 수업에서는 학생들이 자신의 이미지를 준비하지만,
테스트나 시연용으로 사용할 수 있습니다.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
from pathlib import Path

class SampleImageGenerator:
    """샘플 이미지 생성 클래스"""
    
    def __init__(self, output_folder="sample_images"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # 카테고리별 폴더 생성
        self.categories = ["people", "animals", "landscapes", "food", "objects"]
        for category in self.categories:
            (self.output_folder / category).mkdir(exist_ok=True)
    
    def generate_people_samples(self):
        """사람 카테고리 샘플 생성"""
        print("👥 사람 카테고리 샘플 생성 중...")
        
        # 샘플 1: 간단한 인물 실루엣
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img.fill(240)  # 밝은 배경
        
        # 머리
        cv2.circle(img, (200, 120), 50, (100, 150, 200), -1)
        # 몸통
        cv2.rectangle(img, (160, 170), (240, 320), (100, 150, 200), -1)
        # 팔
        cv2.rectangle(img, (120, 180), (160, 250), (100, 150, 200), -1)
        cv2.rectangle(img, (240, 180), (280, 250), (100, 150, 200), -1)
        # 다리
        cv2.rectangle(img, (175, 320), (200, 380), (100, 150, 200), -1)
        cv2.rectangle(img, (200, 320), (225, 380), (100, 150, 200), -1)
        
        cv2.imwrite(str(self.output_folder / "people" / "person_01.jpg"), img)
        
        # 샘플 2: 그룹 실루엣
        img2 = np.zeros((400, 600, 3), dtype=np.uint8)
        img2.fill(220)  # 배경
        
        # 3명의 사람 그리기
        positions = [(150, 120), (300, 130), (450, 125)]
        colors = [(80, 120, 180), (120, 180, 80), (180, 80, 120)]
        
        for (x, y), color in zip(positions, colors):
            # 머리
            cv2.circle(img2, (x, y), 40, color, -1)
            # 몸통
            cv2.rectangle(img2, (x-30, y+40), (x+30, y+150), color, -1)
            # 팔
            cv2.rectangle(img2, (x-50, y+50), (x-30, y+100), color, -1)
            cv2.rectangle(img2, (x+30, y+50), (x+50, y+100), color, -1)
            # 다리
            cv2.rectangle(img2, (x-20, y+150), (x-5, y+200), color, -1)
            cv2.rectangle(img2, (x+5, y+150), (x+20, y+200), color, -1)
        
        cv2.imwrite(str(self.output_folder / "people" / "group_01.jpg"), img2)
        print("  ✅ 2개 샘플 생성 완료")
    
    def generate_animal_samples(self):
        """동물 카테고리 샘플 생성"""
        print("🐾 동물 카테고리 샘플 생성 중...")
        
        # 샘플 1: 간단한 고양이
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img.fill(200)  # 배경
        
        # 몸통 (타원)
        cv2.ellipse(img, (200, 250), (80, 50), 0, 0, 360, (100, 100, 100), -1)
        # 머리 (원)
        cv2.circle(img, (200, 160), 60, (120, 120, 120), -1)
        # 귀 (삼각형)
        pts = np.array([[160, 120], [180, 80], [200, 120]], np.int32)
        cv2.fillPoly(img, [pts], (120, 120, 120))
        pts = np.array([[200, 120], [220, 80], [240, 120]], np.int32)
        cv2.fillPoly(img, [pts], (120, 120, 120))
        # 눈
        cv2.circle(img, (180, 150), 8, (0, 0, 0), -1)
        cv2.circle(img, (220, 150), 8, (0, 0, 0), -1)
        # 코
        cv2.circle(img, (200, 170), 5, (200, 150, 150), -1)
        # 꼬리
        cv2.ellipse(img, (280, 220), (30, 80), 45, 0, 360, (100, 100, 100), -1)
        
        cv2.imwrite(str(self.output_folder / "animals" / "cat_01.jpg"), img)
        
        # 샘플 2: 간단한 강아지
        img2 = np.zeros((400, 500, 3), dtype=np.uint8)
        img2.fill(180)  # 배경
        
        # 몸통
        cv2.ellipse(img2, (250, 280), (100, 60), 0, 0, 360, (150, 100, 80), -1)
        # 머리
        cv2.ellipse(img2, (250, 180), (70, 60), 0, 0, 360, (150, 100, 80), -1)
        # 귀 (늘어진)
        cv2.ellipse(img2, (200, 160), (25, 40), 20, 0, 360, (130, 80, 60), -1)
        cv2.ellipse(img2, (300, 160), (25, 40), -20, 0, 360, (130, 80, 60), -1)
        # 눈
        cv2.circle(img2, (230, 170), 8, (0, 0, 0), -1)
        cv2.circle(img2, (270, 170), 8, (0, 0, 0), -1)
        # 코
        cv2.circle(img2, (250, 190), 6, (0, 0, 0), -1)
        # 다리
        cv2.rectangle(img2, (200, 320), (220, 370), (130, 80, 60), -1)
        cv2.rectangle(img2, (240, 320), (260, 370), (130, 80, 60), -1)
        cv2.rectangle(img2, (280, 320), (300, 370), (130, 80, 60), -1)
        cv2.rectangle(img2, (220, 320), (240, 370), (130, 80, 60), -1)
        
        cv2.imwrite(str(self.output_folder / "animals" / "dog_01.jpg"), img2)
        print("  ✅ 2개 샘플 생성 완료")
    
    def generate_landscape_samples(self):
        """풍경 카테고리 샘플 생성"""
        print("🌄 풍경 카테고리 샘플 생성 중...")
        
        # 샘플 1: 산과 하늘
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 하늘 그라데이션
        for y in range(250):
            color = int(200 + (50 * y / 250))
            img[y, :] = [color, color - 20, color - 40]
        
        # 산 실루엣
        mountain_points = [
            [0, 250], [100, 200], [200, 150], [300, 180], 
            [400, 120], [500, 160], [600, 200], [600, 400], [0, 400]
        ]
        pts = np.array(mountain_points, np.int32)
        cv2.fillPoly(img, [pts], (80, 100, 60))
        
        # 구름
        cv2.ellipse(img, (150, 80), (40, 20), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (170, 85), (35, 18), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (400, 60), (50, 25), 0, 0, 360, (255, 255, 255), -1)
        
        cv2.imwrite(str(self.output_folder / "landscapes" / "mountain_01.jpg"), img)
        
        # 샘플 2: 도시 스카이라인
        img2 = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 하늘 (저녁)
        for y in range(200):
            r = int(100 + (100 * y / 200))
            g = int(50 + (80 * y / 200))
            b = int(150 + (50 * y / 200))
            img2[y, :] = [b, g, r]
        
        # 건물들
        buildings = [
            ([50, 200], [120, 400]),   # 건물 1
            ([140, 180], [200, 400]),  # 건물 2  
            ([220, 220], [280, 400]),  # 건물 3
            ([300, 160], [360, 400]),  # 건물 4
            ([380, 190], [440, 400]),  # 건물 5
            ([460, 170], [520, 400])   # 건물 6
        ]
        
        colors = [(60, 60, 80), (70, 70, 90), (50, 50, 70), (80, 80, 100), (65, 65, 85), (55, 55, 75)]
        
        for (p1, p2), color in zip(buildings, colors):
            cv2.rectangle(img2, p1, p2, color, -1)
            # 창문
            for i in range(p1[1] + 20, p2[1] - 10, 30):
                for j in range(p1[0] + 10, p2[0] - 5, 20):
                    if random.random() > 0.3:  # 일부 창문만 켜진 상태
                        cv2.rectangle(img2, (j, i), (j + 8, i + 12), (200, 200, 100), -1)
        
        cv2.imwrite(str(self.output_folder / "landscapes" / "city_01.jpg"), img2)
        print("  ✅ 2개 샘플 생성 완료")
    
    def generate_food_samples(self):
        """음식 카테고리 샘플 생성"""
        print("🍕 음식 카테고리 샘플 생성 중...")
        
        # 샘플 1: 피자
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img.fill(250)  # 밝은 배경
        
        # 피자 베이스 (원)
        cv2.circle(img, (200, 200), 120, (200, 150, 100), -1)
        
        # 토핑들
        # 토마토 소스
        cv2.circle(img, (200, 200), 110, (180, 100, 80), -1)
        
        # 치즈
        for _ in range(20):
            x = random.randint(100, 300)
            y = random.randint(100, 300)
            if (x - 200) ** 2 + (y - 200) ** 2 < 100 ** 2:
                cv2.circle(img, (x, y), random.randint(8, 15), (255, 230, 180), -1)
        
        # 페퍼로니
        positions = [(170, 160), (230, 170), (180, 220), (240, 240), (200, 180)]
        for x, y in positions:
            cv2.circle(img, (x, y), 15, (150, 50, 50), -1)
        
        cv2.imwrite(str(self.output_folder / "food" / "pizza_01.jpg"), img)
        
        # 샘플 2: 과일 바구니
        img2 = np.zeros((400, 500, 3), dtype=np.uint8)
        img2.fill(230)  # 배경
        
        # 바구니
        cv2.ellipse(img2, (250, 350), (120, 40), 0, 0, 360, (139, 69, 19), -1)
        cv2.rectangle(img2, (130, 310), (370, 350), (139, 69, 19), -1)
        
        # 과일들
        # 사과 (빨간색)
        cv2.circle(img2, (200, 280), 25, (80, 80, 200), -1)
        cv2.circle(img2, (300, 290), 25, (80, 80, 200), -1)
        
        # 오렌지
        cv2.circle(img2, (250, 260), 30, (0, 165, 255), -1)
        cv2.circle(img2, (180, 320), 28, (0, 165, 255), -1)
        
        # 바나나
        cv2.ellipse(img2, (320, 320), (35, 15), 45, 0, 360, (0, 255, 255), -1)
        
        cv2.imwrite(str(self.output_folder / "food" / "fruits_01.jpg"), img2)
        print("  ✅ 2개 샘플 생성 완료")
    
    def generate_object_samples(self):
        """사물 카테고리 샘플 생성"""
        print("📱 사물 카테고리 샘플 생성 중...")
        
        # 샘플 1: 노트북
        img = np.zeros((400, 500, 3), dtype=np.uint8)
        img.fill(220)  # 배경
        
        # 노트북 본체
        cv2.rectangle(img, (100, 200), (400, 350), (100, 100, 100), -1)
        # 화면
        cv2.rectangle(img, (120, 80), (380, 200), (50, 50, 50), -1)
        # 화면 내용 (파란색)
        cv2.rectangle(img, (140, 100), (360, 180), (150, 100, 50), -1)
        # 키보드 영역
        cv2.rectangle(img, (120, 220), (380, 330), (80, 80, 80), -1)
        
        # 키보드 키들
        for y in range(240, 310, 20):
            for x in range(140, 360, 25):
                cv2.rectangle(img, (x, y), (x + 18, y + 12), (120, 120, 120), -1)
        
        # 트랙패드
        cv2.rectangle(img, (220, 280), (280, 320), (90, 90, 90), -1)
        
        cv2.imwrite(str(self.output_folder / "objects" / "laptop_01.jpg"), img)
        
        # 샘플 2: 스마트폰
        img2 = np.zeros((500, 400, 3), dtype=np.uint8)
        img2.fill(240)  # 밝은 배경
        
        # 스마트폰 외형
        cv2.rectangle(img2, (150, 80), (250, 420), (30, 30, 30), -1, cv2.LINE_AA)
        # 화면
        cv2.rectangle(img2, (160, 100), (240, 400), (0, 0, 0), -1)
        # 화면 내용
        cv2.rectangle(img2, (165, 105), (235, 120), (100, 150, 255), -1)  # 상태바
        
        # 앱 아이콘들
        app_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
                      (255, 255, 100), (255, 100, 255), (100, 255, 255)]
        positions = [(175, 140), (210, 140), (175, 175), (210, 175), (175, 210), (210, 210)]
        
        for pos, color in zip(positions, app_colors):
            cv2.rectangle(img2, pos, (pos[0] + 20, pos[1] + 20), color, -1)
        
        # 홈 버튼
        cv2.circle(img2, (200, 390), 8, (60, 60, 60), -1)
        
        cv2.imwrite(str(self.output_folder / "objects" / "smartphone_01.jpg"), img2)
        print("  ✅ 2개 샘플 생성 완료")
    
    def create_category_info_files(self):
        """각 카테고리별 정보 파일 생성"""
        print("📝 카테고리 정보 파일 생성 중...")
        
        category_info = {
            "people": {
                "description": "사람이 포함된 이미지들",
                "examples": ["인물 사진", "가족 사진", "운동하는 모습", "회의 장면"],
                "analysis_tips": "사람의 수, 행동, 감정, 상황 등을 중점적으로 분석"
            },
            "animals": {
                "description": "동물이 주제인 이미지들", 
                "examples": ["애완동물", "야생동물", "동물원", "농장 동물"],
                "analysis_tips": "동물의 종류, 행동, 환경, 특징 등을 자세히 설명"
            },
            "landscapes": {
                "description": "풍경과 자연경관 이미지들",
                "examples": ["산", "바다", "도시", "건물", "자연 풍경"],
                "analysis_tips": "지형, 날씨, 시간대, 분위기, 특징적 요소들 분석"
            },
            "food": {
                "description": "음식 관련 이미지들",
                "examples": ["요리", "음료", "식재료", "레스토랑", "디저트"],
                "analysis_tips": "음식의 종류, 상태, 플레이팅, 색깔, 질감 등 묘사"
            },
            "objects": {
                "description": "일상 사물과 도구들",
                "examples": ["전자기기", "도구", "가구", "문구류", "생활용품"],
                "analysis_tips": "사물의 종류, 용도, 상태, 재질, 디자인 등 설명"
            }
        }
        
        for category, info in category_info.items():
            info_file = self.output_folder / category / "category_info.md"
            
            content = f"# {category.title()} 카테고리\\n\\n"
            content += f"## 설명\\n{info['description']}\\n\\n"
            content += f"## 예시\\n"
            for example in info['examples']:
                content += f"- {example}\\n"
            content += f"\\n## 분석 팁\\n{info['analysis_tips']}\\n\\n"
            content += f"## 샘플 프롬프트\\n"
            
            if category == "people":
                content += "- 이 사진에 있는 사람들과 그들의 활동을 설명해주세요.\\n"
                content += "- 사람들의 표정과 분위기는 어떤가요?\\n"
            elif category == "animals":
                content += "- 이 동물의 종류와 특징을 설명해주세요.\\n"
                content += "- 동물이 무엇을 하고 있나요?\\n"
            elif category == "landscapes":
                content += "- 이 풍경의 주요 특징을 설명해주세요.\\n"
                content += "- 이 장소의 분위기는 어떤가요?\\n"
            elif category == "food":
                content += "- 이 음식의 종류와 상태를 설명해주세요.\\n"
                content += "- 음식이 어떻게 준비되고 제공되었나요?\\n"
            elif category == "objects":
                content += "- 이 사물의 종류와 용도를 설명해주세요.\\n"
                content += "- 이 물건의 상태와 특징은 어떤가요?\\n"
            
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print("  ✅ 모든 카테고리 정보 파일 생성 완료")
    
    def generate_all_samples(self):
        """모든 샘플 이미지 생성"""
        print("🎨 샘플 이미지 생성 시작...")
        print(f"📁 저장 위치: {self.output_folder}")
        print()
        
        # 각 카테고리별 샘플 생성
        self.generate_people_samples()
        self.generate_animal_samples() 
        self.generate_landscape_samples()
        self.generate_food_samples()
        self.generate_object_samples()
        
        # 카테고리 정보 파일 생성
        self.create_category_info_files()
        
        print("\\n🎉 모든 샘플 이미지 생성 완료!")
        print(f"📊 총 {len(self.categories)}개 카테고리, 각각 2개씩 총 10개 이미지")
        print("\\n📂 생성된 구조:")
        for category in self.categories:
            folder = self.output_folder / category
            files = list(folder.glob("*.jpg"))
            print(f"  {category}/: {len(files)}개 이미지")
        
        return True


def create_readme():
    """전체 README 파일 생성"""
    readme_content = """# Week 1 샘플 이미지 데이터셋

이 폴더는 1주차 실습을 위한 샘플 이미지들을 포함합니다.

## 📁 구조

```
sample_images/
├── people/          # 사람 관련 이미지 (2개)
├── animals/         # 동물 관련 이미지 (2개)
├── landscapes/      # 풍경 관련 이미지 (2개)
├── food/           # 음식 관련 이미지 (2개)
└── objects/        # 사물 관련 이미지 (2개)
```

## 🎯 사용 목적

- Google AI Studio Gemini Vision API 테스트
- 이미지 자동 캡셔닝 실습
- 다양한 프롬프트 실험
- 배치 처리 기능 테스트

## 🚀 사용 방법

### 기본 사용
```python
from weeks.week01.labs.lab01_google_ai_studio import GeminiVisionLab

lab = GeminiVisionLab()
result = lab.analyze_single_image("sample_images/people/person_01.jpg")
```

### 배치 처리
```python
results = lab.batch_image_analysis("sample_images/")
```

## 📋 카테고리별 특징

### 👥 People (사람)
- 인물 실루엣과 그룹 장면
- 사람 수, 행동, 상황 분석에 적합

### 🐾 Animals (동물)  
- 고양이와 강아지 실루엣
- 동물 종류, 특징 인식 테스트용

### 🌄 Landscapes (풍경)
- 산과 도시 스카이라인
- 환경, 분위기 분석 실습용

### 🍕 Food (음식)
- 피자와 과일 바구니
- 음식 종류, 상태 묘사 연습용

### 📱 Objects (사물)
- 노트북과 스마트폰
- 일상용품 인식 및 설명 연습용

## ⚠️ 주의사항

1. **실제 과제용 아님**: 이 이미지들은 테스트용이며, 실제 과제에서는 본인이 준비한 다양한 이미지를 사용하세요.

2. **저작권**: 모든 이미지는 프로그래밍으로 생성된 것으로 상업적 사용에 제한이 없습니다.

3. **품질**: 단순화된 그래픽이므로 실제 사진보다 AI 분석 결과가 제한적일 수 있습니다.

## 🔄 샘플 재생성

```bash
python sample_data_generator.py
```

위 명령으로 새로운 샘플 이미지를 생성할 수 있습니다.
"""
    
    with open("sample_images/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("📖 README.md 파일 생성 완료")


if __name__ == "__main__":
    print("🎨 Week 1 샘플 이미지 생성기")
    print("=" * 50)
    
    # 샘플 이미지 생성
    generator = SampleImageGenerator()
    
    if generator.generate_all_samples():
        create_readme()
        print("\\n✅ 모든 작업 완료!")
        print("\\n🎯 다음 단계:")
        print("1. sample_images/ 폴더 확인")
        print("2. lab01_google_ai_studio.py로 테스트")
        print("3. 실제 과제용 이미지 준비")
    else:
        print("❌ 샘플 생성에 실패했습니다.")