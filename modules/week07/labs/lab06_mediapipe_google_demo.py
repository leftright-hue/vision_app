"""
Lab 06: MediaPipe와 Google Video Intelligence API 데모
실시간 행동 인식 실습
"""

import cv2
import numpy as np
import time
import os
from typing import List, Dict, Any, Tuple
import json

print("=" * 60)
print("Week 7 - Lab 06: MediaPipe & Google Video Intelligence Demo")
print("=" * 60)

# ============================================
# Part 1: MediaPipe 실시간 행동 인식
# ============================================

def demo_mediapipe():
    """MediaPipe를 이용한 실시간 포즈 감지 및 운동 카운팅"""
    print("\n📹 MediaPipe 데모 시작...")

    try:
        import mediapipe as mp

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        print("✅ MediaPipe 로드 성공!")

        # 포즈 감지기 초기화
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 웹캠 열기 (또는 비디오 파일)
        cap = cv2.VideoCapture(0)  # 0 = 기본 웹캠

        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다. 비디오 파일로 대체...")
            # 샘플 비디오 생성
            create_sample_video("sample_exercise.mp4")
            cap = cv2.VideoCapture("sample_exercise.mp4")

        # 운동 카운터 변수
        counter = 0
        stage = None

        print("\n🏃 운동 카운팅 시작!")
        print("ESC 키를 누르면 종료됩니다.\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # RGB 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 포즈 감지
            results = pose.process(image)

            # BGR 변환 (OpenCV 표시용)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 랜드마크 그리기
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                # 간단한 스쿼트 카운터 (무릎 각도 기반)
                landmarks = results.pose_landmarks.landmark

                # 왼쪽 무릎 각도 계산
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)

                # 스쿼트 카운팅 로직
                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == 'up':
                    stage = "down"
                    counter += 1
                    print(f"✅ 스쿼트 카운트: {counter}")

            # 정보 표시
            cv2.rectangle(image, (0,0), (250,100), (245,117,16), -1)

            # 카운터 표시
            cv2.putText(image, 'REPS', (15,12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                       (10,60),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # 스테이지 표시
            cv2.putText(image, 'STAGE', (125,12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage if stage else "Ready",
                       (100,60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # 이미지 표시
            cv2.imshow('MediaPipe Pose Detection', image)

            if cv2.waitKey(10) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n📊 최종 결과: {counter}회 운동 완료!")

    except ImportError:
        print("❌ MediaPipe가 설치되지 않았습니다!")
        print("설치: pip install mediapipe opencv-python")


def calculate_angle(a, b, c):
    """세 점 사이의 각도 계산"""
    import math

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


# ============================================
# Part 2: Google Video Intelligence API 데모
# ============================================

def demo_google_video_intelligence():
    """Google Video Intelligence API를 이용한 비디오 분석"""
    print("\n☁️ Google Video Intelligence API 데모...")

    try:
        from google.cloud import videointelligence
        print("✅ Google Cloud SDK 로드 성공!")

        # API 클라이언트 초기화
        video_client = videointelligence.VideoIntelligenceServiceClient()

        # 분석할 비디오 (GCS URI 또는 로컬 파일)
        # 예시: "gs://cloud-samples-data/video/gbikes_dinosaur.mp4"
        video_uri = input("GCS URI 입력 (또는 Enter로 샘플 사용): ").strip()

        if not video_uri:
            video_uri = "gs://cloud-samples-data/video/gbikes_dinosaur.mp4"
            print(f"샘플 비디오 사용: {video_uri}")

        # 분석 기능 선택
        features = [
            videointelligence.Feature.LABEL_DETECTION,
            videointelligence.Feature.SHOT_CHANGE_DETECTION,
        ]

        print("\n🔄 비디오 분석 중... (1-2분 소요)")

        # API 호출
        operation = video_client.annotate_video(
            request={"features": features, "input_uri": video_uri}
        )

        # 결과 대기
        result = operation.result(timeout=180)
        print("✅ 분석 완료!\n")

        # 결과 파싱
        segment = result.annotation_results[0]

        # 레이블 표시
        print("🏷️ 감지된 레이블:")
        for label_annotation in segment.segment_label_annotations[:10]:
            print(f"  - {label_annotation.entity.description}")

            for category in label_annotation.category_entities:
                print(f"    카테고리: {category.description}")

            for segment in label_annotation.segments[:1]:
                confidence = segment.confidence
                print(f"    신뢰도: {confidence:.2%}")

        # 장면 전환
        print(f"\n🎬 감지된 장면: {len(segment.shot_annotations)}개")

    except ImportError:
        print("❌ Google Cloud SDK가 설치되지 않았습니다!")
        print("설치: pip install google-cloud-videointelligence")
        print("\n시뮬레이션 결과 표시...")
        simulate_google_results()

    except Exception as e:
        print(f"❌ API 호출 실패: {str(e)}")
        print("\n💡 팁:")
        print("1. GOOGLE_APPLICATION_CREDENTIALS 환경변수 확인")
        print("2. API 활성화 여부 확인")
        print("3. 서비스 계정 권한 확인")


def simulate_google_results():
    """Google API 시뮬레이션 결과"""
    print("\n📋 시뮬레이션 결과:")

    labels = [
        ("person", 0.95),
        ("bicycle", 0.89),
        ("outdoor", 0.87),
        ("vehicle", 0.82),
        ("road", 0.78),
        ("cycling", 0.92),
        ("sports", 0.75)
    ]

    print("\n🏷️ 감지된 레이블:")
    for label, confidence in labels:
        print(f"  - {label} (신뢰도: {confidence:.1%})")

    print("\n🎬 감지된 장면: 8개")
    print("  - 장면 1: 0.0s - 3.5s")
    print("  - 장면 2: 3.5s - 7.2s")
    print("  - 장면 3: 7.2s - 12.1s")
    print("  - ...")


# ============================================
# Part 3: 비교 및 통합
# ============================================

def compare_approaches():
    """MediaPipe vs Google Video Intelligence 비교"""
    print("\n" + "=" * 60)
    print("📊 MediaPipe vs Google Video Intelligence 비교")
    print("=" * 60)

    comparison = """
    | 특성 | MediaPipe | Google Video Intelligence |
    |------|-----------|--------------------------|
    | 실시간 처리 | ✅ 가능 (30+ FPS) | ❌ 배치 처리 |
    | 오프라인 | ✅ 가능 | ❌ 인터넷 필요 |
    | 비용 | 무료 | 유료 (월 1000분 무료) |
    | 정확도 | 중상 | 상 |
    | 커스터마이징 | ✅ 가능 | ❌ 제한적 |
    | 행동 종류 | 제한적 | 400+ 사전정의 |
    | 설치 난이도 | 쉬움 | 중간 (API 설정) |
    | 용도 | 실시간 앱 | 대용량 분석 |
    """
    print(comparison)

    print("\n💡 사용 권장사항:")
    print("- 실시간 피트니스 앱 → MediaPipe")
    print("- CCTV 영상 분석 → Google Video Intelligence")
    print("- 프로토타입 → MediaPipe")
    print("- 프로덕션 (대규모) → Google Video Intelligence")


def create_sample_video(filename="sample.mp4"):
    """간단한 샘플 비디오 생성"""
    print(f"📹 샘플 비디오 생성 중: {filename}")

    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    # 간단한 애니메이션 (움직이는 원)
    for i in range(100):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # 움직이는 원
        x = int(320 + 100 * np.sin(i * 0.1))
        y = int(240 + 100 * np.cos(i * 0.1))
        cv2.circle(frame, (x, y), 30, (0, 0, 255), -1)

        # 텍스트
        cv2.putText(frame, f"Frame {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        out.write(frame)

    out.release()
    print(f"✅ 샘플 비디오 생성 완료: {filename}")


# ============================================
# 메인 실행
# ============================================

def main():
    """메인 실행 함수"""
    while True:
        print("\n" + "=" * 60)
        print("🎬 Week 7 - 행동 인식 데모")
        print("=" * 60)
        print("\n옵션을 선택하세요:")
        print("1. MediaPipe 실시간 포즈 감지")
        print("2. Google Video Intelligence API 분석")
        print("3. 두 방식 비교")
        print("4. 종료")

        choice = input("\n선택 (1-4): ").strip()

        if choice == '1':
            demo_mediapipe()
        elif choice == '2':
            demo_google_video_intelligence()
        elif choice == '3':
            compare_approaches()
        elif choice == '4':
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 다시 시도하세요.")


if __name__ == "__main__":
    main()