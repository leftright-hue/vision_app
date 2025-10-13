"""
Lab 05: 실전 응용 (Practical Applications)

이 실습에서는 행동 인식의 실전 응용 사례를 구현합니다:
- 운동 카운터 (푸시업, 스쿼트, 점핑잭)
- 제스처 인식
- 이상 행동 감지
- 스포츠 동작 분석

사용법:
    python lab05_practical_apps.py --app exercise --exercise pushup
    python lab05_practical_apps.py --app gesture
    python lab05_practical_apps.py --app anomaly --input video.mp4
"""

import argparse
import time
from collections import deque
from typing import List, Tuple, Optional, Deque, Dict
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class ExerciseCounter:
    """
    MediaPipe Pose를 사용한 운동 카운터
    """

    def __init__(self, exercise_type: str = 'pushup'):
        """
        Args:
            exercise_type: 'pushup', 'squat', 'jumping_jack'
        """
        if not HAS_MEDIAPIPE:
            raise ImportError("mediapipe 패키지 필요: pip install mediapipe")

        if not HAS_OPENCV:
            raise ImportError("opencv-python 패키지 필요: pip install opencv-python")

        self.exercise_type = exercise_type

        # MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 카운터 상태
        self.count = 0
        self.state = "up"  # "up" or "down"
        self.angle_history: Deque[float] = deque(maxlen=30)

        print(f"✅ {exercise_type} 카운터 준비 완료")

    def calculate_angle(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        point3: Tuple[float, float]
    ) -> float:
        """
        세 점 사이의 각도를 계산합니다 (point2가 꼭지점)
        """
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360.0 - angle

        return angle

    def get_key_angle(self, landmarks) -> Optional[float]:
        """
        운동 종류에 따라 핵심 관절 각도를 계산합니다.
        """
        try:
            if self.exercise_type == 'pushup':
                # 팔꿈치 각도 (어깨-팔꿈치-손목)
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                return self.calculate_angle(shoulder, elbow, wrist)

            elif self.exercise_type == 'squat':
                # 무릎 각도 (엉덩이-무릎-발목)
                hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                return self.calculate_angle(hip, knee, ankle)

            elif self.exercise_type == 'jumping_jack':
                # 팔 각도 (어깨-팔꿈치-손목)
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                return self.calculate_angle(shoulder, elbow, wrist)

        except Exception as e:
            return None

        return None

    def update(self, angle: float) -> bool:
        """
        각도를 업데이트하고 카운트가 증가했는지 반환합니다.

        Args:
            angle: 관절 각도

        Returns:
            카운트 증가 여부
        """
        self.angle_history.append(angle)

        count_increased = False

        # 운동별 임계값
        if self.exercise_type in ['pushup', 'squat']:
            # 내려갔다 올라오기
            if angle < 100 and self.state == "up":
                self.state = "down"
            elif angle > 140 and self.state == "down":
                self.state = "up"
                self.count += 1
                count_increased = True

        elif self.exercise_type == 'jumping_jack':
            # 팔 올렸다 내리기
            if angle > 140 and self.state == "down":
                self.state = "up"
            elif angle < 100 and self.state == "up":
                self.state = "down"
                self.count += 1
                count_increased = True

        return count_increased

    def run(self, source: int = 0):
        """
        운동 카운터 실행

        Args:
            source: 비디오 소스 (0 = 웹캠, 또는 파일 경로)
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"비디오 소스를 열 수 없습니다: {source}")

        print(f"\n{self.exercise_type.upper()} 카운터 시작!")
        print("종료: 'q' 키\n")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # BGR -> RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Pose 검출
            results = self.pose.process(image)

            # BGR로 다시 변환
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Pose 랜드마크 그리기
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # 각도 계산
                angle = self.get_key_angle(results.pose_landmarks.landmark)

                if angle is not None:
                    # 카운트 업데이트
                    count_increased = self.update(angle)

                    if count_increased:
                        # 카운트 증가 효과
                        cv2.circle(image, (100, 100), 50, (0, 255, 0), -1)

                    # 각도 표시
                    cv2.putText(
                        image,
                        f"Angle: {int(angle)}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    # 상태 표시
                    cv2.putText(
                        image,
                        f"State: {self.state.upper()}",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2
                    )

            # 카운트 표시
            cv2.putText(
                image,
                f"Count: {self.count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3
            )

            cv2.imshow(f'{self.exercise_type.upper()} Counter', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n✅ 총 {self.count}회 완료!")


class GestureRecognizer:
    """
    MediaPipe Hands를 사용한 제스처 인식
    """

    def __init__(self):
        if not HAS_MEDIAPIPE:
            raise ImportError("mediapipe 패키지 필요: pip install mediapipe")

        # MediaPipe Hands 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        print("✅ 제스처 인식기 준비 완료")

    def count_fingers(self, landmarks) -> int:
        """
        펼쳐진 손가락 개수를 세습니다.
        """
        # 손가락 끝과 관절 인덱스
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]

        finger_joints = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        count = 0

        # 각 손가락 체크
        for tip, joint in zip(finger_tips, finger_joints):
            tip_y = landmarks[tip.value].y
            joint_y = landmarks[joint.value].y

            # 손가락 끝이 관절보다 위에 있으면 펼쳐진 것
            if tip_y < joint_y:
                count += 1

        return count

    def recognize_gesture(self, finger_count: int) -> str:
        """
        손가락 개수로 제스처 인식
        """
        gestures = {
            0: "Fist ✊",
            1: "One ☝️",
            2: "Peace ✌️",
            3: "Three 👌",
            4: "Four 🤚",
            5: "Five ✋"
        }

        return gestures.get(finger_count, "Unknown")

    def run(self, source: int = 0):
        """
        제스처 인식 실행
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"비디오 소스를 열 수 없습니다: {source}")

        print("\n제스처 인식 시작!")
        print("종료: 'q' 키\n")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # BGR -> RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Hands 검출
            results = self.hands.process(image)

            # BGR로 다시 변환
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Hands 랜드마크 그리기
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    # 손가락 개수 세기
                    finger_count = self.count_fingers(hand_landmarks.landmark)

                    # 제스처 인식
                    gesture = self.recognize_gesture(finger_count)

                    # 결과 표시
                    cv2.putText(
                        image,
                        f"Fingers: {finger_count}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        image,
                        f"Gesture: {gesture}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 0, 0),
                        3
                    )

            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


class AnomalyDetector:
    """
    간단한 이상 행동 감지기 (모션 크기 기반)
    """

    def __init__(self, threshold: float = 50.0):
        """
        Args:
            threshold: 이상 행동 판정 임계값
        """
        if not HAS_OPENCV:
            raise ImportError("opencv-python 패키지 필요: pip install opencv-python")

        self.threshold = threshold
        self.motion_history: Deque[float] = deque(maxlen=60)

        print(f"✅ 이상 감지기 준비 완료 (임계값: {threshold})")

    def detect(self, video_path: str):
        """
        비디오에서 이상 행동을 감지합니다.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

        ret, prev_frame = cap.read()

        if not ret:
            raise ValueError("첫 번째 프레임을 읽을 수 없습니다")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_idx = 0
        anomaly_frames = []

        print("\n이상 행동 감지 중...")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optical Flow 계산
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            # 모션 크기
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = np.mean(mag)

            self.motion_history.append(mean_mag)

            # 이상 감지 (평균보다 크게 벗어남)
            if len(self.motion_history) > 30:
                avg = np.mean(self.motion_history)
                std = np.std(self.motion_history)

                if mean_mag > avg + 2 * std or mean_mag > self.threshold:
                    anomaly_frames.append(frame_idx)
                    cv2.putText(
                        frame,
                        "ANOMALY DETECTED!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3
                    )

            # 모션 그래프 표시
            cv2.putText(
                frame,
                f"Motion: {mean_mag:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow('Anomaly Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_gray = gray
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"  {frame_idx} 프레임 처리됨...")

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n✅ 분석 완료")
        print(f"총 {len(anomaly_frames)}개 이상 프레임 감지: {anomaly_frames[:10]}...")


def main():
    parser = argparse.ArgumentParser(description="Lab 05: 실전 응용")
    parser.add_argument("--app", type=str, required=True,
                       choices=['exercise', 'gesture', 'anomaly'],
                       help="실행할 앱")
    parser.add_argument("--exercise", type=str, default='pushup',
                       choices=['pushup', 'squat', 'jumping_jack'],
                       help="운동 종류 (exercise 앱용)")
    parser.add_argument("--input", type=str, help="입력 비디오 파일 (anomaly 앱용)")
    parser.add_argument("--threshold", type=float, default=50.0,
                       help="이상 감지 임계값 (anomaly 앱용)")

    args = parser.parse_args()

    try:
        if args.app == 'exercise':
            counter = ExerciseCounter(exercise_type=args.exercise)
            counter.run(source=0)

        elif args.app == 'gesture':
            recognizer = GestureRecognizer()
            recognizer.run(source=0)

        elif args.app == 'anomaly':
            if not args.input:
                print("❌ --input 파라미터가 필요합니다")
                return

            detector = AnomalyDetector(threshold=args.threshold)
            detector.detect(args.input)

    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
