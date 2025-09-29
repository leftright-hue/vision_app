#!/usr/bin/env python3
"""
Week 5 Lab: 실시간 비디오 객체 탐지
웹캠과 비디오 파일을 사용한 실시간 교실 물건 탐지

이 실습에서는:
1. 웹캠을 통한 실시간 객체 탐지
2. 비디오 파일 처리 및 저장
3. 성능 모니터링 및 최적화
4. 다양한 시각화 옵션
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import deque, Counter
import json
from datetime import datetime
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# YOLOv8 import
try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLOv8 패키지 로드 완료")
except ImportError:
    print("❌ Ultralytics 패키지가 설치되지 않았습니다.")
    print("설치 명령어: pip install ultralytics")
    exit(1)

class RealTimeDetector:
    """
    실시간 객체 탐지 클래스
    """
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25, iou_threshold=0.7):
        """
        실시간 탐지기 초기화
        
        Args:
            model_path: YOLO 모델 경로
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 모델 로드
        self.model = None
        self.load_model()
        
        # 교실 물건 클래스 (커스텀 모델용)
        self.classroom_classes = {
            0: 'book',
            1: 'laptop', 
            2: 'chair',
            3: 'whiteboard',
            4: 'bag'
        }
        
        # COCO 클래스에서 교실 관련 클래스들
        self.coco_classroom_filter = [
            'book', 'laptop', 'chair', 'backpack', 'handbag', 'suitcase',
            'bottle', 'cup', 'cell phone', 'clock', 'mouse', 'keyboard', 'remote'
        ]
        
        # 클래스별 색상
        self.class_colors = {
            'book': (0, 0, 255),        # 빨강 (BGR)
            'laptop': (0, 255, 0),      # 초록
            'chair': (255, 0, 0),       # 파랑
            'whiteboard': (0, 255, 255), # 노랑
            'bag': (255, 0, 255),       # 마젠타
            'backpack': (255, 0, 255),
            'handbag': (0, 165, 255),   # 주황
            'suitcase': (0, 255, 128),
            'bottle': (255, 255, 0),    # 시안
            'cup': (203, 192, 255),     # 핑크
            'cell phone': (128, 0, 128),
            'clock': (0, 165, 255),
            'mouse': (255, 128, 0),
            'keyboard': (128, 128, 128),
            'remote': (208, 224, 64)
        }
        
        # 성능 모니터링
        self.fps_queue = deque(maxlen=30)
        self.detection_history = deque(maxlen=100)
        self.frame_count = 0
        self.total_inference_time = 0
        
        # 통계
        self.session_stats = {
            'start_time': datetime.now(),
            'total_frames': 0,
            'total_detections': 0,
            'class_counts': Counter(),
            'avg_fps': 0,
            'avg_inference_time': 0
        }
        
        # 시각화 옵션
        self.show_fps = True
        self.show_confidence = True
        self.show_class_count = True
        self.show_inference_time = True
        
        # 녹화 설정
        self.is_recording = False
        self.video_writer = None
        self.output_path = None
    
    def load_model(self):
        """YOLO 모델 로드"""
        try:
            print(f"🔄 모델 로딩: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✅ 모델 로드 완료")
            
            # 모델 정보 출력
            if hasattr(self.model, 'info'):
                self.model.info()
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.model = None
    
    def detect_objects(self, frame):
        """
        프레임에서 객체 탐지
        
        Args:
            frame: 입력 프레임
        
        Returns:
            results: 탐지 결과
            inference_time: 추론 시간 (ms)
        """
        if self.model is None:
            return None, 0
        
        start_time = time.time()
        
        try:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                stream=False
            )
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            
            return results[0] if results else None, inference_time
            
        except Exception as e:
            print(f"❌ 탐지 실패: {e}")
            return None, 0
    
    def filter_classroom_objects(self, results):
        """교실 관련 객체만 필터링"""
        if results is None or results.boxes is None:
            return results
        
        # 커스텀 모델인지 확인 (클래스 수로 판단)
        num_classes = len(results.names)
        is_custom_model = num_classes <= 10  # 일반적으로 커스텀 모델은 클래스가 적음
        
        if is_custom_model:
            return results  # 커스텀 모델은 이미 교실 객체만 탐지
        
        # COCO 모델에서 교실 관련 객체만 필터링
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        filtered_indices = []
        for i, class_id in enumerate(class_ids):
            class_name = results.names[class_id]
            if class_name in self.coco_classroom_filter:
                filtered_indices.append(i)
        
        if filtered_indices:
            # 필터링된 결과 생성
            import torch
            results.boxes.xyxy = torch.tensor(boxes[filtered_indices])
            results.boxes.conf = torch.tensor(confidences[filtered_indices])
            results.boxes.cls = torch.tensor(class_ids[filtered_indices])
        else:
            results.boxes = None
        
        return results
    
    def draw_detections(self, frame, results, inference_time):
        """
        탐지 결과를 프레임에 그리기
        
        Args:
            frame: 원본 프레임
            results: 탐지 결과
            inference_time: 추론 시간
        
        Returns:
            annotated_frame: 어노테이션된 프레임
            detection_count: 탐지된 객체 수
        """
        annotated_frame = frame.copy()
        detection_count = 0
        class_counts = Counter()
        
        if results is not None and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            detection_count = len(boxes)
            
            # 각 탐지 결과 그리기
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                
                # 클래스 이름 결정
                class_name = results.names[class_id]
                class_counts[class_name] += 1
                
                # 색상 선택
                color = self.class_colors.get(class_name, (128, 128, 128))
                
                # 바운딩 박스 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # 라벨 텍스트
                if self.show_confidence:
                    label = f"{class_name}: {conf:.2f}"
                else:
                    label = class_name
                
                # 라벨 배경
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # 라벨 텍스트
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 통계 업데이트
        self.update_statistics(detection_count, class_counts, inference_time)
        
        # 정보 패널 그리기
        self.draw_info_panel(annotated_frame, detection_count, class_counts, inference_time)
        
        return annotated_frame, detection_count
    
    def update_statistics(self, detection_count, class_counts, inference_time):
        """통계 정보 업데이트"""
        self.session_stats['total_frames'] += 1
        self.session_stats['total_detections'] += detection_count
        self.session_stats['class_counts'].update(class_counts)
        
        # 평균 추론 시간 업데이트
        self.total_inference_time += inference_time
        self.session_stats['avg_inference_time'] = self.total_inference_time / self.session_stats['total_frames']
        
        # 탐지 히스토리 업데이트
        self.detection_history.append({
            'timestamp': time.time(),
            'detections': detection_count,
            'inference_time': inference_time,
            'classes': dict(class_counts)
        })
    
    def draw_info_panel(self, frame, detection_count, class_counts, inference_time):
        """정보 패널 그리기"""
        h, w = frame.shape[:2]
        
        # 배경 패널
        panel_height = 120
        cv2.rectangle(frame, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        y_offset = 30
        
        # FPS 표시
        if self.show_fps and len(self.fps_queue) > 0:
            avg_fps = sum(self.fps_queue) / len(self.fps_queue)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 20
        
        # 추론 시간 표시
        if self.show_inference_time:
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 20
        
        # 탐지 객체 수 표시
        cv2.putText(frame, f"Objects: {detection_count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 20
        
        # 클래스별 카운트 표시 (상위 3개)
        if self.show_class_count and class_counts:
            most_common = class_counts.most_common(3)
            class_text = ", ".join([f"{cls}:{cnt}" for cls, cnt in most_common])
            cv2.putText(frame, f"Classes: {class_text}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 2)
        
        # 녹화 상태 표시
        if self.is_recording:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 60, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def start_recording(self, output_path, fps=30):
        """비디오 녹화 시작"""
        self.output_path = output_path
        self.is_recording = True
        
        # VideoWriter 설정은 첫 프레임에서 수행
        self.video_writer = None
        print(f"🔴 녹화 시작: {output_path}")
    
    def stop_recording(self):
        """비디오 녹화 중지"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print(f"⏹️ 녹화 완료: {self.output_path}")
    
    def process_webcam(self, camera_id=0, window_name="교실 물건 탐지기"):
        """
        웹캠을 사용한 실시간 탐지
        
        Args:
            camera_id: 카메라 ID (기본값: 0)
            window_name: 윈도우 이름
        """
        print(f"📹 웹캠 시작 (카메라 ID: {camera_id})")
        
        # 웹캠 초기화
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ 카메라 {camera_id}를 열 수 없습니다.")
            return
        
        # 카메라 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ 카메라 설정: {actual_width}x{actual_height} @ {actual_fps}fps")
        print("\n🎮 조작법:")
        print("  ESC: 종료")
        print("  SPACE: 녹화 시작/중지")
        print("  S: 스크린샷 저장")
        print("  F: FPS 표시 토글")
        print("  C: 신뢰도 표시 토글")
        print("  I: 추론 시간 표시 토글")
        
        frame_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다.")
                    break
                
                # FPS 계산
                current_time = time.time()
                fps = 1 / (current_time - frame_time) if (current_time - frame_time) > 0 else 0
                frame_time = current_time
                self.fps_queue.append(fps)
                
                # 객체 탐지
                results, inference_time = self.detect_objects(frame)
                results = self.filter_classroom_objects(results)
                
                # 결과 시각화
                annotated_frame, detection_count = self.draw_detections(frame, results, inference_time)
                
                # 녹화
                if self.is_recording:
                    if self.video_writer is None:
                        # 첫 프레임에서 VideoWriter 초기화
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(
                            self.output_path, fourcc, 30.0, 
                            (annotated_frame.shape[1], annotated_frame.shape[0])
                        )
                    
                    if self.video_writer is not None:
                        self.video_writer.write(annotated_frame)
                
                # 화면 출력
                cv2.imshow(window_name, annotated_frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE - 녹화 토글
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"classroom_detection_{timestamp}.mp4"
                        self.start_recording(output_path)
                elif key == ord('s'):  # S - 스크린샷
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"📸 스크린샷 저장: {screenshot_path}")
                elif key == ord('f'):  # F - FPS 토글
                    self.show_fps = not self.show_fps
                    print(f"FPS 표시: {'ON' if self.show_fps else 'OFF'}")
                elif key == ord('c'):  # C - 신뢰도 토글
                    self.show_confidence = not self.show_confidence
                    print(f"신뢰도 표시: {'ON' if self.show_confidence else 'OFF'}")
                elif key == ord('i'):  # I - 추론 시간 토글
                    self.show_inference_time = not self.show_inference_time
                    print(f"추론 시간 표시: {'ON' if self.show_inference_time else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자 중단")
        
        finally:
            # 정리
            if self.is_recording:
                self.stop_recording()
            
            cap.release()
            cv2.destroyAllWindows()
            
            # 세션 통계 출력
            self.print_session_statistics()
    
    def process_video_file(self, input_path, output_path=None):
        """
        비디오 파일 처리
        
        Args:
            input_path: 입력 비디오 파일 경로
            output_path: 출력 비디오 파일 경로 (선택사항)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            print(f"❌ 입력 파일이 존재하지 않습니다: {input_path}")
            return
        
        print(f"🎬 비디오 파일 처리: {input_path}")
        
        # 비디오 캡처 초기화
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            print(f"❌ 비디오 파일을 열 수 없습니다: {input_path}")
            return
        
        # 비디오 정보
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 비디오 정보: {width}x{height}, {fps:.1f}fps, {total_frames}프레임")
        
        # 출력 비디오 설정
        video_writer = None
        if output_path:
            output_path = Path(output_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 출력 파일: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 객체 탐지
                results, inference_time = self.detect_objects(frame)
                results = self.filter_classroom_objects(results)
                
                # 결과 시각화
                annotated_frame, detection_count = self.draw_detections(frame, results, inference_time)
                
                # 진행률 표시
                progress = (frame_count / total_frames) * 100
                
                # 진행률을 프레임에 표시
                cv2.putText(annotated_frame, f"Progress: {progress:.1f}% ({frame_count}/{total_frames})", 
                           (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 출력 비디오에 저장
                if video_writer is not None:
                    video_writer.write(annotated_frame)
                
                # 화면 출력 (선택사항)
                cv2.imshow('비디오 처리 중...', annotated_frame)
                
                # ESC로 중단 가능
                if cv2.waitKey(1) & 0xFF == 27:
                    print("\n⏹️ 사용자 중단")
                    break
                
                # 진행률 출력 (매 100프레임마다)
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    estimated_total = elapsed_time * total_frames / frame_count
                    remaining_time = estimated_total - elapsed_time
                    
                    print(f"진행률: {progress:.1f}% | "
                          f"경과: {elapsed_time:.1f}s | "
                          f"남은 시간: {remaining_time:.1f}s")
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자 중단")
        
        finally:
            # 정리
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # 처리 완료 메시지
            total_time = time.time() - start_time
            print(f"\n✅ 비디오 처리 완료!")
            print(f"   처리된 프레임: {frame_count}/{total_frames}")
            print(f"   총 처리 시간: {total_time:.1f}초")
            print(f"   평균 FPS: {frame_count/total_time:.1f}")
            
            # 세션 통계 출력
            self.print_session_statistics()
    
    def print_session_statistics(self):
        """세션 통계 출력"""
        duration = datetime.now() - self.session_stats['start_time']
        
        print("\n📊 세션 통계")
        print("=" * 50)
        print(f"세션 시간: {duration}")
        print(f"총 프레임: {self.session_stats['total_frames']}")
        print(f"총 탐지 객체: {self.session_stats['total_detections']}")
        print(f"평균 추론 시간: {self.session_stats['avg_inference_time']:.1f}ms")
        
        if len(self.fps_queue) > 0:
            print(f"평균 FPS: {sum(self.fps_queue)/len(self.fps_queue):.1f}")
        
        if self.session_stats['total_frames'] > 0:
            print(f"프레임당 평균 객체: {self.session_stats['total_detections']/self.session_stats['total_frames']:.1f}")
        
        print("\n클래스별 탐지 수:")
        for class_name, count in self.session_stats['class_counts'].most_common():
            percentage = (count / self.session_stats['total_detections']) * 100 if self.session_stats['total_detections'] > 0 else 0
            print(f"  {class_name}: {count}개 ({percentage:.1f}%)")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="실시간 교실 물건 탐지")
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='YOLO 모델 경로 (기본값: yolov8n.pt)')
    parser.add_argument('--source', type=str, default='webcam', 
                       help='입력 소스: webcam, 비디오파일경로')
    parser.add_argument('--output', type=str, default=None, 
                       help='출력 비디오 파일 경로')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='신뢰도 임계값 (기본값: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7, 
                       help='IoU 임계값 (기본값: 0.7)')
    parser.add_argument('--camera', type=int, default=0, 
                       help='카메라 ID (기본값: 0)')
    
    args = parser.parse_args()
    
    print("🎯 실시간 교실 물건 탐지 시스템")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"신뢰도 임계값: {args.conf}")
    print(f"IoU 임계값: {args.iou}")
    
    # 탐지기 초기화
    detector = RealTimeDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    if detector.model is None:
        print("❌ 모델 로드 실패. 프로그램을 종료합니다.")
        return
    
    try:
        if args.source == 'webcam':
            # 웹캠 모드
            detector.process_webcam(camera_id=args.camera)
        else:
            # 비디오 파일 모드
            if not Path(args.source).exists():
                print(f"❌ 입력 파일이 존재하지 않습니다: {args.source}")
                return
            
            output_path = args.output
            if output_path is None:
                # 자동 출력 파일명 생성
                input_path = Path(args.source)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{input_path.stem}_detected_{timestamp}.mp4"
            
            detector.process_video_file(args.source, output_path)
    
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
    
    print("\n🎉 실시간 탐지 세션 완료!")

if __name__ == "__main__":
    # 명령행 인자 없이 실행 시 대화형 모드
    import sys
    
    if len(sys.argv) == 1:
        print("🎯 실시간 교실 물건 탐지 시스템")
        print("=" * 60)
        
        print("\n📋 실행 모드 선택:")
        print("1. 웹캠 실시간 탐지")
        print("2. 비디오 파일 처리")
        print("3. 명령행 도움말")
        
        choice = input("\n선택하세요 (1-3): ").strip()
        
        if choice == '1':
            # 웹캠 모드
            detector = RealTimeDetector()
            if detector.model is not None:
                detector.process_webcam()
        
        elif choice == '2':
            # 비디오 파일 모드
            video_path = input("비디오 파일 경로를 입력하세요: ").strip()
            if video_path and Path(video_path).exists():
                detector = RealTimeDetector()
                if detector.model is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"output_detected_{timestamp}.mp4"
                    detector.process_video_file(video_path, output_path)
            else:
                print("❌ 유효하지 않은 파일 경로입니다.")
        
        elif choice == '3':
            # 도움말
            print("\n📖 명령행 사용법:")
            print("python realtime_video_detection.py --help")
        
        else:
            print("❌ 잘못된 선택입니다.")
    
    else:
        # 명령행 인자가 있는 경우
        main()
