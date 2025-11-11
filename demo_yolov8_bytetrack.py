import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO
from byte_tracker import BYTETracker

def main():
    # YOLOv8 모델 로드 (없으면 자동 다운로드)
    print("YOLOv8 모델 로딩 중...")
    model = YOLO("yolov8n.pt")
    print("모델 로드 완료!")
    
    # ByteTracker 초기화
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8)
    
    # 비디오 소스 설정 (0: 웹캠, 또는 비디오 파일 경로)
    video_source = 0  # 웹캠 사용
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: 비디오 소스 '{video_source}'를 열 수 없습니다.")
        sys.exit(1)
    
    print("추적 시작... (ESC 키로 종료)")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패 또는 비디오 종료")
                break
            
            frame_count += 1
            
            # YOLOv8 객체 탐지
            results = model(frame, verbose=False)
            
            # 탐지 결과 추출
            if len(results[0].boxes) > 0:
                dets = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
                detections = np.concatenate([dets, confs], axis=1)
            else:
                detections = np.zeros((0, 5))
            
            # ByteTrack 업데이트
            tracks = tracker.update(detections)
            
            # 추적 결과 시각화
            for tid, x1, y1, x2, y2 in tracks:
                # 바운딩 박스
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # ID 라벨
                label = f'ID {int(tid)}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 프레임 정보 표시
            info_text = f'Frame: {frame_count} | Tracks: {len(tracks)}'
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 화면 출력 (WSL에서는 X11 forwarding 필요)
            cv2.imshow("ByteTrack Demo", frame)
            
            # ESC 키로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("사용자 종료 요청")
                break
                
    except KeyboardInterrupt:
        print("\n프로그램 종료 (Ctrl+C)")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("리소스 정리 완료")

if __name__ == "__main__":
    main()
