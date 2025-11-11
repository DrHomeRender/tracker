# ByteTrack Modern Rebuild (for YOLOv8 + Jetson CUDA 12.x)

이 프로젝트는 최신 CUDA / PyTorch 2.x 환경에서도 호환되는 **경량 ByteTrack 리빌드 버전**입니다.  
YOLOv8 감지 결과를 입력받아 실시간으로 객체 추적을 수행하며, Jetson/NPU 환경에서도 안정적으로 동작합니다.

---

## 🚀 주요 구성
| 파일 | 설명 |
|------|------|
| `kalman_filter.py` | 객체 이동 예측용 Kalman 필터 (8차원 상태 공간) |
| `byte_tracker.py` | ByteTrack 핵심 알고리즘 (IoU 매칭 + ID 유지) |
| `demo_yolov8_bytetrack.py` | YOLOv8 감지기 + ByteTrack 연동 데모 |
| `requirements.txt` | 의존 패키지 목록 |
| `README.md` | 설치 및 실행 안내 |

---

## 🧩 1. 가상환경 생성
Python 3.10~3.12 버전 사용 권장합니다.  
Jetson Orin, Ubuntu 22.04 기준 예시입니다.

```bash
# 1️⃣ 프로젝트 폴더 클론 (또는 직접 생성)
mkdir bytetrack_core && cd bytetrack_core

# 2️⃣ 가상환경 생성
python3 -m venv venv
source venv/bin/activate   # (Windows는 venv\\Scripts\\activate)

# 3️⃣ 필수 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
