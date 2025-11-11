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

## 📦 YOLOv8 모델 다운로드

### 자동 다운로드 (권장)
프로그램 첫 실행 시 자동으로 `yolov8n.pt` 모델을 다운로드합니다.

### 수동 다운로드
YOLOv8 모델을 미리 다운로드하려면 아래 링크를 참고하세요:

- **공식 문서**: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- **모델 다운로드**: [GitHub Releases - Ultralytics Assets](https://github.com/ultralytics/assets/releases)
- **YOLOv8 저장소**: [Ultralytics YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

#### 사용 가능한 모델 버전
| 모델 | 파라미터 수 | 정확도 (mAP) | 속도 | 용도 |
|------|------------|--------------|------|------|
| `yolov8n.pt` | 3.2M | 37.3 | 초고속 | 실시간 추적 (권장) |
| `yolov8s.pt` | 11.2M | 44.9 | 고속 | 균형형 |
| `yolov8m.pt` | 25.9M | 50.2 | 중속 | 고정확도 |
| `yolov8l.pt` | 43.7M | 52.9 | 저속 | 오프라인 분석 |
| `yolov8x.pt` | 68.2M | 53.9 | 최저속 | 최고 정확도 |

**권장**: Jetson/임베디드 환경에서는 `yolov8n.pt` 또는 `yolov8s.pt` 사용

---

## 🧩 1. 가상환경 생성

### Windows
```bash
# 프로젝트 폴더로 이동
cd D:\hyconsoft\dip\bytetracker

# 가상환경 생성
python -m venv tracker

# 활성화
tracker\Scripts\activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / WSL / Jetson
```bash
# 프로젝트 폴더로 이동
cd /path/to/bytetracker

# 가상환경 생성
python3 -m venv tracker

# 활성화
source tracker/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
