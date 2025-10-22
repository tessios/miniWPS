# 용접 용융풀 결함 분석 시스템

딥러닝 기반의 용접 용융풀(Weld Pool) 동영상 분석 시스템입니다. CNN + LSTM 하이브리드 모델을 사용하여 용접 과정에서 결함 발생 가능성을 실시간으로 예측합니다.

## 주요 기능

### 1. 비디오 분석
- 용접 용융풀 동영상 자동 분석
- 프레임별 결함 확률 계산
- 시각화된 분석 결과 제공

### 2. 딥러닝 모델
- **CNN (Convolutional Neural Network)**: 각 프레임에서 공간적 특징 추출
- **LSTM (Long Short-Term Memory)**: 시간적 패턴 학습
- **하이브리드 구조**: 시공간적 특징을 모두 활용

### 3. 사용자 인터페이스
- CustomTkinter 기반 GUI
- 실시간 비디오 재생 및 분석
- 결과 시각화 및 내보내기

### 4. 데이터 처리
- 비디오 프레임 추출 및 전처리
- 용융풀 영역 강조 (밝은 영역 검출)
- 데이터 증강 및 정규화

## 시스템 구조

```
miniWPS/
├── welding_ml_analyzer.py      # 핵심 ML 모듈
│   ├── VideoProcessor          # 비디오 처리
│   ├── WeldingDataset          # 데이터셋 클래스
│   ├── CNN_LSTM_Model          # 딥러닝 모델
│   └── WeldingDefectPredictor  # 예측 시스템
│
├── welding_video_ui.py         # GUI 인터페이스
│   └── WeldingVideoAnalyzer    # 메인 UI 클래스
│
├── example_usage.py            # 사용 예제
├── requirements.txt            # 의존성 패키지
└── README_WELDING_ML.md        # 이 문서
```

## 설치 방법

### 1. Python 환경 설정
Python 3.8 이상이 필요합니다.

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. PyTorch 설치
GPU를 사용하는 경우:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU만 사용하는 경우:
```bash
pip install torch torchvision
```

## 사용 방법

### 방법 1: GUI 인터페이스 사용

```python
from welding_video_ui import open_welding_analyzer

# GUI 실행
open_welding_analyzer()
```

또는 직접 실행:
```bash
python welding_video_ui.py
```

#### GUI 사용 단계:
1. **비디오 선택**: "비디오 선택" 버튼을 클릭하여 용접 동영상 로드
2. **모델 로드**: 학습된 모델 파일(.pth) 불러오기
3. **분석 시작**: "비디오 분석 시작" 버튼 클릭
4. **결과 확인**: 오른쪽 패널에서 프레임별 예측 결과 확인
5. **결과 저장**: JSON 또는 비디오 형식으로 저장

### 방법 2: 코드로 직접 사용

#### 단일 비디오 분석
```python
from welding_ml_analyzer import WeldingDefectPredictor

# 예측기 초기화 (학습된 모델 로드)
predictor = WeldingDefectPredictor(
    model_path='welding_defect_model.pth'
)

# 비디오 분석
predictions = predictor.predict_video('welding_video.mp4')

# 결과 출력
for pred in predictions:
    print(f"프레임 {pred.frame_number}: {pred.defect_type} "
          f"(확률: {pred.defect_probability:.2%})")
```

#### 모델 학습
```python
from welding_ml_analyzer import WeldingDefectPredictor, WeldingDataset
from torch.utils.data import DataLoader

# 데이터 준비
train_videos = ['video1.mp4', 'video2.mp4', ...]
train_labels = [0, 1, ...]  # 0: 정상, 1: 결함

# 데이터셋 생성
train_dataset = WeldingDataset(train_videos, train_labels, sequence_length=10)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 검증 데이터도 동일하게 준비
val_dataset = WeldingDataset(val_videos, val_labels, sequence_length=10)
val_loader = DataLoader(val_dataset, batch_size=4)

# 학습
predictor = WeldingDefectPredictor()
predictor.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=0.001
)
```

#### 실시간 예측
```python
from welding_ml_analyzer import WeldingDefectPredictor, VideoProcessor

predictor = WeldingDefectPredictor(model_path='model.pth')
processor = VideoProcessor()

# 프레임 추출
frames = processor.extract_frames('video.mp4', frame_rate=10)

# 슬라이딩 윈도우로 실시간 예측
frame_buffer = []
sequence_length = 10

for frame in frames:
    frame_buffer.append(frame)

    if len(frame_buffer) >= sequence_length:
        recent_frames = frame_buffer[-sequence_length:]
        prediction = predictor.predict_realtime(recent_frames)
        print(f"결함 확률: {prediction.defect_probability:.2%}")
```

## 모델 아키텍처

### CNN + LSTM 하이브리드 모델

```
입력 비디오 (10 프레임, 224x224)
    ↓
[CNN Feature Extractor]
    - Conv2d (3→64, 7x7)
    - BatchNorm + ReLU + MaxPool
    - Conv2d (64→128, 3x3)
    - BatchNorm + ReLU + MaxPool
    - Conv2d (128→256, 3x3)
    - BatchNorm + ReLU + MaxPool
    - Conv2d (256→512, 3x3)
    - AdaptiveAvgPool → 512차원 특징
    ↓
[LSTM Temporal Modeling]
    - Input: (batch, 10, 512)
    - LSTM: 2 layers, hidden_size=256
    - Output: (batch, 256)
    ↓
[Fully Connected]
    - Linear (256→128) + ReLU + Dropout
    - Linear (128→2)  # 정상/결함
    ↓
출력: [정상 확률, 결함 확률]
```

### 모델 파라미터
- **총 파라미터**: 약 1,500만 개
- **CNN 파라미터**: 약 1,200만 개
- **LSTM 파라미터**: 약 300만 개

### 학습 설정
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (ReduceLROnPlateau)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 4 (GPU 메모리에 따라 조정)
- **Sequence Length**: 10 프레임

## 데이터 준비

### 데이터 구조
```
data/
├── train/
│   ├── normal_001.mp4      # 정상 용접
│   ├── normal_002.mp4
│   ├── defect_001.mp4      # 결함 있는 용접
│   └── defect_002.mp4
└── val/
    ├── normal_003.mp4
    └── defect_003.mp4
```

### 레이블링
- **0**: 정상 (결함 없음)
- **1**: 결함 (결함 발생 또는 발생 가능성 높음)

### 데이터 요구사항
- **비디오 형식**: MP4, AVI, MOV, MKV
- **권장 해상도**: 720p 이상
- **프레임레이트**: 30 FPS 이상
- **용융풀이 명확하게 보이는 영상**

### 데이터 증강 (자동)
- 리사이징: 224x224
- 정규화: ImageNet 통계 사용
- 용융풀 강조: 밝은 영역 검출 및 강조

## 성능 최적화

### GPU 사용
```python
# GPU 사용 (자동 감지)
predictor = WeldingDefectPredictor()  # 자동으로 CUDA 사용

# CPU 강제 사용
predictor = WeldingDefectPredictor(device='cpu')
```

### 배치 크기 조정
```python
# GPU 메모리가 충분한 경우
train_loader = DataLoader(dataset, batch_size=8)

# GPU 메모리가 부족한 경우
train_loader = DataLoader(dataset, batch_size=2)
```

### 멀티프로세싱
```python
# Linux/Mac에서 데이터 로딩 속도 향상
train_loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4  # CPU 코어 수에 맞게 조정
)

# Windows에서는 num_workers=0 권장
```

## 결과 분석

### 예측 결과 구조
```python
DefectPrediction(
    frame_number=50,              # 프레임 번호
    defect_probability=0.85,      # 결함 확률 (0~1)
    defect_type='결함',           # '정상' 또는 '결함'
    confidence=0.85,              # 예측 신뢰도
    timestamp=5.0                 # 비디오 상의 시간(초)
)
```

### 결과 시각화
```python
from welding_ml_analyzer import visualize_predictions

# 예측 결과를 비디오에 오버레이
visualize_predictions(
    video_path='input.mp4',
    predictions=predictions,
    output_path='output_annotated.mp4'
)
```

### JSON 결과 저장
```python
import json

results = {
    'video_path': 'video.mp4',
    'predictions': [
        {
            'frame_number': p.frame_number,
            'defect_probability': p.defect_probability,
            'defect_type': p.defect_type,
            'confidence': p.confidence
        }
        for p in predictions
    ]
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## 문제 해결

### CUDA 메모리 부족
```python
# 배치 크기 줄이기
train_loader = DataLoader(dataset, batch_size=1)

# 또는 CPU 사용
predictor = WeldingDefectPredictor(device='cpu')
```

### 비디오 로딩 오류
```python
# OpenCV 설치 확인
pip install opencv-python opencv-contrib-python

# 코덱 문제인 경우 FFmpeg 설치
# Ubuntu: sudo apt-get install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

### 느린 학습 속도
- GPU 사용 확인
- 배치 크기 증가
- num_workers 조정
- 데이터를 SSD에 저장

### 낮은 정확도
- 더 많은 학습 데이터 수집
- 에폭 수 증가
- 데이터 증강 적용
- 학습률 조정
- 모델 크기 증가

## 예제 코드

전체 예제는 `example_usage.py` 파일을 참고하세요:

```bash
python example_usage.py
```

포함된 예제:
1. 단일 비디오 결함 예측
2. 모델 학습
3. 실시간 예측 시뮬레이션
4. 비디오 전처리
5. 여러 비디오 일괄 처리
6. 커스텀 모델 구성

## API 문서

### VideoProcessor
```python
processor = VideoProcessor(frame_size=(224, 224))

# 프레임 추출
frames = processor.extract_frames(video_path, frame_rate=10)

# 용융풀 강조
enhanced = processor.enhance_melt_pool(frame)

# 전처리
tensor = processor.preprocess_frames(frames)
```

### WeldingDefectPredictor
```python
predictor = WeldingDefectPredictor(
    model_path='model.pth',
    device='cuda'  # 또는 'cpu'
)

# 비디오 예측
predictions = predictor.predict_video(video_path)

# 실시간 예측
prediction = predictor.predict_realtime(frame_buffer)

# 모델 학습
predictor.train(train_loader, val_loader, num_epochs=50)

# 모델 저장
predictor.save_model('model.pth')
```

### WeldingDataset
```python
dataset = WeldingDataset(
    video_paths=['v1.mp4', 'v2.mp4'],
    labels=[0, 1],
    sequence_length=10
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

## 라이센스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 참고 자료

- **PyTorch**: https://pytorch.org/
- **OpenCV**: https://opencv.org/
- **CustomTkinter**: https://github.com/TomSchimansky/CustomTkinter

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**버전**: 1.0.0
**최종 업데이트**: 2025-10-22
