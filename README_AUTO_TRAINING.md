# 자동 학습 파이프라인 가이드

## 개요

이 시스템은 용접 용융풀 결함 분석을 위한 **완전 자동화된 학습 파이프라인**입니다.

### 주요 기능

✅ **자동 데이터 수집**: 외부 소스에서 용접 비디오/이미지 자동 다운로드
✅ **간편한 레이블링**: 키보드 조작만으로 빠른 레이블 지정
✅ **자동 전처리**: 학습용 데이터 자동 분할 및 준비
✅ **원클릭 학습**: 전체 과정을 한 번에 실행

---

## 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│  1. 데이터 수집 (auto_data_collector.py)                    │
├─────────────────────────────────────────────────────────────┤
│  • YouTube 검색                                             │
│  • Pexels 스톡 비디오                                       │
│  • 직접 URL 리스트                                          │
│  • 로컬 폴더                                                │
│  • 공개 데이터셋                                            │
│                                                             │
│  출력: welding_data/videos/*.mp4                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 데이터 레이블링 (labeling_tool.py)                      │
├─────────────────────────────────────────────────────────────┤
│  • 비디오 재생 및 검토                                      │
│  • 키보드로 레이블 지정:                                    │
│    - 0: 정상 (Normal)                                       │
│    - 1: 결함 (Defect)                                       │
│    - 2: 불확실 (Uncertain)                                  │
│    - s: 스킵                                                │
│                                                             │
│  출력: welding_data/metadata/labels.json                   │
│        welding_data/metadata/training_data.json            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 데이터 전처리 (자동)                                    │
├─────────────────────────────────────────────────────────────┤
│  • Train/Val 분할 (80/20)                                  │
│  • 데이터 검증                                              │
│  • 통계 정보 생성                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 모델 학습 (welding_ml_analyzer.py)                      │
├─────────────────────────────────────────────────────────────┤
│  • CNN + LSTM 모델 학습                                     │
│  • 자동 검증 및 체크포인트                                  │
│  • 최고 성능 모델 저장                                      │
│                                                             │
│  출력: welding_data/welding_defect_model.pth               │
└─────────────────────────────────────────────────────────────┘
```

---

## 빠른 시작

### 방법 1: 전체 파이프라인 한 번에 실행

```bash
python auto_training_pipeline.py
```

대화형 프롬프트를 따라 진행하면 됩니다.

### 방법 2: 개별 단계 실행

#### 1단계: 데이터 수집

```python
from auto_data_collector import AutoDataCollector

# 수집기 초기화
collector = AutoDataCollector(output_dir='welding_data')

# 로컬 폴더에서 수집 (가장 간단)
collector.collect_from_local_folder(
    folder_path='/path/to/videos',
    pattern='*.mp4',
    label='normal'
)

# 또는 URL 리스트에서 다운로드
urls = [
    'https://example.com/welding1.mp4',
    'https://example.com/welding2.mp4',
]
collector.collect_from_url_list(urls, label='defect')

# 수집 기록 저장
collector.save_collection_log()
collector.print_summary()
```

#### 2단계: 데이터 레이블링

```bash
python labeling_tool.py welding_data
```

**키 조작법:**
- `0`: 정상 (Normal)
- `1`: 결함 (Defect)
- `2`: 불확실 (Uncertain)
- `s`: 스킵 (Skip)
- `q`: 종료 (Quit)
- `Space`: 재생/일시정지
- `←`/`→`: 프레임 이동 (10프레임 단위)

#### 3단계: 모델 학습

```python
from auto_training_pipeline import AutoTrainingPipeline

pipeline = AutoTrainingPipeline(data_dir='welding_data')

# 데이터 준비
train_videos, train_labels, val_videos, val_labels = \
    pipeline.step3_prepare_training_data()

# 학습 시작
pipeline.step4_train_model(
    train_videos, train_labels,
    val_videos, val_labels
)
```

---

## 데이터 수집 방법

### 1. YouTube 검색

```python
collector.collect_from_youtube(
    search_query='welding process',
    max_videos=10
)
```

**요구사항:** `yt-dlp` 패키지
```bash
pip install yt-dlp
```

### 2. Pexels 스톡 비디오

```python
collector.collect_from_pexels(
    query='welding',
    max_items=20
)
```

**요구사항:** Pexels API 키 (무료)
1. https://www.pexels.com/api/ 에서 API 키 발급
2. 환경 변수 설정:
   ```bash
   export PEXELS_API_KEY='your_api_key'
   ```

### 3. URL 리스트

```python
# URL 리스트 파일 작성 (urls.txt)
# https://example.com/video1.mp4
# https://example.com/video2.mp4

with open('urls.txt', 'r') as f:
    urls = [line.strip() for line in f if line.strip()]

collector.collect_from_url_list(urls)
```

### 4. 로컬 폴더 (가장 간단!)

```python
collector.collect_from_local_folder(
    folder_path='/home/user/welding_videos',
    pattern='*.mp4'
)
```

---

## 레이블링 팁

### 효율적인 레이블링 전략

1. **빠른 검토 모드**
   - 비디오를 재생하면서 전체적인 품질 확인
   - 확실한 경우 즉시 레이블 지정

2. **세밀한 검토 모드**
   - `Space`로 일시정지
   - `←`/`→`로 프레임 이동하며 세부 확인
   - 용융풀 안정성, 밝기, 모양 등 관찰

3. **레이블 기준**
   - **정상 (0)**: 용융풀이 안정적이고 균일한 밝기
   - **결함 (1)**: 용융풀이 불안정하거나 불균일한 패턴
   - **불확실 (2)**: 판단이 어려운 경우 (학습에서 제외됨)

4. **배치 작업**
   - 비슷한 종류의 비디오를 모아서 레이블링
   - 일관된 기준 유지

---

## 학습 설정

### 기본 설정

```python
sequence_length = 10     # 연속 프레임 수
batch_size = 2          # 배치 크기 (GPU 메모리에 따라 조정)
num_epochs = 50         # 에폭 수
learning_rate = 0.001   # 학습률
```

### GPU 메모리별 권장 배치 크기

| GPU 메모리 | 배치 크기 | 시퀀스 길이 |
|-----------|----------|------------|
| 4GB       | 1        | 10         |
| 6GB       | 2        | 10         |
| 8GB       | 4        | 10         |
| 12GB+     | 8        | 10         |

### 최적화 팁

1. **데이터가 적을 때** (< 20개)
   - `num_epochs` 증가 (100+)
   - 데이터 증강 활성화

2. **과적합 방지**
   - Dropout 비율 증가 (0.5 → 0.7)
   - Early stopping 활용

3. **학습 속도 향상**
   - GPU 사용
   - 배치 크기 증가
   - 멀티프로세싱 (`num_workers`)

---

## 디렉토리 구조

```
welding_data/
├── videos/                      # 수집된 비디오
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
│
├── images/                      # 수집된 이미지
│   ├── image1.jpg
│   └── ...
│
├── metadata/                    # 메타데이터
│   ├── collection_log.json      # 수집 기록
│   ├── labels.json              # 레이블 데이터
│   └── training_data.json       # 학습용 데이터
│
└── welding_defect_model.pth     # 학습된 모델
```

---

## 학습 후 사용

### 학습된 모델로 예측

```python
from welding_ml_analyzer import WeldingDefectPredictor

# 모델 로드
predictor = WeldingDefectPredictor(
    model_path='welding_data/welding_defect_model.pth'
)

# 비디오 분석
predictions = predictor.predict_video('test_video.mp4')

# 결과 확인
for pred in predictions:
    print(f"프레임 {pred.frame_number}: {pred.defect_type} "
          f"(확률: {pred.defect_probability:.2%})")
```

### GUI로 분석

```bash
python welding_video_ui.py
```

1. "비디오 선택" → 분석할 비디오 선택
2. "모델 불러오기" → `welding_data/welding_defect_model.pth` 선택
3. "비디오 분석 시작" 클릭
4. 결과 확인 및 저장

---

## 실전 예제

### 예제 1: 처음부터 끝까지

```bash
# 1. 로컬 비디오 수집
python -c "
from auto_data_collector import AutoDataCollector
c = AutoDataCollector('welding_data')
c.collect_from_local_folder('/path/to/videos', '*.mp4')
c.save_collection_log()
"

# 2. 레이블링
python labeling_tool.py welding_data

# 3. 학습
python auto_training_pipeline.py
# → 모드 선택: 2 (개별 단계)
# → 단계 선택: 4 (모델 학습)
```

### 예제 2: YouTube에서 데이터 수집

```python
from auto_data_collector import AutoDataCollector

collector = AutoDataCollector('welding_data')

# 다양한 검색어로 수집
queries = [
    'welding process',
    'TIG welding',
    'MIG welding',
    'welding defects'
]

for query in queries:
    print(f"\n검색: {query}")
    collector.collect_from_youtube(query, max_videos=5)

collector.save_collection_log()
```

### 예제 3: 점진적 학습 (데이터 추가 후 재학습)

```python
# 1. 새로운 데이터 추가
collector = AutoDataCollector('welding_data')
collector.collect_from_local_folder('/new/videos', '*.mp4')

# 2. 새 데이터만 레이블링
# (기존 레이블은 자동으로 보존됨)
python labeling_tool.py welding_data

# 3. 재학습
pipeline = AutoTrainingPipeline('welding_data')
train_v, train_l, val_v, val_l = pipeline.step3_prepare_training_data()
pipeline.step4_train_model(train_v, train_l, val_v, val_l)
```

---

## 문제 해결

### 데이터 수집 문제

**YouTube 다운로드 실패**
```bash
pip install --upgrade yt-dlp
```

**Pexels API 오류**
```bash
# API 키 확인
echo $PEXELS_API_KEY

# API 키 설정
export PEXELS_API_KEY='your_key'
```

### 레이블링 문제

**비디오가 열리지 않음**
```bash
pip install --upgrade opencv-python
```

**한글 깨짐**
- 시스템 폰트 설정 확인
- 또는 영문 파일명 사용

### 학습 문제

**GPU 메모리 부족**
```python
# 배치 크기 줄이기
batch_size = 1

# 또는 CPU 사용
predictor = WeldingDefectPredictor(device='cpu')
```

**과적합**
- 더 많은 데이터 수집
- Dropout 증가
- 데이터 증강 활용

---

## 데이터 품질 가이드

### 좋은 학습 데이터

✅ 용융풀이 명확하게 보이는 비디오
✅ 다양한 용접 조건 (전류, 속도 등)
✅ 정상/결함 비율이 비슷함 (50:50 권장)
✅ 일관된 레이블링 기준
✅ 충분한 데이터 양 (최소 20개 이상)

### 피해야 할 것

❌ 용융풀이 잘 안 보이는 비디오
❌ 너무 짧거나 긴 비디오 (3-30초 권장)
❌ 극단적으로 불균형한 데이터
❌ 모호한 레이블

---

## 고급 기능

### 커스텀 데이터 소스 추가

```python
class CustomDataCollector(AutoDataCollector):
    def collect_from_custom_api(self, api_url, params):
        """커스텀 API에서 데이터 수집"""
        response = requests.get(api_url, params=params)
        # ... 구현 ...
```

### 자동 레이블링 (휴리스틱)

```python
# 밝기 기반 간단한 자동 레이블링
def auto_label_by_brightness(video_path):
    cap = cv2.VideoCapture(video_path)
    avg_brightness = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness.append(np.mean(gray))

    cap.release()

    # 밝기 변동이 크면 결함으로 분류
    if np.std(avg_brightness) > 30:
        return 'defect'
    else:
        return 'normal'
```

---

## 성능 벤치마크

### 예상 처리 시간

| 작업 | 1개당 시간 | 비고 |
|-----|-----------|-----|
| YouTube 다운로드 | 1-3분 | 네트워크 속도 의존 |
| 비디오 레이블링 | 30초-2분 | 비디오 길이 의존 |
| 프레임 추출 | 5-10초 | 비디오 길이 의존 |
| 학습 (1 에폭) | 5-30분 | 데이터 수, GPU 의존 |

### 권장 하드웨어

- **최소**: CPU, 8GB RAM
- **권장**: GPU (4GB+), 16GB RAM, SSD
- **최적**: RTX 3060+ (12GB), 32GB RAM, NVMe SSD

---

## 라이센스 및 주의사항

### 데이터 수집 시 주의

- YouTube: 저작권 확인 필요
- Pexels: 무료 라이센스 (상업적 사용 가능)
- 공개 데이터셋: 각 데이터셋의 라이센스 확인

### 학습 데이터 개인정보

수집한 데이터에 개인정보가 포함되지 않도록 주의하세요.

---

## 추가 자료

- **전체 문서**: README_WELDING_ML.md
- **설치 가이드**: INSTALL.md
- **사용 예제**: example_usage.py
- **시스템 테스트**: test_system.py

---

**버전**: 1.0.0
**최종 업데이트**: 2025-10-22
