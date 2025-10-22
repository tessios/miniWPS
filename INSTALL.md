# 설치 가이드

## 빠른 시작

### 1단계: 필수 패키지 설치

```bash
# requirements.txt에 있는 모든 패키지 설치
pip install -r requirements.txt
```

또는 개별 설치:

```bash
# 기본 패키지
pip install numpy
pip install opencv-python opencv-contrib-python
pip install Pillow

# GUI 프레임워크
pip install customtkinter

# 머신러닝 (CPU 버전)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU 사용시 (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 기타 유틸리티
pip install python-Levenshtein
pip install tqdm
pip install pandas
pip install matplotlib seaborn
pip install scipy
```

### 2단계: 설치 확인

```bash
python test_system.py
```

모든 테스트가 통과하면 설치가 완료된 것입니다.

## 상세 설치 가이드

### Windows

1. **Python 3.8+ 설치**
   - https://www.python.org/downloads/ 에서 다운로드
   - 설치 시 "Add Python to PATH" 체크

2. **패키지 설치**
   ```cmd
   pip install -r requirements.txt
   ```

3. **PyTorch 설치** (GPU 사용시)
   ```cmd
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Linux (Ubuntu/Debian)

1. **시스템 패키지 업데이트**
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip
   ```

2. **OpenCV 의존성**
   ```bash
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   ```

3. **Python 패키지 설치**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **PyTorch 설치** (GPU 사용시)
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### macOS

1. **Homebrew로 Python 설치**
   ```bash
   brew install python
   ```

2. **패키지 설치**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **PyTorch 설치** (Apple Silicon)
   ```bash
   pip3 install torch torchvision
   ```

## 가상환경 사용 (권장)

### venv 사용

```bash
# 가상환경 생성
python -m venv venv

# 활성화
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### conda 사용

```bash
# 환경 생성
conda create -n welding-ml python=3.11

# 활성화
conda activate welding-ml

# 패키지 설치
pip install -r requirements.txt
```

## GPU 지원

### CUDA 확인

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### CUDA 버전별 PyTorch 설치

- **CUDA 11.8**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

- **CUDA 12.1**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

- **CPU만**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

## 문제 해결

### OpenCV 설치 오류

```bash
# opencv-python 재설치
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

### CustomTkinter 오류

```bash
pip install --upgrade customtkinter
```

### PyTorch 설치 오류

공식 웹사이트에서 설치 명령어 확인:
https://pytorch.org/get-started/locally/

### 버전 충돌

```bash
# 가상환경을 새로 만들고 다시 설치
python -m venv venv_new
source venv_new/bin/activate  # Windows: venv_new\Scripts\activate
pip install -r requirements.txt
```

## 최소 요구사항

- **Python**: 3.8 이상
- **RAM**: 8GB 이상 (16GB 권장)
- **GPU**: NVIDIA GPU (선택사항, 학습 속도 향상)
- **저장공간**: 5GB 이상 (모델 및 데이터용)

## 설치 확인

모든 패키지가 정상적으로 설치되었는지 확인:

```bash
python test_system.py
```

출력 예시:
```
============================================================
테스트 결과 요약
============================================================
imports             : ✓ 통과
model               : ✓ 통과
processor           : ✓ 통과
predictor           : ✓ 통과
realtime            : ✓ 통과
ui                  : ✓ 통과

============================================================
✓ 모든 테스트 통과!
============================================================
```

## 다음 단계

설치가 완료되면:

1. **README 읽기**: `README_WELDING_ML.md` 참고
2. **예제 실행**: `example_usage.py` 참고
3. **GUI 실행**: `python welding_video_ui.py`
4. **데이터 준비**: 용접 비디오 수집 및 레이블링
5. **모델 학습**: 준비된 데이터로 모델 학습

## 지원

설치 중 문제가 발생하면:
1. Python 버전 확인: `python --version`
2. pip 업그레이드: `pip install --upgrade pip`
3. 가상환경 사용
4. 이슈 등록
