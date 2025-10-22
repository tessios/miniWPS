"""
동작 시퀀스 데모 (PyTorch 없이 실행 가능)
비디오 처리 부분만 테스트합니다.
"""

import cv2
import numpy as np
from pathlib import Path


class SimpleVideoProcessor:
    """간단한 비디오 프로세서 (데모용)"""

    def __init__(self):
        print("=" * 70)
        print("용접 용융풀 결함 분석 시스템 - 동작 시퀀스 데모")
        print("=" * 70)

    def create_sample_video(self, output_path='sample_welding.mp4'):
        """샘플 용접 비디오 생성 (시뮬레이션)"""
        print("\n[단계 1] 샘플 용접 비디오 생성")
        print("-" * 70)

        # 비디오 설정
        width, height = 640, 480
        fps = 30
        duration = 3  # 3초
        total_frames = fps * duration

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"  - 해상도: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - 길이: {duration}초 ({total_frames}프레임)")

        # 용접 시뮬레이션 비디오 생성
        for i in range(total_frames):
            # 검은 배경
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # 용융풀 시뮬레이션 (밝은 원)
            center_x = 320 + int(i * 2)  # 오른쪽으로 이동
            center_y = 240 + int(np.sin(i * 0.2) * 20)  # 약간 위아래로

            # 밝기 변화 (결함 시뮬레이션)
            if i > 60 and i < 70:  # 결함 구간
                brightness = 150 + int(np.random.random() * 50)  # 불안정
                color = (brightness, brightness // 2, 0)  # 푸르스름
            else:  # 정상 구간
                brightness = 255
                color = (brightness, brightness, brightness)  # 하얀색

            # 용융풀 그리기
            cv2.circle(frame, (center_x, center_y), 30, color, -1)
            cv2.circle(frame, (center_x, center_y), 40,
                      (color[0]//2, color[1]//2, color[2]//2), 2)

            # 프레임 번호 표시
            cv2.putText(frame, f"Frame: {i}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)

        out.release()
        print(f"  ✓ 샘플 비디오 생성 완료: {output_path}")
        return output_path

    def analyze_video(self, video_path):
        """비디오 분석 시퀀스"""
        print(f"\n[단계 2] 비디오 분석 시작")
        print("-" * 70)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  ✗ 비디오를 열 수 없습니다: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  - 비디오 정보:")
        print(f"    • 해상도: {width}x{height}")
        print(f"    • FPS: {fps}")
        print(f"    • 총 프레임: {total_frames}")
        print(f"    • 길이: {total_frames/fps:.2f}초")

        print(f"\n[단계 3] 프레임별 특징 추출")
        print("-" * 70)

        frame_idx = 0
        analysis_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 분석 (10프레임마다)
            if frame_idx % 10 == 0:
                # 밝기 분석
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
                max_brightness = np.max(gray)

                # 용융풀 영역 검출 (밝은 영역)
                _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                melt_pool_area = np.sum(threshold > 0)

                # 안정성 평가 (간단한 휴리스틱)
                is_stable = max_brightness > 240 and melt_pool_area > 1000

                result = {
                    'frame': frame_idx,
                    'time': frame_idx / fps,
                    'avg_brightness': avg_brightness,
                    'max_brightness': max_brightness,
                    'melt_pool_area': melt_pool_area,
                    'status': '정상' if is_stable else '불안정'
                }

                analysis_results.append(result)

                print(f"  프레임 {frame_idx:3d} ({result['time']:5.2f}초): "
                      f"밝기={max_brightness:3.0f}, 면적={melt_pool_area:5.0f}px² "
                      f"→ {result['status']}")

            frame_idx += 1

        cap.release()

        print(f"\n[단계 4] 분석 결과 요약")
        print("-" * 70)

        total_analyzed = len(analysis_results)
        stable_count = sum(1 for r in analysis_results if r['status'] == '정상')
        unstable_count = total_analyzed - stable_count

        print(f"  - 분석된 구간: {total_analyzed}개")
        print(f"  - 정상 구간: {stable_count}개 ({stable_count/total_analyzed*100:.1f}%)")
        print(f"  - 불안정 구간: {unstable_count}개 ({unstable_count/total_analyzed*100:.1f}%)")

        if unstable_count > 0:
            print(f"\n  ⚠️  불안정 구간 발견:")
            for r in analysis_results:
                if r['status'] == '불안정':
                    print(f"     • {r['time']:.2f}초 (프레임 {r['frame']})")

        return analysis_results

    def show_workflow(self):
        """전체 워크플로우 설명"""
        print("\n" + "=" * 70)
        print("시스템 동작 시퀀스 (전체 워크플로우)")
        print("=" * 70)

        workflow = """
┌─────────────────────────────────────────────────────────────────┐
│                    1. 데이터 준비 단계                          │
├─────────────────────────────────────────────────────────────────┤
│  [사용자] 용접 비디오 선택                                      │
│     ↓                                                           │
│  [시스템] 비디오 로드 및 정보 추출                              │
│     • FPS, 해상도, 총 프레임 수 확인                            │
│     • 메타데이터 분석                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. 프레임 추출 단계                          │
├─────────────────────────────────────────────────────────────────┤
│  [VideoProcessor.extract_frames()]                             │
│     • 설정된 FPS로 프레임 추출 (예: 10 FPS)                     │
│     • BGR → RGB 변환                                            │
│     • 프레임 리스트로 저장                                      │
│                                                                 │
│  예: 30 FPS 비디오 → 10 FPS로 추출 = 1/3 프레임만 사용          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    3. 전처리 단계                               │
├─────────────────────────────────────────────────────────────────┤
│  [VideoProcessor.preprocess_frames()]                          │
│     1) 리사이징: 224x224 크기로 조정                            │
│     2) 정규화: ImageNet 통계로 정규화                           │
│        - Mean: [0.485, 0.456, 0.406]                           │
│        - Std:  [0.229, 0.224, 0.225]                           │
│     3) Tensor 변환: numpy array → PyTorch tensor               │
│                                                                 │
│  [선택] enhance_melt_pool(): 용융풀 영역 강조                   │
│     • 히스토그램 평활화                                         │
│     • 가우시안 블러                                             │
│     • 밝은 영역 임계값 처리                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    4. 시퀀스 구성 단계                          │
├─────────────────────────────────────────────────────────────────┤
│  연속된 프레임을 시퀀스로 묶기 (기본 10프레임)                  │
│                                                                 │
│  예시: 100개 프레임 → 슬라이딩 윈도우                           │
│     • 시퀀스 1: 프레임 0-9                                      │
│     • 시퀀스 2: 프레임 5-14   (50% 오버랩)                      │
│     • 시퀀스 3: 프레임 10-19                                    │
│     • ...                                                       │
│                                                                 │
│  각 시퀀스 형태: (10, 3, 224, 224)                              │
│     (프레임 수, 채널, 높이, 너비)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                5. 딥러닝 모델 예측 단계                         │
├─────────────────────────────────────────────────────────────────┤
│  [CNN_LSTM_Model.forward()]                                    │
│                                                                 │
│  입력: (batch, 10, 3, 224, 224)                                │
│                                                                 │
│  ┌─────────────────────────────────┐                          │
│  │   CNN Feature Extraction        │                          │
│  │   각 프레임별로 처리:            │                          │
│  │   • Conv2D (3→64)              │                          │
│  │   • Conv2D (64→128)            │                          │
│  │   • Conv2D (128→256)           │                          │
│  │   • Conv2D (256→512)           │                          │
│  │   • AdaptiveAvgPool → 512차원   │                          │
│  └─────────────────────────────────┘                          │
│                ↓                                               │
│  출력: (batch, 10, 512) - 10개 프레임의 512차원 특징            │
│                                                                 │
│                ↓                                               │
│  ┌─────────────────────────────────┐                          │
│  │   LSTM Temporal Modeling        │                          │
│  │   시간적 패턴 학습:              │                          │
│  │   • 2-layer LSTM                │                          │
│  │   • Hidden size: 256            │                          │
│  │   • 시퀀스 전체를 분석           │                          │
│  └─────────────────────────────────┘                          │
│                ↓                                               │
│  출력: (batch, 256) - 마지막 타임스텝의 은닉 상태               │
│                                                                 │
│                ↓                                               │
│  ┌─────────────────────────────────┐                          │
│  │   Fully Connected Classifier    │                          │
│  │   • Linear (256→128) + ReLU     │                          │
│  │   • Dropout (50%)               │                          │
│  │   • Linear (128→2)              │                          │
│  └─────────────────────────────────┘                          │
│                ↓                                               │
│  출력: (batch, 2) - [정상 확률, 결함 확률]                      │
│                                                                 │
│  Softmax 적용 후 최종 예측:                                     │
│     • 결함 확률 > 0.5 → "결함"                                  │
│     • 결함 확률 ≤ 0.5 → "정상"                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    6. 결과 후처리 단계                          │
├─────────────────────────────────────────────────────────────────┤
│  [WeldingDefectPredictor.predict_video()]                      │
│                                                                 │
│  각 시퀀스별 예측 결과를 DefectPrediction 객체로 변환:          │
│                                                                 │
│  DefectPrediction(                                             │
│     frame_number=50,        # 시퀀스 중심 프레임               │
│     defect_probability=0.85, # 결함 확률                       │
│     defect_type='결함',      # 분류 결과                       │
│     confidence=0.85,         # 예측 신뢰도                     │
│     timestamp=5.0            # 비디오 상 시간(초)              │
│  )                                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    7. 결과 시각화 단계                          │
├─────────────────────────────────────────────────────────────────┤
│  [Option 1] GUI 표시                                           │
│     • 오른쪽 패널에 프레임별 결과 리스트 표시                   │
│     • 통계 정보 (정상/결함 비율)                                │
│     • 비디오 재생 시 예측 결과 오버레이                         │
│                                                                 │
│  [Option 2] JSON 저장                                          │
│     {                                                          │
│       "video_path": "welding.mp4",                            │
│       "predictions": [                                         │
│         {"frame": 50, "defect_prob": 0.85, ...},              │
│         ...                                                    │
│       ]                                                        │
│     }                                                          │
│                                                                 │
│  [Option 3] 주석 비디오 생성                                    │
│     • 원본 비디오에 예측 결과 텍스트 오버레이                   │
│     • 결함 구간은 빨간색, 정상은 녹색 표시                      │
│     • output_with_predictions.mp4로 저장                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    학습 모드 (Training)                         │
├─────────────────────────────────────────────────────────────────┤
│  [WeldingDefectPredictor.train()]                              │
│                                                                 │
│  1단계: 데이터 준비                                             │
│     • WeldingDataset으로 비디오 + 레이블 로드                   │
│     • DataLoader로 배치 구성                                    │
│                                                                 │
│  2단계: 학습 루프 (각 에폭마다)                                 │
│     for epoch in range(num_epochs):                            │
│       for batch in train_loader:                               │
│         1) Forward pass: 모델 예측                             │
│         2) Loss 계산: CrossEntropyLoss                         │
│         3) Backward pass: 그래디언트 계산                       │
│         4) Optimizer step: 가중치 업데이트                      │
│                                                                 │
│  3단계: 검증                                                    │
│     • 검증 데이터로 성능 평가                                   │
│     • 최고 성능 모델 저장                                       │
│                                                                 │
│  4단계: Learning Rate 조정                                     │
│     • ReduceLROnPlateau: 손실이 개선 안 되면 LR 감소           │
│                                                                 │
│  출력: welding_defect_model.pth (학습된 가중치)                │
└─────────────────────────────────────────────────────────────────┘
"""
        print(workflow)


def main():
    """메인 데모 실행"""
    processor = SimpleVideoProcessor()

    # 워크플로우 설명
    processor.show_workflow()

    print("\n" + "=" * 70)
    print("실제 동작 테스트 시작")
    print("=" * 70)

    # 샘플 비디오 생성
    video_path = processor.create_sample_video()

    # 비디오 분석
    results = processor.analyze_video(video_path)

    print("\n" + "=" * 70)
    print("데모 완료!")
    print("=" * 70)
    print("\n실제 시스템 사용 시:")
    print("  1. PyTorch 설치 필요")
    print("  2. GUI 실행: python welding_video_ui.py")
    print("  3. 또는 코드로 직접 사용: example_usage.py 참고")
    print("=" * 70)


if __name__ == "__main__":
    main()
