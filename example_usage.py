"""
용접 용융풀 비디오 분석 시스템 사용 예제
"""

import numpy as np
import torch
from pathlib import Path

from welding_ml_analyzer import (
    WeldingDefectPredictor,
    WeldingDataset,
    VideoProcessor,
    visualize_predictions
)
from torch.utils.data import DataLoader


def example_1_video_prediction():
    """
    예제 1: 단일 비디오 결함 예측
    학습된 모델을 사용하여 비디오를 분석합니다.
    """
    print("=" * 60)
    print("예제 1: 단일 비디오 결함 예측")
    print("=" * 60)

    # 1. Predictor 초기화 (학습된 모델 로드)
    predictor = WeldingDefectPredictor(
        model_path='welding_defect_model.pth'  # 학습된 모델 경로
    )

    # 2. 비디오 분석
    video_path = 'test_welding_video.mp4'  # 분석할 비디오 경로

    print(f"\n비디오 분석 중: {video_path}")
    predictions = predictor.predict_video(video_path, sequence_length=10)

    # 3. 결과 출력
    print(f"\n총 {len(predictions)}개의 예측 결과:")
    for i, pred in enumerate(predictions[:5]):  # 처음 5개만 출력
        print(f"\n[{i+1}] 프레임 {pred.frame_number}:")
        print(f"  - 결함 유형: {pred.defect_type}")
        print(f"  - 결함 확률: {pred.defect_probability:.2%}")
        print(f"  - 신뢰도: {pred.confidence:.2%}")
        print(f"  - 타임스탬프: {pred.timestamp:.2f}초")

    # 4. 결과 시각화 (비디오로 저장)
    print("\n\n결과를 비디오로 저장 중...")
    visualize_predictions(
        video_path=video_path,
        predictions=predictions,
        output_path='output_with_predictions.mp4'
    )
    print("저장 완료: output_with_predictions.mp4")


def example_2_model_training():
    """
    예제 2: 모델 학습
    용접 비디오 데이터셋으로 모델을 학습합니다.
    """
    print("\n\n" + "=" * 60)
    print("예제 2: 모델 학습")
    print("=" * 60)

    # 1. 데이터 준비
    # 실제로는 여러분의 데이터 경로를 사용하세요
    train_videos = [
        'data/train/video_1.mp4',
        'data/train/video_2.mp4',
        'data/train/video_3.mp4',
        'data/train/video_4.mp4',
    ]

    # 레이블: 0 = 정상, 1 = 결함
    train_labels = [0, 1, 0, 1]

    val_videos = [
        'data/val/video_5.mp4',
        'data/val/video_6.mp4',
    ]

    val_labels = [0, 1]

    # 2. 데이터셋 생성
    print("\n데이터셋 생성 중...")
    train_dataset = WeldingDataset(
        video_paths=train_videos,
        labels=train_labels,
        sequence_length=10
    )

    val_dataset = WeldingDataset(
        video_paths=val_videos,
        labels=val_labels,
        sequence_length=10
    )

    # 3. 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # GPU 메모리에 맞게 조정
        shuffle=True,
        num_workers=0  # Windows에서는 0으로 설정
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    # 4. Predictor 초기화
    predictor = WeldingDefectPredictor()

    # 5. 학습 시작
    print("\n모델 학습 시작...")
    predictor.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.001,
        save_path='welding_defect_model.pth'
    )

    print("\n학습 완료! 모델이 'welding_defect_model.pth'에 저장되었습니다.")


def example_3_realtime_prediction():
    """
    예제 3: 실시간 예측 (프레임 버퍼 사용)
    """
    print("\n\n" + "=" * 60)
    print("예제 3: 실시간 예측 시뮬레이션")
    print("=" * 60)

    # 1. Predictor 및 Processor 초기화
    predictor = WeldingDefectPredictor(
        model_path='welding_defect_model.pth'
    )
    processor = VideoProcessor()

    # 2. 비디오에서 프레임 추출
    video_path = 'test_welding_video.mp4'
    frames = processor.extract_frames(video_path, frame_rate=10)

    print(f"\n총 {len(frames)}개 프레임 추출")

    # 3. 슬라이딩 윈도우로 실시간 예측 시뮬레이션
    sequence_length = 10
    frame_buffer = []

    print("\n실시간 예측 시뮬레이션 시작:")
    for i, frame in enumerate(frames[:50]):  # 처음 50프레임만
        # 버퍼에 프레임 추가
        frame_buffer.append(frame)

        # 버퍼가 충분히 차면 예측
        if len(frame_buffer) >= sequence_length:
            # 최근 sequence_length 개의 프레임만 사용
            recent_frames = frame_buffer[-sequence_length:]

            # 예측
            prediction = predictor.predict_realtime(recent_frames)

            print(f"프레임 {i}: {prediction.defect_type} "
                  f"(결함 확률: {prediction.defect_probability:.2%})")


def example_4_video_preprocessing():
    """
    예제 4: 비디오 전처리 및 용융풀 강조
    """
    print("\n\n" + "=" * 60)
    print("예제 4: 비디오 전처리")
    print("=" * 60)

    processor = VideoProcessor()

    # 1. 프레임 추출
    video_path = 'test_welding_video.mp4'
    frames = processor.extract_frames(video_path, frame_rate=10)

    print(f"\n{len(frames)}개 프레임 추출")

    # 2. 용융풀 영역 강조
    print("\n용융풀 강조 처리 중...")
    enhanced_frames = []
    for i, frame in enumerate(frames[:10]):  # 처음 10개만
        enhanced = processor.enhance_melt_pool(frame)
        enhanced_frames.append(enhanced)

    print("용융풀 강조 완료")

    # 3. 전처리 (텐서 변환)
    print("\n텐서로 변환 중...")
    frames_tensor = processor.preprocess_frames(frames[:10])
    print(f"텐서 형태: {frames_tensor.shape}")
    # 출력: torch.Size([10, 3, 224, 224]) - (프레임 수, 채널, 높이, 너비)


def example_5_batch_processing():
    """
    예제 5: 여러 비디오 일괄 처리
    """
    print("\n\n" + "=" * 60)
    print("예제 5: 여러 비디오 일괄 처리")
    print("=" * 60)

    predictor = WeldingDefectPredictor(
        model_path='welding_defect_model.pth'
    )

    # 분석할 비디오 목록
    video_paths = [
        'video1.mp4',
        'video2.mp4',
        'video3.mp4',
    ]

    results_all = {}

    for video_path in video_paths:
        print(f"\n처리 중: {video_path}")

        try:
            # 예측
            predictions = predictor.predict_video(video_path)

            # 결과 저장
            results_all[video_path] = predictions

            # 간단한 통계
            defect_count = sum(1 for p in predictions if p.defect_type == '결함')
            total = len(predictions)

            print(f"  - 총 분석: {total}개 구간")
            print(f"  - 결함 발견: {defect_count}개 ({defect_count/total*100:.1f}%)")

        except Exception as e:
            print(f"  - 오류: {str(e)}")

    print("\n\n일괄 처리 완료!")


def example_6_custom_model_config():
    """
    예제 6: 커스텀 모델 구성
    """
    print("\n\n" + "=" * 60)
    print("예제 6: 커스텀 모델 구성")
    print("=" * 60)

    from welding_ml_analyzer import CNN_LSTM_Model

    # 커스텀 모델 파라미터로 생성
    custom_model = CNN_LSTM_Model(
        num_classes=3,  # 정상, 경미한 결함, 심각한 결함
        hidden_size=512,  # LSTM 은닉층 크기 증가
        num_lstm_layers=3,  # LSTM 레이어 3개
        dropout=0.3  # 드롭아웃 30%
    )

    # 모델 정보 출력
    total_params = sum(p.numel() for p in custom_model.parameters())
    trainable_params = sum(p.numel() for p in custom_model.parameters() if p.requires_grad)

    print(f"\n모델 구성:")
    print(f"  - 클래스 수: 3")
    print(f"  - LSTM 은닉층 크기: 512")
    print(f"  - LSTM 레이어 수: 3")
    print(f"  - 드롭아웃: 30%")
    print(f"\n파라미터:")
    print(f"  - 전체: {total_params:,}")
    print(f"  - 학습 가능: {trainable_params:,}")


def main():
    """
    메인 함수 - 원하는 예제를 선택하여 실행하세요
    """
    print("\n" + "=" * 60)
    print("용접 용융풀 비디오 분석 시스템 - 사용 예제")
    print("=" * 60)

    print("\n사용 가능한 예제:")
    print("1. 단일 비디오 결함 예측")
    print("2. 모델 학습")
    print("3. 실시간 예측 시뮬레이션")
    print("4. 비디오 전처리")
    print("5. 여러 비디오 일괄 처리")
    print("6. 커스텀 모델 구성")

    print("\n" + "-" * 60)
    print("주의: 이 예제들을 실행하려면 비디오 데이터가 필요합니다.")
    print("비디오 경로를 실제 데이터 경로로 변경한 후 실행하세요.")
    print("-" * 60)

    # 예제 6은 데이터 없이도 실행 가능
    print("\n\n데이터 없이 실행 가능한 예제 6 실행:")
    example_6_custom_model_config()

    # 다른 예제를 실행하려면 주석을 해제하세요
    # example_1_video_prediction()
    # example_2_model_training()
    # example_3_realtime_prediction()
    # example_4_video_preprocessing()
    # example_5_batch_processing()


if __name__ == "__main__":
    main()
