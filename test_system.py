"""
시스템 테스트 스크립트
실제 비디오 데이터 없이 기본 기능을 테스트합니다.
"""

import sys
import traceback


def test_imports():
    """필수 패키지 임포트 테스트"""
    print("=" * 60)
    print("1. 패키지 임포트 테스트")
    print("=" * 60)

    results = {}

    # 기본 패키지
    packages = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('customtkinter', 'CustomTkinter'),
    ]

    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {display_name}: 설치됨")
            results[display_name] = True
        except ImportError as e:
            print(f"✗ {display_name}: 설치 필요 - {e}")
            results[display_name] = False

    return all(results.values())


def test_model_creation():
    """모델 생성 테스트"""
    print("\n" + "=" * 60)
    print("2. 딥러닝 모델 생성 테스트")
    print("=" * 60)

    try:
        from welding_ml_analyzer import CNN_LSTM_Model
        import torch

        model = CNN_LSTM_Model(
            num_classes=2,
            hidden_size=256,
            num_lstm_layers=2
        )

        print(f"✓ 모델 생성 성공")

        # 파라미터 개수 확인
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - 총 파라미터 수: {total_params:,}")

        # 더미 입력으로 Forward pass 테스트
        batch_size = 2
        sequence_length = 10
        channels = 3
        height = 224
        width = 224

        dummy_input = torch.randn(batch_size, sequence_length, channels, height, width)
        print(f"  - 입력 형태: {dummy_input.shape}")

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        print(f"  - 출력 형태: {output.shape}")
        print(f"✓ Forward pass 성공")

        return True

    except Exception as e:
        print(f"✗ 모델 생성 실패: {e}")
        traceback.print_exc()
        return False


def test_video_processor():
    """비디오 프로세서 테스트"""
    print("\n" + "=" * 60)
    print("3. 비디오 프로세서 테스트")
    print("=" * 60)

    try:
        from welding_ml_analyzer import VideoProcessor
        import numpy as np

        processor = VideoProcessor(frame_size=(224, 224))
        print("✓ VideoProcessor 초기화 성공")

        # 더미 프레임 생성
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"  - 더미 프레임 생성: {dummy_frame.shape}")

        # 전처리 테스트
        preprocessed = processor.preprocess_frame(dummy_frame)
        print(f"  - 전처리 결과: {preprocessed.shape}")
        print(f"✓ 프레임 전처리 성공")

        # 용융풀 강조 테스트
        enhanced = processor.enhance_melt_pool(dummy_frame)
        print(f"  - 용융풀 강조 결과: {enhanced.shape}")
        print(f"✓ 용융풀 강조 성공")

        return True

    except Exception as e:
        print(f"✗ VideoProcessor 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_predictor():
    """예측기 테스트"""
    print("\n" + "=" * 60)
    print("4. 예측기 초기화 테스트")
    print("=" * 60)

    try:
        from welding_ml_analyzer import WeldingDefectPredictor
        import torch

        # CPU 모드로 초기화
        predictor = WeldingDefectPredictor(device='cpu')
        print(f"✓ WeldingDefectPredictor 초기화 성공")
        print(f"  - 디바이스: {predictor.device}")
        print(f"  - 결함 타입: {predictor.defect_types}")

        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            print(f"  - CUDA 사용 가능")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  - CUDA 사용 불가 (CPU 모드)")

        return True

    except Exception as e:
        print(f"✗ Predictor 초기화 실패: {e}")
        traceback.print_exc()
        return False


def test_realtime_prediction():
    """실시간 예측 테스트 (더미 데이터)"""
    print("\n" + "=" * 60)
    print("5. 실시간 예측 테스트 (더미 데이터)")
    print("=" * 60)

    try:
        from welding_ml_analyzer import WeldingDefectPredictor
        import numpy as np

        predictor = WeldingDefectPredictor(device='cpu')

        # 더미 프레임 버퍼 생성
        sequence_length = 10
        frame_buffer = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(sequence_length)
        ]

        print(f"  - 더미 프레임 버퍼 생성: {len(frame_buffer)}개 프레임")

        # 예측 실행
        prediction = predictor.predict_realtime(frame_buffer)

        print(f"✓ 실시간 예측 성공:")
        print(f"  - 결함 유형: {prediction.defect_type}")
        print(f"  - 결함 확률: {prediction.defect_probability:.4f}")
        print(f"  - 신뢰도: {prediction.confidence:.4f}")

        return True

    except Exception as e:
        print(f"✗ 실시간 예측 실패: {e}")
        traceback.print_exc()
        return False


def test_ui_import():
    """UI 모듈 임포트 테스트"""
    print("\n" + "=" * 60)
    print("6. UI 모듈 임포트 테스트")
    print("=" * 60)

    try:
        from welding_video_ui import WeldingVideoAnalyzer
        print("✓ WeldingVideoAnalyzer 임포트 성공")
        print("  - GUI를 실행하려면 welding_video_ui.py를 직접 실행하세요")
        return True

    except Exception as e:
        print(f"✗ UI 모듈 임포트 실패: {e}")
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("용접 용융풀 결함 분석 시스템 - 시스템 테스트")
    print("=" * 60)
    print()

    test_results = {}

    # 각 테스트 실행
    test_results['imports'] = test_imports()
    test_results['model'] = test_model_creation()
    test_results['processor'] = test_video_processor()
    test_results['predictor'] = test_predictor()
    test_results['realtime'] = test_realtime_prediction()
    test_results['ui'] = test_ui_import()

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    for test_name, result in test_results.items():
        status = "✓ 통과" if result else "✗ 실패"
        print(f"{test_name:20s}: {status}")

    all_passed = all(test_results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 모든 테스트 통과!")
        print("=" * 60)
        print("\n시스템이 정상적으로 작동합니다.")
        print("\n다음 단계:")
        print("1. 용접 비디오 데이터 준비")
        print("2. 데이터 레이블링 (정상/결함)")
        print("3. 모델 학습: example_usage.py의 example_2 참고")
        print("4. 비디오 분석: GUI 실행 또는 코드 사용")
        print("\nGUI 실행 방법:")
        print("  python welding_video_ui.py")
    else:
        print("✗ 일부 테스트 실패")
        print("=" * 60)
        print("\n실패한 테스트를 확인하고 필요한 패키지를 설치하세요:")
        print("  pip install -r requirements.txt")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
