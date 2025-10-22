"""
자동 학습 파이프라인
데이터 수집 → 레이블링 → 전처리 → 학습까지 자동화
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional
import sys


class AutoTrainingPipeline:
    """
    자동 학습 파이프라인
    """

    def __init__(self, data_dir: str = 'welding_data'):
        """
        Args:
            data_dir: 데이터 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.metadata_dir = self.data_dir / 'metadata'

        print("=" * 70)
        print("자동 학습 파이프라인")
        print("=" * 70)

    def step1_collect_data(self):
        """단계 1: 데이터 수집"""
        print("\n" + "=" * 70)
        print("단계 1: 데이터 수집")
        print("=" * 70)

        from auto_data_collector import AutoDataCollector

        collector = AutoDataCollector(output_dir=str(self.data_dir))

        print("\n데이터 수집 방법을 선택하세요:")
        print("  1. 로컬 폴더에서 가져오기")
        print("  2. URL 리스트에서 다운로드")
        print("  3. YouTube 검색 (yt-dlp 필요)")
        print("  4. Pexels API (API 키 필요)")
        print("  5. 건너뛰기 (이미 수집됨)")

        choice = input("\n선택 (1-5): ").strip()

        if choice == '1':
            folder_path = input("폴더 경로: ")
            pattern = input("파일 패턴 (예: *.mp4): ") or "*.mp4"
            collector.collect_from_local_folder(folder_path, pattern)

        elif choice == '2':
            urls_file = input("URL 리스트 파일 경로 (.txt): ")
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            collector.collect_from_url_list(urls)

        elif choice == '3':
            query = input("검색 쿼리 (기본: welding process): ") or "welding process"
            max_videos = int(input("최대 개수 (기본: 10): ") or "10")
            collector.collect_from_youtube(query, max_videos)

        elif choice == '4':
            query = input("검색 쿼리 (기본: welding): ") or "welding"
            max_items = int(input("최대 개수 (기본: 20): ") or "20")
            collector.collect_from_pexels(query, max_items)

        else:
            print("  건너뛰기")

        collector.save_collection_log()
        collector.print_summary()

    def step2_label_data(self):
        """단계 2: 데이터 레이블링"""
        print("\n" + "=" * 70)
        print("단계 2: 데이터 레이블링")
        print("=" * 70)

        from labeling_tool import SimpleLabelingTool

        tool = SimpleLabelingTool(str(self.data_dir))

        print("\n레이블링을 시작하시겠습니까?")
        print("  y: 예")
        print("  n: 아니오 (이미 레이블링됨)")

        choice = input("\n선택 (y/n): ").strip().lower()

        if choice == 'y':
            tool.label_videos()
            tool.export_for_training()
        else:
            print("  건너뛰기")

    def step3_prepare_training_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """단계 3: 학습 데이터 준비"""
        print("\n" + "=" * 70)
        print("단계 3: 학습 데이터 준비")
        print("=" * 70)

        training_data_file = self.metadata_dir / 'training_data.json'

        if not training_data_file.exists():
            print("✗ 학습 데이터 파일이 없습니다.")
            print("  레이블링을 먼저 완료하세요.")
            return [], [], [], []

        with open(training_data_file, 'r') as f:
            data = json.load(f)

        videos = data['videos']
        labels = data['labels']

        # 검증
        valid_videos = []
        valid_labels = []

        for video, label in zip(videos, labels):
            if Path(video).exists():
                valid_videos.append(video)
                valid_labels.append(label)
            else:
                print(f"  ⚠️  파일 없음, 스킵: {video}")

        print(f"\n✓ 유효한 데이터: {len(valid_videos)}개")
        print(f"  정상: {sum(1 for l in valid_labels if l == 0)}개")
        print(f"  결함: {sum(1 for l in valid_labels if l == 1)}개")

        # Train/Val 분할 (80/20)
        split_idx = int(len(valid_videos) * 0.8)

        train_videos = valid_videos[:split_idx]
        train_labels = valid_labels[:split_idx]
        val_videos = valid_videos[split_idx:]
        val_labels = valid_labels[split_idx:]

        print(f"\n데이터 분할:")
        print(f"  학습 데이터: {len(train_videos)}개")
        print(f"  검증 데이터: {len(val_videos)}개")

        return train_videos, train_labels, val_videos, val_labels

    def step4_train_model(self, train_videos: List[str], train_labels: List[int],
                         val_videos: List[str], val_labels: List[int]):
        """단계 4: 모델 학습"""
        print("\n" + "=" * 70)
        print("단계 4: 모델 학습")
        print("=" * 70)

        try:
            from welding_ml_analyzer import WeldingDefectPredictor, WeldingDataset
            import torch
            from torch.utils.data import DataLoader
        except ImportError as e:
            print(f"✗ 필요한 패키지가 없습니다: {e}")
            print("  pip install torch torchvision 을 먼저 실행하세요.")
            return

        if not train_videos:
            print("✗ 학습 데이터가 없습니다.")
            return

        # 학습 설정
        print("\n학습 설정:")
        sequence_length = int(input("  시퀀스 길이 (기본: 10): ") or "10")
        batch_size = int(input("  배치 크기 (기본: 2): ") or "2")
        num_epochs = int(input("  에폭 수 (기본: 50): ") or "50")
        learning_rate = float(input("  학습률 (기본: 0.001): ") or "0.001")

        # 데이터셋 생성
        print("\n데이터셋 생성 중...")
        train_dataset = WeldingDataset(
            video_paths=train_videos,
            labels=train_labels,
            sequence_length=sequence_length
        )

        val_dataset = WeldingDataset(
            video_paths=val_videos,
            labels=val_labels,
            sequence_length=sequence_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # 모델 학습
        print("\n모델 학습 시작...")
        predictor = WeldingDefectPredictor()

        model_save_path = str(self.data_dir / 'welding_defect_model.pth')

        predictor.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            save_path=model_save_path
        )

        print(f"\n✓ 학습 완료! 모델 저장: {model_save_path}")

    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        print("\n자동 학습 파이프라인을 시작합니다.")
        print("\n진행 단계:")
        print("  1. 데이터 수집")
        print("  2. 데이터 레이블링")
        print("  3. 학습 데이터 준비")
        print("  4. 모델 학습")

        proceed = input("\n계속하시겠습니까? (y/n): ").strip().lower()

        if proceed != 'y':
            print("취소됨")
            return

        # 단계 1: 데이터 수집
        self.step1_collect_data()

        # 단계 2: 레이블링
        self.step2_label_data()

        # 단계 3: 데이터 준비
        train_videos, train_labels, val_videos, val_labels = self.step3_prepare_training_data()

        if not train_videos:
            print("\n✗ 파이프라인 중단: 학습 데이터가 없습니다.")
            return

        # 단계 4: 학습
        self.step4_train_model(train_videos, train_labels, val_videos, val_labels)

        print("\n" + "=" * 70)
        print("✓ 전체 파이프라인 완료!")
        print("=" * 70)


def main():
    """메인 실행"""

    print("""
╔══════════════════════════════════════════════════════════════════╗
║         용접 용융풀 결함 분석 - 자동 학습 파이프라인            ║
╚══════════════════════════════════════════════════════════════════╝

이 도구는 다음 과정을 자동화합니다:
  1. 용접 비디오/이미지 자동 수집
  2. 수집된 데이터 레이블링 (정상/결함)
  3. 학습 데이터 전처리 및 분할
  4. 딥러닝 모델 학습

    """)

    data_dir = input("데이터 저장 디렉토리 (기본: welding_data): ").strip() or "welding_data"

    pipeline = AutoTrainingPipeline(data_dir=data_dir)

    print("\n실행 모드를 선택하세요:")
    print("  1. 전체 파이프라인 실행 (수집→레이블링→학습)")
    print("  2. 개별 단계 실행")

    mode = input("\n선택 (1-2): ").strip()

    if mode == '1':
        pipeline.run_full_pipeline()

    elif mode == '2':
        print("\n실행할 단계를 선택하세요:")
        print("  1. 데이터 수집")
        print("  2. 데이터 레이블링")
        print("  3. 학습 데이터 준비")
        print("  4. 모델 학습")

        step = input("\n선택 (1-4): ").strip()

        if step == '1':
            pipeline.step1_collect_data()
        elif step == '2':
            pipeline.step2_label_data()
        elif step == '3':
            pipeline.step3_prepare_training_data()
        elif step == '4':
            train_videos, train_labels, val_videos, val_labels = pipeline.step3_prepare_training_data()
            pipeline.step4_train_model(train_videos, train_labels, val_videos, val_labels)
        else:
            print("잘못된 선택")

    else:
        print("잘못된 선택")


if __name__ == "__main__":
    main()
