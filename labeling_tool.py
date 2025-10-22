"""
데이터 레이블링 도구
수집된 비디오/이미지에 레이블(정상/결함)을 쉽게 지정할 수 있습니다.
"""

import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class SimpleLabelingTool:
    """
    간단한 레이블링 도구 (OpenCV 기반)
    키보드로 레이블을 지정합니다.
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 데이터 디렉토리 (auto_data_collector의 출력)
        """
        self.data_dir = Path(data_dir)
        self.video_dir = self.data_dir / 'videos'
        self.metadata_dir = self.data_dir / 'metadata'

        # 레이블 정의
        self.labels = {
            '0': 'normal',    # 정상
            '1': 'defect',    # 결함
            '2': 'uncertain', # 불확실
            's': 'skip',      # 스킵
        }

        # 레이블 저장
        self.labeled_data = {}
        self.load_existing_labels()

        print("=" * 70)
        print("비디오 레이블링 도구")
        print("=" * 70)
        print("\n키 조작법:")
        print("  0: 정상 (Normal)")
        print("  1: 결함 (Defect)")
        print("  2: 불확실 (Uncertain)")
        print("  s: 스킵 (Skip)")
        print("  q: 종료 (Quit)")
        print("  Space: 재생/일시정지")
        print("  ←/→: 이전/다음 프레임")
        print("=" * 70)

    def load_existing_labels(self):
        """기존 레이블 로드"""
        labels_file = self.metadata_dir / 'labels.json'

        if labels_file.exists():
            with open(labels_file, 'r') as f:
                self.labeled_data = json.load(f)
            print(f"\n✓ 기존 레이블 {len(self.labeled_data)}개 로드")
        else:
            print("\n• 새로운 레이블링 세션 시작")

    def save_labels(self):
        """레이블 저장"""
        labels_file = self.metadata_dir / 'labels.json'

        with open(labels_file, 'w') as f:
            json.dump(self.labeled_data, f, indent=2)

        print(f"\n✓ 레이블 저장 완료: {labels_file}")

    def label_videos(self):
        """비디오 레이블링 시작"""

        if not self.video_dir.exists():
            print(f"✗ 비디오 디렉토리가 존재하지 않습니다: {self.video_dir}")
            return

        video_files = list(self.video_dir.glob('*.mp4')) + \
                      list(self.video_dir.glob('*.avi')) + \
                      list(self.video_dir.glob('*.mov'))

        if not video_files:
            print(f"✗ 비디오 파일이 없습니다: {self.video_dir}")
            return

        print(f"\n총 {len(video_files)}개 비디오 발견")

        # 아직 레이블되지 않은 비디오만 필터링
        unlabeled_videos = [
            v for v in video_files
            if v.name not in self.labeled_data
        ]

        if not unlabeled_videos:
            print("✓ 모든 비디오가 이미 레이블링되었습니다.")
            return

        print(f"레이블링 필요: {len(unlabeled_videos)}개")

        for idx, video_path in enumerate(unlabeled_videos):
            print(f"\n[{idx+1}/{len(unlabeled_videos)}] {video_path.name}")
            print("-" * 70)

            label = self.label_single_video(video_path)

            if label == 'quit':
                print("\n레이블링 중단")
                break
            elif label == 'skip':
                print("  → 스킵")
                continue
            else:
                self.labeled_data[video_path.name] = label
                print(f"  → 레이블: {label}")

                # 자동 저장
                self.save_labels()

        self.print_summary()

    def label_single_video(self, video_path: Path) -> str:
        """
        단일 비디오 레이블링

        Returns:
            레이블 ('normal', 'defect', 'uncertain', 'skip', 'quit')
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"  ✗ 비디오를 열 수 없습니다: {video_path}")
            return 'skip'

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  정보: {width}x{height}, {fps:.1f} FPS, {total_frames} 프레임")

        window_name = f'레이블링: {video_path.name}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        playing = True
        current_frame = 0
        label_result = None

        while True:
            if playing:
                ret, frame = cap.read()
                if not ret:
                    # 비디오 끝 → 처음으로
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    current_frame = 0
                    continue

                current_frame += 1
            else:
                # 일시정지 상태
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()

            if not ret:
                break

            # 정보 오버레이
            display_frame = frame.copy()
            self._draw_info(display_frame, current_frame, total_frames)

            cv2.imshow(window_name, display_frame)

            # 키 입력 (30fps 기준)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF

            if key == ord('q'):
                label_result = 'quit'
                break
            elif key == ord('0'):
                label_result = 'normal'
                break
            elif key == ord('1'):
                label_result = 'defect'
                break
            elif key == ord('2'):
                label_result = 'uncertain'
                break
            elif key == ord('s'):
                label_result = 'skip'
                break
            elif key == ord(' '):
                playing = not playing
            elif key == 81:  # 왼쪽 화살표
                current_frame = max(0, current_frame - 10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == 83:  # 오른쪽 화살표
                current_frame = min(total_frames - 1, current_frame + 10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        cap.release()
        cv2.destroyWindow(window_name)

        return label_result if label_result else 'skip'

    def _draw_info(self, frame: np.ndarray, current_frame: int, total_frames: int):
        """프레임에 정보 오버레이"""
        h, w = frame.shape[:2]

        # 배경
        cv2.rectangle(frame, (10, 10), (w - 10, 120), (0, 0, 0), -1)

        # 텍스트
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"프레임: {current_frame}/{total_frames}", (20, 40),
                   font, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "0:정상  1:결함  2:불확실  s:스킵  q:종료  Space:재생/일시정지",
                   (20, 70), font, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "←/→: 이전/다음 프레임 (10프레임 단위)",
                   (20, 100), font, 0.5, (200, 200, 200), 1)

    def print_summary(self):
        """레이블링 요약"""
        if not self.labeled_data:
            print("\n레이블링된 데이터가 없습니다.")
            return

        # 통계
        normal_count = sum(1 for v in self.labeled_data.values() if v == 'normal')
        defect_count = sum(1 for v in self.labeled_data.values() if v == 'defect')
        uncertain_count = sum(1 for v in self.labeled_data.values() if v == 'uncertain')

        print("\n" + "=" * 70)
        print("레이블링 요약")
        print("=" * 70)
        print(f"  총 레이블: {len(self.labeled_data)}개")
        print(f"  정상: {normal_count}개")
        print(f"  결함: {defect_count}개")
        print(f"  불확실: {uncertain_count}개")
        print("=" * 70)

    def export_for_training(self, output_file: str = 'training_data.json'):
        """학습용 데이터 형식으로 내보내기"""
        training_data = {
            'videos': [],
            'labels': []
        }

        for filename, label in self.labeled_data.items():
            if label in ['normal', 'defect']:  # 불확실은 제외
                video_path = str(self.video_dir / filename)
                training_data['videos'].append(video_path)
                training_data['labels'].append(0 if label == 'normal' else 1)

        output_path = self.metadata_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"\n✓ 학습 데이터 내보내기 완료: {output_path}")
        print(f"  사용 가능한 데이터: {len(training_data['videos'])}개")

        return output_path


def main():
    """메인 실행"""
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'welding_data'

    tool = SimpleLabelingTool(data_dir)
    tool.label_videos()
    tool.export_for_training()


if __name__ == "__main__":
    main()
