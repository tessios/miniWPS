"""
Welding Melt Pool Video Analysis Module
용접 용융풀 비디오 분석 및 결함 예측 시스템

이 모듈은 용접 용융풀 동영상을 분석하여 결함 발생 가능성을 예측합니다.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DefectPrediction:
    """결함 예측 결과"""
    frame_number: int
    defect_probability: float
    defect_type: str
    confidence: float
    timestamp: float


class VideoProcessor:
    """
    비디오 처리 클래스
    용접 용융풀 동영상에서 프레임을 추출하고 전처리합니다.
    """

    def __init__(self, frame_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            frame_size: 모델 입력을 위한 프레임 크기 (width, height)
        """
        self.frame_size = frame_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path: str,
                       frame_rate: Optional[int] = None) -> List[np.ndarray]:
        """
        비디오에서 프레임 추출

        Args:
            video_path: 비디오 파일 경로
            frame_rate: 초당 추출할 프레임 수 (None이면 모든 프레임)

        Returns:
            프레임 리스트
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_rate is None:
            frame_interval = 1
        else:
            frame_interval = max(1, int(fps / frame_rate))

        logger.info(f"비디오 FPS: {fps}, 총 프레임: {total_frames}, 추출 간격: {frame_interval}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # BGR to RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_idx += 1

        cap.release()
        logger.info(f"총 {len(frames)}개 프레임 추출 완료")
        return frames

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        단일 프레임 전처리

        Args:
            frame: 원본 프레임 (numpy array)

        Returns:
            전처리된 텐서
        """
        return self.transform(frame)

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        여러 프레임을 배치로 전처리

        Args:
            frames: 프레임 리스트

        Returns:
            배치 텐서 (batch_size, channels, height, width)
        """
        preprocessed = [self.preprocess_frame(frame) for frame in frames]
        return torch.stack(preprocessed)

    def enhance_melt_pool(self, frame: np.ndarray) -> np.ndarray:
        """
        용융풀 영역 강조 (밝은 영역 검출)

        Args:
            frame: 입력 프레임

        Returns:
            용융풀이 강조된 프레임
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 히스토그램 평활화로 명암 강조
        equalized = cv2.equalizeHist(gray)

        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

        # 밝은 영역(용융풀) 임계값 처리
        _, threshold = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # 마스크를 컬러 이미지에 적용
        mask = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
        enhanced = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

        return enhanced


class WeldingDataset(Dataset):
    """
    용접 비디오 데이터셋
    학습용 데이터 로더
    """

    def __init__(self, video_paths: List[str], labels: List[int],
                 sequence_length: int = 10, transform=None):
        """
        Args:
            video_paths: 비디오 파일 경로 리스트
            labels: 각 비디오의 레이블 (0: 정상, 1: 결함)
            sequence_length: 시퀀스 길이 (연속 프레임 개수)
            transform: 전처리 변환
        """
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.processor = VideoProcessor()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 프레임 추출
        frames = self.processor.extract_frames(video_path, frame_rate=10)

        # 시퀀스 길이에 맞게 샘플링
        if len(frames) > self.sequence_length:
            # 균등하게 샘플링
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.sequence_length:
            # 부족하면 마지막 프레임 반복
            frames += [frames[-1]] * (self.sequence_length - len(frames))

        # 전처리
        frames_tensor = self.processor.preprocess_frames(frames)

        return frames_tensor, label


class CNN_LSTM_Model(nn.Module):
    """
    CNN + LSTM 하이브리드 모델

    CNN: 각 프레임에서 공간적 특징 추출
    LSTM: 시간적 패턴 학습
    """

    def __init__(self, num_classes: int = 2, hidden_size: int = 256,
                 num_lstm_layers: int = 2, dropout: float = 0.5):
        """
        Args:
            num_classes: 클래스 개수 (결함 유형 개수)
            hidden_size: LSTM 은닉층 크기
            num_lstm_layers: LSTM 레이어 개수
            dropout: 드롭아웃 비율
        """
        super(CNN_LSTM_Model, self).__init__()

        # CNN Feature Extractor (ResNet-inspired)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Feature dimension from CNN
        self.cnn_output_size = 512

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch_size, sequence_length, channels, height, width)

        Returns:
            (batch_size, num_classes)
        """
        batch_size, seq_len, c, h, w = x.size()

        # CNN feature extraction for each frame
        # Reshape to (batch_size * seq_len, c, h, w)
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn(x)  # (batch_size * seq_len, 512, 1, 1)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 512)

        # LSTM temporal modeling
        lstm_out, (hn, cn) = self.lstm(cnn_features)  # lstm_out: (batch_size, seq_len, hidden_size)

        # Use the last time step output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Classification
        output = self.fc(last_output)  # (batch_size, num_classes)

        return output


class WeldingDefectPredictor:
    """
    용접 결함 예측기
    학습된 모델을 사용하여 실시간 또는 배치로 결함을 예측합니다.
    """

    def __init__(self, model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model_path: 학습된 모델 가중치 경로
            device: 'cuda' 또는 'cpu'
        """
        self.device = torch.device(device)
        self.model = CNN_LSTM_Model(num_classes=2)
        self.model.to(self.device)

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            logger.info(f"모델 로드 완료: {model_path}")

        self.processor = VideoProcessor()
        self.defect_types = ['정상', '결함']

    def load_model(self, model_path: str):
        """모델 가중치 로드"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def save_model(self, model_path: str):
        """모델 가중치 저장"""
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"모델 저장 완료: {model_path}")

    def predict_video(self, video_path: str,
                      sequence_length: int = 10) -> List[DefectPrediction]:
        """
        비디오 전체에 대한 결함 예측

        Args:
            video_path: 비디오 파일 경로
            sequence_length: 분석할 연속 프레임 개수

        Returns:
            각 시퀀스에 대한 예측 결과 리스트
        """
        self.model.eval()

        # 프레임 추출
        frames = self.processor.extract_frames(video_path, frame_rate=10)

        predictions = []

        with torch.no_grad():
            # 슬라이딩 윈도우로 시퀀스 생성
            for i in range(0, len(frames) - sequence_length + 1, sequence_length // 2):
                sequence = frames[i:i + sequence_length]

                if len(sequence) < sequence_length:
                    break

                # 전처리
                frames_tensor = self.processor.preprocess_frames(sequence)
                frames_tensor = frames_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

                # 예측
                output = self.model(frames_tensor)
                probabilities = torch.softmax(output, dim=1)
                defect_prob = probabilities[0, 1].item()  # 결함 확률
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()

                prediction = DefectPrediction(
                    frame_number=i + sequence_length // 2,
                    defect_probability=defect_prob,
                    defect_type=self.defect_types[predicted_class],
                    confidence=confidence,
                    timestamp=i / 10.0  # Assuming 10 FPS
                )

                predictions.append(prediction)
                logger.info(f"프레임 {prediction.frame_number}: "
                          f"{prediction.defect_type} (확률: {defect_prob:.2%}, "
                          f"신뢰도: {confidence:.2%})")

        return predictions

    def predict_realtime(self, frame_buffer: List[np.ndarray]) -> DefectPrediction:
        """
        실시간 결함 예측 (프레임 버퍼 기반)

        Args:
            frame_buffer: 최근 프레임들의 버퍼 (sequence_length 개)

        Returns:
            예측 결과
        """
        self.model.eval()

        with torch.no_grad():
            frames_tensor = self.processor.preprocess_frames(frame_buffer)
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

            output = self.model(frames_tensor)
            probabilities = torch.softmax(output, dim=1)
            defect_prob = probabilities[0, 1].item()
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

            return DefectPrediction(
                frame_number=len(frame_buffer),
                defect_probability=defect_prob,
                defect_type=self.defect_types[predicted_class],
                confidence=confidence,
                timestamp=0.0
            )

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 0.001,
              save_path: str = 'welding_defect_model.pth'):
        """
        모델 학습

        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            num_epochs: 에폭 수
            learning_rate: 학습률
            save_path: 모델 저장 경로
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.5, patience=5)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()

            val_accuracy = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            logger.info(f'Epoch [{epoch+1}/{num_epochs}] '
                       f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(save_path)
                logger.info(f'최고 모델 저장 (검증 손실: {best_val_loss:.4f})')

        logger.info('학습 완료!')


def visualize_predictions(video_path: str, predictions: List[DefectPrediction],
                         output_path: str = 'output_with_predictions.mp4'):
    """
    예측 결과를 비디오에 시각화

    Args:
        video_path: 원본 비디오 경로
        predictions: 예측 결과 리스트
        output_path: 출력 비디오 경로
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 예측을 프레임 번호로 인덱싱
    pred_dict = {p.frame_number: p for p in predictions}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 해당 프레임에 예측이 있으면 표시
        if frame_idx in pred_dict:
            pred = pred_dict[frame_idx]

            # 배경 박스
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)

            # 텍스트 표시
            text1 = f"Status: {pred.defect_type}"
            text2 = f"Defect Prob: {pred.defect_probability:.2%}"
            text3 = f"Confidence: {pred.confidence:.2%}"

            color = (0, 0, 255) if pred.defect_type == '결함' else (0, 255, 0)

            cv2.putText(frame, text1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2)
            cv2.putText(frame, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
            cv2.putText(frame, text3, (20, 85), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    logger.info(f"시각화 완료: {output_path}")


if __name__ == "__main__":
    # 사용 예제

    # 1. 비디오 프로세서 테스트
    print("=== 비디오 프로세서 테스트 ===")
    processor = VideoProcessor()

    # 2. 모델 초기화
    print("\n=== 모델 초기화 ===")
    predictor = WeldingDefectPredictor()
    print(f"사용 디바이스: {predictor.device}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in predictor.model.parameters()):,}")

    # 3. 학습 예제 (데이터가 있을 때)
    # train_videos = ['video1.mp4', 'video2.mp4', ...]
    # train_labels = [0, 1, ...]  # 0: 정상, 1: 결함
    # train_dataset = WeldingDataset(train_videos, train_labels, sequence_length=10)
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    #
    # val_dataset = WeldingDataset(val_videos, val_labels, sequence_length=10)
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    #
    # predictor.train(train_loader, val_loader, num_epochs=50)

    # 4. 예측 예제
    # predictions = predictor.predict_video('test_video.mp4')
    # visualize_predictions('test_video.mp4', predictions, 'output.mp4')

    print("\n=== 준비 완료 ===")
    print("학습 데이터를 준비한 후 train() 메서드를 사용하여 모델을 학습하세요.")
    print("학습된 모델로 predict_video()를 사용하여 결함을 예측할 수 있습니다.")
