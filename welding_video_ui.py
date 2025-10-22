"""
Welding Video Analysis UI Module
용접 비디오 분석 UI 모듈

CustomTkinter 기반의 비디오 분석 인터페이스를 제공합니다.
"""

import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from pathlib import Path
from typing import Optional, List
import json

from welding_ml_analyzer import (
    WeldingDefectPredictor,
    VideoProcessor,
    DefectPrediction,
    visualize_predictions
)


class WeldingVideoAnalyzer(ctk.CTkToplevel):
    """
    용접 비디오 분석 전용 윈도우
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("용접 용융풀 결함 분석 시스템")
        self.geometry("1400x800")

        # 변수 초기화
        self.video_path: Optional[str] = None
        self.predictor = WeldingDefectPredictor()
        self.processor = VideoProcessor()
        self.current_predictions: List[DefectPrediction] = []
        self.is_playing = False
        self.current_frame_idx = 0
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.playback_speed = 1.0

        # UI 구성
        self._setup_ui()

    def _setup_ui(self):
        """UI 레이아웃 설정"""

        # 그리드 레이아웃 설정
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # === 상단 컨트롤 패널 ===
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.control_frame.grid_columnconfigure(1, weight=1)

        # 파일 선택
        ctk.CTkButton(
            self.control_frame,
            text="비디오 선택",
            command=self.load_video,
            width=120
        ).grid(row=0, column=0, padx=5, pady=5)

        self.video_path_label = ctk.CTkLabel(
            self.control_frame,
            text="비디오를 선택하세요",
            anchor="w"
        )
        self.video_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # 모델 선택
        ctk.CTkButton(
            self.control_frame,
            text="모델 불러오기",
            command=self.load_model,
            width=120
        ).grid(row=1, column=0, padx=5, pady=5)

        self.model_status_label = ctk.CTkLabel(
            self.control_frame,
            text="모델: 초기화됨 (학습 필요)",
            anchor="w"
        )
        self.model_status_label.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # === 메인 콘텐츠 영역 (3열 레이아웃) ===
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # 왼쪽: 분석 컨트롤
        self._setup_left_panel()

        # 중앙: 비디오 디스플레이
        self._setup_center_panel()

        # 오른쪽: 결과 표시
        self._setup_right_panel()

    def _setup_left_panel(self):
        """왼쪽 패널 (분석 컨트롤) 설정"""
        self.left_frame = ctk.CTkFrame(self.main_frame, width=250)
        self.left_frame.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)

        # 타이틀
        ctk.CTkLabel(
            self.left_frame,
            text="분석 제어",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=10, pady=10)

        # 분석 실행 버튼
        self.analyze_btn = ctk.CTkButton(
            self.left_frame,
            text="비디오 분석 시작",
            command=self.run_analysis,
            fg_color="green",
            height=40,
            state="disabled"
        )
        self.analyze_btn.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # 진행률 표시
        self.progress_label = ctk.CTkLabel(
            self.left_frame,
            text="진행: 0%",
            anchor="w"
        )
        self.progress_label.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.left_frame)
        self.progress_bar.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        self.progress_bar.set(0)

        # 구분선
        ctk.CTkFrame(self.left_frame, height=2, fg_color="gray").grid(
            row=4, column=0, padx=10, pady=15, sticky="ew"
        )

        # 재생 컨트롤
        ctk.CTkLabel(
            self.left_frame,
            text="재생 제어",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=5, column=0, padx=10, pady=(10, 5))

        # 재생/일시정지 버튼
        self.play_pause_btn = ctk.CTkButton(
            self.left_frame,
            text="▶ 재생",
            command=self.toggle_playback,
            state="disabled"
        )
        self.play_pause_btn.grid(row=6, column=0, padx=10, pady=5, sticky="ew")

        # 속도 조절
        ctk.CTkLabel(self.left_frame, text="재생 속도:").grid(
            row=7, column=0, padx=10, pady=(10, 0), sticky="w"
        )

        self.speed_slider = ctk.CTkSlider(
            self.left_frame,
            from_=0.25,
            to=2.0,
            number_of_steps=7,
            command=self.update_playback_speed
        )
        self.speed_slider.set(1.0)
        self.speed_slider.grid(row=8, column=0, padx=10, pady=5, sticky="ew")

        self.speed_label = ctk.CTkLabel(self.left_frame, text="1.0x")
        self.speed_label.grid(row=9, column=0, padx=10, pady=0)

        # 결과 저장
        ctk.CTkFrame(self.left_frame, height=2, fg_color="gray").grid(
            row=10, column=0, padx=10, pady=15, sticky="ew"
        )

        self.save_results_btn = ctk.CTkButton(
            self.left_frame,
            text="결과 저장",
            command=self.save_results,
            state="disabled"
        )
        self.save_results_btn.grid(row=11, column=0, padx=10, pady=5, sticky="ew")

        self.export_video_btn = ctk.CTkButton(
            self.left_frame,
            text="분석 비디오 내보내기",
            command=self.export_annotated_video,
            state="disabled"
        )
        self.export_video_btn.grid(row=12, column=0, padx=10, pady=5, sticky="ew")

    def _setup_center_panel(self):
        """중앙 패널 (비디오 디스플레이) 설정"""
        self.center_frame = ctk.CTkFrame(self.main_frame)
        self.center_frame.grid(row=0, column=1, padx=5, pady=0, sticky="nsew")
        self.center_frame.grid_rowconfigure(0, weight=1)
        self.center_frame.grid_columnconfigure(0, weight=1)

        # 비디오 캔버스
        self.video_canvas = ctk.CTkCanvas(
            self.center_frame,
            bg="black",
            highlightthickness=0
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        # 타임라인 슬라이더
        self.timeline_slider = ctk.CTkSlider(
            self.center_frame,
            from_=0,
            to=100,
            command=self.seek_video
        )
        self.timeline_slider.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # 프레임 정보
        self.frame_info_label = ctk.CTkLabel(
            self.center_frame,
            text="프레임: 0 / 0 | 시간: 00:00 / 00:00"
        )
        self.frame_info_label.grid(row=2, column=0, pady=5)

    def _setup_right_panel(self):
        """오른쪽 패널 (결과 표시) 설정"""
        self.right_frame = ctk.CTkFrame(self.main_frame, width=300)
        self.right_frame.grid(row=0, column=2, padx=(5, 0), pady=0, sticky="nsew")
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # 타이틀
        ctk.CTkLabel(
            self.right_frame,
            text="분석 결과",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=10, pady=10)

        # 결과 스크롤 프레임
        self.results_frame = ctk.CTkScrollableFrame(
            self.right_frame,
            label_text="결함 예측"
        )
        self.results_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # 통계 정보
        self.stats_frame = ctk.CTkFrame(self.right_frame)
        self.stats_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="통계:\n분석 대기 중...",
            justify="left"
        )
        self.stats_label.pack(padx=10, pady=10)

    # === 기능 메서드 ===

    def load_video(self):
        """비디오 파일 불러오기"""
        file_path = filedialog.askopenfilename(
            title="용접 비디오 선택",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.video_path = file_path
        self.video_path_label.configure(text=os.path.basename(file_path))

        # 비디오 열기
        if self.video_capture:
            self.video_capture.release()

        self.video_capture = cv2.VideoCapture(file_path)
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

        # 첫 프레임 표시
        self.current_frame_idx = 0
        self.display_current_frame()

        # 버튼 활성화
        self.analyze_btn.configure(state="normal")
        self.play_pause_btn.configure(state="normal")

    def load_model(self):
        """학습된 모델 불러오기"""
        file_path = filedialog.askopenfilename(
            title="모델 파일 선택",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.predictor.load_model(file_path)
            self.model_status_label.configure(
                text=f"모델: {os.path.basename(file_path)} (로드됨)"
            )
        except Exception as e:
            self.model_status_label.configure(text=f"모델 로드 실패: {str(e)}")

    def run_analysis(self):
        """비디오 분석 실행 (스레드)"""
        if not self.video_path:
            return

        self.analyze_btn.configure(state="disabled", text="분석 중...")
        self.progress_bar.set(0)

        # 분석을 별도 스레드에서 실행
        thread = threading.Thread(target=self._analyze_video_thread)
        thread.daemon = True
        thread.start()

    def _analyze_video_thread(self):
        """비디오 분석 (백그라운드 스레드)"""
        try:
            # 예측 실행
            self.current_predictions = self.predictor.predict_video(
                self.video_path,
                sequence_length=10
            )

            # UI 업데이트 (메인 스레드에서)
            self.after(0, self._update_analysis_results)

        except Exception as e:
            self.after(0, lambda: self._show_error(f"분석 중 오류: {str(e)}"))

        finally:
            self.after(0, lambda: self.analyze_btn.configure(
                state="normal",
                text="비디오 분석 시작"
            ))

    def _update_analysis_results(self):
        """분석 결과 UI 업데이트"""

        # 기존 결과 제거
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # 결과 표시
        for pred in self.current_predictions:
            result_frame = ctk.CTkFrame(self.results_frame)
            result_frame.pack(fill="x", padx=5, pady=5)

            # 색상 설정
            color = "red" if pred.defect_type == "결함" else "green"

            # 프레임 정보
            ctk.CTkLabel(
                result_frame,
                text=f"프레임 {pred.frame_number}",
                font=ctk.CTkFont(weight="bold")
            ).pack(anchor="w", padx=5, pady=2)

            # 결함 유형
            ctk.CTkLabel(
                result_frame,
                text=f"상태: {pred.defect_type}",
                text_color=color
            ).pack(anchor="w", padx=5, pady=2)

            # 확률
            ctk.CTkLabel(
                result_frame,
                text=f"결함 확률: {pred.defect_probability:.1%}"
            ).pack(anchor="w", padx=5, pady=2)

            # 신뢰도
            ctk.CTkLabel(
                result_frame,
                text=f"신뢰도: {pred.confidence:.1%}"
            ).pack(anchor="w", padx=5, pady=2)

        # 통계 업데이트
        self._update_statistics()

        # 버튼 활성화
        self.save_results_btn.configure(state="normal")
        self.export_video_btn.configure(state="normal")

    def _update_statistics(self):
        """통계 정보 업데이트"""
        if not self.current_predictions:
            return

        total = len(self.current_predictions)
        defects = sum(1 for p in self.current_predictions if p.defect_type == "결함")
        normal = total - defects

        avg_defect_prob = np.mean([p.defect_probability for p in self.current_predictions])
        avg_confidence = np.mean([p.confidence for p in self.current_predictions])

        stats_text = f"""통계:
총 분석 구간: {total}
정상: {normal} ({normal/total*100:.1f}%)
결함: {defects} ({defects/total*100:.1f}%)

평균 결함 확률: {avg_defect_prob:.1%}
평균 신뢰도: {avg_confidence:.1%}"""

        self.stats_label.configure(text=stats_text)

    def display_current_frame(self):
        """현재 프레임 표시"""
        if not self.video_capture:
            return

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.video_capture.read()

        if not ret:
            return

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 분석 결과가 있으면 오버레이
        if self.current_predictions:
            frame_rgb = self._overlay_prediction(frame_rgb, self.current_frame_idx)

        # 캔버스 크기에 맞게 리사이즈
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            h, w = frame_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        # PIL Image로 변환
        img = Image.fromarray(frame_rgb)
        self.current_photo = ImageTk.PhotoImage(img)

        # 캔버스에 표시
        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.current_photo,
            anchor="center"
        )

        # 프레임 정보 업데이트
        current_time = self.current_frame_idx / self.fps if self.fps > 0 else 0
        total_time = self.total_frames / self.fps if self.fps > 0 else 0

        self.frame_info_label.configure(
            text=f"프레임: {self.current_frame_idx} / {self.total_frames} | "
                 f"시간: {self._format_time(current_time)} / {self._format_time(total_time)}"
        )

    def _overlay_prediction(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """프레임에 예측 결과 오버레이"""
        # 해당 프레임의 예측 찾기
        pred = None
        for p in self.current_predictions:
            if abs(p.frame_number - frame_idx) < 5:  # 범위 내
                pred = p
                break

        if not pred:
            return frame

        # 텍스트 오버레이
        color = (255, 0, 0) if pred.defect_type == "결함" else (0, 255, 0)

        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {pred.defect_type}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Defect Prob: {pred.defect_probability:.1%}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {pred.confidence:.1%}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def toggle_playback(self):
        """재생/일시정지 토글"""
        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_pause_btn.configure(text="⏸ 일시정지")
            self._playback_loop()
        else:
            self.play_pause_btn.configure(text="▶ 재생")

    def _playback_loop(self):
        """재생 루프"""
        if not self.is_playing:
            return

        if self.current_frame_idx >= self.total_frames - 1:
            self.is_playing = False
            self.play_pause_btn.configure(text="▶ 재생")
            return

        self.current_frame_idx += 1
        self.display_current_frame()

        # 타임라인 업데이트
        progress = (self.current_frame_idx / self.total_frames) * 100
        self.timeline_slider.set(progress)

        # 다음 프레임 스케줄링
        delay = int(1000 / (self.fps * self.playback_speed))
        self.after(delay, self._playback_loop)

    def seek_video(self, value):
        """비디오 탐색"""
        if not self.video_capture:
            return

        self.current_frame_idx = int((float(value) / 100) * self.total_frames)
        self.display_current_frame()

    def update_playback_speed(self, value):
        """재생 속도 업데이트"""
        self.playback_speed = float(value)
        self.speed_label.configure(text=f"{self.playback_speed:.2f}x")

    def save_results(self):
        """분석 결과 JSON으로 저장"""
        if not self.current_predictions:
            return

        file_path = filedialog.asksaveasfilename(
            title="결과 저장",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # 결과를 딕셔너리로 변환
        results = {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "predictions": [
                {
                    "frame_number": p.frame_number,
                    "defect_probability": p.defect_probability,
                    "defect_type": p.defect_type,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp
                }
                for p in self.current_predictions
            ]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self._show_info(f"결과 저장 완료: {file_path}")

    def export_annotated_video(self):
        """분석 결과가 표시된 비디오 내보내기"""
        if not self.current_predictions or not self.video_path:
            return

        output_path = filedialog.asksaveasfilename(
            title="비디오 저장",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )

        if not output_path:
            return

        # 백그라운드 스레드에서 처리
        thread = threading.Thread(
            target=self._export_video_thread,
            args=(output_path,)
        )
        thread.daemon = True
        thread.start()

        self.export_video_btn.configure(state="disabled", text="내보내는 중...")

    def _export_video_thread(self, output_path: str):
        """비디오 내보내기 (백그라운드)"""
        try:
            visualize_predictions(
                self.video_path,
                self.current_predictions,
                output_path
            )
            self.after(0, lambda: self._show_info(f"비디오 저장 완료: {output_path}"))
        except Exception as e:
            self.after(0, lambda: self._show_error(f"비디오 내보내기 실패: {str(e)}"))
        finally:
            self.after(0, lambda: self.export_video_btn.configure(
                state="normal",
                text="분석 비디오 내보내기"
            ))

    def _show_info(self, message: str):
        """정보 메시지 표시"""
        dialog = ctk.CTkInputDialog(text=message, title="알림")

    def _show_error(self, message: str):
        """에러 메시지 표시"""
        dialog = ctk.CTkInputDialog(text=message, title="오류")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """시간 포맷팅 (MM:SS)"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def __del__(self):
        """소멸자"""
        if self.video_capture:
            self.video_capture.release()


def open_welding_analyzer():
    """독립 실행 함수"""
    app = ctk.CTk()
    app.withdraw()  # 메인 윈도우 숨기기
    analyzer = WeldingVideoAnalyzer(app)
    analyzer.mainloop()


if __name__ == "__main__":
    open_welding_analyzer()
