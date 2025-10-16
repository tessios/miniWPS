import customtkinter as ctk
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import json
import os
import re
import logging
from tkinter import filedialog
import tempfile
import subprocess
from typing import Optional, Dict, List, Tuple
from Levenshtein import distance

# --- 로깅 설정 ---
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 상수 정의 ---
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
KNOWLEDGE_BASE_FILE = "wps_knowledge_base.json"
OUTPUT_FOLDER = "WPS-OUTPUT"
OCR_DPI = 300
OCR_DEFAULT_PSM = '--psm 7'


class WorkbenchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("학습형 WPS 워크벤치 v4.1 (최적화)")
        self.geometry("1600x900")

        # --- 변수 초기화 ---
        self._init_variables()

        # --- GUI 레이아웃 ---
        self._setup_layout()
        self._setup_left_frame()
        self._setup_center_frame()
        self._setup_right_frame()
        self._setup_bindings()

        self.rebuild_data_entries()

    def _init_variables(self):
        """변수 초기화"""
        self.template_data: Optional[Dict] = None
        self.pdf_doc: Optional[fitz.Document] = None
        self.current_page: int = 0
        self.zoom_level: float = 1.0
        self.input_files: List[str] = []
        self.current_file_index: int = -1
        self.rect_start_pos: Optional[Tuple] = None
        self.current_rect_id: Optional[int] = None
        self.selected_field: Optional[str] = None
        self.data_entries: Dict = {}
        self.processes: List[str] = []
        self.knowledge_base: Dict = self.load_knowledge_base()
        self.file_list_buttons: List = []

        # 설정
        self.VALID_WELDING_PROCESSES = ['SMAW', 'GTAW', 'FCAW', 'GMAW', 'PAW', 'SAW']
        self.base_fields = ['Welding_Process_Type', 'WPS_No', 'Preheat_Temp_Min', 'Gas_Flow_Rate']
        self.dynamic_field_templates = {
            'AWS_Class': {'whitelist': 'ER-SH0123456789,TFG'},
            'Current': {'whitelist': '~-0123456789'},
            'Voltage': {'whitelist': '~-0123456789'},
            'Travel_Speed': {'whitelist': '~-0123456789'},
        }

    def _setup_layout(self):
        """메인 레이아웃 설정"""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=280)
        self.grid_columnconfigure(1, weight=5)
        self.grid_columnconfigure(2, weight=2, minsize=350)

    def _setup_left_frame(self):
        """왼쪽 프레임 설정"""
        self.left_frame = ctk.CTkFrame(self, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_rowconfigure(3, weight=1)

        # 파일 경로 입력
        self.file_frame = ctk.CTkFrame(self.left_frame)
        self.file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.file_frame.grid_columnconfigure(0, weight=1)

        self.pdf_path_entry = ctk.CTkEntry(self.file_frame, placeholder_text="PDF 폴더 경로")
        self.pdf_path_entry.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")

        ctk.CTkButton(self.file_frame, text="폴더...", width=60,
                      command=self.browse_for_pdf_folder).grid(row=0, column=1, pady=5)

        # 템플릿 버튼
        ctk.CTkButton(self.left_frame, text="템플릿 불러오기/저장",
                      command=self.manage_template).grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # 상태 레이블
        self.status_label = ctk.CTkLabel(self.left_frame, text="PDF 폴더를 불러오세요.",
                                         wraplength=260, justify="left")
        self.status_label.grid(row=2, column=0, padx=10, pady=10)

        # 파일 목록
        self.file_list_frame = ctk.CTkScrollableFrame(self.left_frame, label_text="PDF 파일 목록")
        self.file_list_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")

        # 페이지 네비게이션
        self.page_nav_frame = ctk.CTkFrame(self.left_frame)
        self.page_nav_frame.grid(row=4, column=0, padx=10, pady=10)

        ctk.CTkButton(self.page_nav_frame, text="< 이전 페이지",
                      command=self.prev_page).pack(side="left")
        self.page_label = ctk.CTkLabel(self.page_nav_frame, text="Page 0/0")
        self.page_label.pack(side="left", padx=10)
        ctk.CTkButton(self.page_nav_frame, text="다음 페이지 >",
                      command=self.next_page).pack(side="left")

        # 파일 네비게이션
        self.bottom_frame = ctk.CTkFrame(self.left_frame)
        self.bottom_frame.grid(row=5, column=0, padx=10, pady=10, sticky="s")

        ctk.CTkButton(self.bottom_frame, text="<< 이전 파일",
                      command=self.prev_pdf).pack(fill="x", pady=(0, 5))
        ctk.CTkButton(self.bottom_frame, text="다음 파일 >>",
                      command=self.next_pdf).pack(fill="x")

    def _setup_center_frame(self):
        """중앙 프레임 (캔버스) 설정"""
        self.center_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="gray20")
        self.center_frame.grid(row=0, column=1, sticky="nsew")
        self.center_frame.grid_rowconfigure(0, weight=1)
        self.center_frame.grid_columnconfigure(0, weight=1)

        # 캔버스
        self.canvas = ctk.CTkCanvas(self.center_frame, bg="gray20", highlightthickness=0)

        # 스크롤바
        self.v_scrollbar = ctk.CTkScrollbar(self.center_frame, orientation="vertical",
                                            command=self.canvas.yview)
        self.h_scrollbar = ctk.CTkScrollbar(self.center_frame, orientation="horizontal",
                                            command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=self.v_scrollbar.set,
                              xscrollcommand=self.h_scrollbar.set)

        self.v_scrollbar.grid(row=0, column=1, sticky='ns')
        self.h_scrollbar.grid(row=1, column=0, sticky='ew')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        # 파일명 레이블
        self.file_label = ctk.CTkLabel(self.canvas, text="PDF를 불러오세요",
                                       fg_color="black", text_color="white", corner_radius=5)

    def _setup_right_frame(self):
        """오른쪽 프레임 설정"""
        self.right_frame = ctk.CTkFrame(self, width=350)
        self.right_frame.grid(row=0, column=2, sticky="nsew")
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # 자동 추출 버튼
        ctk.CTkButton(self.right_frame, text="현재 파일 자동 추출",
                      command=self.run_extraction).grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # 데이터 입력 프레임
        self.data_entry_frame = ctk.CTkScrollableFrame(self.right_frame,
                                                       label_text="추출 데이터 (라벨 클릭 후 영역 지정)")
        self.data_entry_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.data_entry_frame.grid_columnconfigure(0, weight=1)

        # 저장 버튼
        ctk.CTkButton(self.right_frame, text="최종 결과 저장 및 학습",
                      command=self.save_results, fg_color="green").grid(row=2, column=0,
                                                                        padx=10, pady=10, sticky="ew")

    def _setup_bindings(self):
        """이벤트 바인딩 설정"""
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)

    # --- 파일 관리 ---
    def browse_for_pdf_folder(self):
        """PDF 폴더 선택"""
        folder_path = filedialog.askdirectory(title="WPS PDF 폴더 선택")
        if not folder_path:
            return

        self.pdf_path_entry.delete(0, "end")
        self.pdf_path_entry.insert(0, folder_path)
        self.load_pdf_folder()

    def load_pdf_folder(self):
        """폴더 내 PDF 파일 로드"""
        folder_path = self.pdf_path_entry.get()
        if not os.path.isdir(folder_path):
            self.status_label.configure(text="유효하지 않은 폴더 경로입니다.", text_color="red")
            return

        # PDF 파일 목록 생성
        self.input_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith('.pdf')
        ])

        if not self.input_files:
            self.status_label.configure(text="PDF 파일이 없습니다.", text_color="orange")
            return

        # 파일 목록 버튼 생성
        for btn in self.file_list_buttons:
            btn.destroy()
        self.file_list_buttons = []

        for i, f in enumerate(self.input_files):
            btn = ctk.CTkButton(
                self.file_list_frame,
                text=os.path.basename(f),
                fg_color="gray",
                command=lambda idx=i: self.jump_to_pdf(idx)
            )
            btn.pack(fill="x", padx=5, pady=2)
            self.file_list_buttons.append(btn)

        # 첫 번째 파일 로드
        if self.input_files:
            self.jump_to_pdf(0)

    def jump_to_pdf(self, index: int):
        """특정 PDF로 이동"""
        if not 0 <= index < len(self.input_files):
            return

        self.current_file_index = index

        # 버튼 색상 업데이트
        for i, btn in enumerate(self.file_list_buttons):
            btn.configure(fg_color="green" if i == index else "gray")

        self.load_pdf_document()

    def load_pdf_document(self):
        """현재 PDF 문서 로드"""
        if not 0 <= self.current_file_index < len(self.input_files):
            return

        # 이전 문서 정리
        if self.pdf_doc:
            self.pdf_doc.close()

        # 새 문서 열기
        pdf_path = self.input_files[self.current_file_index]
        try:
            self.pdf_doc = fitz.open(pdf_path)
            self.current_page = 0
            self.zoom_level = 1.0
            self.display_page(fit_to_screen=True)

            self.status_label.configure(
                text=f"로딩 완료:\n{os.path.basename(pdf_path)}",
                text_color="white"
            )

            # 프로세스 초기화 및 재구성
            self.processes = []
            self.rebuild_data_entries()

            # 템플릿이 있으면 자동 추출
            if self.template_data:
                self.after(100, self.run_extraction)

        except Exception as e:
            logging.error(f"PDF 로드 실패: {e}")
            self.status_label.configure(text=f"PDF 로드 실패: {e}", text_color="red")

    # --- 템플릿 관리 ---
    def manage_template(self):
        """템플릿 저장/로드 관리"""
        action_dialog = ctk.CTkInputDialog(
            text="템플릿 관리:\n'load' 또는 'save'를 입력하세요.",
            title="템플릿 관리"
        )
        action = action_dialog.get_input()

        if not action:
            return

        action = action.lower()

        if action == 'save':
            self._save_template()
        elif action == 'load':
            self._load_template()
        else:
            self.status_label.configure(text="'load' 또는 'save'를 입력하세요.", text_color="orange")

    def _save_template(self):
        """템플릿 저장"""
        if not self.template_data:
            self.status_label.configure(text="저장할 템플릿이 없습니다.", text_color="orange")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(self.template_data, f, indent=4, ensure_ascii=False)
                self.status_label.configure(text=f"템플릿 저장 완료", text_color="green")
            except Exception as e:
                logging.error(f"템플릿 저장 실패: {e}")
                self.status_label.configure(text=f"저장 실패: {e}", text_color="red")

    def _load_template(self):
        """템플릿 로드"""
        filepath = filedialog.askopenfilename(
            title="템플릿 JSON 파일 선택",
            filetypes=(("JSON files", "*.json"),)
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.template_data = json.load(f)

            self.status_label.configure(text=f"템플릿 로딩 성공", text_color="green")
            self.rebuild_data_entries()
            self.update()

            if self.pdf_doc:
                self.run_extraction()

        except Exception as e:
            logging.error(f"템플릿 로드 실패: {e}")
            self.status_label.configure(text=f"로드 실패: {e}", text_color="red")

    # --- 데이터 입력 필드 관리 ---
    def rebuild_data_entries(self):
        """데이터 입력 필드 재구성"""
        # 기존 위젯 제거
        for widget in self.data_entry_frame.winfo_children():
            widget.destroy()

        self.data_entries = {}

        # 표시할 필드 결정
        fields_to_show = self.base_fields.copy()

        if self.processes:
            for process in self.processes:
                for suffix_template in self.dynamic_field_templates:
                    fields_to_show.append(f"{suffix_template}_{process.strip()}")

        # 필드 생성
        for field in fields_to_show:
            frame = ctk.CTkFrame(self.data_entry_frame)
            frame.pack(fill="x", padx=5, pady=2, anchor="w")
            frame.grid_columnconfigure(1, weight=1)

            label = ctk.CTkLabel(frame, text=field, width=150, anchor="w")
            label.grid(row=0, column=0)

            entry = ctk.CTkEntry(frame)
            entry.grid(row=0, column=1, sticky="ew")

            self.data_entries[field] = entry

            # 라벨 클릭 이벤트
            label.bind("<Button-1>", lambda event, f=field: self.start_defining(f))

    def start_defining(self, field_name: str):
        """필드 영역 정의 시작"""
        self.selected_field = field_name
        self.status_label.configure(
            text=f"정의 중: {field_name}\n영역을 드래그하세요.",
            text_color="cyan"
        )

    # --- 프로세스 인식 ---
    def generate_dynamic_fields(self):
        """Welding Process Type 인식 및 동적 필드 생성"""
        if not self.template_data or not self.pdf_doc:
            return

        info = self.template_data.get('Welding_Process_Type')
        if not info or 'rect' not in info:
            return

        page = self.pdf_doc.load_page(info['page'])
        rect = info['rect']

        try:
            config = self.get_ocr_config('Welding_Process_Type')
            raw_text = self.ocr_from_area_direct(page, rect, config)

            # 프로세스 패턴 매칭
            process_pattern = "|".join(self.VALID_WELDING_PROCESSES)
            found_processes = re.findall(process_pattern, raw_text, re.IGNORECASE)

            # 자동 인식 실패 시 수동 입력
            if not found_processes and raw_text:
                dialog = ctk.CTkInputDialog(
                    text=f"자동 인식 실패!\n인식된 텍스트: '{raw_text}'\n\n"
                         f"올바른 프로세스를 입력 (예: GTAW+SMAW):",
                    title="수동 입력"
                )
                user_input = dialog.get_input()

                if user_input:
                    found_processes = re.findall(process_pattern, user_input, re.IGNORECASE)

            if found_processes:
                self.processes = [p.upper() for p in found_processes]
                self.status_label.configure(
                    text=f"프로세스 인식: {self.processes}",
                    text_color="cyan"
                )
                self.rebuild_data_entries()
            else:
                self.status_label.configure(
                    text="프로세스를 인식하지 못했습니다.",
                    text_color="orange"
                )
                self.processes = []
                self.rebuild_data_entries()

        except Exception as e:
            logging.error(f"프로세스 인식 오류: {e}")
            self.status_label.configure(
                text=f"프로세스 인식 중 에러: {e}",
                text_color="red"
            )

    # --- OCR 관련 ---
    def get_ocr_config(self, field_name: str) -> str:
        """필드별 OCR 설정 반환"""
        # 기본 설정
        if field_name == 'Welding_Process_Type':
            return '--psm 6'  # 단일 블록
        elif 'AWS_Class' in field_name or 'WPS_No' in field_name:
            return '--psm 7'  # 단일 라인
        elif any(x in field_name for x in ['Current', 'Voltage', 'Speed', 'Temp']):
            return '--psm 7 -c tessedit_char_whitelist=0123456789.~-'
        else:
            return OCR_DEFAULT_PSM

    def ocr_from_area_direct(self, page: fitz.Page, rect: Tuple, config: str = '') -> str:
        """영역에서 직접 OCR 수행"""
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                temp_image_path = os.path.join(tempdir, "temp.png")

                # 이미지 추출
                clip_rect = fitz.Rect(rect)
                pix = page.get_pixmap(clip=clip_rect, dpi=OCR_DPI)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.save(temp_image_path)

                # Tesseract 실행
                temp_output_base = os.path.join(tempdir, "output")
                command = [TESSERACT_CMD, temp_image_path, temp_output_base, "-l", "eng"]

                if config:
                    command.extend(config.split())

                result = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10
                )

                # 결과 읽기
                output_file = temp_output_base + ".txt"
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    return text
                else:
                    return ""

        except subprocess.TimeoutExpired:
            logging.error("OCR 타임아웃")
            return "OCR Timeout"
        except subprocess.CalledProcessError as e:
            logging.error(f"Tesseract 실행 오류: {e.stderr}")
            return "OCR Error"
        except Exception as e:
            logging.error(f"OCR 처리 오류: {e}")
            return "OCR Error"

    # --- 추출 실행 ---
    def run_extraction(self):
        """자동 추출 실행"""
        if not self.template_data or not self.pdf_doc:
            self.status_label.configure(
                text="템플릿 또는 PDF가 로드되지 않았습니다.",
                text_color="orange"
            )
            return

        self.status_label.configure(text="자동 추출 중...", text_color="cyan")

        # 프로세스 인식 먼저
        self.after(50, self.generate_dynamic_fields)
        self.update()

        # 나머지 필드 추출
        self.after(100, self._run_extraction_remaining)

    def _run_extraction_remaining(self):
        """나머지 필드 추출"""
        for field in self.data_entries.keys():
            if field != 'Welding_Process_Type':
                self.run_extraction_for_field(field)

        self.status_label.configure(
            text="자동 추출 완료! 값을 검토/수정하세요.",
            text_color="green"
        )

    def run_extraction_for_field(self, field: str):
        """특정 필드 추출"""
        entry = self.data_entries.get(field)
        info = self.template_data.get(field, {})

        if not entry or not info:
            return

        page_num = info.get('page')
        rect = info.get('rect')

        if page_num is None or not rect:
            return

        try:
            page = self.pdf_doc.load_page(page_num)
            config = self.get_ocr_config(field)
            value = self.ocr_from_area_direct(page, rect, config)

            # 지능형 보정
            corrected_value = self.intelligent_correction(field, value)

            # 엔트리에 입력
            entry.delete(0, "end")
            entry.insert(0, corrected_value)

        except Exception as e:
            logging.error(f"필드 추출 오류 ({field}): {e}")
            entry.delete(0, "end")
            entry.insert(0, "추출 실패")

    # --- 지식 베이스 및 학습 ---
    def load_knowledge_base(self) -> Dict:
        """지식 베이스 로드"""
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            try:
                with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"지식 베이스 로드 실패: {e}")
        return {}

    def save_knowledge_base(self):
        """지식 베이스 저장"""
        try:
            with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.knowledge_base, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"지식 베이스 저장 실패: {e}")

    def intelligent_correction(self, field: str, ocr_text: str) -> str:
        """지식 베이스 기반 지능형 보정"""
        if not ocr_text or ocr_text in ["OCR Error", "OCR Timeout"]:
            return ocr_text

        base_field = field.split('_')[0]
        known_values = set(
            self.knowledge_base.get(field, []) +
            self.knowledge_base.get(base_field, [])
        )

        if not known_values:
            return ocr_text

        try:
            # Levenshtein 거리 기반 매칭
            closest_match = min(known_values, key=lambda x: distance(str(ocr_text), str(x)))

            if distance(str(ocr_text), str(closest_match)) <= 2 and len(ocr_text) > 2:
                return closest_match
        except (TypeError, ValueError) as e:
            logging.error(f"보정 오류: {e}")

        return ocr_text

    def learn_value(self, field: str, value: str):
        """값 학습"""
        if not value or value in ["OCR Error", "OCR Timeout", "추출 실패"]:
            return

        base_field = field.split('_')[0]

        for f in [field, base_field]:
            if f not in self.knowledge_base:
                self.knowledge_base[f] = []

            if value not in self.knowledge_base[f]:
                self.knowledge_base[f].append(value)

    # --- 결과 저장 ---
    def save_results(self):
        """결과 저장 및 학습"""
        if self.current_file_index < 0:
            self.status_label.configure(text="저장할 파일이 없습니다.", text_color="orange")
            return

        # 데이터 수집
        output_data = {field: entry.get() for field, entry in self.data_entries.items()}

        # 학습
        for field, value in output_data.items():
            self.learn_value(field, value)

        self.save_knowledge_base()

        # 파일 저장
        current_pdf_path = self.input_files[self.current_file_index]
        output_filename = os.path.splitext(os.path.basename(current_pdf_path))[0] + '_result.json'

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

            self.status_label.configure(
                text=f"'{output_filename}' 저장 및 학습 완료!",
                text_color="green"
            )

        except Exception as e:
            logging.error(f"결과 저장 실패: {e}")
            self.status_label.configure(text=f"저장 실패: {e}", text_color="red")

    # --- PDF 네비게이션 ---
    def next_pdf(self):
        """다음 PDF 파일로 이동"""
        if self.current_file_index < len(self.input_files) - 1:
            self.jump_to_pdf(self.current_file_index + 1)

    def prev_pdf(self):
        """이전 PDF 파일로 이동"""
        if self.current_file_index > 0:
            self.jump_to_pdf(self.current_file_index - 1)

    def prev_page(self):
        """이전 페이지로 이동"""
        if self.pdf_doc and self.current_page > 0:
            self.current_page -= 1
            self.display_page()

    def next_page(self):
        """다음 페이지로 이동"""
        if self.pdf_doc and self.current_page < len(self.pdf_doc) - 1:
            self.current_page += 1
            self.display_page()

    # --- 캔버스 표시 ---
    def display_page(self, fit_to_screen: bool = False):
        """PDF 페이지 표시"""
        if not self.pdf_doc:
            return

        page = self.pdf_doc.load_page(self.current_page)

        # 화면에 맞추기
        if fit_to_screen:
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            if canvas_w < 2:
                canvas_w = self.winfo_width() - 650
            if canvas_h < 2:
                canvas_h = self.winfo_height() - 50

            img_w, img_h = page.rect.width, page.rect.height
            if img_w > 0 and img_h > 0:
                self.zoom_level = min(canvas_w / img_w, canvas_h / img_h)

        # 페이지 렌더링
        mat = fitz.Matrix(self.zoom_level, self.zoom_level)
        pix = page.get_pixmap(matrix=mat)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        tk_image = ImageTk.PhotoImage(img)

        # 캔버스에 표시
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
        self.canvas.image = tk_image

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.update_page_label()

        # 파일명 표시
        if self.input_files and self.current_file_index != -1:
            self.file_label.configure(text=os.path.basename(self.input_files[self.current_file_index]))
            self.file_label.place(x=10, y=10)
        else:
            self.file_label.place_forget()

    def update_page_label(self):
        """페이지 레이블 업데이트"""
        pdf_total = len(self.input_files)
        pdf_current = self.current_file_index + 1 if pdf_total > 0 else 0

        page_total = len(self.pdf_doc) if self.pdf_doc else 0
        page_current = self.current_page + 1 if page_total > 0 else 0

        self.page_label.configure(
            text=f"PDF {pdf_current}/{pdf_total} | Page {page_current}/{page_total}"
        )

    # --- 마우스 이벤트 ---
    def on_zoom(self, event):
        """마우스 휠 줌"""
        if not self.pdf_doc:
            return

        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_level *= factor
        self.display_page()

    def on_press(self, event):
        """마우스 누름"""
        if not self.selected_field:
            self.status_label.configure(
                text="먼저 오른쪽 목록의 라벨을 클릭하세요.",
                text_color="orange"
            )
            return

        self.canvas.focus_set()
        self.rect_start_pos = (
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y)
        )

        # 이전 사각형 제거
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)

        # 새 사각형 생성
        self.current_rect_id = self.canvas.create_rectangle(
            self.rect_start_pos[0], self.rect_start_pos[1],
            self.rect_start_pos[0], self.rect_start_pos[1],
            outline="red", width=2
        )

    def on_drag(self, event):
        """마우스 드래그"""
        if not self.rect_start_pos:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        self.canvas.coords(
            self.current_rect_id,
            self.rect_start_pos[0], self.rect_start_pos[1],
            x, y
        )

    def on_release(self, event):
        """마우스 릴리즈"""
        if not self.selected_field or not self.rect_start_pos:
            return

        end_pos = (
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y)
        )

        # 이미지 좌표로 변환
        x0_img = min(self.rect_start_pos[0], end_pos[0]) / self.zoom_level
        y0_img = min(self.rect_start_pos[1], end_pos[1]) / self.zoom_level
        x1_img = max(self.rect_start_pos[0], end_pos[0]) / self.zoom_level
        y1_img = max(self.rect_start_pos[1], end_pos[1]) / self.zoom_level

        # 템플릿에 저장
        if not self.template_data:
            self.template_data = {}

        self.template_data[self.selected_field] = {
            'page': self.current_page,
            'rect': (int(x0_img), int(y0_img), int(x1_img), int(y1_img))
        }

        self.status_label.configure(
            text=f"'{self.selected_field}' 좌표 저장! 즉시 추출 실행.",
            text_color="green"
        )

        # 필드 추출
        field_to_extract = self.selected_field
        self.selected_field = None

        self.after(50, self.run_extraction_for_field, field_to_extract)

        # Welding_Process_Type이면 동적 필드 생성
        if field_to_extract == 'Welding_Process_Type':
            self.after(100, self.generate_dynamic_fields)

    # --- 종료 시 정리 ---
    def destroy(self):
        """앱 종료 시 리소스 정리"""
        if self.pdf_doc:
            self.pdf_doc.close()
        super().destroy()


if __name__ == "__main__":
    app = WorkbenchApp()
    app.mainloop()
