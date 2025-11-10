# -*- coding: utf-8 -*-
"""
WPS 워크벤치 v6.7 - 공란 행 처리 추가 (병합셀 지원)
- ★ v6.5: 깨진 유니코드 복구 (â¤ → ≤ → <=)
- ★ v6.5: AWS_Class 스프레드시트 보정 우선 적용
- ★ v6.5: 사용자 수기 입력값 보존 개선
- ★ v6.5: Preheat 온도 형식 개선 (깨진 문자 복구)
- ★ v6.5: OCR whitelist ASCII 전용 (≤ → <= 변환)
- ★ v6.6: 프로세스별 다중 행 처리 (1 WPS = 2 rows)
- ★★★ v6.7: WPS_No 공란 행 자동 포함 (병합셀 지원) ★★★
"""

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

# OpenCV 선택적 import
try:
    import cv2
    import numpy as np

    OPENCV_AVAILABLE = True
    logging.info("✓ OpenCV 사용 가능 - 전처리 기능 활성화")
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("⚠️ OpenCV 미설치 - 전처리 기능 비활성화")
    logging.warning("   설치 명령: pip install opencv-python numpy")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'  # ★ UTF-8 로깅 명시
)

# 상수 정의
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
KNOWLEDGE_BASE_FILE = "wps_knowledge_base.json"
OUTPUT_FOLDER = "WPS-OUTPUT"
WORKSPACE_STATE_FILE = "workspace_state.json"
PREPROCESSING_PROFILE_FILE = "preprocessing_profile.json"
TEMPLATE_USAGE_FILE = "template_usage_history.json"
OCR_DPI = 300
OCR_DEFAULT_PSM = '--psm 7'
FILES_PER_PAGE = 50


class ImagePreprocessor:
    """이미지 전처리 클래스"""

    def __init__(self):
        self.debug_mode = False
        self.available = OPENCV_AVAILABLE

    def auto_preprocess(self, pil_image, field_type='default'):
        """필드 타입에 맞는 최적 전처리 자동 선택"""
        if not self.available:
            return pil_image

        try:
            img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            processed = self.pipeline(img, field_type)
            return Image.fromarray(processed)
        except Exception as e:
            logging.warning(f"전처리 실패, 원본 사용: {e}")
            return pil_image

    def pipeline(self, img, field_type):
        """전처리 파이프라인"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        denoised = cv2.fastNlMeansDenoising(gray, h=7)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        if field_type == 'wps_no':
            sharpened = self.sharpen_light(enhanced)
            result = cv2.adaptiveThreshold(
                sharpened, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif field_type == 'number':
            _, result = cv2.threshold(
                enhanced, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif field_type == 'mixed':
            result = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 15, 10
            )
        else:
            result = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

        result = self.deskew(result)
        result = cv2.copyMakeBorder(
            result, 5, 5, 5, 5,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        return result

    def sharpen_light(self, img):
        """가벼운 샤프닝"""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    def deskew(self, img):
        """기울기 자동 보정"""
        try:
            coords = np.column_stack(np.where(img > 0))
            if len(coords) == 0:
                return img

            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            if abs(angle) > 1.0:
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(
                    img, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                if abs(angle) > 2.0:
                    logging.info(f"  🔄 기울기 보정: {angle:.1f}도")
        except Exception as e:
            logging.debug(f"기울기 보정 실패: {e}")

        return img


class AdaptivePreprocessor:
    """적응형 전처리"""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.success_history = {}
        self.method_cache = {}
        self.available = OPENCV_AVAILABLE
        self.load_profile()

    def load_profile(self):
        """저장된 프로파일 로드"""
        if os.path.exists(PREPROCESSING_PROFILE_FILE):
            try:
                with open(PREPROCESSING_PROFILE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.success_history = data.get('success_history', {})
                    self.method_cache = data.get('method_cache', {})
                    if self.available:
                        logging.info("✓ 전처리 프로파일 로드 완료")
            except Exception as e:
                logging.error(f"프로파일 로드 실패: {e}")

    def save_profile(self):
        """프로파일 저장"""
        try:
            data = {
                'success_history': self.success_history,
                'method_cache': self.method_cache
            }
            with open(PREPROCESSING_PROFILE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"프로파일 저장 실패: {e}")

    def preprocess_adaptive(self, pil_image, field_type='default'):
        """적응형 전처리"""
        if not self.available:
            return pil_image

        if field_type in self.method_cache:
            best_method = self.method_cache[field_type]
            return self.apply_single_method(pil_image, best_method, field_type)

        return self.find_best_method(pil_image, field_type)

    def apply_single_method(self, pil_image, method, field_type):
        """단일 전처리 방법 적용"""
        if not self.available:
            return pil_image

        try:
            img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            denoised = cv2.fastNlMeansDenoising(gray, h=7)

            if method == 'adaptive_gaussian':
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
                result = cv2.adaptiveThreshold(
                    enhanced, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            elif method == 'otsu':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
                _, result = cv2.threshold(
                    enhanced, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            elif method == 'sharpen':
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(denoised, -1, kernel)
                result = cv2.adaptiveThreshold(
                    sharpened, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            else:
                result = self.preprocessor.pipeline(img, field_type)

            result = self.preprocessor.deskew(result)
            result = cv2.copyMakeBorder(
                result, 5, 5, 5, 5,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )

            return Image.fromarray(result)

        except Exception as e:
            logging.warning(f"전처리 실패: {e}")
            return pil_image

    def find_best_method(self, pil_image, field_type):
        """최적 전처리 방법 선택"""
        default_result = self.preprocessor.auto_preprocess(pil_image, field_type)

        if field_type == 'wps_no':
            self.method_cache[field_type] = 'sharpen'
        elif field_type == 'number':
            self.method_cache[field_type] = 'otsu'
        else:
            self.method_cache[field_type] = 'adaptive_gaussian'

        self.save_profile()
        return default_result

    def learn_success(self, field_type, ocr_result, actual_result=None):
        """OCR 성공 여부 학습"""
        if field_type not in self.success_history:
            self.success_history[field_type] = {'success': 0, 'total': 0}

        self.success_history[field_type]['total'] += 1

        if actual_result is None or ocr_result == actual_result:
            self.success_history[field_type]['success'] += 1

        if self.success_history[field_type]['total'] % 10 == 0:
            self.save_profile()
            success_rate = (
                    self.success_history[field_type]['success'] /
                    self.success_history[field_type]['total'] * 100
            )
            logging.info(f"📊 {field_type} 성공률: {success_rate:.1f}%")


class TemplateRecommender:
    """템플릿 자동 추천 시스템"""

    def __init__(self):
        self.usage_history = self.load_usage_history()

    def load_usage_history(self):
        """템플릿 사용 이력 로드"""
        if os.path.exists(TEMPLATE_USAGE_FILE):
            try:
                with open(TEMPLATE_USAGE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"템플릿 이력 로드 실패: {e}")
        return {}

    def save_usage_history(self):
        """템플릿 사용 이력 저장"""
        try:
            with open(TEMPLATE_USAGE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.usage_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"템플릿 이력 저장 실패: {e}")

    def record_usage(self, pdf_path, template_name):
        """템플릿 사용 기록"""
        pdf_basename = os.path.basename(pdf_path)
        patterns = self.extract_patterns(pdf_basename)

        if pdf_basename not in self.usage_history:
            self.usage_history[pdf_basename] = {
                'template': template_name,
                'patterns': patterns,
                'usage_count': 1
            }
        else:
            self.usage_history[pdf_basename]['template'] = template_name
            self.usage_history[pdf_basename]['usage_count'] += 1

        self.save_usage_history()
        logging.info(f"✓ 템플릿 사용 기록: {pdf_basename} → {template_name}")

    def extract_patterns(self, filename):
        """파일명에서 패턴 추출"""
        patterns = {
            'wps_number': None,
            'company': None,
            'revision': None,
            'keywords': []
        }

        wps_match = re.search(r'(P-WPS-[\w\-\.]+)', filename, re.IGNORECASE)
        if wps_match:
            patterns['wps_number'] = wps_match.group(1)

        rev_match = re.search(r'Rev\.?(\d+)', filename, re.IGNORECASE)
        if rev_match:
            patterns['revision'] = int(rev_match.group(1))

        keywords = re.findall(r'\b[A-Z]{2,}\b', filename)
        patterns['keywords'] = list(set(keywords))

        return patterns

    def calculate_similarity(self, pdf_path, template_name):
        """PDF와 템플릿 간 유사도 계산"""
        pdf_basename = os.path.basename(pdf_path)
        score = 0

        if pdf_basename in self.usage_history:
            if self.usage_history[pdf_basename]['template'] == template_name:
                score += 50

        current_patterns = self.extract_patterns(pdf_basename)

        for history_file, history_data in self.usage_history.items():
            if history_data['template'] != template_name:
                continue

            history_patterns = history_data.get('patterns', {})

            if current_patterns['wps_number'] and history_patterns.get('wps_number'):
                current_base = current_patterns['wps_number'].split('_')[0]
                history_base = history_patterns['wps_number'].split('_')[0]
                if current_base == history_base:
                    score += 30
                    break

            current_kw = set(current_patterns.get('keywords', []))
            history_kw = set(history_patterns.get('keywords', []))
            if current_kw and history_kw:
                overlap = len(current_kw & history_kw)
                if overlap > 0:
                    score += min(20, overlap * 10)
                    break

        template_usage_count = sum(
            1 for data in self.usage_history.values()
            if data['template'] == template_name
        )
        if template_usage_count > 0:
            score += min(20, template_usage_count * 2)

        return min(100, score)

    def recommend_template(self, pdf_path, available_templates):
        """PDF에 가장 적합한 템플릿 추천"""
        if not available_templates:
            return None, 0, False

        scores = {}
        for template_file in available_templates:
            template_name = template_file.replace('template_', '').replace('.json', '')
            score = self.calculate_similarity(pdf_path, template_name)
            scores[template_name] = score

        best_template = max(scores, key=scores.get)
        best_score = scores[best_template]
        auto_load = best_score >= 80

        logging.info(f"🎯 템플릿 추천: {best_template} (신뢰도: {best_score}점)")

        return best_template, best_score, auto_load


class SpreadsheetValidator:
    """
    구글 스프레드시트 기반 검증 시스템
    ★★★ v6.7: 공란 행 처리 추가 (병합셀 지원) ★★★
    ★★★ v6.6: 다중 행 처리 (1 WPS = 2 rows) + 유니코드 복구 ★★★
    """

    def __init__(self, spreadsheet_url):
        self.spreadsheet_url = spreadsheet_url
        self.df = None

        # 스프레드시트 컬럼 매핑
        self.column_map = {
            'WPS_No': 1,  # B열
            'REV': 2,  # C열
            'Metal_1': 3,  # D열
            'Metal_2': 5,  # F열
            'Support_PQR': 7,  # H열
            'Process_1': 8,  # I열 - IG PRO
            'Process_2': 9,  # J열 - JG PRO
            'Min_THK': 10,  # K열
            'Max_THK': 11,  # L열
            'Qualified_WM': 12,  # M열
            'Position': 13,  # N열
            'F_No': 14,  # O열 ★ 프로세스별
            'A_No': 15,  # P열 ★ 프로세스별
            'SFA_No': 16,  # Q열 ★ 프로세스별
            'Size': 17,  # R열 ★ 프로세스별
            'AWS_Class': 18,  # S열 ★ 프로세스별
            'Preheat_Temp_Min': 19,  # T열 ★ 프로세스별
            'PWHT_Temp': 20,  # U열
            'Shield_Gas': 21,  # V열
            'Impact_Test': 22,  # W열
            'Commodity': 23,  # X열
            'Remark': 24,  # Y열
        }

        self.load_spreadsheet()

    def fix_broken_unicode(self, text):
        """
        ★★★ v6.5 신규: 깨진 유니코드 복구 ★★★
        â¤ → ≤ → <=
        â¥ → ≥ → >=
        â  → ≠ → !=
        """
        if not text:
            return text

        original = text

        # 깨진 UTF-8 바이트 시퀀스 복구
        text = text.replace('â¤', '≤')
        text = text.replace('â¥', '≥')
        text = text.replace('â ', '≠')
        text = text.replace('Â≤', '≤')
        text = text.replace('Â≥', '≥')

        # 유니코드를 ASCII로 변환
        text = self.normalize_unicode_symbols(text)

        if text != original:
            logging.info(f"  🔧 깨진 유니코드 복구: '{original}' → '{text}'")

        return text

    def normalize_unicode_symbols(self, text):
        """
        ★★★ v6.4 신규: 유니코드 기호 정규화 ★★★
        ≤ → <=
        ≥ → >=
        ≠ → !=
        """
        if not text:
            return text

        original = text

        # 유니코드 기호를 ASCII로 변환
        text = text.replace('≤', '<=')
        text = text.replace('≥', '>=')
        text = text.replace('≠', '!=')

        if text != original:
            logging.info(f"  🔄 유니코드 정규화: '{original}' → '{text}'")

        return text

    def load_spreadsheet(self):
        """스프레드시트 로드"""
        try:
            import pandas as pd
            import warnings
            warnings.filterwarnings('ignore', message='Unverified HTTPS request')

            try:
                import requests
                response = requests.get(self.spreadsheet_url, verify=False, timeout=10)
                response.raise_for_status()
                from io import StringIO
                self.df = pd.read_csv(StringIO(response.text), header=2, encoding='utf-8')
                logging.info(f"✓ 스프레드시트 로드 (requests): {len(self.df)} 행")

                if len(self.df) > 0:
                    sample_row = self.df.iloc[0]
                    logging.info(f"  📋 첫 행 샘플: WPS_No={sample_row.iloc[1]}")

                return
            except ImportError:
                logging.debug("requests 미설치, urllib 사용")
            except Exception as e:
                logging.debug(f"requests 실패: {e}")

            import ssl
            import urllib.request

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(self.spreadsheet_url)
            with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
                self.df = pd.read_csv(response, header=2, encoding='utf-8')

            logging.info(f"✓ 스프레드시트 로드 (urllib): {len(self.df)} 행")

        except ImportError:
            logging.warning("⚠️ pandas 미설치 - 스프레드시트 검증 비활성화")
            self.df = None
        except Exception as e:
            logging.warning(f"⚠️ 스프레드시트 로드 실패: {e}")
            self.df = None

    def extract_wps_no_from_filename(self, filename):
        """파일명에서 WPS No 추출"""
        match = re.search(r'(P-WPS-[\w\-\.]+?)(?:_Rev|\.pdf)', filename, re.IGNORECASE)
        if match:
            wps_no = match.group(1)
            logging.info(f"📁 파일명에서 WPS No 추출: '{filename}' → '{wps_no}'")
            return wps_no
        return None

    def find_all_rows_by_wps_no(self, wps_no):
        """
        ★★★ v6.6 신규: WPS No로 모든 매칭 행 찾기 ★★★
        같은 WPS_No를 가진 행이 여러 개일 수 있음 (프로세스별)

        ★★★ v6.7: 공란 행 처리 추가 ★★★
        - WPS_No가 있는 첫 행을 찾음
        - 바로 다음 행이 WPS_No 공란이면서 프로세스 정보가 있으면 포함
        """
        if self.df is None:
            return None

        try:
            import pandas as pd

            # 1단계: WPS_No가 일치하는 행 찾기
            mask = self.df.iloc[:, self.column_map['WPS_No']] == wps_no

            if not mask.any():
                return None

            # 2단계: 일치하는 행의 인덱스 가져오기
            matched_indices = self.df.index[mask].tolist()

            # 3단계: 각 일치 행의 바로 다음 행 확인
            all_indices = set(matched_indices)

            for idx in matched_indices:
                next_idx = idx + 1

                # 다음 행이 존재하는지 확인
                if next_idx >= len(self.df):
                    continue

                try:
                    next_row = self.df.iloc[next_idx]
                    next_wps_no = next_row.iloc[self.column_map['WPS_No']]

                    # 다음 행의 WPS_No가 비어있는지 확인
                    is_empty = pd.isna(next_wps_no) or str(next_wps_no).strip() in ['', 'nan', '-']

                    if is_empty:
                        # 프로세스 정보가 있는지 확인
                        process1 = next_row.iloc[self.column_map['Process_1']]
                        process2 = next_row.iloc[self.column_map['Process_2']]

                        has_process = False
                        if not pd.isna(process1) and str(process1).strip() not in ['', 'nan', '-']:
                            has_process = True
                        if not pd.isna(process2) and str(process2).strip() not in ['', 'nan', '-']:
                            has_process = True

                        # AWS_Class가 있는지도 확인
                        aws_class = next_row.iloc[self.column_map['AWS_Class']]
                        has_aws = not pd.isna(aws_class) and str(aws_class).strip() not in ['', 'nan', '-']

                        # 프로세스나 AWS_Class가 있으면 같은 그룹으로 포함
                        if has_process or has_aws:
                            all_indices.add(next_idx)
                            logging.info(f"  📌 공란 행 포함: 행 {next_idx} (다음 프로세스 데이터)")

                except Exception as e:
                    logging.debug(f"다음 행 확인 실패: {e}")
                    continue

            # 4단계: 모든 인덱스로 DataFrame 생성
            all_indices_sorted = sorted(list(all_indices))
            rows = self.df.iloc[all_indices_sorted]

            logging.info(f"✓ 스프레드시트에서 {wps_no} 발견 ({len(rows)}행, 인덱스: {all_indices_sorted})")
            return rows

        except Exception as e:
            logging.debug(f"행 찾기 실패: {e}")
            import traceback
            logging.debug(traceback.format_exc())

        return None

    def find_row_by_wps_no(self, wps_no):
        """
        ★ v6.6: 하위 호환성 유지 (첫 번째 행만 반환)
        """
        rows = self.find_all_rows_by_wps_no(wps_no)
        if rows is not None and len(rows) > 0:
            return rows.iloc[0]
        return None

    def get_process_row_index(self, rows, process_name):
        """
        ★★★ v6.6 신규: 프로세스명으로 행 인덱스 찾기 ★★★

        Args:
            rows: DataFrame (같은 WPS_No의 모든 행)
            process_name: 'GTAW', 'SMAW' 등

        Returns:
            행 인덱스 (0, 1, ...) 또는 None
        """
        import pandas as pd

        for idx in range(len(rows)):
            row = rows.iloc[idx]

            # Process_1, Process_2 컬럼 확인
            process1 = row.iloc[self.column_map['Process_1']]
            process2 = row.iloc[self.column_map['Process_2']]

            # 값 정리
            if pd.isna(process1):
                process1_str = ''
            else:
                process1_str = str(process1).strip().upper()

            if pd.isna(process2):
                process2_str = ''
            else:
                process2_str = str(process2).strip().upper()

            # 프로세스명 매칭
            if process_name.upper() in process1_str or process_name.upper() in process2_str:
                logging.info(f"  🎯 {process_name} → 행 {idx} (Process_1={process1_str}, Process_2={process2_str})")
                return idx

        # 매칭 실패 시 기본값
        logging.warning(f"  ⚠️ {process_name} 매칭 실패, 기본 행 인덱스 반환")

        # GTAW/FCAW/GMAW/PAW → 첫 번째 행
        if process_name.upper() in ['GTAW', 'FCAW', 'GMAW', 'PAW', 'SAW']:
            return 0
        # SMAW → 두 번째 행 (있으면)
        elif process_name.upper() == 'SMAW':
            return 1 if len(rows) > 1 else 0
        else:
            return 0

    def get_spreadsheet_value(self, wps_no, field_name):
        """
        스프레드시트에서 값 가져오기
        ★★★ v6.6: 프로세스별 다중 행 처리 ★★★
        """
        rows = self.find_all_rows_by_wps_no(wps_no)
        if rows is None or len(rows) == 0:
            return None

        try:
            import pandas as pd

            # ★★★ Welding_Process_Type 처리 (단일 행 기준) ★★★
            if field_name == 'Welding_Process_Type':
                row = rows.iloc[0]  # 첫 번째 행
                process1 = row.iloc[self.column_map['Process_1']]
                process2 = row.iloc[self.column_map['Process_2']]

                if pd.isna(process1) or str(process1).strip() in ['-', '', 'nan']:
                    process1_str = ''
                else:
                    process1_str = str(process1).strip()

                if pd.isna(process2) or str(process2).strip() in ['-', '', 'nan']:
                    process2_str = ''
                else:
                    process2_str = str(process2).strip()

                if process1_str and process2_str:
                    return f"{process1_str} + {process2_str}"
                elif process1_str:
                    return process1_str
                elif process2_str:
                    return process2_str
                else:
                    return None

            # ★★★ v6.6: 프로세스별 필드 처리 ★★★
            # F_No_GTAW, AWS_Class_SMAW 등
            process_suffix = None
            base_field = field_name

            for process_name in ['GTAW', 'SMAW', 'FCAW', 'GMAW', 'SAW', 'PAW']:
                if field_name.endswith(f'_{process_name}'):
                    process_suffix = process_name
                    base_field = field_name.replace(f'_{process_name}', '')
                    break

            # 프로세스별 필드인 경우
            if process_suffix:
                # 해당 프로세스의 행 인덱스 찾기
                row_idx = self.get_process_row_index(rows, process_suffix)

                if row_idx is None or row_idx >= len(rows):
                    logging.debug(f"  ⚠️ {field_name}: 행 인덱스 없음")
                    return None

                row = rows.iloc[row_idx]
                logging.info(f"  📊 {field_name}: 행 {row_idx} 사용")

                # ★ AWS_Class는 특별 처리 (줄바꿈 파싱)
                if base_field == 'AWS_Class':
                    value = row.iloc[self.column_map['AWS_Class']]
                    if pd.isna(value) or str(value).strip() in ['-', '', 'nan']:
                        return None

                    value_str = str(value).strip()

                    # 줄바꿈으로 분리
                    lines = []
                    if '\n' in value_str:
                        lines = [line.strip() for line in value_str.split('\n') if
                                 line.strip() and line.strip() not in ['-', '']]
                    elif '|' in value_str:
                        lines = [line.strip() for line in value_str.split('|') if
                                 line.strip() and line.strip() not in ['-', '']]
                    elif ',' in value_str:
                        lines = [line.strip() for line in value_str.split(',') if
                                 line.strip() and line.strip() not in ['-', '']]
                    else:
                        if value_str not in ['-', '']:
                            lines = [value_str]

                    # ★ 해당 행의 값 반환 (row_idx가 0이면 첫 줄, 1이면 둘째 줄)
                    if row_idx < len(lines):
                        result = lines[row_idx]
                    else:
                        result = lines[0] if lines else None

                    logging.info(f"  ✓ AWS_Class (행{row_idx}): '{result}'")
                    return result

                # ★ Preheat 처리 (유니코드 복구)
                elif base_field == 'Preheat_Temp_Min' or 'Preheat' in base_field:
                    value = row.iloc[self.column_map['Preheat_Temp_Min']]
                    if pd.isna(value) or str(value).strip() in ['-', '', 'nan']:
                        return None

                    raw_value = str(value).strip()
                    fixed = self.fix_broken_unicode(raw_value)
                    logging.info(f"  ✓ Preheat (행{row_idx}): '{fixed}'")
                    return fixed

                # ★ 기타 프로세스별 필드 (F_No, A_No, SFA_No, Size 등)
                elif base_field in self.column_map:
                    col_idx = self.column_map[base_field]
                    value = row.iloc[col_idx]

                    if pd.isna(value) or str(value).strip() in ['-', '', 'nan']:
                        return None

                    result = str(value).strip()
                    logging.info(f"  ✓ {base_field} (행{row_idx}): '{result}'")
                    return result

                # 컬럼 매핑에 없는 필드
                else:
                    logging.debug(f"  ⚠️ {base_field}: 컬럼 매핑 없음")
                    return None

            # ★★★ 프로세스 구분 없는 필드 (단일 값) ★★★
            else:
                row = rows.iloc[0]  # 첫 번째 행 사용

                # 기타 필드 처리
                field_to_column = {
                    'F_No': 'F_No',
                    'A_No': 'A_No',
                    'Size': 'Size',
                    'SFA_No': 'SFA_No',
                    'Shield_Gas': 'Shield_Gas',
                    'Position': 'Position',
                    'PWHT_Temp': 'PWHT_Temp',
                    'Impact_Test': 'Impact_Test',
                    'Preheat_Temp_Min': 'Preheat_Temp_Min',
                }

                if field_name in field_to_column:
                    column_key = field_to_column[field_name]
                    if column_key in self.column_map:
                        value = row.iloc[self.column_map[column_key]]
                        if pd.isna(value) or str(value).strip() in ['-', '', 'nan']:
                            return None

                        result = str(value).strip()

                        # Preheat는 유니코드 복구
                        if 'Preheat' in field_name:
                            result = self.fix_broken_unicode(result)

                        return result

            return None

        except Exception as e:
            logging.debug(f"값 가져오기 실패 ({field_name}): {e}")
            import traceback
            logging.debug(traceback.format_exc())
            return None

    def validate_and_correct(self, filename, ocr_data, preserve_manual_edits=False):
        """
        OCR 데이터를 스프레드시트로 검증 및 보정
        ★★★ v6.6: 다중 행 처리 + 스프레드시트 값 우선 적용 ★★★

        Args:
            preserve_manual_edits: True면 이미 값이 있는 필드는 보정 안 함
        """
        if self.df is None:
            wps_no = self.extract_wps_no_from_filename(filename)
            if wps_no and 'WPS_No' in ocr_data:
                ocr_data['WPS_No'] = wps_no
            return ocr_data

        wps_no = self.extract_wps_no_from_filename(filename)
        if not wps_no:
            wps_no = ocr_data.get('WPS_No', '')

        corrected_data = {}
        corrections_made = []

        for field, ocr_value in ocr_data.items():
            if field == 'WPS_No':
                corrected_data[field] = wps_no
                if wps_no != ocr_value:
                    corrections_made.append(f"{field}: '{ocr_value}' → '{wps_no}' (📁파일명)")
            else:
                spreadsheet_value = self.get_spreadsheet_value(wps_no, field)

                # ★★★ v6.6: 스프레드시트 값이 있으면 우선 사용 ★★★
                if spreadsheet_value is not None:
                    # 수기 입력 보존 모드이고, 이미 값이 있으면 건너뛰기
                    if preserve_manual_edits and ocr_value and ocr_value not in ["", "None", "OCR Error", "OCR Timeout",
                                                                                 "추출 실패", "❌ 페이지 없음"]:
                        corrected_data[field] = ocr_value
                        logging.debug(f"  🔒 {field}: 수기 입력값 보존 '{ocr_value}'")
                        continue

                    if str(spreadsheet_value) != str(ocr_value):
                        corrections_made.append(
                            f"{field}: '{ocr_value}' → '{spreadsheet_value}' (📊스프레드시트)"
                        )
                    corrected_data[field] = spreadsheet_value
                else:
                    # 스프레드시트에 값이 없을 때만 OCR 값 사용
                    corrected_data[field] = ocr_value
                    if ocr_value:
                        logging.debug(f"  📝 {field}: 스프레드시트 없음, OCR '{ocr_value}' 사용")

        if corrections_made:
            logging.info(f"📋 스프레드시트 보정 ({len(corrections_made)}건):")
            for correction in corrections_made:
                logging.info(f"  ✓ {correction}")

        return corrected_data


class WorkbenchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("학습형 WPS 워크벤치 v6.7 (병합셀 지원)")
        self.geometry("1600x900")

        self._init_variables()
        self._setup_layout()
        self._setup_left_frame()
        self._setup_center_frame()
        self._setup_right_frame()
        self._setup_bindings()

        self.rebuild_data_entries()
        self.after(100, self.load_workspace_state)

    def _init_variables(self):
        """변수 초기화"""
        self.template_data = None
        self.template_name = None
        self.template_coords_variations = {}
        self.pdf_doc = None
        self.current_page = 0
        self.zoom_level = 1.0
        self.input_files = []
        self.current_file_index = -1
        self.rect_start_pos = None
        self.current_rect_id = None
        self.selected_field = None
        self.data_entries = {}
        self.ocr_raw_results = {}
        self.manual_extraction_results = {}
        self.processes = []
        self.knowledge_base = self.load_knowledge_base()
        self.file_list_buttons = []
        self.completed_files = {}

        self.template_rects = {}
        self.template_labels = {}
        self.rect_resize_mode = None
        self.rect_drag_start = None
        self.rect_original_coords = None
        self.editing_field = None

        self.file_list_page = 0
        self.total_file_pages = 0

        self.anchor_field = 'WPS_No'
        self.anchor_position = None
        self.use_anchor_system = True

        if OPENCV_AVAILABLE:
            self.adaptive_preprocessor = AdaptivePreprocessor()
            self.use_preprocessing = True
            logging.info("✓ 전처리 시스템 활성화")
        else:
            self.adaptive_preprocessor = None
            self.use_preprocessing = False
            logging.info("⚠️ OpenCV 미설치 - 기본 OCR만 사용")

        self.template_recommender = TemplateRecommender()

        # ★★★ 스프레드시트 URL (실제 URL로 변경) ★★★
        spreadsheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSkUzQMPJm5ZMyLqz68hUoNNKaUVa4KYshUGA2rUdLYYlwTtt1bdTmvZETGPJWyL9Q6VJ87j5NqbjAo/pub?output=csv"
        self.spreadsheet_validator = SpreadsheetValidator(spreadsheet_url)

        self.VALID_WELDING_PROCESSES = ['SMAW', 'GTAW', 'FCAW', 'GMAW', 'PAW', 'SAW']
        self.base_fields = ['Welding_Process_Type', 'WPS_No', 'Preheat_Temp_Min', 'Gas_Flow_Rate']

        self.dynamic_field_templates = {
            'AWS_Class': {'whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,./ '},
            'Current': {'whitelist': '0123456789~-'},
            'Voltage': {'whitelist': '0123456789~-'},
            'Travel_Speed': {'whitelist': '0123456789.~-'},
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
        self.left_frame.grid_rowconfigure(5, weight=1)

        self.file_frame = ctk.CTkFrame(self.left_frame)
        self.file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.file_frame.grid_columnconfigure(0, weight=1)

        self.pdf_path_entry = ctk.CTkEntry(self.file_frame, placeholder_text="PDF 폴더 경로")
        self.pdf_path_entry.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")

        ctk.CTkButton(self.file_frame, text="폴더...", width=60,
                      command=self.browse_for_pdf_folder).grid(row=0, column=1, pady=5)

        ctk.CTkButton(self.left_frame, text="📋 템플릿 관리 (Load/Save)",
                      command=self.manage_template, height=35).grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.template_name_frame = ctk.CTkFrame(self.left_frame, fg_color="#2b2b2b", border_width=1,
                                                border_color="#444444")
        self.template_name_frame.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="ew")

        self.template_name_label = ctk.CTkLabel(
            self.template_name_frame,
            text="📋 템플릿: 없음",
            font=("Arial", 11, "bold"),
            text_color="#ffa500"
        )
        self.template_name_label.pack(pady=5, padx=10)

        self.progress_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.progress_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.progress_label = ctk.CTkLabel(self.progress_frame, text="완료 0/0 (0%)",
                                           font=("Arial", 14, "bold"), text_color="#00ff00")
        self.progress_label.pack()

        self.file_page_nav = ctk.CTkFrame(self.left_frame)
        self.file_page_nav.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        self.file_page_nav.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(self.file_page_nav, text="◀", width=50, height=32,
                      command=self.prev_file_page, font=("Arial", 14, "bold")).grid(row=0, column=0, padx=2)

        self.file_page_label = ctk.CTkLabel(self.file_page_nav, text="페이지 1/1",
                                            font=("Arial", 12))
        self.file_page_label.grid(row=0, column=1, padx=5)

        ctk.CTkButton(self.file_page_nav, text="▶", width=50, height=32,
                      command=self.next_file_page, font=("Arial", 14, "bold")).grid(row=0, column=2, padx=2)

        jump_frame = ctk.CTkFrame(self.file_page_nav, fg_color="transparent")
        jump_frame.grid(row=1, column=0, columnspan=3, pady=5)

        ctk.CTkLabel(jump_frame, text="파일 번호:", font=("Arial", 10)).pack(side="left", padx=2)
        self.jump_entry = ctk.CTkEntry(jump_frame, width=60, height=24)
        self.jump_entry.pack(side="left", padx=2)
        self.jump_entry.bind("<Return>", lambda e: self.jump_to_file_number())
        ctk.CTkButton(jump_frame, text="이동", width=50, height=24,
                      command=self.jump_to_file_number).pack(side="left", padx=2)

        self.file_list_frame = ctk.CTkScrollableFrame(self.left_frame, label_text="📄 PDF 파일 목록")
        self.file_list_frame.grid(row=5, column=0, padx=10, pady=5, sticky="nsew")

        self.status_label = ctk.CTkLabel(self.left_frame, text="준비",
                                         wraplength=260, justify="center",
                                         font=("Arial", 11))
        self.status_label.grid(row=6, column=0, padx=10, pady=5)

        self.page_nav_frame = ctk.CTkFrame(self.left_frame)
        self.page_nav_frame.grid(row=7, column=0, padx=10, pady=5)

        ctk.CTkButton(self.page_nav_frame, text="◀", width=40,
                      command=self.prev_page).pack(side="left", padx=2)
        self.page_label = ctk.CTkLabel(self.page_nav_frame, text="Page 0/0", width=100)
        self.page_label.pack(side="left", padx=5)
        ctk.CTkButton(self.page_nav_frame, text="▶", width=40,
                      command=self.next_page).pack(side="left", padx=2)

        self.bottom_frame = ctk.CTkFrame(self.left_frame)
        self.bottom_frame.grid(row=8, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkButton(self.bottom_frame, text="⮜ 이전 파일",
                      command=self.prev_pdf, height=35).pack(fill="x", pady=(0, 5))
        ctk.CTkButton(self.bottom_frame, text="다음 파일 ⮞",
                      command=self.next_pdf, height=35).pack(fill="x")

    def _setup_center_frame(self):
        """중앙 프레임 설정"""
        self.center_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="gray20")
        self.center_frame.grid(row=0, column=1, sticky="nsew")
        self.center_frame.grid_rowconfigure(0, weight=1)
        self.center_frame.grid_columnconfigure(0, weight=1)

        self.canvas = ctk.CTkCanvas(self.center_frame, bg="gray20", highlightthickness=0)

        self.v_scrollbar = ctk.CTkScrollbar(self.center_frame, orientation="vertical", command=self.canvas.yview)
        self.h_scrollbar = ctk.CTkScrollbar(self.center_frame, orientation="horizontal", command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.v_scrollbar.grid(row=0, column=1, sticky='ns')
        self.h_scrollbar.grid(row=1, column=0, sticky='ew')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        self.file_label = ctk.CTkLabel(self.canvas, text="PDF를 불러오세요",
                                       fg_color="black", text_color="white", corner_radius=5)

    def _setup_right_frame(self):
        """오른쪽 프레임 설정"""
        self.right_frame = ctk.CTkFrame(self, width=350)
        self.right_frame.grid(row=0, column=2, sticky="nsew")
        self.right_frame.grid_rowconfigure(3, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        preprocess_info = ctk.CTkFrame(self.right_frame,
                                       fg_color="#1a4d2e" if OPENCV_AVAILABLE else "#4d3a1a",
                                       border_width=1,
                                       border_color="#2fa572" if OPENCV_AVAILABLE else "#a57a2f")
        preprocess_info.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")

        if OPENCV_AVAILABLE:
            self.preprocess_var = ctk.BooleanVar(value=True)

            preprocess_toggle = ctk.CTkCheckBox(
                preprocess_info,
                text="🎨 이미지 전처리 사용",
                variable=self.preprocess_var,
                command=self.toggle_preprocessing,
                font=("Arial", 11, "bold"),
                fg_color="#2fa572",
                hover_color="#27ae60"
            )
            preprocess_toggle.pack(pady=5, padx=10)

            self.preprocess_status_label = ctk.CTkLabel(
                preprocess_info,
                text="✓ PDF 텍스트 우선, 없으면 전처리+OCR",
                font=("Arial", 9),
                text_color="#90ee90"
            )
            self.preprocess_status_label.pack(pady=(0, 5))
        else:
            ctk.CTkLabel(
                preprocess_info,
                text="⚠️ 전처리 비활성화",
                font=("Arial", 11, "bold"),
                text_color="#ffa500"
            ).pack(pady=5)

            ctk.CTkLabel(
                preprocess_info,
                text="OpenCV 미설치\npip install opencv-python",
                font=("Arial", 9),
                text_color="#ffcc00"
            ).pack(pady=(0, 5))

        ctk.CTkButton(
            self.right_frame,
            text="🔍 전체 필드 자동 찾기",
            command=self.auto_find_all_fields,
            fg_color="#3498db",
            hover_color="#2980b9",
            height=35,
            font=("Arial", 12, "bold")
        ).grid(row=1, column=0, padx=10, pady=(5, 2), sticky="ew")

        ctk.CTkButton(self.right_frame, text="⚡ 현재 파일 자동 추출",
                      command=self.run_extraction).grid(row=2, column=0, padx=10, pady=(2, 5), sticky="ew")

        self.data_entry_frame = ctk.CTkScrollableFrame(self.right_frame,
                                                       label_text="추출 데이터 (라벨 클릭 후 영역 지정)")
        self.data_entry_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        self.data_entry_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(self.right_frame, text="💾 최종 결과 저장 및 학습",
                      command=self.save_results, fg_color="green", height=40,
                      font=("Arial", 12, "bold")).grid(row=4, column=0,
                                                       padx=10, pady=10, sticky="ew")

    def _setup_bindings(self):
        """이벤트 바인딩"""
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)
        self.canvas.bind("<Motion>", self.on_hover)

    def toggle_preprocessing(self):
        """전처리 ON/OFF"""
        self.use_preprocessing = self.preprocess_var.get()

        if self.use_preprocessing:
            status_text = "✓ PDF 텍스트 우선, 없으면 전처리+OCR"
            status_color = "#90ee90"
            msg = "🎨 전처리 활성화"
            msg_color = "cyan"
        else:
            status_text = "✓ PDF 텍스트 우선, 없으면 기본 OCR"
            status_color = "#ffa500"
            msg = "⚠️ 전처리 비활성화"
            msg_color = "orange"

        if hasattr(self, 'preprocess_status_label'):
            self.preprocess_status_label.configure(text=status_text, text_color=status_color)

        self.status_label.configure(text=msg, text_color=msg_color)
        logging.info(f"{'✓' if self.use_preprocessing else '✗'} 전처리: {self.use_preprocessing}")

    # === 파일 목록 관리 ===
    def update_file_list_display(self):
        """파일 목록 표시"""
        for btn in self.file_list_buttons:
            btn.destroy()
        self.file_list_buttons = []

        start_idx = self.file_list_page * FILES_PER_PAGE
        end_idx = min(start_idx + FILES_PER_PAGE, len(self.input_files))

        for i in range(start_idx, end_idx):
            f = self.input_files[i]
            is_completed = f in self.completed_files

            file_num = i + 1
            btn_text = f"{file_num}. " + ("✓ " if is_completed else "") + os.path.basename(f)

            if i == self.current_file_index:
                btn_color = "#1f6aa5"
                border_width = 2
            elif is_completed:
                btn_color = "#2fa572"
                border_width = 0
            else:
                btn_color = "gray"
                border_width = 0

            btn = ctk.CTkButton(
                self.file_list_frame,
                text=btn_text,
                fg_color=btn_color,
                border_width=border_width,
                border_color="white",
                command=lambda idx=i: self.jump_to_pdf(idx),
                anchor="w",
                height=32
            )
            btn.pack(fill="x", padx=5, pady=2)
            self.file_list_buttons.append(btn)

        self.total_file_pages = (len(self.input_files) + FILES_PER_PAGE - 1) // FILES_PER_PAGE
        self.file_page_label.configure(
            text=f"페이지 {self.file_list_page + 1}/{self.total_file_pages}\n({start_idx + 1}-{end_idx}/{len(self.input_files)})"
        )

        completed_count = len(self.completed_files)
        total_count = len(self.input_files)
        if total_count > 0:
            percentage = (completed_count / total_count) * 100
            self.progress_label.configure(
                text=f"✓ 완료 {completed_count}/{total_count} ({percentage:.1f}%)"
            )

    def prev_file_page(self):
        """이전 페이지"""
        if self.file_list_page > 0:
            self.file_list_page -= 1
            self.update_file_list_display()

    def next_file_page(self):
        """다음 페이지"""
        if self.file_list_page < self.total_file_pages - 1:
            self.file_list_page += 1
            self.update_file_list_display()

    def jump_to_file_number(self):
        """파일 번호로 이동"""
        try:
            file_num = int(self.jump_entry.get())
            if 1 <= file_num <= len(self.input_files):
                self.jump_to_pdf(file_num - 1)
                self.jump_entry.delete(0, "end")
                self.status_label.configure(
                    text=f"✓ 파일 #{file_num}로 이동",
                    text_color="green"
                )
            else:
                self.status_label.configure(
                    text=f"⚠️ 1~{len(self.input_files)} 범위",
                    text_color="orange"
                )
        except ValueError:
            self.status_label.configure(text="⚠️ 숫자 입력", text_color="orange")

    # === 작업 상태 저장/복원 ===
    def save_workspace_state(self):
        """작업 상태 저장"""
        try:
            state = {
                'folder_path': self.pdf_path_entry.get(),
                'current_file_index': self.current_file_index,
                'completed_files': self.completed_files,
                'template_data': self.template_data,
                'template_name': self.template_name,
                'template_coords_variations': self.template_coords_variations,
                'anchor_position': getattr(self, 'template_anchor_position', None),
                'file_list_page': self.file_list_page,
                'ocr_raw_results': self.ocr_raw_results,
                'manual_extraction_results': self.manual_extraction_results
            }
            with open(WORKSPACE_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"작업 상태 저장 실패: {e}")

    def load_workspace_state(self):
        """작업 상태 복원"""
        if not os.path.exists(WORKSPACE_STATE_FILE):
            return
        try:
            with open(WORKSPACE_STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)

            folder_path = state.get('folder_path', '')
            if folder_path and os.path.isdir(folder_path):
                self.pdf_path_entry.delete(0, "end")
                self.pdf_path_entry.insert(0, folder_path)
                self.completed_files = state.get('completed_files', {})
                self.template_data = state.get('template_data')
                self.template_name = state.get('template_name')
                self.template_coords_variations = state.get('template_coords_variations', {})

                anchor_pos = state.get('anchor_position')
                if anchor_pos:
                    self.template_anchor_position = tuple(anchor_pos) if isinstance(anchor_pos, list) else anchor_pos

                self.file_list_page = state.get('file_list_page', 0)
                self.ocr_raw_results = state.get('ocr_raw_results', {})
                self.manual_extraction_results = state.get('manual_extraction_results', {})

                if self.template_name:
                    self.template_name_label.configure(
                        text=f"📋 {self.template_name}",
                        text_color="#00ff00"
                    )
                    self.template_name_frame.configure(border_color="#00ff00")

                self.load_pdf_folder()

                prev_index = state.get('current_file_index', 0)
                if 0 <= prev_index < len(self.input_files):
                    self.after(100, lambda: self.jump_to_pdf(prev_index))

                self.status_label.configure(text="✓ 이전 작업 복원 완료", text_color="cyan")
        except Exception as e:
            logging.error(f"작업 상태 복원 실패: {e}")

    def load_completed_data(self, file_path):
        """완료된 파일 데이터 로드"""
        if file_path not in self.completed_files:
            return

        data = self.completed_files[file_path].get('data', {})

        process_value = data.get('Welding_Process_Type', '')
        if process_value:
            process_pattern = "|".join(self.VALID_WELDING_PROCESSES)
            found_processes = re.findall(process_pattern, process_value, re.IGNORECASE)
            if found_processes:
                self.processes = [p.upper() for p in found_processes]
                self.rebuild_data_entries()

        for field, value in data.items():
            if field in self.data_entries:
                entry = self.data_entries[field]
                entry.delete(0, "end")
                entry.insert(0, value)

    # === 파일 관리 ===
    def browse_for_pdf_folder(self):
        """PDF 폴더 선택"""
        folder_path = filedialog.askdirectory(title="WPS PDF 폴더 선택")
        if not folder_path:
            return
        self.pdf_path_entry.delete(0, "end")
        self.pdf_path_entry.insert(0, folder_path)
        self.load_pdf_folder()

    def load_pdf_folder(self):
        """폴더 내 PDF 로드"""
        folder_path = self.pdf_path_entry.get()
        if not os.path.isdir(folder_path):
            self.status_label.configure(text="유효하지 않은 폴더 경로입니다.", text_color="red")
            return

        all_pdfs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        self.input_files = self.filter_latest_revisions(all_pdfs)

        if not self.input_files:
            self.status_label.configure(text="PDF 파일이 없습니다.", text_color="orange")
            return

        self.load_completed_files_from_output()

        self.file_list_page = 0
        self.update_file_list_display()

        self.status_label.configure(
            text=f"✓ {len(self.input_files)}개 파일 로드 완료",
            text_color="green"
        )

        if self.input_files:
            self.jump_to_pdf(0)

    def filter_latest_revisions(self, pdf_files):
        """최신 Revision만 필터링"""
        wps_groups = {}

        for filepath in pdf_files:
            filename = os.path.basename(filepath)
            wps_match = re.search(r'(P-WPS-[\w\-\.]+?)(?:_Rev\.(\d+))?\.pdf', filename, re.IGNORECASE)

            if wps_match:
                wps_base = wps_match.group(1)
                rev_num = int(wps_match.group(2)) if wps_match.group(2) else 0

                if wps_base not in wps_groups:
                    wps_groups[wps_base] = []

                wps_groups[wps_base].append({
                    'path': filepath,
                    'rev': rev_num,
                    'filename': filename
                })

        latest_files = []
        for wps_base, versions in wps_groups.items():
            latest = max(versions, key=lambda x: x['rev'])
            latest_files.append(latest['path'])

            if len(versions) > 1:
                logging.info(f"{wps_base}: {len(versions)}개 버전 중 Rev.{latest['rev']} 선택")

        return sorted(latest_files)

    def load_completed_files_from_output(self):
        """OUTPUT 폴더에서 완료 파일 로드"""
        if not os.path.exists(OUTPUT_FOLDER):
            return

        for pdf_file in self.input_files:
            pdf_basename = os.path.splitext(os.path.basename(pdf_file))[0]
            result_file = os.path.join(OUTPUT_FOLDER, f"{pdf_basename}_result.json")

            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.completed_files[pdf_file] = {
                        'data': data,
                        'timestamp': os.path.getmtime(result_file)
                    }
                except Exception as e:
                    logging.error(f"결과 파일 로드 실패: {e}")

    def jump_to_pdf(self, index):
        """특정 PDF로 이동"""
        if not 0 <= index < len(self.input_files):
            return

        self.current_file_index = index

        target_page = index // FILES_PER_PAGE
        if target_page != self.file_list_page:
            self.file_list_page = target_page
            self.update_file_list_display()

        self.update_file_button_status()
        self.load_pdf_document()

    def load_pdf_document(self):
        """현재 PDF 문서 로드"""
        if not 0 <= self.current_file_index < len(self.input_files):
            return

        if self.pdf_doc:
            self.pdf_doc.close()

        pdf_path = self.input_files[self.current_file_index]
        try:
            self.pdf_doc = fitz.open(pdf_path)
            self.current_page = 0
            self.zoom_level = 1.0
            self.ocr_raw_results = {}
            self.manual_extraction_results = {}
            self.display_page(fit_to_screen=True)

            is_completed = pdf_path in self.completed_files

            if is_completed:
                self.status_label.configure(
                    text=f"✓ 완료 파일\n{os.path.basename(pdf_path)}",
                    text_color="green"
                )
                self.load_completed_data(pdf_path)
            else:
                self.status_label.configure(
                    text=f"📄 작업 중\n{os.path.basename(pdf_path)}",
                    text_color="white"
                )
                self.processes = []
                self.rebuild_data_entries()

                self.recommend_and_load_template(pdf_path)

        except Exception as e:
            logging.error(f"PDF 로드 실패: {e}")
            self.status_label.configure(text=f"PDF 로드 실패: {e}", text_color="red")

    def recommend_and_load_template(self, pdf_path):
        """템플릿 자동 추천 및 로드"""
        if self.template_data:
            logging.info("✓ 템플릿이 이미 로드되어 있음")
            self.after(300, self.run_extraction)
            return

        available_templates = self._get_template_files()

        if not available_templates:
            logging.info("⚠️ 사용 가능한 템플릿이 없습니다.")
            return

        logging.info(f"📋 템플릿 {len(available_templates)}개 발견")

        recommended_template, confidence, auto_load = self.template_recommender.recommend_template(
            pdf_path, available_templates
        )

        logging.info(f"🎯 추천 결과: {recommended_template} (신뢰도: {confidence}%, 자동로드: {auto_load})")

        if not recommended_template:
            return

        ALWAYS_SHOW_DIALOG = False

        if not ALWAYS_SHOW_DIALOG and auto_load and confidence >= 70:
            template_file = f"template_{recommended_template}.json"
            self._load_template_file_silent(template_file)

            self.status_label.configure(
                text=f"🎯 템플릿 자동 로드\n{recommended_template}\n(신뢰도: {confidence}%)",
                text_color="cyan"
            )

            self.after(300, self.run_extraction)

        elif confidence >= 30:
            self.show_template_recommendation_dialog(recommended_template, confidence, pdf_path)

    def show_template_recommendation_dialog(self, recommended_template, confidence, pdf_path):
        """템플릿 추천 다이얼로그"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("템플릿 추천")
        dialog.geometry("550x350")
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 275
        y = (dialog.winfo_screenheight() // 2) - 175
        dialog.geometry(f"550x350+{x}+{y}")

        ctk.CTkLabel(
            dialog,
            text="🎯",
            font=("Arial", 48)
        ).pack(pady=15)

        ctk.CTkLabel(
            dialog,
            text=f"추천 템플릿: {recommended_template}",
            font=("Arial", 16, "bold")
        ).pack(pady=5)

        ctk.CTkLabel(
            dialog,
            text=f"신뢰도: {confidence}%",
            font=("Arial", 14),
            text_color="#00ff00" if confidence >= 70 else "#ffa500"
        ).pack(pady=5)

        ctk.CTkLabel(
            dialog,
            text="이 템플릿을 사용하시겠습니까?",
            font=("Arial", 12)
        ).pack(pady=10)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        def load_recommended():
            template_file = f"template_{recommended_template}.json"
            self._load_template_file_silent(template_file)
            dialog.destroy()

            self.status_label.configure(
                text=f"✓ 템플릿 로드\n{recommended_template}",
                text_color="green"
            )

            self.after(300, self.run_extraction)

        def choose_other():
            dialog.destroy()
            self.manage_template()

        def skip():
            dialog.destroy()

        ctk.CTkButton(
            btn_frame,
            text="✓ 사용",
            width=120,
            height=40,
            fg_color="#2fa572",
            hover_color="#27ae60",
            command=load_recommended
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="📋 다른 템플릿 선택",
            width=140,
            height=40,
            fg_color="#1f6aa5",
            hover_color="#2980b9",
            command=choose_other
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="건너뛰기",
            width=100,
            height=40,
            fg_color="gray",
            command=skip
        ).pack(side="left", padx=5)

    def _load_template_file_silent(self, filename):
        """템플릿 파일 로드 (다이얼로그 없이)"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.template_data = data.get('template_data', data)
            self.template_name = filename.replace('template_', '').replace('.json', '')
            self.template_coords_variations = data.get('coords_variations', {})

            anchor_pos = data.get('anchor_position')
            if anchor_pos:
                self.template_anchor_position = tuple(anchor_pos) if isinstance(anchor_pos, list) else anchor_pos

            self.template_name_label.configure(
                text=f"📋 {self.template_name}",
                text_color="#00ff00"
            )
            self.template_name_frame.configure(border_color="#00ff00")

            self.rebuild_data_entries()
            self.save_workspace_state()

            if self.pdf_doc:
                self.display_page()

            logging.info(f"✓ 템플릿 로드: {self.template_name}")

        except Exception as e:
            logging.error(f"템플릿 로드 실패: {e}")

    # === 템플릿 관리 ===
    def manage_template(self):
        """템플릿 Load/Save UI"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("템플릿 관리")
        dialog.geometry("550x500")
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 275
        y = (dialog.winfo_screenheight() // 2) - 250
        dialog.geometry(f"550x500+{x}+{y}")

        ctk.CTkLabel(dialog, text="📋 템플릿 관리",
                     font=("Arial", 20, "bold")).pack(pady=15)

        load_frame = ctk.CTkFrame(dialog, fg_color="#2b2b2b", border_width=2, border_color="#1f6aa5")
        load_frame.pack(pady=10, padx=20, fill="both", expand=True)

        ctk.CTkLabel(load_frame, text="📤 불러오기",
                     font=("Arial", 16, "bold"),
                     text_color="#1f6aa5").pack(pady=10)

        list_frame = ctk.CTkScrollableFrame(load_frame, height=200, fg_color="#1a1a1a")
        list_frame.pack(pady=5, padx=10, fill="both", expand=True)

        template_files = self._get_template_files()

        if not template_files:
            ctk.CTkLabel(list_frame, text="💡 저장된 템플릿이 없습니다.",
                         text_color="#ffa500",
                         font=("Arial", 12)).pack(pady=30)
        else:
            for tmpl_file in template_files:
                name = tmpl_file.replace('template_', '').replace('.json', '')

                item_frame = ctk.CTkFrame(list_frame, fg_color="#2b2b2b", border_width=2, border_color="#444444")
                item_frame.pack(fill="x", padx=5, pady=3)

                btn = ctk.CTkButton(
                    item_frame,
                    text=f"📄 {name}",
                    anchor="w",
                    height=40,
                    font=("Arial", 13, "bold"),
                    fg_color="#1f6aa5",
                    hover_color="#2980b9",
                    text_color="white",
                    command=lambda f=tmpl_file: self._load_template_file(f, dialog)
                )
                btn.pack(fill="x", padx=3, pady=3)

        save_frame = ctk.CTkFrame(dialog, fg_color="#2b2b2b", border_width=2, border_color="#2fa572")
        save_frame.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(save_frame, text="💾 저장",
                     font=("Arial", 16, "bold"),
                     text_color="#2fa572").pack(pady=10)

        save_desc = ctk.CTkLabel(save_frame,
                                 text="현재 설정된 영역 좌표를 템플릿으로 저장합니다.",
                                 font=("Arial", 11),
                                 text_color="gray")
        save_desc.pack(pady=(0, 10))

        ctk.CTkButton(save_frame, text="✨ 새 템플릿으로 저장", height=45,
                      font=("Arial", 14, "bold"),
                      fg_color="#2fa572",
                      hover_color="#27ae60",
                      text_color="white",
                      command=lambda: self._save_new_template(dialog)).pack(pady=(0, 15), padx=20, fill="x")

        ctk.CTkButton(dialog, text="✖ 닫기", width=120, height=35,
                      fg_color="gray",
                      command=dialog.destroy).pack(pady=10)

    def _get_template_files(self):
        """템플릿 파일 목록"""
        try:
            files = [f for f in os.listdir('.')
                     if f.startswith('template_') and f.endswith('.json')]
            return sorted(files)
        except Exception as e:
            logging.error(f"템플릿 파일 검색 실패: {e}")
            return []

    def _load_template_file(self, filename, dialog):
        """템플릿 파일 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.template_data = data.get('template_data', data)
            self.template_name = filename.replace('template_', '').replace('.json', '')
            self.template_coords_variations = data.get('coords_variations', {})

            anchor_pos = data.get('anchor_position')
            if anchor_pos:
                self.template_anchor_position = tuple(anchor_pos) if isinstance(anchor_pos, list) else anchor_pos

            self.template_name_label.configure(
                text=f"📋 {self.template_name}",
                text_color="#00ff00"
            )
            self.template_name_frame.configure(border_color="#00ff00")

            self.status_label.configure(
                text=f"✓ 템플릿 로드:\n{self.template_name}",
                text_color="green"
            )

            self.rebuild_data_entries()
            self.save_workspace_state()
            dialog.destroy()

            if self.pdf_doc:
                self.display_page()
                self.run_extraction()

        except Exception as e:
            logging.error(f"템플릿 로드 실패: {e}")
            self.status_label.configure(
                text=f"로드 실패: {e}",
                text_color="red"
            )

    def _save_new_template(self, dialog):
        """새 템플릿 저장"""
        if not self.template_data:
            self.status_label.configure(text="저장할 템플릿 없음", text_color="orange")
            return

        name_dialog = ctk.CTkInputDialog(
            text="템플릿 이름 입력:",
            title="템플릿 저장"
        )
        template_name = name_dialog.get_input()

        if not template_name:
            return

        template_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', template_name)
        filename = f"template_{template_name}.json"

        try:
            self.save_template_anchor_position()

            save_data = {
                'template_data': self.template_data,
                'coords_variations': self.template_coords_variations,
                'anchor_position': getattr(self, 'template_anchor_position', None),
                'name': template_name
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            self.template_name = template_name

            self.template_name_label.configure(
                text=f"📋 {template_name}",
                text_color="#00ff00"
            )
            self.template_name_frame.configure(border_color="#00ff00")

            self.status_label.configure(
                text=f"✓ 템플릿 저장:\n{template_name}",
                text_color="green"
            )

            logging.info(f"✓ 템플릿 저장 성공: {filename}")
            dialog.destroy()

        except Exception as e:
            logging.error(f"템플릿 저장 실패: {e}")
            self.status_label.configure(
                text=f"저장 실패: {e}",
                text_color="red"
            )

    # === 데이터 입력 필드 ===
    def rebuild_data_entries(self):
        """데이터 입력 필드 재구성"""
        for widget in self.data_entry_frame.winfo_children():
            widget.destroy()

        self.data_entries = {}

        fields_to_show = self.base_fields.copy()

        if self.processes:
            for process in self.processes:
                for suffix_template in self.dynamic_field_templates:
                    fields_to_show.append(f"{suffix_template}_{process.strip()}")

        for field in fields_to_show:
            frame = ctk.CTkFrame(self.data_entry_frame, fg_color="#1a1a1a", border_width=1, border_color="#333333")
            frame.pack(fill="x", padx=5, pady=3, anchor="w")
            frame.grid_columnconfigure(1, weight=1)

            has_coords = self.template_data and field in self.template_data and self.template_data[field].get('rect')

            label_frame = ctk.CTkFrame(frame, fg_color="#000000", corner_radius=3)
            label_frame.grid(row=0, column=0, padx=3, pady=3, sticky="w")

            label_text = f"{'✓' if has_coords else '○'} {field}"
            label_color = "#00ff00" if has_coords else "#999999"

            label = ctk.CTkLabel(label_frame, text=label_text, width=150, anchor="w",
                                 text_color=label_color, font=("Arial", 10, "bold"))
            label.pack(padx=5, pady=2)

            entry = ctk.CTkEntry(frame, fg_color="#2b2b2b", text_color="#ffffff", border_color="#555555")
            entry.grid(row=0, column=1, sticky="ew", padx=3, pady=3)

            self.data_entries[field] = entry
            label.bind("<Button-1>", lambda event, f=field: self.start_defining(f))

            if field == 'Welding_Process_Type':
                entry.bind("<FocusOut>", self.on_process_type_changed)
                entry.bind("<Return>", self.on_process_type_changed)

    def on_process_type_changed(self, event):
        """Welding_Process_Type 수동 변경 감지"""
        entry = event.widget
        new_value = entry.get().strip()

        if not new_value:
            return

        logging.info(f"🔄 Welding_Process_Type 수동 변경 감지: '{new_value}'")

        process_pattern = "|".join(self.VALID_WELDING_PROCESSES)
        found_processes = re.findall(process_pattern, new_value, re.IGNORECASE)

        if not found_processes:
            logging.warning(f"⚠️ 유효한 프로세스 발견 안됨: '{new_value}'")
            return

        new_processes = sorted(list(set([p.upper() for p in found_processes])))
        old_processes = self.processes if hasattr(self, 'processes') else []

        if new_processes == old_processes:
            return

        logging.info(f"✓ 프로세스 변경: {old_processes} → {new_processes}")

        old_dynamic_data = {}
        if old_processes:
            for process in old_processes:
                for suffix_template in self.dynamic_field_templates:
                    old_field = f"{suffix_template}_{process}"
                    if old_field in self.data_entries:
                        value = self.data_entries[old_field].get()
                        if value:
                            old_dynamic_data[suffix_template] = value

        self.processes = new_processes

        self.status_label.configure(
            text=f"✓ 프로세스 변경\n{' + '.join(self.processes)}",
            text_color="cyan"
        )

        self.rebuild_data_entries()

        if 'Welding_Process_Type' in self.data_entries:
            self.data_entries['Welding_Process_Type'].delete(0, "end")
            self.data_entries['Welding_Process_Type'].insert(0, new_value)

        if old_dynamic_data:
            for suffix_template, value in old_dynamic_data.items():
                if self.processes:
                    new_field = f"{suffix_template}_{self.processes[0]}"
                    if new_field in self.data_entries:
                        self.data_entries[new_field].delete(0, "end")
                        self.data_entries[new_field].insert(0, value)

        if self.template_data and self.pdf_doc:
            self.after(200, self._extract_new_process_fields)

    def _extract_new_process_fields(self):
        """새로 생성된 프로세스 필드 자동 추출"""
        logging.info("🔍 새 프로세스 필드 자동 추출 시도...")

        for process in self.processes:
            for suffix_template in self.dynamic_field_templates:
                field = f"{suffix_template}_{process}"

                if field in self.data_entries:
                    current_value = self.data_entries[field].get()
                    if current_value and current_value not in ["추출 실패", "❌ 페이지 없음"]:
                        continue

                    self.run_extraction_for_field(field)

    def start_defining(self, field_name):
        """필드 영역 정의 시작"""
        self.selected_field = field_name
        self.status_label.configure(
            text=f"🎯 {field_name}\n영역을 드래그하세요",
            text_color="cyan"
        )

    # === 프로세스 인식 ===
    def generate_dynamic_fields(self):
        """Welding Process Type 인식"""
        if not self.template_data or not self.pdf_doc:
            return

        manual_value = None
        if 'Welding_Process_Type' in self.data_entries:
            manual_value = self.data_entries['Welding_Process_Type'].get().strip()
            if manual_value:
                process_pattern = "|".join(self.VALID_WELDING_PROCESSES)
                found_processes = re.findall(process_pattern, manual_value, re.IGNORECASE)

                if found_processes:
                    new_processes = sorted(list(set([p.upper() for p in found_processes])))
                    old_processes = self.processes if hasattr(self, 'processes') else []

                    if new_processes != old_processes:
                        all_current_data = {}
                        for field, entry in self.data_entries.items():
                            value = entry.get()
                            if value and value not in ["OCR Error", "OCR Timeout", "추출 실패"]:
                                all_current_data[field] = value

                        self.processes = new_processes
                        self.rebuild_data_entries()

                        for field in self.base_fields:
                            if field in all_current_data and field in self.data_entries:
                                self.data_entries[field].delete(0, "end")
                                self.data_entries[field].insert(0, all_current_data[field])

                        self.after(200, self._run_extraction_remaining)
                    return

        info = self.template_data.get('Welding_Process_Type')
        if not info or 'rect' not in info:
            return

        page = self.pdf_doc.load_page(info['page'])
        rect = info['rect']

        try:
            config = self.get_ocr_config('Welding_Process_Type')
            raw_text = self.ocr_from_area_direct(page, rect, config, 'Welding_Process_Type')

            self.ocr_raw_results['Welding_Process_Type'] = raw_text

            process_pattern = "|".join(self.VALID_WELDING_PROCESSES)
            found_processes = re.findall(process_pattern, raw_text, re.IGNORECASE)

            if not found_processes and raw_text:
                if 'Welding_Process_Type' in self.data_entries:
                    entry = self.data_entries['Welding_Process_Type']
                    entry.delete(0, "end")
                    entry.insert(0, raw_text)
                return

            if found_processes:
                new_processes = sorted(list(set([p.upper() for p in found_processes])))

                all_current_data = {}
                for field, entry in self.data_entries.items():
                    value = entry.get()
                    if value and value not in ["OCR Error", "OCR Timeout", "추출 실패"]:
                        all_current_data[field] = value

                self.processes = new_processes
                self.rebuild_data_entries()

                for field in self.base_fields:
                    if field in all_current_data and field in self.data_entries:
                        self.data_entries[field].delete(0, "end")
                        self.data_entries[field].insert(0, all_current_data[field])

                if 'Welding_Process_Type' in self.data_entries:
                    entry = self.data_entries['Welding_Process_Type']
                    entry.delete(0, "end")
                    entry.insert(0, raw_text)

                # ★★★ 스프레드시트 보정 ★★★
                if self.current_file_index >= 0:
                    current_pdf_path = self.input_files[self.current_file_index]
                    filename = os.path.basename(current_pdf_path)

                    temp_data = {'Welding_Process_Type': raw_text}
                    corrected_data = self.spreadsheet_validator.validate_and_correct(filename, temp_data)

                    corrected_process = corrected_data.get('Welding_Process_Type', raw_text)
                    if corrected_process != raw_text:
                        if 'Welding_Process_Type' in self.data_entries:
                            self.data_entries['Welding_Process_Type'].delete(0, "end")
                            self.data_entries['Welding_Process_Type'].insert(0, corrected_process)

                self.after(200, self._run_extraction_remaining)

        except Exception as e:
            logging.error(f"프로세스 인식 오류: {e}")

    # === OCR 관련 ===
    def get_field_type_for_preprocessing(self, field_name):
        """필드명으로 전처리 타입 판단"""
        if 'WPS_No' in field_name:
            return 'wps_no'
        elif any(x in field_name for x in ['Current', 'Voltage', 'Speed', 'Temp', 'Preheat']):
            return 'number'
        elif 'AWS_Class' in field_name or 'Process' in field_name:
            return 'mixed'
        else:
            return 'default'

    def auto_find_all_fields(self):
        """모든 필드 자동 찾기"""
        if not self.pdf_doc:
            self.status_label.configure(text="⚠️ PDF를 먼저 열어주세요", text_color="orange")
            return

        page = self.pdf_doc.load_page(self.current_page)

        if not self.template_data:
            self.template_data = {}

        found_count = 0
        for field in self.data_entries.keys():
            if field in self.template_data and self.template_data[field].get('rect'):
                continue

            rect = self.auto_find_field_region(page, field)
            if rect:
                self.template_data[field] = {
                    'page': self.current_page,
                    'rect': rect
                }
                found_count += 1

        self.display_page()

        if found_count > 0:
            self.status_label.configure(
                text=f"✓ {found_count}개 필드 자동 탐지 완료!",
                text_color="green"
            )
            self.save_workspace_state()
            self.after(500, self.run_extraction)
        else:
            self.status_label.configure(
                text="⚠️ 자동 탐지 실패\n수동으로 지정하세요",
                text_color="orange"
            )

    def auto_find_field_region(self, page, field_name):
        """텍스트 패턴으로 필드 영역 자동 탐지"""
        keywords = self.get_field_keywords(field_name)
        if not keywords:
            return None

        try:
            blocks = page.get_text("dict")["blocks"]

            best_match = None
            best_score = 0

            for keyword in keywords:
                for block in blocks:
                    if "lines" not in block:
                        continue

                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].upper()

                            score = 0
                            if keyword.upper() == text:
                                score = 100
                            elif keyword.upper() in text:
                                score = 80
                            elif any(word in text for word in keyword.upper().split()):
                                score = 60

                            if score > best_score:
                                best_score = score
                                bbox = span["bbox"]

                                data_x0 = bbox[2] + 10
                                data_y0 = bbox[1] - 5
                                data_x1 = bbox[2] + 200
                                data_y1 = bbox[3] + 5

                                best_match = (int(data_x0), int(data_y0), int(data_x1), int(data_y1))

            if best_match and best_score >= 60:
                logging.info(f"✓ 자동 탐지: {field_name} (점수: {best_score})")
                return best_match

            return None

        except Exception as e:
            logging.error(f"자동 탐지 실패 ({field_name}): {e}")
            return None

    def get_field_keywords(self, field_name):
        """필드별 검색 키워드"""
        keywords_map = {
            'WPS_No': ['WPS No', 'WPS NO', 'SUPPORTING PQR', 'P-WPS', 'WPS Number'],
            'Welding_Process_Type': ['WELDING PROCESS', 'Process Type', 'GTAW', 'SMAW', 'FCAW', 'GMAW'],
            'AWS_Class': ['AWS', 'CLASSIFICATION', 'FILLER METAL', 'ELECTRODE', 'CLASS'],
            'Current': ['CURRENT', 'AMPERAGE', 'AMP', 'AMPS'],
            'Voltage': ['VOLTAGE', 'VOLT', 'VOLTS'],
            'Travel_Speed': ['TRAVEL SPEED', 'SPEED', 'TRAVEL'],
            'Preheat_Temp_Min': ['PREHEAT', 'MIN TEMP', 'MINIMUM TEMPERATURE', 'PREHEAT TEMP', 'INTERPASS'],
            'Gas_Flow_Rate': ['GAS FLOW', 'FLOW RATE', 'SHIELDING GAS', 'FLOW', 'GAS'],
        }

        for base_keyword, keywords in keywords_map.items():
            if base_keyword in field_name:
                return keywords

        return []

    def get_ocr_config(self, field_name):
        """
        필드별 OCR 설정
        ★★★ v6.5: Preheat whitelist ASCII 전용 ★★★
        """
        if field_name == 'Welding_Process_Type':
            return '--psm 6'
        elif 'WPS_No' in field_name:
            return '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.'
        elif 'AWS_Class' in field_name or 'AWS' in field_name or 'Class' in field_name:
            return '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,./ '
        elif 'Preheat' in field_name or 'Temp' in field_name:
            # ★★★ v6.5: ASCII만 사용 (유니코드 제거: ≤ → <, ≥ → >) ★★★
            return '--psm 7 -c tessedit_char_whitelist=0123456789.~-<>=TMin :,'
        elif any(x in field_name for x in ['Current', 'Voltage', 'Speed', 'Flow']):
            return '--psm 7 -c tessedit_char_whitelist=0123456789.~- '
        else:
            return OCR_DEFAULT_PSM

    def get_whitelist_chars(self, field):
        """
        필드별 허용 문자
        ★★★ v6.5: Preheat whitelist ASCII 전용 ★★★
        """
        if 'WPS_No' in field:
            return set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_. ')
        elif 'Preheat' in field or 'Temp' in field:
            # ★★★ v6.5: ASCII만 (유니코드 제거) ★★★
            return set('0123456789.~-<>=TMin :,')
        elif any(x in field for x in ['Current', 'Voltage', 'Speed', 'Flow']):
            return set('0123456789.~- ')
        elif 'AWS' in field or 'Class' in field:
            return set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,./ ')
        else:
            return None

    def apply_whitelist_filter(self, text, field):
        """화이트리스트 필터링"""
        if not text or text.strip() == '':
            return ""

        whitelist = self.get_whitelist_chars(field)
        if whitelist is None:
            return text

        filtered = ''.join(c for c in text if c in whitelist)
        filtered = filtered.strip()

        if filtered != text.strip():
            removed = ''.join(c for c in text if c not in whitelist)
            if removed:
                logging.info(f"🔒 화이트리스트 필터: '{text}' → '{filtered}'")

        return filtered

    def preprocess_preheat(self, text):
        """
        Preheat OCR 결과 전처리
        ★★★ v6.5: 깨진 유니코드 복구 + T, Min, <, >, =, : 기호 보존 ★★★
        """
        if not text:
            return text

        original = text

        # ★ v6.5: 깨진 유니코드 복구
        text = self.spreadsheet_validator.fix_broken_unicode(text)

        # ★ 불필요한 문자 제거 (T, Min, <, >, =, :, 숫자, 공백, 쉼표는 보존)
        # C, F, ℃, ° 등은 제거
        text = text.replace('℃', '').replace('°C', '').replace('°', '').replace('C', '').replace('F', '')

        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text).strip()

        if text != original:
            logging.info(f"  🧹 Preheat 전처리: '{original}' → '{text}'")

        return text

    def postprocess_current_voltage(self, text, field_name):
        """Current/Voltage 후처리 - ~ 강제 삽입"""
        if not text or text.strip() == '':
            return text

        numbers = re.findall(r'\d+\.?\d*', text)

        if len(numbers) == 0:
            return text
        elif len(numbers) == 1:
            return numbers[0]
        elif len(numbers) == 2:
            result = f"{numbers[0]}~{numbers[1]}"
            if result != text:
                logging.info(f"{field_name} 후처리: '{text}' → '{result}'")
            return result
        else:
            result = f"{numbers[0]}~{numbers[-1]}"
            logging.info(f"{field_name} 후처리: '{text}' → '{result}'")
            return result

    def extract_text_from_pdf_area(self, page, rect, field_name=''):
        """PDF 영역에서 텍스트 직접 추출"""
        try:
            clip_rect = fitz.Rect(rect)
            text = page.get_text("text", clip=clip_rect).strip()

            if not text or len(text) == 0:
                return None

            if len(text) > 200:
                logging.info(f"  ⚠️ PDF 텍스트가 너무 김 (OCR 사용)")
                return None

            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '-.,~()/<>:')
            if len(text) > 5 and special_chars / len(text) > 0.5:
                logging.info(f"  ⚠️ PDF 텍스트 품질 의심 (OCR 사용)")
                return None

            if field_name and any(x in field_name for x in ['Current', 'Voltage', 'Speed', 'Flow']):
                if not any(c.isdigit() for c in text):
                    logging.info(f"  ⚠️ 숫자 필드에 숫자 없음 (OCR 사용)")
                    return None

            logging.info(f"  📄 PDF 텍스트 추출: '{text}'")
            return text

        except Exception as e:
            logging.debug(f"PDF 텍스트 추출 실패: {e}")
            return None

    def ocr_from_area_direct(self, page, rect, config='', field_name=''):
        """
        영역에서 텍스트/OCR 수행
        ★★★ v6.5: OCR 결과에 깨진 유니코드 복구 추가 ★★★
        """
        pdf_text = self.extract_text_from_pdf_area(page, rect, field_name)
        if pdf_text:
            # ★ v6.5: PDF 텍스트도 깨진 유니코드 복구
            if 'Preheat' in field_name or 'Temp' in field_name:
                pdf_text = self.spreadsheet_validator.fix_broken_unicode(pdf_text)
            return pdf_text

        logging.info(f"  🖼️ 텍스트 레이어 없음 → 이미지 OCR 실행")

        try:
            with tempfile.TemporaryDirectory() as tempdir:
                temp_image_path = os.path.join(tempdir, "temp.png")

                clip_rect = fitz.Rect(rect)
                pix = page.get_pixmap(clip=clip_rect, dpi=OCR_DPI)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                if self.use_preprocessing and self.adaptive_preprocessor:
                    field_type = self.get_field_type_for_preprocessing(field_name)
                    img = self.adaptive_preprocessor.preprocess_adaptive(img, field_type)

                img.save(temp_image_path)

                temp_output_base = os.path.join(tempdir, "output")
                command = [TESSERACT_CMD, temp_image_path, temp_output_base, "-l", "eng"]

                if config:
                    command.extend(config.split())

                subprocess.run(command, check=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=10)

                output_file = temp_output_base + ".txt"
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    logging.info(f"OCR 원본: '{text}'")

                    # ★ v6.5: OCR 결과에도 깨진 유니코드 복구 적용
                    if 'Preheat' in field_name or 'Temp' in field_name:
                        text = self.spreadsheet_validator.fix_broken_unicode(text)
                        logging.info(f"OCR 복구: '{text}'")

                    return text
                return ""

        except subprocess.TimeoutExpired:
            logging.error("OCR 타임아웃")
            return "OCR Timeout"
        except subprocess.CalledProcessError as e:
            logging.error(f"Tesseract 오류: {e.stderr}")
            return "OCR Error"
        except Exception as e:
            logging.error(f"OCR 오류: {e}")
            return "OCR Error"

    # === 추출 실행 ===
    def run_extraction(self):
        """자동 추출"""
        if not self.template_data or not self.pdf_doc:
            self.status_label.configure(text="템플릿 또는 PDF 필요", text_color="orange")
            return

        self.status_label.configure(text="🔍 자동 추출 중...", text_color="cyan")
        self.ocr_raw_results = {}
        self.manual_extraction_results = {}
        self.after(50, self.generate_dynamic_fields)
        self.update()
        self.after(100, self._run_extraction_remaining)
        self.after(200, self.display_page)

    def _run_extraction_remaining(self):
        """
        나머지 필드 추출 + 스프레드시트 검증
        ★★★ v6.5: AWS_Class 필드도 추출하도록 개선 ★★★
        """
        extraction_count = 0
        skipped_count = 0

        for field in self.data_entries.keys():
            if field == 'Welding_Process_Type':
                continue

            current_value = self.data_entries[field].get()

            # ★ v6.5: AWS_Class 필드는 무조건 추출 (스프레드시트 값 우선)
            if 'AWS_Class' not in field:
                if field in self.base_fields and current_value and current_value not in ["", "OCR Error", "OCR Timeout",
                                                                                         "추출 실패"]:
                    skipped_count += 1
                    continue

                if field not in self.base_fields:
                    if current_value and current_value not in ["", "OCR Error", "OCR Timeout", "추출 실패"]:
                        skipped_count += 1
                        continue

            self.run_extraction_for_field(field)
            extraction_count += 1

        logging.info(f"📊 추출 완료: {extraction_count}개 추출, {skipped_count}개 보존")

        # ★★★ v6.5: 스프레드시트 검증 (무조건 적용) ★★★
        if self.current_file_index >= 0:
            current_pdf_path = self.input_files[self.current_file_index]
            filename = os.path.basename(current_pdf_path)

            ocr_data = {field: entry.get() for field, entry in self.data_entries.items()}

            # ★ v6.5: 수기 입력 보존 모드 OFF (스프레드시트 값 최우선)
            corrected_data = self.spreadsheet_validator.validate_and_correct(
                filename,
                ocr_data,
                preserve_manual_edits=False  # 스프레드시트 값 항상 우선!
            )

            updates_made = False
            for field, corrected_value in corrected_data.items():
                if field in self.data_entries:
                    original_value = self.data_entries[field].get()
                    if corrected_value and str(corrected_value) not in ["", "None"]:
                        if str(corrected_value) != str(original_value):
                            self.data_entries[field].delete(0, "end")
                            self.data_entries[field].insert(0, corrected_value)
                            updates_made = True

            if updates_made:
                self.status_label.configure(text="✓ 추출 + 스프레드시트 보정 완료!", text_color="green")
            else:
                self.status_label.configure(text="✓ 추출 완료!", text_color="green")

        for field in self.data_entries.keys():
            if field not in self.template_data:
                self.template_data[field] = {
                    'page': self.current_page,
                    'rect': None
                }

    def run_extraction_for_field(self, field):
        """특정 필드 추출"""
        entry = self.data_entries.get(field)
        info = self.template_data.get(field, {})

        if not entry or not info:
            return

        page_num = info.get('page')
        rect = info.get('rect')

        if page_num is None or not rect:
            return

        if not self.pdf_doc:
            logging.error(f"PDF 문서가 로드되지 않음: {field}")
            return

        if page_num >= len(self.pdf_doc):
            logging.error(f"⚠️ {field}: 페이지 범위 초과")
            entry.delete(0, "end")
            entry.insert(0, f"❌ 페이지 없음")
            return

        try:
            page = self.pdf_doc.load_page(page_num)
            config = self.get_ocr_config(field)

            raw_value = self.ocr_from_area_direct(page, rect, config, field)
            self.ocr_raw_results[field] = raw_value

            if not raw_value or raw_value in ["OCR Error", "OCR Timeout", ""]:
                raw_value = self.try_anchor_based_extraction(page, field, config)

            if not raw_value or raw_value in ["OCR Error", "OCR Timeout", ""]:
                raw_value = self._try_learned_coords(page, field, config)

            filtered = self.apply_whitelist_filter(raw_value, field)

            # ★ Preheat 전처리 (깨진 유니코드 복구 포함)
            if 'Preheat' in field or 'Temp' in field:
                filtered = self.preprocess_preheat(filtered)

            # ★ Current/Voltage 후처리
            if any(x in field for x in ['Current', 'Voltage']) and not any(x in field for x in ['Preheat', 'Temp']):
                filtered = self.postprocess_current_voltage(filtered, field)

            corrected_value = self.intelligent_correction(field, filtered)

            entry.delete(0, "end")
            entry.insert(0, corrected_value)

            logging.info(f"필드 추출: {field} | '{raw_value}' → '{corrected_value}'")

        except Exception as e:
            logging.error(f"필드 추출 오류 ({field}): {e}")
            entry.delete(0, "end")
            entry.insert(0, "추출 실패")

    def _try_learned_coords(self, page, field, config):
        """학습된 좌표로 재시도"""
        if field not in self.template_coords_variations:
            return ""

        variations = self.template_coords_variations[field]
        for rect in variations:
            result = self.ocr_from_area_direct(page, rect, config, field)
            if result and result not in ["OCR Error", "OCR Timeout", ""]:
                logging.info(f"✓ 학습된 좌표로 추출 성공: {field}")
                return result
        return ""

    # === 앵커 기반 좌표 ===
    def detect_anchor_position(self, page):
        """앵커 필드 위치 감지"""
        if self.anchor_field not in self.template_data:
            return None

        anchor_info = self.template_data[self.anchor_field]
        anchor_rect = anchor_info.get('rect')

        if not anchor_rect:
            return None

        x0, y0, x1, y1 = anchor_rect
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2

        return (center_x, center_y)

    def calculate_coord_shift(self, current_anchor, template_anchor):
        """앵커 위치 변화로 shift 계산"""
        if not current_anchor or not template_anchor:
            return (0, 0)

        dx = current_anchor[0] - template_anchor[0]
        dy = current_anchor[1] - template_anchor[1]

        return (dx, dy)

    def apply_shift_to_rect(self, rect, shift):
        """좌표에 shift 적용"""
        x0, y0, x1, y1 = rect
        dx, dy = shift

        return (
            int(x0 + dx),
            int(y0 + dy),
            int(x1 + dx),
            int(y1 + dy)
        )

    def try_anchor_based_extraction(self, page, field, config):
        """앵커 기반 좌표 조정"""
        if not self.use_anchor_system:
            return ""

        if field == self.anchor_field:
            return ""

        current_anchor = self.detect_anchor_position(page)
        if not current_anchor:
            return ""

        template_anchor = getattr(self, 'template_anchor_position', current_anchor)

        shift = self.calculate_coord_shift(current_anchor, template_anchor)

        if abs(shift[0]) > 5 or abs(shift[1]) > 5:
            template_rect = self.template_data.get(field, {}).get('rect')
            if template_rect:
                adjusted_rect = self.apply_shift_to_rect(template_rect, shift)

                try:
                    result = self.ocr_from_area_direct(page, adjusted_rect, config, field)
                    if result and result not in ["OCR Error", "OCR Timeout", ""]:
                        filtered = self.apply_whitelist_filter(result, field)
                        if len(filtered) > 0:
                            if field not in self.template_coords_variations:
                                self.template_coords_variations[field] = []
                            if adjusted_rect not in self.template_coords_variations[field]:
                                self.template_coords_variations[field].append(adjusted_rect)
                            return result
                except Exception as e:
                    logging.debug(f"앵커 기반 추출 실패: {e}")

        return ""

    def save_template_anchor_position(self):
        """템플릿 저장 시 앵커 위치 저장"""
        if not self.template_data or not self.pdf_doc:
            return

        try:
            page = self.pdf_doc.load_page(self.current_page)
            anchor_pos = self.detect_anchor_position(page)
            if anchor_pos:
                self.template_anchor_position = anchor_pos
        except Exception as e:
            logging.error(f"앵커 위치 저장 실패: {e}")

    # === 지식 베이스 ===
    def load_knowledge_base(self):
        """지식 베이스 로드"""
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            try:
                with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                    kb = json.load(f)
                    if not isinstance(kb, dict) or 'corrections' not in kb:
                        return {'values': kb if isinstance(kb, dict) else {}, 'corrections': {}}
                    return kb
            except Exception as e:
                logging.error(f"지식 베이스 로드 실패: {e}")
        return {'values': {}, 'corrections': {}}

    def save_knowledge_base(self):
        """지식 베이스 저장"""
        try:
            with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.knowledge_base, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"지식 베이스 저장 실패: {e}")

    def intelligent_correction(self, field, ocr_text):
        """지능형 보정"""
        if not ocr_text or ocr_text in ["OCR Error", "OCR Timeout"]:
            return ocr_text

        corrections = self.knowledge_base.get('corrections', {})
        field_corrections = corrections.get(field, {})

        if ocr_text in field_corrections:
            corrected = field_corrections[ocr_text]
            logging.info(f"✓ 변환 맵 적용: '{ocr_text}' → '{corrected}'")
            return corrected

        base_field = field.split('_')[0]
        values = self.knowledge_base.get('values', {})
        known_values = set(values.get(field, []) + values.get(base_field, []))

        if not known_values:
            return ocr_text

        try:
            closest_match = min(known_values, key=lambda x: distance(str(ocr_text), str(x)))
            dist = distance(str(ocr_text), str(closest_match))
            threshold = 1 if len(ocr_text) <= 5 else 2

            if dist <= threshold and len(ocr_text) > 2:
                logging.info(f"✓ 유사도 보정: '{ocr_text}' → '{closest_match}' (거리: {dist})")
                return closest_match
        except (TypeError, ValueError) as e:
            logging.error(f"보정 오류: {e}")

        return ocr_text

    def learn_correction(self, field, ocr_raw, user_corrected):
        """OCR 원본 → 사용자 보정 값 학습"""
        if not ocr_raw or not user_corrected:
            return

        if ocr_raw == user_corrected:
            return

        if 'corrections' not in self.knowledge_base:
            self.knowledge_base['corrections'] = {}

        if field not in self.knowledge_base['corrections']:
            self.knowledge_base['corrections'][field] = {}

        self.knowledge_base['corrections'][field][ocr_raw] = user_corrected
        logging.info(f"✓ 학습: {field} | '{ocr_raw}' → '{user_corrected}'")

    def learn_value(self, field, value):
        """정답 값 학습"""
        if not value or value in ["OCR Error", "OCR Timeout", "추출 실패"]:
            return

        if 'values' not in self.knowledge_base:
            self.knowledge_base['values'] = {}

        base_field = field.split('_')[0]

        for f in [field, base_field]:
            if f not in self.knowledge_base['values']:
                self.knowledge_base['values'][f] = []

            if value not in self.knowledge_base['values'][f]:
                self.knowledge_base['values'][f].append(value)

    # === 결과 저장 ===
    def save_results(self):
        """
        결과 저장 및 학습
        ★★★ v6.5: 수기 입력값 보존하되 스프레드시트 보정 적용 ★★★
        """
        if self.current_file_index < 0:
            self.status_label.configure(text="저장할 파일 없음", text_color="orange")
            return

        output_data = {field: entry.get() for field, entry in self.data_entries.items()}

        current_pdf_path = self.input_files[self.current_file_index]
        filename = os.path.basename(current_pdf_path)

        # ★★★ v6.5: 스프레드시트로 최종 검증 (단, 수기 입력값 보존 모드) ★★★
        corrected_data = self.spreadsheet_validator.validate_and_correct(
            filename,
            output_data,
            preserve_manual_edits=True  # 사용자가 이미 입력한 값은 건드리지 않음
        )

        # 화면에 보정된 값 업데이트 (사용자가 입력하지 않은 필드만)
        updates_made = False
        for field, corrected_value in corrected_data.items():
            if field in self.data_entries:
                original_value = output_data.get(field, '')

                # 사용자가 수기로 입력한 값이 아니거나, 오류 값인 경우만 업데이트
                if original_value in ["", "None", "OCR Error", "OCR Timeout", "추출 실패", "❌ 페이지 없음"]:
                    if str(corrected_value) != str(original_value):
                        self.data_entries[field].delete(0, "end")
                        self.data_entries[field].insert(0, corrected_value)
                        updates_made = True

        if updates_made:
            logging.info("🔄 화면 업데이트: 스프레드시트 보정값 반영 (수기 입력은 보존)")

        # 학습
        for field, final_value in corrected_data.items():
            if not final_value or final_value in ["OCR Error", "OCR Timeout", "추출 실패"]:
                continue

            auto_value = self.ocr_raw_results.get(field, '')
            manual_value = self.manual_extraction_results.get(field, '')

            if auto_value and manual_value and auto_value != manual_value:
                self.learn_correction(field, auto_value, manual_value)

            if manual_value and final_value and manual_value != final_value:
                self.learn_correction(field, manual_value, final_value)

            if auto_value and final_value and auto_value != final_value:
                self.learn_correction(field, auto_value, final_value)

            self.learn_value(field, final_value)

            if self.adaptive_preprocessor:
                field_type = self.get_field_type_for_preprocessing(field)
                self.adaptive_preprocessor.learn_success(field_type, auto_value, final_value)

        self.save_knowledge_base()

        output_filename = os.path.splitext(filename)[0] + '_result.json'

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(corrected_data, f, indent=4, ensure_ascii=False)

            import time
            self.completed_files[current_pdf_path] = {
                'data': corrected_data,
                'timestamp': time.time()
            }

            if self.template_name:
                self.template_recommender.record_usage(current_pdf_path, self.template_name)

            self.save_workspace_state()
            self.update_file_button_status()

            self.save_to_excel(corrected_data, current_pdf_path)

            self.status_label.configure(
                text=f"✓ 저장 완료!\n{filename}",
                text_color="green"
            )

        except Exception as e:
            logging.error(f"결과 저장 실패: {e}")
            self.status_label.configure(text=f"저장 실패: {e}", text_color="red")

    def save_to_excel(self, data, pdf_path):
        """Excel 파일에 데이터 추가"""
        try:
            import openpyxl
            from openpyxl import Workbook

            excel_file = os.path.join(OUTPUT_FOLDER, "WPS_추출결과.xlsx")

            if not os.path.exists(excel_file):
                wb = Workbook()
                ws = wb.active
                ws.title = "WPS Data"
                headers = ['PDF파일명', '추출일시'] + list(data.keys())
                ws.append(headers)
            else:
                wb = openpyxl.load_workbook(excel_file)
                ws = wb.active

            import datetime
            row_data = [
                           os.path.basename(pdf_path),
                           datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                       ] + list(data.values())

            ws.append(row_data)
            wb.save(excel_file)
            logging.info(f"Excel 저장 완료: {excel_file}")

        except ImportError:
            logging.warning("openpyxl 라이브러리가 없습니다. pip install openpyxl")
        except Exception as e:
            logging.error(f"Excel 저장 실패: {e}")

    def update_file_button_status(self):
        """파일 버튼 상태 업데이트"""
        start_idx = self.file_list_page * FILES_PER_PAGE

        for i, btn in enumerate(self.file_list_buttons):
            actual_idx = start_idx + i
            if actual_idx >= len(self.input_files):
                break

            file_path = self.input_files[actual_idx]
            is_completed = file_path in self.completed_files

            file_num = actual_idx + 1
            btn_text = f"{file_num}. " + ("✓ " if is_completed else "") + os.path.basename(file_path)
            btn.configure(text=btn_text)

            if actual_idx == self.current_file_index:
                btn.configure(fg_color="#1f6aa5", border_width=2, border_color="white")
            elif is_completed:
                btn.configure(fg_color="#2fa572", border_width=0)
            else:
                btn.configure(fg_color="gray", border_width=0)

        completed_count = len(self.completed_files)
        total_count = len(self.input_files)
        if total_count > 0:
            percentage = (completed_count / total_count) * 100
            self.progress_label.configure(
                text=f"✓ 완료 {completed_count}/{total_count} ({percentage:.1f}%)"
            )

    # === PDF 네비게이션 ===
    def next_pdf(self):
        """다음 PDF"""
        if self.current_file_index < len(self.input_files) - 1:
            self.jump_to_pdf(self.current_file_index + 1)

    def prev_pdf(self):
        """이전 PDF"""
        if self.current_file_index > 0:
            self.jump_to_pdf(self.current_file_index - 1)

    def prev_page(self):
        """이전 페이지"""
        if self.pdf_doc and self.current_page > 0:
            self.current_page -= 1
            self.display_page()

    def next_page(self):
        """다음 페이지"""
        if self.pdf_doc and self.current_page < len(self.pdf_doc) - 1:
            self.current_page += 1
            self.display_page()

    # === 템플릿 좌표 시각화 ===
    def draw_template_rects(self):
        """템플릿 좌표 시각화"""
        self.template_rects = {}
        self.template_labels = {}

        if not self.template_data:
            return

        for field, info in self.template_data.items():
            if info.get('page') != self.current_page:
                continue

            rect = info.get('rect')
            if not rect:
                continue

            if field not in self.data_entries:
                continue

            x0, y0, x1, y1 = rect
            x0_canvas = x0 * self.zoom_level
            y0_canvas = y0 * self.zoom_level
            x1_canvas = x1 * self.zoom_level
            y1_canvas = y1 * self.zoom_level

            rect_id = self.canvas.create_rectangle(
                x0_canvas, y0_canvas, x1_canvas, y1_canvas,
                outline="red", width=3, tags=("template_rect", f"rect_{field}")
            )
            self.template_rects[field] = rect_id

            label_text = field
            label_bg = self.canvas.create_rectangle(
                x0_canvas - 2, y0_canvas - 22,
                x0_canvas + len(label_text) * 7 + 4, y0_canvas - 2,
                fill="black", outline="red", width=1,
                tags=("template_label_bg", f"label_bg_{field}")
            )

            label_id = self.canvas.create_text(
                x0_canvas + 2, y0_canvas - 12,
                text=label_text,
                fill="white",
                anchor="w",
                font=("Arial", 10, "bold"),
                tags=("template_label", f"label_{field}")
            )
            self.template_labels[field] = label_id

            # ★ 핸들 크기 증가 (4 → 6) - 클릭하기 쉽게
            handle_size = 6
            handle_id = self.canvas.create_rectangle(
                x1_canvas - handle_size, y1_canvas - handle_size,
                x1_canvas + handle_size, y1_canvas + handle_size,
                fill="red", outline="yellow", width=2,
                tags=("resize_handle", f"handle_{field}")
            )

    def get_field_at_pos(self, x, y):
        """해당 위치의 필드명 반환"""
        if self.get_resize_field_at_pos(x, y):
            return None

        tolerance = 8

        for field, rect_id in self.template_rects.items():
            coords = self.canvas.coords(rect_id)
            if coords is None or len(coords) != 4:
                continue

            x0, y0, x1, y1 = coords
            handle_size = 6  # ★ handle_size 업데이트 (4 → 6)
            if (x0 - tolerance <= x <= x1 - handle_size + tolerance and
                    y0 - tolerance <= y <= y1 - handle_size + tolerance):
                return field
        return None

    def get_resize_field_at_pos(self, x, y):
        """리사이즈 핸들 필드 반환"""
        tolerance = 5  # ★ 핸들 인식 범위 최소화 (25 → 5)
        items = self.canvas.find_overlapping(x - tolerance, y - tolerance, x + tolerance, y + tolerance)
        for item in items:
            tags = self.canvas.gettags(item)
            for tag in tags:
                if tag.startswith("handle_"):
                    return tag.replace("handle_", "")
        return None

    def on_hover(self, event):
        """마우스 호버 시 커서 변경"""
        if not self.template_data:
            self.canvas.config(cursor="")
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        if self.get_resize_field_at_pos(x, y):
            self.canvas.config(cursor="bottom_right_corner")
        elif self.get_field_at_pos(x, y):
            self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="")

    # === 캔버스 표시 ===
    def display_page(self, fit_to_screen=False):
        """PDF 페이지 표시"""
        if not self.pdf_doc:
            return

        page = self.pdf_doc.load_page(self.current_page)

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

        mat = fitz.Matrix(self.zoom_level, self.zoom_level)
        pix = page.get_pixmap(matrix=mat)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        tk_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
        self.canvas.image = tk_image

        self.draw_template_rects()

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.update_page_label()

        if self.input_files and self.current_file_index != -1:
            file_path = self.input_files[self.current_file_index]
            is_completed = file_path in self.completed_files
            filename = ("✓ " if is_completed else "") + os.path.basename(file_path)

            self.file_label.configure(text=filename)
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
            text=f"PDF {pdf_current}/{pdf_total} | 페이지 {page_current}/{page_total}"
        )

    # === 마우스 이벤트 ===
    def on_zoom(self, event):
        """마우스 휠 줌"""
        if not self.pdf_doc:
            return

        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_level *= factor
        self.display_page()

    def on_press(self, event):
        """마우스 누름"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        if self.template_data:
            resize_field = self.get_resize_field_at_pos(x, y)
            if resize_field:
                self.rect_resize_mode = 'resize'
                self.editing_field = resize_field
                self.rect_drag_start = (x, y)
                rect_id = self.template_rects.get(resize_field)
                if rect_id is not None:
                    self.rect_original_coords = self.canvas.coords(rect_id)
                    if self.rect_original_coords:
                        self.status_label.configure(
                            text=f"🔧 {resize_field} 크기 조절 중...",
                            text_color="yellow"
                        )
                        return

            move_field = self.get_field_at_pos(x, y)
            if move_field:
                self.rect_resize_mode = 'move'
                self.editing_field = move_field
                self.rect_drag_start = (x, y)
                rect_id = self.template_rects.get(move_field)
                if rect_id is not None:
                    self.rect_original_coords = self.canvas.coords(rect_id)
                    if self.rect_original_coords:
                        self.status_label.configure(
                            text=f"🎯 {move_field} 이동 중...",
                            text_color="cyan"
                        )
                        return

        if not self.selected_field:
            self.status_label.configure(
                text="⚠️ 먼저 라벨 클릭",
                text_color="orange"
            )
            return

        self.canvas.focus_set()
        self.rect_start_pos = (x, y)

        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)

        self.current_rect_id = self.canvas.create_rectangle(
            self.rect_start_pos[0], self.rect_start_pos[1],
            self.rect_start_pos[0], self.rect_start_pos[1],
            outline="blue", width=2, tags="new_rect"
        )

    def on_drag(self, event):
        """마우스 드래그"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        if self.rect_resize_mode and self.editing_field:
            if self.rect_drag_start is None or self.rect_original_coords is None:
                return

            dx = x - self.rect_drag_start[0]
            dy = y - self.rect_drag_start[1]

            rect_id = self.template_rects.get(self.editing_field)
            if rect_id is None:
                return

            x0, y0, x1, y1 = self.rect_original_coords

            if self.rect_resize_mode == 'move':
                self.canvas.coords(rect_id, x0 + dx, y0 + dy, x1 + dx, y1 + dy)
                label_id = self.template_labels.get(self.editing_field)
                if label_id:
                    self.canvas.coords(label_id, x0 + dx, y0 + dy - 5)

            elif self.rect_resize_mode == 'resize':
                new_x1 = x1 + dx
                new_y1 = y1 + dy
                if new_x1 > x0 + 20 and new_y1 > y0 + 20:
                    self.canvas.coords(rect_id, x0, y0, new_x1, new_y1)

            return

        if not self.rect_start_pos:
            return

        self.canvas.coords(
            self.current_rect_id,
            self.rect_start_pos[0], self.rect_start_pos[1],
            x, y
        )

    def on_release(self, event):
        """마우스 릴리즈"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        if self.rect_resize_mode and self.editing_field:
            field = self.editing_field
            rect_id = self.template_rects.get(field)
            if rect_id is None:
                self.rect_resize_mode = None
                self.editing_field = None
                return

            coords = self.canvas.coords(rect_id)
            if coords is None or len(coords) != 4:
                self.rect_resize_mode = None
                self.editing_field = None
                return

            x0, y0, x1, y1 = coords
            x0_pdf = int(x0 / self.zoom_level)
            y0_pdf = int(y0 / self.zoom_level)
            x1_pdf = int(x1 / self.zoom_level)
            y1_pdf = int(y1 / self.zoom_level)

            new_rect = (x0_pdf, y0_pdf, x1_pdf, y1_pdf)

            old_rect = self.template_data[field]['rect']
            if old_rect != new_rect:
                self.template_data[field]['rect'] = new_rect

                if field not in self.template_coords_variations:
                    self.template_coords_variations[field] = []
                if old_rect not in self.template_coords_variations[field]:
                    self.template_coords_variations[field].append(old_rect)
                if new_rect not in self.template_coords_variations[field]:
                    self.template_coords_variations[field].append(new_rect)

                self.save_workspace_state()

                self.status_label.configure(
                    text=f"✓ {field} 좌표 학습 완료",
                    text_color="green"
                )

                if self.pdf_doc:
                    try:
                        page = self.pdf_doc.load_page(self.current_page)
                        config = self.get_ocr_config(field)
                        ocr_result = self.ocr_from_area_direct(page, new_rect, config, field)

                        self.manual_extraction_results[field] = ocr_result

                        filtered = self.apply_whitelist_filter(ocr_result, field)
                        if 'Preheat' in field or 'Temp' in field:
                            filtered = self.preprocess_preheat(filtered)
                        corrected = self.intelligent_correction(field, filtered)

                        if field in self.data_entries:
                            self.data_entries[field].delete(0, "end")
                            self.data_entries[field].insert(0, corrected)

                    except Exception as e:
                        logging.error(f"OCR 재추출 실패: {e}")

            self.rect_resize_mode = None
            self.editing_field = None
            self.rect_drag_start = None
            self.rect_original_coords = None

            self.display_page()
            return

        if not self.selected_field or not self.rect_start_pos:
            return

        end_pos = (x, y)

        x0_img = min(self.rect_start_pos[0], end_pos[0]) / self.zoom_level
        y0_img = min(self.rect_start_pos[1], end_pos[1]) / self.zoom_level
        x1_img = max(self.rect_start_pos[0], end_pos[0]) / self.zoom_level
        y1_img = max(self.rect_start_pos[1], end_pos[1]) / self.zoom_level

        if not self.template_data:
            self.template_data = {}

        rect = (int(x0_img), int(y0_img), int(x1_img), int(y1_img))

        old_rect = self.template_data.get(self.selected_field, {}).get('rect')

        self.template_data[self.selected_field] = {
            'page': self.current_page,
            'rect': rect
        }

        if old_rect and old_rect != rect:
            if self.selected_field not in self.template_coords_variations:
                self.template_coords_variations[self.selected_field] = []

            if rect not in self.template_coords_variations[self.selected_field]:
                self.template_coords_variations[self.selected_field].append(rect)

        self.status_label.configure(
            text=f"✓ {self.selected_field}\nOCR 실행 중...",
            text_color="green"
        )

        self.save_workspace_state()

        field_to_extract = self.selected_field
        self.selected_field = None

        if self.pdf_doc:
            try:
                page = self.pdf_doc.load_page(self.current_page)
                config = self.get_ocr_config(field_to_extract)
                ocr_result = self.ocr_from_area_direct(page, rect, config, field_to_extract)

                self.manual_extraction_results[field_to_extract] = ocr_result

                if field_to_extract in self.data_entries:
                    self.data_entries[field_to_extract].delete(0, "end")
                    self.data_entries[field_to_extract].insert(0, ocr_result)

                self.status_label.configure(
                    text=f"✓ {field_to_extract}\n추출: {ocr_result[:20]}...",
                    text_color="green"
                )
            except Exception as e:
                logging.error(f"즉시 OCR 실패: {e}")

        self.display_page()

        if field_to_extract == 'Welding_Process_Type':
            self.after(100, self.generate_dynamic_fields)

    # === 종료 시 정리 ===
    def destroy(self):
        """앱 종료 시 리소스 정리"""
        self.save_workspace_state()

        if self.adaptive_preprocessor:
            self.adaptive_preprocessor.save_profile()

        if self.pdf_doc:
            self.pdf_doc.close()

        super().destroy()


if __name__ == "__main__":
    if not OPENCV_AVAILABLE:
        print("\n" + "=" * 60)
        print("⚠️  OpenCV가 설치되지 않았습니다.")
        print("   전처리 기능 없이 기본 OCR만 사용합니다.")
        print("\n   더 나은 OCR 정확도를 위해 OpenCV 설치를 권장합니다:")
        print("   pip install opencv-python numpy")
        print("=" * 60 + "\n")

    app = WorkbenchApp()
    app.mainloop()
