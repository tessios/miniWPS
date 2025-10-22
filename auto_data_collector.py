"""
자동 데이터 수집 시스템
외부 소스에서 용접 관련 비디오/이미지를 자동으로 수집합니다.
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import time
from urllib.parse import urljoin, urlparse
import hashlib


class AutoDataCollector:
    """
    자동 데이터 수집기
    다양한 소스에서 용접 관련 데이터를 자동으로 수집합니다.
    """

    def __init__(self, output_dir: str = 'collected_data'):
        """
        Args:
            output_dir: 수집된 데이터를 저장할 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 하위 디렉토리 생성
        self.video_dir = self.output_dir / 'videos'
        self.image_dir = self.output_dir / 'images'
        self.metadata_dir = self.output_dir / 'metadata'

        for dir_path in [self.video_dir, self.image_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)

        # 수집 기록
        self.collection_log = []
        self.downloaded_urls = set()

        print(f"✓ 데이터 수집기 초기화 완료: {self.output_dir}")

    def collect_from_youtube(self, search_query: str = "welding process",
                            max_videos: int = 10):
        """
        YouTube에서 용접 관련 비디오 수집

        Args:
            search_query: 검색 쿼리
            max_videos: 최대 다운로드 개수
        """
        print(f"\n[YouTube 수집] 검색어: '{search_query}'")
        print("-" * 70)

        try:
            import yt_dlp
        except ImportError:
            print("⚠️  yt-dlp가 설치되지 않았습니다.")
            print("   설치 방법: pip install yt-dlp")
            return []

        ydl_opts = {
            'format': 'best[height<=720]',  # 720p 이하
            'outtmpl': str(self.video_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'max_downloads': max_videos,
        }

        search_url = f"ytsearch{max_videos}:{search_query}"

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"  검색 중...")
                info = ydl.extract_info(search_url, download=False)

                if 'entries' in info:
                    downloaded = []
                    for i, entry in enumerate(info['entries'][:max_videos]):
                        try:
                            video_id = entry['id']
                            title = entry.get('title', 'Unknown')
                            duration = entry.get('duration', 0)

                            print(f"  [{i+1}/{max_videos}] {title[:50]}...")

                            # 다운로드
                            ydl.download([entry['webpage_url']])

                            metadata = {
                                'source': 'youtube',
                                'video_id': video_id,
                                'title': title,
                                'duration': duration,
                                'url': entry['webpage_url'],
                                'timestamp': time.time()
                            }

                            downloaded.append(metadata)
                            self.collection_log.append(metadata)

                        except Exception as e:
                            print(f"  ✗ 다운로드 실패: {e}")

                    print(f"\n✓ {len(downloaded)}개 비디오 다운로드 완료")
                    return downloaded

        except Exception as e:
            print(f"✗ YouTube 수집 실패: {e}")
            return []

    def collect_from_pexels(self, query: str = "welding", max_items: int = 20):
        """
        Pexels (무료 스톡 사이트)에서 이미지/비디오 수집

        Args:
            query: 검색 쿼리
            max_items: 최대 다운로드 개수

        Note: API 키가 필요합니다.
              https://www.pexels.com/api/ 에서 무료로 발급 가능
        """
        print(f"\n[Pexels 수집] 검색어: '{query}'")
        print("-" * 70)

        # API 키 확인
        api_key = os.environ.get('PEXELS_API_KEY')
        if not api_key:
            print("⚠️  PEXELS_API_KEY 환경 변수가 설정되지 않았습니다.")
            print("   설정 방법:")
            print("   1. https://www.pexels.com/api/ 에서 무료 API 키 발급")
            print("   2. export PEXELS_API_KEY='your_api_key'")
            return []

        headers = {'Authorization': api_key}

        # 비디오 검색
        video_url = f"https://api.pexels.com/videos/search"
        params = {'query': query, 'per_page': max_items}

        try:
            response = requests.get(video_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            downloaded = []
            for i, video in enumerate(data.get('videos', [])[:max_items]):
                try:
                    video_id = video['id']

                    # 가장 낮은 화질 선택 (빠른 다운로드)
                    video_files = video.get('video_files', [])
                    if not video_files:
                        continue

                    # HD 화질 선택
                    hd_file = next((f for f in video_files if f.get('quality') == 'hd'), video_files[0])
                    download_url = hd_file['link']

                    print(f"  [{i+1}/{len(data['videos'])}] 다운로드 중...")

                    # 다운로드
                    video_response = requests.get(download_url, stream=True)
                    video_response.raise_for_status()

                    filename = f"pexels_{video_id}.mp4"
                    filepath = self.video_dir / filename

                    with open(filepath, 'wb') as f:
                        for chunk in video_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    metadata = {
                        'source': 'pexels',
                        'video_id': video_id,
                        'url': video['url'],
                        'duration': video.get('duration', 0),
                        'width': video.get('width', 0),
                        'height': video.get('height', 0),
                        'filename': filename,
                        'timestamp': time.time()
                    }

                    downloaded.append(metadata)
                    self.collection_log.append(metadata)

                except Exception as e:
                    print(f"  ✗ 다운로드 실패: {e}")

            print(f"\n✓ {len(downloaded)}개 비디오 다운로드 완료")
            return downloaded

        except Exception as e:
            print(f"✗ Pexels 수집 실패: {e}")
            return []

    def collect_from_url_list(self, urls: List[str], label: Optional[str] = None):
        """
        URL 리스트에서 직접 다운로드

        Args:
            urls: 다운로드할 URL 리스트
            label: 레이블 (정상/결함)
        """
        print(f"\n[직접 다운로드] {len(urls)}개 URL")
        print("-" * 70)

        downloaded = []

        for i, url in enumerate(urls):
            try:
                # 이미 다운로드한 URL은 스킵
                url_hash = hashlib.md5(url.encode()).hexdigest()
                if url_hash in self.downloaded_urls:
                    print(f"  [{i+1}/{len(urls)}] 이미 다운로드됨, 스킵")
                    continue

                print(f"  [{i+1}/{len(urls)}] 다운로드 중...")

                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                # 파일 타입 확인
                content_type = response.headers.get('content-type', '')

                if 'video' in content_type:
                    ext = '.mp4'
                    save_dir = self.video_dir
                elif 'image' in content_type:
                    ext = '.jpg'
                    save_dir = self.image_dir
                else:
                    # URL에서 확장자 추측
                    parsed = urlparse(url)
                    ext = Path(parsed.path).suffix or '.mp4'
                    save_dir = self.video_dir if ext in ['.mp4', '.avi', '.mov'] else self.image_dir

                filename = f"{url_hash}{ext}"
                filepath = save_dir / filename

                # 다운로드
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = filepath.stat().st_size / (1024 * 1024)  # MB
                print(f"    ✓ 저장 완료: {filename} ({file_size:.2f} MB)")

                metadata = {
                    'source': 'direct_url',
                    'url': url,
                    'filename': filename,
                    'label': label,
                    'file_size_mb': file_size,
                    'timestamp': time.time()
                }

                downloaded.append(metadata)
                self.collection_log.append(metadata)
                self.downloaded_urls.add(url_hash)

                # 예의상 대기
                time.sleep(1)

            except Exception as e:
                print(f"  ✗ 다운로드 실패: {e}")

        print(f"\n✓ {len(downloaded)}개 파일 다운로드 완료")
        return downloaded

    def collect_from_public_datasets(self):
        """
        공개 데이터셋에서 수집
        """
        print(f"\n[공개 데이터셋 수집]")
        print("-" * 70)

        datasets = {
            'roboflow_welding': {
                'description': 'Roboflow 용접 결함 데이터셋',
                'type': 'image',
                'note': 'Roboflow API 키 필요'
            },
            'kaggle_welding': {
                'description': 'Kaggle 용접 데이터셋',
                'type': 'mixed',
                'note': 'Kaggle API 설정 필요'
            }
        }

        print("사용 가능한 공개 데이터셋:")
        for name, info in datasets.items():
            print(f"  • {name}: {info['description']}")
            print(f"    타입: {info['type']}, {info['note']}")

        print("\n⚠️  자동 다운로드를 위해서는 각 플랫폼의 API 키가 필요합니다.")
        print("   수동으로 다운로드 후 collect_from_local_folder() 사용을 권장합니다.")

    def collect_from_local_folder(self, folder_path: str,
                                  pattern: str = "*.mp4",
                                  label: Optional[str] = None):
        """
        로컬 폴더에서 파일 수집 (복사)

        Args:
            folder_path: 소스 폴더 경로
            pattern: 파일 패턴 (예: "*.mp4", "*.jpg")
            label: 레이블
        """
        print(f"\n[로컬 폴더 수집] {folder_path}")
        print("-" * 70)

        source_path = Path(folder_path)
        if not source_path.exists():
            print(f"✗ 폴더가 존재하지 않습니다: {folder_path}")
            return []

        files = list(source_path.glob(pattern))
        print(f"  발견된 파일: {len(files)}개")

        imported = []
        for i, file_path in enumerate(files):
            try:
                # 파일 타입에 따라 저장 위치 결정
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    dest_dir = self.video_dir
                else:
                    dest_dir = self.image_dir

                dest_path = dest_dir / file_path.name

                # 이미 존재하면 스킵
                if dest_path.exists():
                    print(f"  [{i+1}/{len(files)}] 이미 존재함, 스킵: {file_path.name}")
                    continue

                # 복사
                import shutil
                shutil.copy2(file_path, dest_path)

                print(f"  [{i+1}/{len(files)}] 복사 완료: {file_path.name}")

                metadata = {
                    'source': 'local_folder',
                    'original_path': str(file_path),
                    'filename': file_path.name,
                    'label': label,
                    'timestamp': time.time()
                }

                imported.append(metadata)
                self.collection_log.append(metadata)

            except Exception as e:
                print(f"  ✗ 복사 실패: {e}")

        print(f"\n✓ {len(imported)}개 파일 가져오기 완료")
        return imported

    def save_collection_log(self, filename: str = 'collection_log.json'):
        """수집 기록 저장"""
        log_path = self.metadata_dir / filename

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.collection_log, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 수집 기록 저장 완료: {log_path}")
        print(f"  총 {len(self.collection_log)}개 항목")

    def get_statistics(self):
        """수집 통계"""
        video_count = len(list(self.video_dir.glob('*')))
        image_count = len(list(self.image_dir.glob('*')))

        # 파일 크기 계산
        total_size = 0
        for file in self.video_dir.glob('*'):
            total_size += file.stat().st_size
        for file in self.image_dir.glob('*'):
            total_size += file.stat().st_size

        total_size_mb = total_size / (1024 * 1024)

        stats = {
            'video_count': video_count,
            'image_count': image_count,
            'total_count': video_count + image_count,
            'total_size_mb': total_size_mb,
            'collection_entries': len(self.collection_log)
        }

        return stats

    def print_summary(self):
        """수집 요약 출력"""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("데이터 수집 요약")
        print("=" * 70)
        print(f"  비디오: {stats['video_count']}개")
        print(f"  이미지: {stats['image_count']}개")
        print(f"  총 파일: {stats['total_count']}개")
        print(f"  총 크기: {stats['total_size_mb']:.2f} MB")
        print(f"  수집 기록: {stats['collection_entries']}개 항목")
        print("=" * 70)


def main():
    """사용 예제"""
    print("=" * 70)
    print("자동 데이터 수집 시스템")
    print("=" * 70)

    collector = AutoDataCollector(output_dir='welding_data')

    print("\n사용 가능한 수집 방법:")
    print("  1. YouTube 검색")
    print("  2. Pexels 스톡 비디오")
    print("  3. 직접 URL 리스트")
    print("  4. 로컬 폴더")
    print("  5. 공개 데이터셋")

    # 예제 1: 로컬 폴더에서 수집 (가장 간단함)
    print("\n[예제] 로컬 폴더에서 수집")
    print("  사용법:")
    print("  collector.collect_from_local_folder(")
    print("      folder_path='/path/to/videos',")
    print("      pattern='*.mp4',")
    print("      label='normal'  # or 'defect'")
    print("  )")

    # 예제 2: URL 리스트
    print("\n[예제] URL 리스트에서 직접 다운로드")
    print("  urls = [")
    print("      'https://example.com/welding_video1.mp4',")
    print("      'https://example.com/welding_video2.mp4',")
    print("  ]")
    print("  collector.collect_from_url_list(urls, label='normal')")

    # 예제 3: YouTube (yt-dlp 필요)
    print("\n[예제] YouTube에서 수집 (yt-dlp 설치 필요)")
    print("  collector.collect_from_youtube(")
    print("      search_query='welding process',")
    print("      max_videos=10")
    print("  )")

    # 예제 4: Pexels (API 키 필요)
    print("\n[예제] Pexels에서 수집 (API 키 필요)")
    print("  collector.collect_from_pexels(")
    print("      query='welding',")
    print("      max_items=20")
    print("  )")

    # 통계 표시
    collector.print_summary()

    print("\n✓ 데이터 수집 후 다음 단계:")
    print("  1. 수집된 데이터 확인: collected_data/ 폴더")
    print("  2. 레이블링: labeling_tool.py 사용")
    print("  3. 학습: example_usage.py의 example_2 참고")


if __name__ == "__main__":
    main()
