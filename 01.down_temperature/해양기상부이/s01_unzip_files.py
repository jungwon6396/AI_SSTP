# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import zipfile
import os

def extract_all_zips():
    # 1. 현재 파이썬 파일이 실행되는 경로를 가져옵니다.
    current_dir = os.getcwd()
    print(f"현재 작업 디렉토리: {current_dir}")

    # 2. 현재 폴더 내의 모든 파일을 리스트업합니다.
    files = os.listdir(current_dir)

    # 3. 파일 확장자가 .zip인 것만 골라냅니다.
    zip_files = [f for f in files if f.endswith('.zip')]

    if not zip_files:
        print("압축 파일(.zip)을 찾을 수 없습니다.")
        return

    for zip_file in zip_files:
        # 파일명에서 확장자를 제외한 이름으로 폴더 생성 (예: data.zip -> data 폴더)
        folder_name = os.path.splitext(zip_file)[0]
        extract_path = os.path.join(current_dir, folder_name)

        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # 압축 해제 시 한글 깨짐 방지 (CP437 -> UTF-8 변환이 필요한 경우 대비)
                zip_ref.extractall(extract_path)
                print(f"성공: [{zip_file}] -> [{folder_name}] 폴더에 해제됨")
        except Exception as e:
            print(f"실패: [{zip_file}] 해제 중 오류 발생: {e}")

if __name__ == "__main__":
    extract_all_zips()