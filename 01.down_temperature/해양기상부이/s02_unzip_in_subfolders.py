# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:09:42 2026

@author: user
"""

import zipfile
import os

def extract_zips_in_subfolders():
    # 1. 현재 파이썬 파일이 있는 위치를 시작점으로 설정
    root_dir = os.getcwd()
    print(f"탐색 시작 디렉토리: {root_dir}")
    print("-" * 50)

    # 2. os.walk를 이용해 모든 하위 폴더를 순회
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 확장자가 .zip인 파일만 선택
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                # 압축 파일 이름과 동일한 폴더명 생성 (확장자 제외)
                folder_name = os.path.splitext(file)[0]
                extract_path = os.path.join(root, folder_name)

                print(f"압축 해제 중: {zip_path}")

                # 압축 해제용 폴더가 없으면 생성
                if not os.path.exists(extract_path):
                    os.makedirs(extract_path)

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    print(f"   => 완료: {extract_path}")
                except Exception as e:
                    print(f"   => 실패: {file} 에러 발생: {e}")

    print("-" * 50)
    print("모든 하위 폴더의 압축 해제 작업이 끝났습니다.")

if __name__ == "__main__":
    extract_zips_in_subfolders()