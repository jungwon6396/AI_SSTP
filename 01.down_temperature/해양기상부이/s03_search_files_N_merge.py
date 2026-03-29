# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:59:01 2026

@author: user
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_csv_with_fallback(path, **kwargs):
    last_error = None
    for encoding in ("cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    return pd.read_csv(path, **kwargs)


def precise_classify_by_row(meta_path, data_root='old_data', output_root='merge_data'):
    meta_path = os.path.join(BASE_DIR, meta_path)
    data_root = os.path.join(BASE_DIR, data_root)
    output_root = os.path.join(BASE_DIR, output_root)

    # --- [Step 1: 메타데이터 규칙 로드] ---
    meta_df = read_csv_with_fallback(meta_path, skipinitialspace=True)

    # 지점별 규칙 정리
    rules = []
    for _, row in meta_df.iterrows():
        try:
            rules.append({
                'id': int(row["지점"]),
                'start': pd.to_datetime(row["시작일"], errors="raise"),
                'end': pd.to_datetime(row["종료일"], errors="raise") if pd.notnull(row["종료일"]) else datetime(2099, 12, 31),
                'name': str(row["지점명"]).strip()
            })
        except (KeyError, TypeError, ValueError):
            continue

    os.makedirs(output_root, exist_ok=True)

    # 지점명별로 데이터를 모으기 위한 딕셔너리
    collected_data = {}

    # --- [Step 2: 모든 파일의 행을 읽어서 판별] ---
    print(f"🚀 데이터를 행 단위로 정밀 분석 중입니다. 잠시만 기다려 주세요...")

    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.csv') and 'META' not in file:
                file_path = os.path.join(root, file)
                try:
                    # 파일 전체 로드
                    df = read_csv_with_fallback(file_path)
                    if df.empty or '일시' not in df.columns:
                        continue

                    df['일시'] = pd.to_datetime(df['일시'], errors="coerce")
                    df = df.dropna(subset=['일시'])
                    if df.empty:
                        continue

                    station_id_col = '지점' if '지점' in df.columns else df.columns[0]
                    f_id = int(df[station_id_col].iloc[0])

                    # 해당 ID를 가진 메타 규칙들만 필터링
                    relevant_rules = [r for r in rules if r['id'] == f_id]
                    if not relevant_rules:
                        continue

                    for r in relevant_rules:
                        # [핵심] 각 행의 날짜가 메타데이터 범위 내에 있는지 필터링
                        mask = (df['일시'] >= r['start']) & (df['일시'] <= r['end'])
                        matched_rows = df[mask]

                        if not matched_rows.empty:
                            s_name = r['name']
                            if s_name not in collected_data:
                                collected_data[s_name] = []
                            collected_data[s_name].append(matched_rows)

                except Exception as e:
                    print(f"⚠️ 오류 발생 ({file}): {e}")

    # --- [Step 3: 지점별 통합 및 저장] ---
    print(f"\n📂 지점별 파일 생성 중...")
    if not collected_data:
        print("⚠️ 통합할 데이터가 없습니다. 메타데이터/원천 데이터 경로와 인코딩을 확인하세요.")
        return

    for s_name, df_list in collected_data.items():
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        final_df = final_df.sort_values('일시').drop_duplicates() # 중복 제거 및 정렬

        output_path = os.path.join(output_root, f"{s_name}_통합데이터.csv")
        final_df.to_csv(output_path, index=False, encoding='cp949')

        # 정보 출력
        print(f"✅ {s_name:<15}: {len(final_df):>7,}건 통합 완료 ({final_df['일시'].min().date()} ~ {final_df['일시'].max().date()})")

    print(f"\n✨ 모든 작업이 완료되었습니다. 결과 폴더: {os.path.abspath(output_root)}")

if __name__ == "__main__":
    precise_classify_by_row('META_관측지점정보_해양기상부이.csv')
