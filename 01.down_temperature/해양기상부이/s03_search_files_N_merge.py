# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:59:01 2026

@author: user
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
from datetime import datetime

def precise_classify_by_row(meta_path, data_root='old_data', output_root='merge_data'):
    # --- [Step 1: 메타데이터 규칙 로드] ---
    try:
        meta_df = pd.read_csv(meta_path, encoding='cp949', skipinitialspace=True)
    except:
        meta_df = pd.read_csv(meta_path, encoding='utf-8', skipinitialspace=True)

    # 지점별 규칙 정리
    rules = []
    for _, row in meta_df.iterrows():
        try:
            rules.append({
                'id': int(row.iloc[0]),
                'start': pd.to_datetime(row.iloc[1]),
                'end': pd.to_datetime(row.iloc[2]) if pd.notnull(row.iloc[2]) else datetime(2099, 12, 31),
                'name': str(row.iloc[3]).strip()
            })
        except: continue

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 지점명별로 데이터를 모으기 위한 딕셔너리
    collected_data = {} 

    # --- [Step 2: 모든 파일의 행을 읽어서 판별] ---
    print(f"🚀 데이터를 행 단위로 정밀 분석 중입니다. 잠시만 기다려 주세요...")

    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.csv') and 'META' not in file:
                file_path = os.path.join(root, file)
                try:
                    # 파일 전체 로드
                    df = pd.read_csv(file_path, encoding='cp949')
                    if df.empty or '일시' not in df.columns: continue
                    
                    df['일시'] = pd.to_datetime(df['일시'])
                    f_id = int(df.iloc[0, 0])
                    
                    # 해당 ID를 가진 메타 규칙들만 필터링
                    relevant_rules = [r for r in rules if r['id'] == f_id]
                    
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