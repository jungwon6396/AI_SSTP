# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:48:44 2026

@author: user
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import folium
from folium.features import DivIcon
import matplotlib.pyplot as plt
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_meta(meta_path):
    try:
        meta_df = pd.read_csv(meta_path, encoding='cp949', skipinitialspace=True)
    except UnicodeDecodeError:
        meta_df = pd.read_csv(meta_path, encoding='utf-8', skipinitialspace=True)

    meta_df = meta_df.copy()
    meta_df["지점"] = meta_df.iloc[:, 0].astype(str).str.strip()
    meta_df["지점명"] = meta_df.iloc[:, 3].astype(str).str.strip()
    meta_df["위도"] = pd.to_numeric(meta_df["위도"], errors="coerce")
    meta_df["경도"] = pd.to_numeric(meta_df["경도"], errors="coerce")

    meta_df = meta_df.dropna(subset=["지점명", "위도", "경도"]).copy()
    return meta_df


def get_station_name_from_filename(filename):
    base = os.path.splitext(filename)[0]
    return base.split("_")[0].strip()


def read_station_file(path):
    try:
        df = pd.read_csv(path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8")

    if df.empty:
        return df

    if "일시" in df.columns:
        df["일시"] = pd.to_datetime(df["일시"], errors="coerce")

    if "수온(°C)" in df.columns:
        df["수온(°C)"] = pd.to_numeric(df["수온(°C)"], errors="coerce")
        df["수온(°C)"] = df["수온(°C)"].replace([-99, -999, -9999], pd.NA)

    return df


def infer_time_step(df):
    """
    관측 시간 간격 추정
    가장 대표적인 시간 간격(중앙값)을 사용
    """
    time_series = df["일시"].dropna().sort_values().drop_duplicates()
    if len(time_series) < 2:
        return None

    diffs = time_series.diff().dropna()
    if diffs.empty:
        return None

    step = diffs.median()

    # 이상한 값 방지
    if pd.isna(step) or step <= pd.Timedelta(0):
        return None

    return step


def compute_sst_coverage(df, start_date, end_date):
    """
    start_date ~ end_date 구간에서
    기대 시각 대비 유효 수온값 비율 계산
    """
    if df.empty or "일시" not in df.columns or "수온(°C)" not in df.columns:
        return 0.0, 0, 0, None

    work = df.copy()
    work = work.dropna(subset=["일시"]).sort_values("일시")

    if work.empty:
        return 0.0, 0, 0, None

    # 기간을 먼저 자름
    work = work[(work["일시"] >= start_date) & (work["일시"] <= end_date)]
    if work.empty:
        return 0.0, 0, 0, None

    step = infer_time_step(work)
    if step is None:
        return 0.0, 0, 0, None

    # 기대 시각 생성
    expected_index = pd.date_range(start=start_date, end=end_date, freq=step)
    expected_count = len(expected_index)

    # 같은 시각 중복 제거 후 재색인
    work = work.drop_duplicates(subset=["일시"])
    work = work.set_index("일시").reindex(expected_index)

    valid_count = work["수온(°C)"].notna().sum()

    coverage = valid_count / expected_count if expected_count > 0 else 0.0
    return coverage, valid_count, expected_count, step


def select_good_stations(meta_df, data_dir, start_date, end_date, threshold=0.90):
    """
    10년 기간 동안 수온 유효값 비율이 threshold 이상인 지점만 선택
    """
    selected = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    print("📌 10년 자료 완전성(coverage) 평가 시작")
    print(f"   기간: {start_date.date()} ~ {end_date.date()}")
    print(f"   기준: 수온 유효값 비율 >= {threshold:.0%}")
    print("-" * 70)

    for file in files:
        try:
            station_name = get_station_name_from_filename(file)
            path = os.path.join(data_dir, file)
            df = read_station_file(path)

            if df.empty or "일시" not in df.columns or "수온(°C)" not in df.columns:
                print(f"⚠️ 제외: {station_name} (필수 컬럼 없음)")
                continue

            coverage, valid_count, expected_count, step = compute_sst_coverage(
                df, start_date, end_date
            )

            if step is None:
                print(f"⚠️ 제외: {station_name} (시간 간격 추정 실패)")
                continue

            if coverage >= threshold:
                meta_row = meta_df[meta_df["지점명"] == station_name]
                if meta_row.empty:
                    print(f"⚠️ 제외: {station_name} (메타 매칭 실패)")
                    continue

                selected.append({
                    "file": file,
                    "station_name": station_name,
                    "station_id": meta_row.iloc[0]["지점"],
                    "lat": float(meta_row.iloc[0]["위도"]),
                    "lon": float(meta_row.iloc[0]["경도"]),
                    "coverage": coverage,
                    "valid_count": valid_count,
                    "expected_count": expected_count,
                    "step": step
                })

                print(f"✅ 채택: {station_name:15s}  "
                      f"coverage={coverage:6.2%}  "
                      f"valid={valid_count:7d} / expected={expected_count:7d}  "
                      f"step={step}")

            else:
                print(f"⏭️ 제외: {station_name:15s}  "
                      f"coverage={coverage:6.2%}  "
                      f"valid={valid_count:7d} / expected={expected_count:7d}  "
                      f"step={step}")

        except Exception as e:
            print(f"⚠️ 오류: {file} -> {e}")

    print("-" * 70)
    print(f"총 채택 지점 수: {len(selected)}")
    return selected


def plot_station_map(selected_stations, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    m = folium.Map(
        location=[36.0, 127.5],
        zoom_start=7,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Esri Ocean'
    )

    for st in selected_stations:
        popup_html = f"""
        <div style='width:240px; font-family:Malgun Gothic; font-size:10pt;'>
            <b style='font-size:11pt; color:darkblue;'>{st['station_name']}</b><br>
            <hr style='margin:5px 0;'>
            <b>지점번호:</b> {st['station_id']}<br>
            <b>위도:</b> {st['lat']:.4f}<br>
            <b>경도:</b> {st['lon']:.4f}<br>
            <b>Coverage:</b> {st['coverage']:.2%}<br>
            <b>유효수온:</b> {st['valid_count']:,}<br>
            <b>기대개수:</b> {st['expected_count']:,}<br>
            <b>시간간격:</b> {st['step']}
        </div>
        """

        folium.CircleMarker(
            location=[st["lat"], st["lon"]],
            radius=6,
            color='red',
            fill=True,
            fill_opacity=0.8
        ).add_to(m)

        folium.Marker(
            [st["lat"], st["lon"]],
            icon=DivIcon(
                icon_size=(120, 20),
                icon_anchor=(0, 0),
                html=(
                    f'<div style="font-size:9pt; font-weight:bold; '
                    f'background:rgba(255,255,255,0.8); border:1px solid gray; '
                    f'padding:2px; border-radius:3px; width:fit-content;">'
                    f'{st["station_name"]}</div>'
                )
            ),
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(m)

    out_html = os.path.join(output_dir, "00_Station_Map_10yr_90pct.html")
    m.save(out_html)
    print(f"🗺️ 정점도 저장 완료: {out_html}")


def plot_station_timeseries(selected_stations, data_dir, output_dir, plot_start, plot_end):
    os.makedirs(output_dir, exist_ok=True)

    for st in selected_stations:
        try:
            path = os.path.join(data_dir, st["file"])
            df = read_station_file(path)

            if df.empty:
                continue

            df = df.dropna(subset=["일시", "수온(°C)"]).sort_values("일시")
            df = df[(df["일시"] >= plot_start) & (df["일시"] <= plot_end)]

            if df.empty:
                continue

            plt.figure(figsize=(15, 5))
            plt.scatter(
                df["일시"],
                df["수온(°C)"],
                s=3,
                alpha=0.5
            )

            plt.xlim(plot_start, plot_end)
            plt.ylim(-1, 35)

            plt.title(
                f'[{st["station_name"]} / {st["station_id"]}] 수온 시계열 '
                f'(coverage={st["coverage"]:.2%})',
                fontsize=14
            )
            plt.xlabel("일시")
            plt.ylabel("수온(°C)")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()

            out_png = os.path.join(
                output_dir,
                f'Chart_{st["station_name"]}_10yr90pct.png'
            )
            plt.savefig(out_png, dpi=200)
            plt.close()

            print(f"✅ 시계열 저장 완료: {st['station_name']}")

        except Exception as e:
            print(f"⚠️ 시계열 오류: {st['station_name']} -> {e}")


def main():
    meta_path = "META_관측지점정보_해양기상부이.csv"
    data_dir = "merge_data"
    output_dir = "Final_Results"

    # 10년 완전성 평가 구간
    coverage_start = datetime(2021, 1, 1)
    coverage_end = datetime(2025, 12, 31, 23, 59, 59)

    # 그림 표시 구간
    plot_start = datetime(2021, 1, 1)
    plot_end = datetime(2025, 12, 31, 23, 59, 59)

    meta_df = load_meta(meta_path)

    selected_stations = select_good_stations(
        meta_df=meta_df,
        data_dir=data_dir,
        start_date=coverage_start,
        end_date=coverage_end,
        threshold=0.0
    )

    plot_station_map(selected_stations, output_dir)
    plot_station_timeseries(selected_stations, data_dir, output_dir, plot_start, plot_end)

    print(f"\n✨ 모든 작업 완료: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()