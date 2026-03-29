# -*- coding: utf-8 -*-
"""
LSTM 학습에 사용한 해양기상부이 관측소와 Copernicus 표층 수온(thetao) 시계열 비교 스크립트

기본 동작:
1. LSTM 결과 폴더를 읽어 비교 대상 관측소를 자동 탐색
2. 관측소 메타에서 위/경도를 읽음
3. Copernicus thetao 격자 중 관측소와 가장 가까운 해양 격자를 선택
   - 가장 가까운 격자가 육지(전 기간 NaN)면, 가장 가까운 비육지 격자를 다시 선택
4. 관측소 자료를 KST 기준 원시 시각으로 읽고 UTC로 변환한 뒤 일평균 생성
5. 지점별 비교 CSV, 비교 그림, 격자 매칭 요약 CSV 저장

주의:
- Copernicus `P1D-m` 일자료는 UTC 하루 평균으로 해석하는 것이 안전합니다.
- 따라서 본 스크립트는 관측자료(KST)를 UTC로 변환한 뒤 UTC 일평균으로 비교합니다.
"""

import argparse
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_CACHE_DIR = os.path.join(tempfile.gettempdir(), "ai_sstp_cache")
os.makedirs(TEMP_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(TEMP_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", TEMP_CACHE_DIR)

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


DEFAULT_THETAO_PATH = os.path.join(
    BASE_DIR,
    "s01_copernicus_nc_data",
    "copernicus_phy_thetao_20210101_20251231.nc",
)

STATION_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "..", "01.down_temperature", "해양기상부이"),
    os.path.join(BASE_DIR, "..", "01.down_temperature", "해양기상부이"),
]

DEFAULT_TARGET_STATIONS = ["서해190", "서해170", "인천", "외연도", "부안", "칠발도"]
KST_OFFSET = pd.Timedelta(hours=9)
SLIDE_STYLE_GROUPS = [
    ["서해170", "인천", "서해190"],
    ["외연도", "부안", "칠발도"],
]


def choose_plot_font() -> str:
    candidates = [
        "AppleGothic",
        "Malgun Gothic",
        "NanumGothic",
        "NanumBarunGothic",
        "Noto Sans CJK KR",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = choose_plot_font()
plt.rcParams["axes.unicode_minus"] = False


def normalize_text(value) -> str:
    return unicodedata.normalize("NFC", str(value).strip())


def resolve_station_base_dir() -> str:
    for path in STATION_DIR_CANDIDATES:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("해양기상부이 자료 폴더를 찾지 못했습니다.")


def find_meta_path(station_base_dir: str) -> str:
    candidates = [
        os.path.join(station_base_dir, "META_관측지점정보_해양기상부이.csv"),
        os.path.join(station_base_dir, "META_관측지점정보_해양기상부이.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("관측소 메타 CSV를 찾지 못했습니다.")


def load_station_metadata(meta_path: str) -> pd.DataFrame:
    last_error = None
    for encoding in ("cp949", "utf-8"):
        try:
            meta_df = pd.read_csv(meta_path, encoding=encoding, skipinitialspace=True)
            meta_df = meta_df.copy()
            meta_df["지점"] = meta_df.iloc[:, 0].astype(str).map(normalize_text)
            meta_df["지점명"] = meta_df.iloc[:, 3].astype(str).map(normalize_text)
            meta_df["위도"] = pd.to_numeric(meta_df["위도"], errors="coerce")
            meta_df["경도"] = pd.to_numeric(meta_df["경도"], errors="coerce")
            return meta_df.dropna(subset=["지점명", "위도", "경도"]).copy()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"메타 파일을 읽지 못했습니다: {last_error}")


def discover_lstm_stations(station_base_dir: str) -> List[str]:
    lstm_dir = os.path.join(station_base_dir, "lstm_results")
    found = []

    if os.path.isdir(lstm_dir):
        for child in sorted(Path(lstm_dir).iterdir()):
            if not child.is_dir():
                continue
            metrics_files = list(child.glob("*_metrics.json"))
            if metrics_files:
                found.append(normalize_text(child.name))

    if found:
        return found
    return [normalize_text(name) for name in DEFAULT_TARGET_STATIONS]


def find_station_csv_path(station_base_dir: str, station_name: str) -> str:
    merge_dir = os.path.join(station_base_dir, "merge_data")
    if not os.path.isdir(merge_dir):
        raise FileNotFoundError(f"merge_data 폴더가 없습니다: {merge_dir}")

    station_name = normalize_text(station_name)
    for file_name in os.listdir(merge_dir):
        normalized_name = normalize_text(file_name)
        if normalized_name == f"{station_name}_통합데이터.csv":
            return os.path.join(merge_dir, file_name)

    raise FileNotFoundError(f"{station_name} 관측소 통합 CSV를 찾지 못했습니다.")


def load_station_observation(csv_path: str) -> pd.DataFrame:
    last_error = None
    for encoding in ("cp949", "utf-8"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"관측소 CSV를 읽지 못했습니다: {last_error}")

    if df.empty:
        raise ValueError("관측소 CSV가 비어 있습니다.")

    if "일시" not in df.columns or "수온(°C)" not in df.columns:
        raise KeyError(f"필수 컬럼이 없습니다. columns={list(df.columns)}")

    obs_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(df["일시"], errors="coerce"),
            "obs_sst": pd.to_numeric(df["수온(°C)"], errors="coerce"),
        }
    )
    obs_df["obs_sst"] = obs_df["obs_sst"].replace([-99, -999, -9999], np.nan)
    obs_df = obs_df.dropna(subset=["datetime"]).sort_values("datetime")
    obs_df = obs_df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    return obs_df


def to_daily_observation(obs_df: pd.DataFrame) -> pd.DataFrame:
    work = obs_df.copy()
    work["aggregation_time"] = work["datetime"] - KST_OFFSET

    daily = (
        work.set_index("aggregation_time")
        .resample("1D")
        .mean(numeric_only=True)
        .rename_axis("date")
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    earth_radius_km = 6371.0

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius_km * c


def prepare_thetao_surface(thetao_da: xr.DataArray) -> xr.DataArray:
    work = thetao_da
    if "depth" in work.dims:
        work = work.isel(depth=0)

    extra_dims = [dim for dim in work.dims if dim not in ("time", "latitude", "longitude")]
    if extra_dims:
        work = work.isel({dim: 0 for dim in extra_dims})

    if set(work.dims) != {"time", "latitude", "longitude"}:
        raise ValueError(f"예상과 다른 thetao 차원입니다: {work.dims}")

    return work


def select_nearest_ocean_grid(
    station_lat: float,
    station_lon: float,
    thetao_surface: xr.DataArray,
) -> Dict[str, float]:
    lat_values = thetao_surface["latitude"].values
    lon_values = thetao_surface["longitude"].values
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

    ocean_mask = thetao_surface.notnull().any(dim="time").values
    if not np.any(ocean_mask):
        raise ValueError("thetao 자료에서 유효한 해양 격자를 찾지 못했습니다.")

    distances = haversine_km(station_lat, station_lon, lat_grid, lon_grid)
    flat_order = np.argsort(distances, axis=None)

    for rank, flat_idx in enumerate(flat_order, start=1):
        lat_idx, lon_idx = np.unravel_index(flat_idx, distances.shape)
        is_ocean = bool(ocean_mask[lat_idx, lon_idx])
        if not is_ocean:
            continue

        return {
            "lat_idx": int(lat_idx),
            "lon_idx": int(lon_idx),
            "grid_lat": float(lat_grid[lat_idx, lon_idx]),
            "grid_lon": float(lon_grid[lat_idx, lon_idx]),
            "distance_km": float(distances[lat_idx, lon_idx]),
            "search_rank": int(rank),
        }

    raise ValueError("가장 가까운 비육지 격자를 찾지 못했습니다.")


def extract_copernicus_series(
    thetao_surface: xr.DataArray,
    lat_idx: int,
    lon_idx: int,
) -> pd.DataFrame:
    point = thetao_surface.isel(latitude=lat_idx, longitude=lon_idx)
    series_df = point.to_dataframe(name="copernicus_sst").reset_index()
    series_df["time"] = pd.to_datetime(series_df["time"])
    series_df["date"] = series_df["time"].dt.normalize()
    series_df = series_df[["date", "copernicus_sst"]].copy()
    series_df = series_df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return series_df


def compute_metrics(merged_df: pd.DataFrame) -> Dict[str, float]:
    valid = merged_df.dropna(subset=["obs_sst", "copernicus_sst"]).copy()
    if valid.empty:
        return {
            "n_days": 0,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "correlation": np.nan,
        }

    diff = valid["copernicus_sst"] - valid["obs_sst"]
    bias = float(diff.mean())
    mae = float(diff.abs().mean())
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if len(valid) >= 2:
        correlation = float(valid["obs_sst"].corr(valid["copernicus_sst"]))
    else:
        correlation = np.nan

    return {
        "n_days": int(len(valid)),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
    }


def summarize_group_performance(
    merged_df: pd.DataFrame,
    group_key: str,
    group_label: str,
) -> pd.DataFrame:
    valid = merged_df.dropna(subset=["obs_sst", "copernicus_sst"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=[group_key, group_label, "n_days", "bias", "mae", "rmse", "correlation"])

    valid["diff"] = valid["copernicus_sst"] - valid["obs_sst"]
    rows = []

    for group_value, sub in valid.groupby(group_key):
        if pd.isna(group_value) or sub.empty:
            continue

        corr = float(sub["obs_sst"].corr(sub["copernicus_sst"])) if len(sub) >= 2 else np.nan
        rows.append(
            {
                group_key: int(group_value),
                group_label: str(group_value),
                "n_days": int(len(sub)),
                "bias": float(sub["diff"].mean()),
                "mae": float(sub["diff"].abs().mean()),
                "rmse": float(np.sqrt(np.mean(sub["diff"] ** 2))),
                "correlation": corr,
            }
        )

    return pd.DataFrame(rows).sort_values(group_key).reset_index(drop=True)


def build_monthly_performance(merged_df: pd.DataFrame) -> pd.DataFrame:
    work = merged_df.copy()
    work["month"] = pd.to_datetime(work["date"]).dt.month
    month_df = summarize_group_performance(work, group_key="month", group_label="month_label")
    month_labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    if not month_df.empty:
        month_df["month_label"] = month_df["month"].map(month_labels)
    return month_df


def build_seasonal_performance(merged_df: pd.DataFrame) -> pd.DataFrame:
    work = merged_df.copy()
    month = pd.to_datetime(work["date"]).dt.month
    season_order = {"winter": 1, "spring": 2, "summer": 3, "autumn": 4}
    work["season"] = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        ["winter", "spring", "summer", "autumn"],
        default=np.nan,
    )
    work["season_order"] = work["season"].map(season_order)
    season_df = summarize_group_performance(work, group_key="season_order", group_label="season")
    if not season_df.empty:
        season_df = season_df.drop(columns=["season_order"]).reset_index(drop=True)
    return season_df


def plot_comparison(
    merged_df: pd.DataFrame,
    station_name: str,
    station_id: str,
    station_lat: float,
    station_lon: float,
    grid_info: Dict[str, float],
    metrics: Dict[str, float],
    out_path: str,
) -> None:
    valid = merged_df.dropna(subset=["obs_sst", "copernicus_sst"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [2.3, 1.0]})
    ax_ts, ax_scatter = axes

    ax_ts.plot(
        merged_df["date"],
        merged_df["obs_sst"],
        color="#2563eb",
        linewidth=1.2,
        label="관측소 일평균 수온",
    )
    ax_ts.plot(
        merged_df["date"],
        merged_df["copernicus_sst"],
        color="#dc2626",
        linewidth=1.2,
        label="Copernicus 표층 수온",
    )

    title = (
        f"[{station_name} / {station_id}] 관측소 vs Copernicus 표층 수온 비교\n"
        f"일평균 기준=UTC | "
        f"관측소({station_lat:.4f}, {station_lon:.4f}) -> "
        f"격자({grid_info['grid_lat']:.4f}, {grid_info['grid_lon']:.4f}), "
        f"{grid_info['distance_km']:.2f} km"
    )
    ax_ts.set_title(title, fontsize=12, weight="bold")
    ax_ts.set_xlabel("날짜")
    ax_ts.set_ylabel("수온 (°C)")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc="upper right")

    summary_text = (
        f"비교일수={metrics['n_days']} | "
        f"Bias={metrics['bias']:.3f} | "
        f"MAE={metrics['mae']:.3f} | "
        f"RMSE={metrics['rmse']:.3f} | "
        f"R={metrics['correlation']:.3f} | "
        f"탐색순위={grid_info['search_rank']}"
    )
    ax_ts.text(
        0.01,
        0.02,
        summary_text,
        transform=ax_ts.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    if not valid.empty:
        ax_ts.set_xlim(valid["date"].min(), valid["date"].max())

        ax_scatter.scatter(
            valid["obs_sst"],
            valid["copernicus_sst"],
            s=13,
            alpha=0.55,
            color="#0f766e",
            edgecolors="none",
        )
        min_val = float(np.nanmin([valid["obs_sst"].min(), valid["copernicus_sst"].min()]))
        max_val = float(np.nanmax([valid["obs_sst"].max(), valid["copernicus_sst"].max()]))
        ax_scatter.plot([min_val, max_val], [min_val, max_val], color="#7c3aed", linewidth=1.0, linestyle="--")
        ax_scatter.set_xlim(min_val, max_val)
        ax_scatter.set_ylim(min_val, max_val)
    else:
        ax_scatter.text(0.5, 0.5, "비교 가능한 자료 없음", ha="center", va="center", transform=ax_scatter.transAxes)

    ax_scatter.set_title("산점도")
    ax_scatter.set_xlabel("관측소 수온 (°C)")
    ax_scatter.set_ylabel("Copernicus 수온 (°C)")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.text(
        0.03,
        0.97,
        f"RMSE = {metrics['rmse']:.3f}\nR = {metrics['correlation']:.3f}",
        va="top",
        transform=ax_scatter.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_group_panel(
    group_results: List[Dict[str, object]],
    out_path: str,
    panel_title: str,
) -> None:
    fig = plt.figure(figsize=(15, 8.5), facecolor="white")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 2.4], hspace=0.35, wspace=0.2)

    ax_map = fig.add_subplot(gs[:, 0])

    station_lons = [row["station_lon"] for row in group_results]
    station_lats = [row["station_lat"] for row in group_results]
    grid_lons = [row["grid_lon"] for row in group_results]
    grid_lats = [row["grid_lat"] for row in group_results]

    lon_min = min(station_lons + grid_lons) - 0.35
    lon_max = max(station_lons + grid_lons) + 0.35
    lat_min = min(station_lats + grid_lats) - 0.35
    lat_max = max(station_lats + grid_lats) + 0.35

    ax_map.set_facecolor("#eef6fb")
    ax_map.grid(True, color="white", linewidth=0.9, alpha=0.9)
    ax_map.set_xlim(lon_min, lon_max)
    ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_xlabel("경도")
    ax_map.set_ylabel("위도")
    ax_map.set_title("정점도", fontsize=12, weight="bold")

    for idx, row in enumerate(group_results, start=1):
        ax_map.scatter(row["station_lon"], row["station_lat"], s=80, color="#2563eb", edgecolors="white", zorder=4)
        ax_map.scatter(row["grid_lon"], row["grid_lat"], s=65, marker="s", color="#dc2626", edgecolors="white", zorder=4)
        ax_map.plot(
            [row["station_lon"], row["grid_lon"]],
            [row["station_lat"], row["grid_lat"]],
            color="#6b7280",
            linestyle="--",
            linewidth=0.9,
            zorder=3,
        )
        ax_map.text(
            row["station_lon"] + 0.03,
            row["station_lat"] + 0.03,
            f"{idx}. {row['station_name']}",
            fontsize=10,
            weight="bold",
            color="#111827",
            zorder=5,
        )

    ax_map.text(
        0.03,
        0.02,
        "원: 관측소\n사각형: Copernicus 격자",
        transform=ax_map.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    for row_idx, row in enumerate(group_results):
        ax_ts = fig.add_subplot(gs[row_idx, 1])
        merged_df = pd.read_csv(row["output_csv"], parse_dates=["date"])
        valid = merged_df.dropna(subset=["obs_sst", "copernicus_sst"]).copy()

        ax_ts.plot(merged_df["date"], merged_df["obs_sst"], color="#2563eb", linewidth=1.0, label="관측소")
        ax_ts.plot(merged_df["date"], merged_df["copernicus_sst"], color="#dc2626", linewidth=1.0, label="Copernicus")
        ax_ts.grid(True, alpha=0.3)
        ax_ts.set_ylabel("°C")

        if row_idx == 0:
            ax_ts.legend(loc="upper right", ncol=2, fontsize=9)

        if row_idx < len(group_results) - 1:
            ax_ts.tick_params(labelbottom=False)
        else:
            ax_ts.set_xlabel("날짜 (UTC 일평균)")

        if not valid.empty:
            ax_ts.set_xlim(valid["date"].min(), valid["date"].max())

        ax_ts.set_title(
            f"{row_idx + 1}. {row['station_name']} | RMSE {row['rmse']:.3f} | R {row['correlation']:.3f} | "
            f"격자거리 {row['distance_km']:.2f} km",
            fontsize=10.5,
            loc="left",
        )

    fig.suptitle(panel_title, fontsize=15, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def build_group_panel_figures(summary_rows: List[Dict[str, object]], output_dir: str) -> List[str]:
    by_station = {row["station_name"]: row for row in summary_rows}
    created_files = []

    for index, group in enumerate(SLIDE_STYLE_GROUPS, start=1):
        group_results = [by_station[name] for name in group if name in by_station]
        if not group_results:
            continue

        out_path = os.path.join(output_dir, f"copernicus_obs_panel_part{index}.png")
        plot_group_panel(
            group_results=group_results,
            out_path=out_path,
            panel_title=f"Copernicus vs 관측소 비교 결과 ({index}/{len(SLIDE_STYLE_GROUPS)})",
        )
        created_files.append(out_path)

    return created_files


def compare_single_station(
    station_name: str,
    meta_df: pd.DataFrame,
    thetao_surface: xr.DataArray,
    output_dir: str,
) -> Dict[str, object]:
    station_base_dir = resolve_station_base_dir()
    meta_row = meta_df.loc[meta_df["지점명"] == normalize_text(station_name)]
    if meta_row.empty:
        raise KeyError(f"메타에서 {station_name} 지점을 찾지 못했습니다.")

    meta_row = meta_row.iloc[0]
    station_id = str(meta_row["지점"])
    station_lat = float(meta_row["위도"])
    station_lon = float(meta_row["경도"])

    csv_path = find_station_csv_path(station_base_dir, station_name)
    obs_hourly = load_station_observation(csv_path)
    obs_daily = to_daily_observation(obs_hourly)

    grid_info = select_nearest_ocean_grid(station_lat, station_lon, thetao_surface)
    cop_series = extract_copernicus_series(
        thetao_surface,
        lat_idx=grid_info["lat_idx"],
        lon_idx=grid_info["lon_idx"],
    )

    merged_df = pd.merge(obs_daily, cop_series, on="date", how="outer").sort_values("date").reset_index(drop=True)
    metrics = compute_metrics(merged_df)

    station_out_dir = os.path.join(output_dir, normalize_text(station_name))
    os.makedirs(station_out_dir, exist_ok=True)

    merged_csv_path = os.path.join(station_out_dir, f"{normalize_text(station_name)}_copernicus_vs_obs_daily.csv")
    fig_path = os.path.join(station_out_dir, f"{normalize_text(station_name)}_copernicus_vs_obs_daily.png")
    monthly_path = os.path.join(station_out_dir, f"{normalize_text(station_name)}_monthly_performance.csv")
    seasonal_path = os.path.join(station_out_dir, f"{normalize_text(station_name)}_seasonal_performance.csv")

    merged_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
    monthly_df = build_monthly_performance(merged_df)
    seasonal_df = build_seasonal_performance(merged_df)
    monthly_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    seasonal_df.to_csv(seasonal_path, index=False, encoding="utf-8-sig")

    plot_comparison(
        merged_df=merged_df,
        station_name=normalize_text(station_name),
        station_id=station_id,
        station_lat=station_lat,
        station_lon=station_lon,
        grid_info=grid_info,
        metrics=metrics,
        out_path=fig_path,
    )

    return {
        "station_name": normalize_text(station_name),
        "station_id": station_id,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "grid_lat": grid_info["grid_lat"],
        "grid_lon": grid_info["grid_lon"],
        "distance_km": grid_info["distance_km"],
        "search_rank": grid_info["search_rank"],
        "daily_basis": "utc",
        "matched_days": metrics["n_days"],
        "bias": metrics["bias"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "correlation": metrics["correlation"],
        "output_csv": merged_csv_path,
        "output_plot": fig_path,
        "monthly_csv": monthly_path,
        "seasonal_csv": seasonal_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LSTM 관측소와 Copernicus 표층 수온 시계열 비교"
    )
    parser.add_argument(
        "--thetao",
        type=str,
        default=DEFAULT_THETAO_PATH,
        help="Copernicus thetao NetCDF 파일 경로",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(BASE_DIR, "s02_compare_results"),
        help="비교 결과 저장 폴더",
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="비교할 관측소명 목록. 비우면 LSTM 결과 폴더 기준 자동 선택",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    station_base_dir = resolve_station_base_dir()
    meta_path = find_meta_path(station_base_dir)
    meta_df = load_station_metadata(meta_path)

    target_stations = (
        [normalize_text(name) for name in args.stations]
        if args.stations
        else discover_lstm_stations(station_base_dir)
    )

    if not os.path.exists(args.thetao):
        raise FileNotFoundError(f"Copernicus thetao 파일이 없습니다: {args.thetao}")

    os.makedirs(args.outdir, exist_ok=True)

    print("📌 Copernicus vs 관측소 비교 시작")
    print(f"   thetao 파일: {args.thetao}")
    print(f"   결과 폴더: {args.outdir}")
    print("   일평균 기준: UTC")
    print(f"   대상 지점: {', '.join(target_stations)}")
    print("   관측자료는 KST -> UTC 변환 후 일평균 비교")

    summary_rows: List[Dict[str, object]] = []
    monthly_rows: List[pd.DataFrame] = []
    seasonal_rows: List[pd.DataFrame] = []

    with xr.open_dataset(args.thetao) as ds:
        if "thetao" not in ds.data_vars:
            raise KeyError("NetCDF에서 thetao 변수를 찾지 못했습니다.")

        thetao_surface = prepare_thetao_surface(ds["thetao"]).load()

    for station_name in target_stations:
        print(f"\n⏳ 비교 중: {station_name}")
        try:
            result = compare_single_station(
                station_name=station_name,
                meta_df=meta_df,
                thetao_surface=thetao_surface,
                output_dir=args.outdir,
            )
            summary_rows.append(result)

            month_df = pd.read_csv(result["monthly_csv"])
            if not month_df.empty:
                month_df.insert(0, "station_name", result["station_name"])
                monthly_rows.append(month_df)

            season_df = pd.read_csv(result["seasonal_csv"])
            if not season_df.empty:
                season_df.insert(0, "station_name", result["station_name"])
                seasonal_rows.append(season_df)

            print(
                f"   ✅ 완료 | 격자거리={result['distance_km']:.2f} km | "
                f"비교일수={result['matched_days']} | RMSE={result['rmse']:.3f}"
            )
        except Exception as exc:
            print(f"   ❌ 실패: {exc}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.outdir, "copernicus_obs_match_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n📝 요약 저장 완료: {summary_path}")

        if monthly_rows:
            monthly_summary_path = os.path.join(args.outdir, "copernicus_obs_monthly_performance_summary.csv")
            pd.concat(monthly_rows, ignore_index=True).to_csv(monthly_summary_path, index=False, encoding="utf-8-sig")
            print(f"📝 월별 성능표 저장 완료: {monthly_summary_path}")

        if seasonal_rows:
            seasonal_summary_path = os.path.join(args.outdir, "copernicus_obs_seasonal_performance_summary.csv")
            pd.concat(seasonal_rows, ignore_index=True).to_csv(seasonal_summary_path, index=False, encoding="utf-8-sig")
            print(f"📝 계절별 성능표 저장 완료: {seasonal_summary_path}")

        panel_files = build_group_panel_figures(summary_rows, args.outdir)
        for panel_path in panel_files:
            print(f"🖼️ 슬라이드용 패널 저장 완료: {panel_path}")
    else:
        print("\n⚠️ 저장할 비교 결과가 없습니다.")


if __name__ == "__main__":
    main()
