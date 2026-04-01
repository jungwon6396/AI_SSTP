"""
[필수 사전 작업 안내]
이 파이썬 스크립트를 정상적으로 실행하기 위해서는 터미널에서 반드시 아래 명령어를 먼저 실행해야 합니다.

    👉 명령어: copernicusmarine login

이유:
Copernicus 해양 모델 데이터는 고품질의 대용량 데이터이므로 API 서버 과부하를 막기 위해 사용자 계정 인증을 필수적으로 요구합니다.
터미널에서 위 명령어를 통해 이메일과 비밀번호를 최초 1회 입력해 두면, 인증 토큰이 현재 컴퓨터 내부에 보관되어
이후부터 스크립트가 실행될 때마다 권한 에러 없이 자동으로 데이터를 무제한 다운로드할 수 있게 됩니다.
"""

import os
import argparse
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_CACHE_DIR = os.path.join(tempfile.gettempdir(), "ai_sstp_cache")
os.makedirs(TEMP_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(TEMP_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", TEMP_CACHE_DIR)

import copernicusmarine
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def choose_plot_font():
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


PLOT_LABELS = {
    "thetao": "표층 수온",
    "so": "표층 염분",
    "zos": "해수면 높이",
    "mlotst": "혼합층 두께",
    "uo": "동서 방향 해류",
    "vo": "남북 방향 해류",
    "current_speed": "해류 속도",
}

PLOT_UNITS = {
    "thetao": "degC",
    "so": "",
    "zos": "m",
    "mlotst": "m",
    "uo": "m/s",
    "vo": "m/s",
    "current_speed": "m/s",
}

PLOT_CMAPS = {
    "thetao": "turbo",
    "so": "viridis",
    "zos": "RdBu_r",
    "mlotst": "YlGnBu",
    "uo": "PuOr_r",
    "vo": "BrBG",
    "current_speed": "plasma",
}

STATION_META_CANDIDATES = [
    os.path.join(BASE_DIR, "..", "01.down_temperature", "해양기상부이", "META_관측지점정보_해양기상부이.csv"),
    os.path.join(BASE_DIR, "..", "01.down_temperature", "해양기상부이", "META_관측지점정보_해양기상부이.csv"),
]


def squeeze_spatial_slice(data_array: xr.DataArray, time_index: int) -> xr.DataArray:
    """
    time/depth가 있으면 선택하여 2차원(latitude, longitude) 배열로 정리한다.
    """
    work = data_array

    if "time" in work.dims:
        work = work.isel(time=time_index)

    if "depth" in work.dims:
        work = work.isel(depth=0)

    extra_dims = [dim for dim in work.dims if dim not in ("latitude", "longitude")]
    if extra_dims:
        work = work.isel({dim: 0 for dim in extra_dims})

    return work


def load_station_metadata():
    for meta_path in STATION_META_CANDIDATES:
        if not os.path.exists(meta_path):
            continue

        for encoding in ("cp949", "utf-8"):
            try:
                meta_df = pd.read_csv(meta_path, encoding=encoding, skipinitialspace=True)
                meta_df = meta_df.copy()
                meta_df["지점명"] = meta_df.iloc[:, 3].astype(str).str.strip()
                meta_df["위도"] = pd.to_numeric(meta_df["위도"], errors="coerce")
                meta_df["경도"] = pd.to_numeric(meta_df["경도"], errors="coerce")
                return meta_df.dropna(subset=["지점명", "위도", "경도"])
            except UnicodeDecodeError:
                continue
            except Exception:
                return None
    return None


def draw_station_overlay(ax, lon_min, lon_max, lat_min, lat_max):
    meta_df = load_station_metadata()
    if meta_df is None or meta_df.empty:
        return

    subset = meta_df[
        (meta_df["경도"] >= lon_min) &
        (meta_df["경도"] <= lon_max) &
        (meta_df["위도"] >= lat_min) &
        (meta_df["위도"] <= lat_max)
    ]

    if subset.empty:
        return

    ax.scatter(
        subset["경도"],
        subset["위도"],
        s=18,
        c="#111827",
        edgecolors="white",
        linewidths=0.6,
        alpha=0.9,
        zorder=5
    )

    for _, row in subset.iterrows():
        ax.text(
            row["경도"] + 0.03,
            row["위도"] + 0.03,
            row["지점명"],
            fontsize=7,
            color="#111827",
            weight="bold",
            alpha=0.9,
            zorder=6
        )


def style_map_axes(ax, lon, lat, subtitle: str):
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))

    ax.set_facecolor("#dceffd")
    ax.grid(True, color="white", linewidth=0.9, alpha=0.85)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")

    for spine in ax.spines.values():
        spine.set_color("#355c7d")
        spine.set_linewidth(1.2)

    ax.text(
        0.015,
        0.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=8,
        color="#355c7d",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75)
    )
    draw_station_overlay(ax, lon_min, lon_max, lat_min, lat_max)


def plot_dataarray_map(data_array: xr.DataArray, title: str, out_path: str, cmap: str = "viridis", var_name: str = ""):
    lon = data_array["longitude"].values
    lat = data_array["latitude"].values
    values = data_array.values

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#eef6fb")
    mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap=cmap)
    contour_levels = 8
    try:
        ax.contour(lon, lat, values, levels=contour_levels, colors="#164863", linewidths=0.35, alpha=0.35)
    except Exception:
        pass

    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.92, pad=0.02)
    colorbar.set_label(PLOT_UNITS.get(var_name) or data_array.attrs.get("units", ""))
    style_map_axes(ax, lon, lat, subtitle="서해 연안 Copernicus 격자장")
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_current_vector_map(uo: xr.DataArray, vo: xr.DataArray, title: str, out_path: str):
    lon = uo["longitude"].values
    lat = uo["latitude"].values
    u = np.asarray(uo.values)
    v = np.asarray(vo.values)
    speed = np.sqrt(u ** 2 + v ** 2)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    step = max(1, min(len(lat), len(lon)) // 12)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#eef6fb")
    mesh = ax.pcolormesh(lon, lat, speed, shading="auto", cmap=PLOT_CMAPS["current_speed"])
    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.92, pad=0.02)
    colorbar.set_label(PLOT_UNITS["current_speed"])
    ax.quiver(
        lon_grid[::step, ::step],
        lat_grid[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        color="white",
        scale=10,
        width=0.003
    )
    style_map_axes(ax, lon, lat, subtitle="색상: 유속, 화살표: 유향")
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_copernicus_file(nc_path: str, plot_dir: str):
    """
    다운로드된 NetCDF에서 time별 공간 분포도를 PNG로 저장한다.
    """
    os.makedirs(plot_dir, exist_ok=True)

    with xr.open_dataset(nc_path) as ds:
        times = ds["time"].values if "time" in ds.coords else [None]
        stem = Path(nc_path).stem

        for time_index, raw_time in enumerate(times):
            time_label = (
                np.datetime_as_string(raw_time, unit="D").replace("-", "")
                if raw_time is not None else "notime"
            )

            for var_name in ds.data_vars:
                sliced = squeeze_spatial_slice(ds[var_name], time_index=time_index)
                label = PLOT_LABELS.get(var_name, var_name)
                out_path = os.path.join(plot_dir, f"{stem}_{var_name}_{time_label}.png")
                cmap = PLOT_CMAPS.get(var_name, "viridis")

                plot_dataarray_map(
                    sliced,
                    title=f"{label} 공간 분포\n기준일: {time_label}",
                    out_path=out_path,
                    cmap=cmap,
                    var_name=var_name
                )

            if {"uo", "vo"}.issubset(ds.data_vars):
                uo = squeeze_spatial_slice(ds["uo"], time_index=time_index)
                vo = squeeze_spatial_slice(ds["vo"], time_index=time_index)
                current_out = os.path.join(plot_dir, f"{stem}_current_speed_{time_label}.png")
                plot_current_vector_map(
                    uo,
                    vo,
                    title=f"해류 속도 및 방향\n기준일: {time_label}",
                    out_path=current_out
                )


def plot_downloaded_outputs(output_dir: str):
    plot_root = os.path.join(output_dir, "plots")
    nc_files = sorted(Path(output_dir).glob("*.nc"))

    if not nc_files:
        print("⚠️ 그릴 NetCDF 파일이 없어 시각화를 건너뜁니다.")
        return

    print(f"\n🖼️ NetCDF 시각화 시작 ({len(nc_files)} files)")

    for nc_path in nc_files:
        print(f"   -> 그림 생성: {nc_path.name}")
        try:
            plot_copernicus_file(str(nc_path), plot_root)
            plt.close('all')
            print("      ✅ 완료")
        except Exception as e:
            print(f"      ❌ 실패: {e}")


def download_copernicus_data(output_dir: str, start_date: str, end_date: str, make_plots: bool = True):
    """
    [CMEMS Global Ocean Physics Analysis and Forecast 다운로드 파이프라인]

    이 스크립트는 유럽연합(EU) 코페르니쿠스 해양환경감시서비스(CMEMS)의 전지구 단위 해양 물리 예측 시스템
    (GLOBAL_ANALYSISFORECAST_PHY_001_024) 데이터를 OPeNDAP/API를 통해 자동 수집합니다.

    ■ 구동 엔진: NEMO 수치모델 엔진 기반 및 위성/부이 등 현장 관측 3차원 자료동화(DA) 적용
    ■ 데이터 명세:
      - 공간 해상도: 수평 1/12° (약 8~9km) 에디 분해능 (Eddy-resolving) / 최상단 표준 수심층(Surface, 0.49m)
      - 시간 해상도: 일평균(Daily mean) 타임스텝 기반, 향후 10일간의 선도 예측장(10-day Forecast)
    ■ AI_SSTP 도입 목적:
      - 단일 정점 시계열 딥러닝 모델의 국지적 한계를 보완하기 위해, 수치모델이 사전 계산한 인근 해역의
        거시적 열 수송 및 해류의 공간적 이류(Advection) 정보를 외부 설명 변수(Explanatory Features)로 주입
    """
    os.makedirs(output_dir, exist_ok=True)

    DATASETS_MAP = {
        "thetao": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", # 표층 수온
        "so": "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", # 표층 염분
        "cur": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", # 해류 (uo, vo)
        "2d_vars": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m", # 2차원 변수 (zos, mlotst)
        
    }

    VARIABLES_MAP = {
        "thetao": ["thetao"],
        "so": ["so"],
        "cur": ["uo", "vo"],
        "2d_vars": ["zos", "mlotst"],
    }

    MIN_LON, MAX_LON = 124.0, 127.5
    MIN_LAT, MAX_LAT = 34.0, 38.0
    MIN_DEPTH, MAX_DEPTH = 0.49, 0.50

    print(f"🔄 다운로드 시작: {start_date} ~ {end_date}")
    print(f"📍 영역: Lon[{MIN_LON}~{MAX_LON}], Lat[{MIN_LAT}~{MAX_LAT}]")
    print(f"📂 저장 경로: {output_dir}")

    for group_name, dataset_id in DATASETS_MAP.items():
        target_vars = VARIABLES_MAP[group_name]
        output_filename = f"copernicus_phy_{group_name}_{start_date.replace('-','')}_{end_date.replace('-','')}.nc"
        target_nc_path = os.path.join(output_dir, output_filename)

        # 기존 파일이 존재하면 덮어쓰지 않고 추가로 (1)이 붙는 현상을 방지하기 위해 파일 삭제
        if os.path.exists(target_nc_path):
            try:
                os.remove(target_nc_path)
                print(f"   🧹 기존 파일 삭제 완료: {output_filename}")
            except Exception as e:
                print(f"   ⚠️ 기존 파일 삭제 실패: {e}")

        print(f"\n⏳ [{group_name.upper()}] 다운로드 진행 중... (변수: {target_vars})")
        print(f"   -> 타겟 데이터셋: {dataset_id}")

        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=target_vars,
                minimum_longitude=MIN_LON,
                maximum_longitude=MAX_LON,
                minimum_latitude=MIN_LAT,
                maximum_latitude=MAX_LAT,
                minimum_depth=MIN_DEPTH,
                maximum_depth=MAX_DEPTH,
                start_datetime=f"{start_date}T00:00:00",
                end_datetime=f"{end_date}T23:59:59",
                output_filename=output_filename,
                output_directory=output_dir
            )
            print(f"   ✅ [{group_name}] 정상 다운로드 완료!")

        except Exception as e:
            print(f"   ❌ [{group_name}] 다운로드 실패. 에러 메시지:\n{e}")
            print("\n💡 힌트: 'copernicusmarine login' 로그인이 안 되어 있거나 데이터셋 구조가 또 변경되었을 수 있습니다.")

    if make_plots:
        plot_downloaded_outputs(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copernicus Marine Data Downloader for AI_SSTP")
    default_outdir = os.path.join(BASE_DIR, "s01_copernicus_nc_data")

    parser.add_argument("--outdir", type=str, default=default_outdir, help="저장할 디렉터리 경로")
    parser.add_argument("--start", type=str, default="2021-01-01", help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="다운로드 후 NetCDF 그림 저장 과정을 건너뜁니다. (이 옵션을 주면 그림을 그리지 않음)"
    )

    args = parser.parse_args()

    download_copernicus_data(
        output_dir=args.outdir,
        start_date=args.start,
        end_date=args.end,
        make_plots=not args.skip_plot
    )
