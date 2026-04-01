from pathlib import Path
import copernicusmarine

# ============================================================
# Copernicus Marine 다운로드 설정
# ============================================================
#
# 목적:
#   SST(수온), 염분, 유속, 2D 물리 변수 자료를 자동 다운로드
#
# 주요 제품 설명:
# ------------------------------------------------------------
# thetao :
#   해수 온도 (potential temperature)
#
# so :
#   해수 염분
#
# cur :
#   해류 자료 (주로 uo, vo)
#
# 2d_vars :
#   해면고(zos), 혼합층 깊이(mlotst) 등 2차원 표층 변수
#
# ============================================================


# ------------------------------------------------------------
# 저장 폴더
# ------------------------------------------------------------
BASE_DIR = Path("./03.copernicus_marine_nc_data")
BASE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 다운로드 영역 설정
# 예시: 한반도 주변
# 필요에 따라 수정
# ------------------------------------------------------------
REGION = {
    "min_lon": 115.0,
    "max_lon": 155.0,
    "min_lat": 20.0,
    "max_lat": 55.0,
}

# ------------------------------------------------------------
# 다운로드 기간 설정
# 일평균 자료 기준
# ------------------------------------------------------------
TIME_RANGE = {
    "start_datetime": "2021-01-01T00:00:00",
    "end_datetime": "2025-12-31T23:59:59",
}

# ------------------------------------------------------------
# 수심 범위
# 표층만 받을 경우:
#   0 ~ 5m 정도
#
# 3차원 전체를 받을 경우:
#   depth 관련 인자 제거 또는 넓게 지정
# ------------------------------------------------------------
DEPTH_RANGE = {
    "minimum_depth": 0.0,
    "maximum_depth": 5.0,
}

# ============================================================
# 제품 목록 정의
# ============================================================
PRODUCTS = {
    "thetao": {
        "dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "variables": ["thetao"],
        "description": "해수 온도 (potential temperature)"
    },
    "so": {
        "dataset_id": "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        "variables": ["so"],
        "description": "해수 염분"
    },
    "cur": {
        "dataset_id": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        "variables": ["uo", "vo"],
        "description": "해류 (동서/남북 유속)"
    },
    "2d_vars": {
        "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        # 실제 포함 변수는 제품 버전에 따라 다를 수 있음
        # 보통 예시: zos, mlotst, bottomT
        "variables": ["zos", "mlotst"],
        "description": "2차원 물리 변수 (예: 해면고, 혼합층 깊이)"
    },
}

# ============================================================
# 다운로드 함수
# ============================================================
def download_product(product_key: str):
    """
    Copernicus Marine 단일 제품 다운로드

    Parameters
    ----------
    product_key : str
        PRODUCTS dict의 key
        예: "thetao", "so", "cur", "2d_vars"
    """
    product = PRODUCTS[product_key]

    out_dir = BASE_DIR / product_key
    out_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{product_key}_{TIME_RANGE['start_datetime'][:10]}_{TIME_RANGE['end_datetime'][:10]}.nc"

    print("=" * 60)
    print(f"[다운로드 시작] {product_key}")
    print(f"설명       : {product['description']}")
    print(f"dataset_id : {product['dataset_id']}")
    print(f"variables  : {product['variables']}")
    print(f"저장 경로   : {out_dir / output_filename}")
    print("=" * 60)

    copernicusmarine.subset(
        dataset_id=product["dataset_id"],
        variables=product["variables"],
        minimum_longitude=REGION["min_lon"],
        maximum_longitude=REGION["max_lon"],
        minimum_latitude=REGION["min_lat"],
        maximum_latitude=REGION["max_lat"],
        start_datetime=TIME_RANGE["start_datetime"],
        end_datetime=TIME_RANGE["end_datetime"],
        minimum_depth=DEPTH_RANGE["minimum_depth"],
        maximum_depth=DEPTH_RANGE["maximum_depth"],
        output_directory=str(out_dir),
        output_filename=output_filename,
        force_download=True,
    )

    print(f"[다운로드 완료] {product_key}\n")


# ============================================================
# 실행 예시
# ============================================================
if __name__ == "__main__":
    # 필요한 제품만 선택적으로 다운로드 가능
    download_list = [
        "thetao",
        "so",
        "cur",
        "2d_vars",
    ]

    for key in download_list:
        download_product(key)