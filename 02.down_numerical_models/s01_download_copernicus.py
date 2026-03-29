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
import copernicusmarine

def download_copernicus_data(output_dir: str, start_date: str, end_date: str):
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
    # 저장 디렉터리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # [1] 다운로드 타겟 설정
    # Copernicus 최근 API 업데이트로 인해 огромной 원본 데이터가 '변수(Variable)'별로 각각 독립된 Dataset ID로 물리적으로 쪼개졌습니다.
    # 따라서 수온, 염분, 해류, 해수면고를 각각 다른 ID에서 다운 받아야 합니다.
    DATASETS_MAP = {
        "thetao": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", # 수온
        "so": "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",         # 염분
        "cur": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",       # 유속(해류)
        "2d_vars": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"        # 해수면고(조위), 저층 수온, 혼합층 두께
    }

    VARIABLES_MAP = {
        "thetao": ["thetao"],
        "so": ["so"],
        "cur": ["uo", "vo"],
        "2d_vars": ["zos", "mlotst"]
    }
    
    # [2] 영역 및 수심 설정 (Bounding Box)
    MIN_LON, MAX_LON = 124.0, 127.5
    MIN_LAT, MAX_LAT = 34.0, 38.0
    MIN_DEPTH, MAX_DEPTH = 0.49, 0.50
    # ---------------------------------------------------------

    print(f"🔄 다운로드 시작: {start_date} ~ {end_date}")
    print(f"📍 영역: Lon[{MIN_LON}~{MAX_LON}], Lat[{MIN_LAT}~{MAX_LAT}]")
    print(f"📂 저장 경로: {output_dir}")

    for group_name, dataset_id in DATASETS_MAP.items():
        target_vars = VARIABLES_MAP[group_name]
        output_filename = f"copernicus_phy_{group_name}_{start_date.replace('-','')}_{end_date.replace('-','')}.nc"
        
        print(f"\n⏳ [{group_name.upper()}] 다운로드 진행 중... (변수: {target_vars})")
        print(f"   -> 타겟 데이터셋: {dataset_id}")
        
        try:
            # subset 메서드를 통해 API 호출
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copernicus Marine Data Downloader for AI_SSTP")
    # 현재 스크립트 파일이 위치한 경로를 기준으로 저장 폴더 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_outdir = os.path.join(base_dir, "s01_copernicus_nc_data")
    
    parser.add_argument("--outdir", type=str, default=default_outdir, help="저장할 디렉터리 경로")
    parser.add_argument("--start", type=str, default="2023-01-01", help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-01-02", help="종료 날짜 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    download_copernicus_data(
        output_dir=args.outdir,
        start_date=args.start,
        end_date=args.end
    )
