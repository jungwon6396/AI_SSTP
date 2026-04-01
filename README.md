# AI_SSTP

해양 관측 자료와 Copernicus 수치모델 자료를 기반으로
수온 예측(정점 기반 + 공간장 기반)을 실험/검증하는 저장소입니다.

## 1. 프로젝트 개요

현재 파이프라인은 크게 두 축으로 구성됩니다.

1. 관측소(정점) 기반 분석/모델링
- 관측 자료 정리, 병합, 시각화
- LSTM 기반 단일 정점 수온 예측

2. Copernicus 공간장 기반 모델링
- Copernicus NetCDF 다운로드/시각화
- 관측소 vs Copernicus 검증
- FNO 멀티 리드타임(1/3/7/14일) 학습 및 실험 자동화

## 2. 디렉터리 구조

```text
AI_SSTP/
├── 01.down_temperature/
│   └── 해양기상부이/
│       ├── s01_unzip_files.py
│       ├── s02_unzip_in_subfolders.py
│       ├── s03_search_files_N_merge.py
│       ├── s04_plot_station_N_timeseries.py
│       ├── s05_make_single_point_model_LSTM.py
│       ├── merge_data/
│       ├── lstm_results/
│       └── Final_Results/
├── 02.down_numerical_models/
│   ├── requirements_model.txt
│   ├── environment_fno.yml
│   ├── README_FNO.md
│   ├── s01_download_N_plot_copernicus.py
│   ├── s02_compare_copernicus_N_KMA_obs.py
│   ├── s03_download_N_plot_copernicus_marine.py
│   ├── s04_train_fno_baseline.py
│   ├── s05_run_fno_experiments.py
│   ├── s01_copernicus_nc_data/
│   ├── s04_fno_baseline_multilead/
│   └── s05_fno_runs/
└── README.md
```

## 3. Copernicus 다운로드/검증

### 3.1 다운로드 및 시각화

- 스크립트: `02.down_numerical_models/s01_download_N_plot_copernicus.py`
- 주요 변수: `thetao`, `so`, `uo`, `vo`, `zos`, `mlotst`

예시:

```powershell
python 02.down_numerical_models/s01_download_N_plot_copernicus.py --start 2021-01-01 --end 2025-12-31
```

### 3.2 관측소 비교

- 스크립트: `02.down_numerical_models/s02_compare_copernicus_N_KMA_obs.py`
- 출력: 일별 비교 CSV/PNG, 월별·계절별 성능 요약

## 4. FNO 멀티 리드타임 학습

### 4.1 환경 준비(권장)

`base`(python 3.13) 충돌을 피하기 위해 전용 환경 사용을 권장합니다.

```powershell
conda env create -f 02.down_numerical_models/environment_fno.yml
conda activate fno
python -c "import torch; print(torch.__version__)"
```

### 4.2 단일 학습 실행

- 스크립트: `02.down_numerical_models/s04_train_fno_baseline.py`
- 기본 리드타임: `1,3,7,14`일

```powershell
python 02.down_numerical_models/s04_train_fno_baseline.py --data-dir 02.down_numerical_models/s01_copernicus_nc_data --lookback 7 --leads 1,3,7,14 --epochs 50 --batch-size 8
```

주요 옵션:
- `--lookback`: 입력 과거 일수
- `--leads`: 예측 리드타임 목록
- `--inspect-input`, `--inspect-only`: 입력 텐서 통계 점검
- `--early-stopping-*`: 자동 조기종료
- `--lr-scheduler-*`: ReduceLROnPlateau 학습률 스케줄

출력 폴더(기본):
- `02.down_numerical_models/s04_fno_baseline_multilead/`

주요 산출물:
- `fno_baseline_multilead_best.pt`
- `report.json`
- `train_history.csv`
- `learning_curve.png`, `lr_curve.png`
- `error_maps/` (리드타임별 MAE/RMSE 맵 PNG + npz)

## 5. 다중 실험 자동 실행

- 스크립트: `02.down_numerical_models/s05_run_fno_experiments.py`
- 목적: `lookback x batch_size x weight_decay` 조합을 순차 실행

기본 스윕:
- `lookbacks`: `3,7,14,30`
- `batch-sizes`: `4,8,16`
- `weight-decays`: `0,1e-6,1e-5,1e-4`

예시:

```powershell
python 02.down_numerical_models/s05_run_fno_experiments.py --epochs 80 --lookbacks 3,7,14 --batch-sizes 4,8,16 --weight-decays 0,1e-6,1e-5,1e-4
```

출력:
- `02.down_numerical_models/s05_fno_runs/run_YYYYMMDD_HHMMSS/...`
- 실험별 폴더 + `summary.csv`

## 6. 자주 겪는 이슈

1. `No module named 'torch'`
- `fno` 환경 미활성화 가능성 큼
- `conda activate fno` 후 재실행

2. PowerShell 프로필 보안 경고(`profile.ps1`)
- 실행 정책 이슈로 자주 발생
- 학습 자체와 직접 무관한 경우가 많음

3. Copernicus 로그인 반복 요청
- `copernicusmarine login` 1회 저장 또는 credentials 파일 사용

## 7. 빠른 시작

```powershell
conda activate fno
python 02.down_numerical_models/s04_train_fno_baseline.py --data-dir 02.down_numerical_models/s01_copernicus_nc_data --lookback 7 --leads 1,3,7,14 --epochs 50 --batch-size 8
```

---

필요하면 다음 단계로 PINO 확장(물리 loss 결합)용 학습 스크립트를 별도 추가할 수 있습니다.
