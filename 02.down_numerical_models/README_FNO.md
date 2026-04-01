# FNO Baseline 실행 가이드 (Windows PowerShell)

## 1) 전용 환경 생성
```powershell
conda env create -f 02.down_numerical_models/environment_fno.yml
conda activate fno
```

## 2) 실행 확인
```powershell
python -c "import torch; print(torch.__version__)"
```

## 3) 멀티 리드타임 학습 실행 (1/3/7/14일)
```powershell
python 02.down_numerical_models/s04_train_fno_baseline.py --data-dir 02.down_numerical_models/s01_copernicus_nc_data --lookback 7 --leads 1,3,7,14 --epochs 50 --batch-size 8
```

## 4) 출력 결과
- 기본 출력 폴더: `02.down_numerical_models/s04_fno_baseline_multilead`
- 주요 파일:
  - `fno_baseline_multilead_best.pt`
  - `report.json` (리드타임별 MAE/RMSE)
  - `train_history.csv`
  - `learning_curve.png`
