# AI_SSTP

해양기상부이 자료를 정리하고, 지점별 통합 데이터를 생성한 뒤, 저수온 관련 분석 및 LSTM 기반 단일 지점 수온 예측 결과를 보관하는 저장소입니다.

## 현재 포함된 작업

- `01.down_temperature/`
  - 해양기상부이 원천 압축 자료
  - 압축 해제 및 재정리용 스크립트
  - 지점별 통합 데이터 생성 스크립트
  - 시계열 시각화 결과
  - LSTM 학습 결과와 예측 산출물
  - 최종 HTML/PNG 결과물

## 디렉터리 구조

```text
AI_SSTP/
├── 01.down_temperature/
│   └── 해양기상부이/
│       ├── META_관측지점정보_해양기상부이.csv
│       ├── old_data/
│       ├── merge_data/
│       ├── lstm_results/
│       ├── Final_Results/
│       ├── s01_unzip_files.py
│       ├── s02_unzip_in_subfolders.py
│       ├── s03_search_files_N_merge.py
│       ├── s04_plot_station_N_timeseries.py
│       └── s05_make_single_point_model_LSTM.py
├── LICENSE
└── README.md
```

## 작업 흐름

### 1. 압축 파일 해제

- `s01_unzip_files.py`
  - 현재 작업 디렉터리의 ZIP 파일을 동일한 이름의 폴더로 해제합니다.

- `s02_unzip_in_subfolders.py`
  - 하위 폴더 내부에 있는 ZIP 파일까지 재귀적으로 해제하는 용도로 사용합니다.

### 2. 메타데이터 기준 통합 데이터 생성

- `s03_search_files_N_merge.py`
  - `META_관측지점정보_해양기상부이.csv`를 기준으로 관측 ID와 기간을 읽습니다.
  - `old_data/` 아래 CSV를 순회하면서 행 단위로 관측 기간을 판별합니다.
  - 지점별 통합 결과를 `merge_data/{지점명}_통합데이터.csv`로 저장합니다.

### 3. 시각화

- `s04_plot_station_N_timeseries.py`
  - 지점 위치 및 시계열 결과를 시각화합니다.
  - 주요 결과는 `Final_Results/`에 HTML 및 PNG 형태로 저장됩니다.

### 4. LSTM 기반 수온 예측

- `s05_make_single_point_model_LSTM.py`
  - 입력 변수:
    - 수온
    - 기온
    - 풍속
    - 기압
    - 유의파고
  - 예측 대상:
    - 다음 시점의 수온
  - 분할 방식:
    - 시간 순서 유지
    - train / validation / test = 70 / 15 / 15
  - 결과 저장:
    - `lstm_results/` 아래 지점별 모델, 학습 곡선, 예측 CSV, 예측 그래프, 요약 지표

## 주요 산출물

- `old_data/`
  - 원천 ZIP 및 압축 해제된 관측 자료 보관

- `merge_data/`
  - 메타데이터 기준으로 정리된 지점별 통합 CSV

- `lstm_results/`
  - 지점별 모델 파일(`.pth`)
  - 학습 이력
  - 예측 결과 CSV
  - 예측 시각화 PNG
  - 요약 성능 지표
  - 성능 요약 문서: `01.down_temperature/해양기상부이/lstm_results/README.md`

- `Final_Results/`
  - 지점도 HTML
  - 지점별 최종 차트 PNG

## 실행 환경 메모

- 일부 CSV는 `cp949` 인코딩을 사용합니다.
- 스크립트는 한글 파일명과 폴더명을 전제로 작성되어 있습니다.
- LSTM 학습 스크립트는 `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `torch`가 필요합니다.

## 주의사항

- 현재 저장소에는 원천 데이터, 중간 산출물, 모델 파일, 최종 결과물이 함께 포함되어 있어 용량이 큽니다.
- 분석 재현성과 결과 검토를 위해 산출물을 그대로 보관하는 구조입니다.
