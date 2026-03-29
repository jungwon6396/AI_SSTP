# AI_SSTP

해양기상부이 자료를 정리하고, 지점별 통합 데이터를 생성한 뒤, 저수온 관련 분석 및 LSTM 기반 단일 지점 수온 예측 결과를 보관하는 저장소입니다.
또한 Copernicus 해양 수치모델 자료를 내려받기 위한 보조 스크립트도 포함합니다.
최근에는 Copernicus 표층 수온과 관측소 일평균 수온(UTC 기준)을 직접 비교하는 평가 스크립트와 결과표도 추가했습니다.

## 현재 포함된 작업

- `01.down_temperature/`
  - 해양기상부이 원천 압축 자료
  - 압축 해제 및 재정리용 스크립트
  - 지점별 통합 데이터 생성 스크립트
  - 2021~2025 기간 기준 완전성 필터링 및 시계열 시각화 결과
  - LSTM 학습 결과와 예측 산출물
  - 최종 HTML/PNG 결과물
- `02.down_numerical_models/`
  - Copernicus Marine 수치모델 다운로드 스크립트
  - Copernicus 표층 수온과 관측 수온 비교 스크립트
  - 월별/계절별 성능표와 비교 그래프
  - 수치모델 다운로드용 의존성 목록

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
├── 02.down_numerical_models/
│   ├── requirements_model.txt
│   ├── s01_download_N_plot_copernicus.py
│   ├── s02_compare_copernicus_N_KMA_obs.py
│   ├── s01_copernicus_nc_data/
│   └── s02_compare_results/
├── LICENSE
└── README.md
```

## 작업 흐름

### 1. 압축 파일 해제

- `s01_unzip_files.py`
  - 스크립트가 있는 폴더의 ZIP 파일을 동일한 이름의 폴더로 해제합니다.

- `s02_unzip_in_subfolders.py`
  - 스크립트가 있는 폴더를 기준으로 하위 폴더 내부 ZIP까지 재귀적으로 해제합니다.

### 2. 메타데이터 기준 통합 데이터 생성

- `s03_search_files_N_merge.py`
  - `META_관측지점정보_해양기상부이.csv`를 기준으로 관측 ID와 기간을 읽습니다.
  - `old_data/` 아래 CSV를 순회하면서 행 단위로 관측 기간을 판별합니다.
  - `cp949`와 `utf-8` 인코딩을 순차적으로 시도합니다.
  - 지점별 통합 결과를 `merge_data/{지점명}_통합데이터.csv`로 저장합니다.

### 3. 시각화

- `s04_plot_station_N_timeseries.py`
  - `2021-01-01`부터 `2025-12-31`까지 기간을 기준으로 수온 자료 완전성을 평가합니다.
  - 수온 유효값 비율이 90% 이상인 지점만 선택합니다.
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
  - 입력 컬럼은 CSV 헤더명을 기준으로 탐색합니다.
  - 예측 대상:
    - 다음 시점의 수온
  - 분할 방식:
    - 시간 순서 유지
    - train / validation / test = 70 / 15 / 15
  - 결과 저장:
    - `lstm_results/` 아래 지점별 모델, 학습 곡선, 예측 CSV, 예측 그래프, 요약 지표

### 5. Copernicus 해양 수치모델 다운로드 및 시각화

- `02.down_numerical_models/s01_download_N_plot_copernicus.py`
  - Copernicus Marine 인증 이후 관심 영역의 해양 물리 변수를 NetCDF로 다운로드합니다.
  - 다운로드된 NetCDF를 기준으로 변수별 공간 분포 PNG를 함께 생성할 수 있습니다.
  - 기본 출력 폴더는 `02.down_numerical_models/s01_copernicus_nc_data/`입니다.
  - 그림 생성 결과는 기본적으로 `02.down_numerical_models/s01_copernicus_nc_data/plots/` 아래에 저장됩니다.
  - `--skip-plot` 옵션을 주면 다운로드만 수행하고 그림 생성을 건너뜁니다.
  - 실행 전 `copernicusmarine login`이 필요합니다.

### 6. Copernicus 표층 수온 vs 관측소 수온 비교

- `02.down_numerical_models/s02_compare_copernicus_N_KMA_obs.py`
  - LSTM 학습 대상 관측소를 자동 탐색합니다.
  - 관측소와 가장 가까운 Copernicus 해양 격자를 찾습니다.
  - 가장 가까운 격자가 육지라면, 가장 가까운 비육지 격자를 선택하도록 설계했습니다.
  - 관측자료는 KST 원시 시각을 UTC로 변환한 뒤 UTC 일평균으로 계산합니다.
  - Copernicus `thetao` 일자료와 관측소 일평균 수온을 병합합니다.
  - 지점별 산출물:
    - 일별 비교 CSV
    - 시계열 + 산점도 PNG
    - 월별 성능표 CSV
    - 계절별 성능표 CSV
  - 전체 요약 산출물:
    - `copernicus_obs_match_summary.csv`
    - `copernicus_obs_monthly_performance_summary.csv`
    - `copernicus_obs_seasonal_performance_summary.csv`

## 현재 성능 요약

### LSTM 테스트 성능 요약

`01.down_temperature/해양기상부이/lstm_results/summary_metrics.csv` 기준입니다.

| 순위 | 지점 | Test RMSE | Test MAE | Best Epoch |
| --- | --- | ---: | ---: | ---: |
| 1 | 서해170 | 0.9039 | 0.7418 | 10 |
| 2 | 인천 | 1.1434 | 0.8457 | 20 |
| 3 | 서해190 | 1.2841 | 1.0512 | 98 |
| 4 | 외연도 | 2.2245 | 1.7242 | 25 |
| 5 | 부안 | 2.4821 | 2.0569 | 6 |
| 6 | 칠발도 | 2.5933 | 2.0612 | 2 |

해석 메모:
- LSTM은 서해170, 인천, 서해190에서 상대적으로 안정적입니다.
- 외연도, 부안, 칠발도는 오차가 큰 편이라 추가 피처 보강이나 전처리 개선 여지가 있습니다.
- 칠발도는 표본 수가 많지만 best epoch가 2로 매우 이른 편이라 학습 설정 점검이 필요할 수 있습니다.

### Copernicus vs 관측소 비교 요약

`02.down_numerical_models/s02_compare_results/copernicus_obs_match_summary.csv` 기준이며, 비교 기준은 UTC 일평균입니다.

| 순위 | 지점 | 비교일수 | 격자거리(km) | Bias | RMSE | R |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 외연도 | 935 | 0.000 | -0.497 | 0.781 | 0.9970 |
| 2 | 부안 | 1214 | 1.972 | -0.364 | 0.798 | 0.9954 |
| 3 | 서해190 | 1253 | 3.342 | 0.505 | 0.952 | 0.9961 |
| 4 | 서해170 | 1272 | 4.405 | 0.377 | 1.058 | 0.9943 |
| 5 | 인천 | 1116 | 1.428 | -0.245 | 1.076 | 0.9912 |
| 6 | 칠발도 | 1298 | 5.084 | 0.790 | 1.897 | 0.9806 |

해석 메모:
- 모든 지점에서 상관계수 `R`이 0.98 이상으로 매우 높아, Copernicus는 시계열 변동 패턴을 잘 따라갑니다.
- 반면 절대값은 지점별 편향이 존재합니다.
- 서해170, 서해190, 칠발도는 상대적으로 양의 편향이 있고, 외연도, 부안, 인천은 음의 편향이 있습니다.
- 칠발도는 6개 지점 중 RMSE가 가장 커서 가장 주의가 필요한 지점입니다.

### Copernicus 계절별 특징

`02.down_numerical_models/s02_compare_results/copernicus_obs_seasonal_performance_summary.csv` 기준으로 지점별 최악 계절만 추리면 다음과 같습니다.

| 지점 | 가장 어려운 계절 | Bias | RMSE | R |
| --- | --- | ---: | ---: | ---: |
| 칠발도 | 여름 | 2.445 | 3.010 | 0.8853 |
| 서해170 | 겨울 | 1.586 | 1.820 | 0.8921 |
| 인천 | 가을 | -0.979 | 1.575 | 0.9816 |
| 서해190 | 겨울 | 1.396 | 1.516 | 0.9338 |
| 부안 | 겨울 | -0.813 | 1.089 | 0.9651 |
| 외연도 | 가을 | -0.689 | 0.882 | 0.9930 |

해석 메모:
- 서해170과 서해190은 겨울철에 Copernicus가 관측보다 따뜻하게 나타나는 경향이 큽니다.
- 칠발도는 여름철 과대추정이 특히 커서, 국지 연안 과정 또는 격자 대표성 문제를 의심할 수 있습니다.
- 인천은 가을철 음의 편향이 크게 나타납니다.

## 관측소별 해석 요약

### 서해170

- LSTM은 6개 정점 중 가장 낮은 Test RMSE(`0.9039`)를 보여 단일 정점 예측 기준으로 가장 안정적인 성능을 보였습니다.
- Copernicus 비교에서는 전체 `RMSE 1.058`, `R 0.994`로 시계열 추세는 매우 잘 따라가지만, 겨울철 `Bias +1.586`, `RMSE 1.820`으로 따뜻하게 계산되는 경향이 뚜렷합니다.
- 해석:
  - 순수 시계열 예측에는 강점이 있지만, 물리 배경장을 그대로 쓰기보다는 겨울철 bias correction이 우선 필요합니다.

### 서해190

- LSTM은 `Test RMSE 1.2841`로 상위권이며 validation/test 차이가 크지 않아 일반화 성능이 비교적 안정적입니다.
- Copernicus 비교에서는 `RMSE 0.952`, `R 0.996`로 배경장 재현성이 높고, 특히 여름철 `RMSE 0.480`으로 우수합니다.
- 해석:
  - 서해170과 유사하게 겨울철 양의 편향이 존재하지만, 전반적으로 Copernicus를 외생 입력으로 쓰기 좋은 정점입니다.

### 인천

- LSTM은 `Test RMSE 1.1434`로 양호하지만 validation 대비 test 오차가 증가해 일반화 점검이 필요한 정점입니다.
- Copernicus 비교에서는 `RMSE 1.076`, `R 0.991`로 상관은 높으나 가을철 `RMSE 1.575`, 11월 `RMSE 2.067`로 계절 전환기 취약성이 보입니다.
- 해석:
  - 조석 혼합과 연안 국지성이 강한 지역 특성상, 단순 최근접 격자값만으로는 국지 변동을 충분히 설명하지 못할 가능성이 큽니다.

### 외연도

- LSTM은 `Test RMSE 2.2245`로 단일 정점 예측 성능은 낮은 편입니다.
- 반면 Copernicus 비교에서는 전체 최상위권인 `RMSE 0.781`, `R 0.997`을 기록해 배경장 재현성이 매우 뛰어납니다.
- 해석:
  - 외해 성격이 강해 수치모델의 공간 평균장이 잘 맞는 정점으로 보이며, Physics-guided 모델의 기준 정점으로 활용하기 좋습니다.

### 부안

- LSTM은 `Test RMSE 2.4821`로 오차가 큰 편이지만, Copernicus 비교에서는 `RMSE 0.798`, `R 0.995`로 매우 안정적인 성능을 보였습니다.
- 봄철 Copernicus 성능이 특히 좋고(`RMSE 0.532`), 겨울철 음의 편향이 커지는 경향이 있습니다.
- 해석:
  - 현재 단일 정점 LSTM보다 물리 배경장을 활용하는 접근이 더 유망해 보이는 대표 사례입니다.

### 칠발도

- LSTM은 `Test RMSE 2.5933`으로 가장 큰 오차를 보였고, best epoch가 `2`로 매우 이르게 결정되어 학습 안정성도 낮습니다.
- Copernicus 비교에서도 `RMSE 1.897`, `R 0.981`로 6개 정점 중 가장 불리하며, 여름철 `Bias +2.445`, `RMSE 3.010`으로 급격히 악화됩니다.
- 해석:
  - 국지 연안 혼합, 격자 대표성, 혹은 지형 효과가 크게 작용하는 지점으로 추정되며, 별도 보정 전략 또는 모델 분리가 필요합니다.

## 지역별 종합 해석

- 외해/개방 수역에 가까운 정점(외연도, 부안)은 Copernicus 배경장이 매우 잘 맞습니다.
- 연안 동역학과 계절 전환 영향이 큰 정점(인천, 서해170, 서해190)은 높은 상관계수에도 불구하고 계절별 편향이 남습니다.
- 칠발도는 현재 기준으로 가장 어려운 정점이며, 단일 정점 LSTM과 Copernicus 직접 비교 모두에서 개선이 필요합니다.
- 따라서 후속 모델은 모든 정점에 동일한 전략을 적용하기보다, 정점군별로 다른 보정/학습 전략을 채택하는 편이 합리적입니다.

## 다음 워크 프로세스 제안

### 1. Copernicus 시계열 비교 결과를 기준선(baseline)으로 고정

- 현재 확보된 `copernicus_obs_match_summary.csv`, 월별/계절별 성능표를 기준선으로 문서화합니다.
- 이후 모든 후속 실험은 이 기준선 대비 개선 여부를 비교하는 방식으로 정리합니다.

### 2. 정점군별 전략 분리

- 안정적 재현군: 외연도, 부안
  - Copernicus `thetao`를 바로 외생 입력에 넣는 실험을 우선 진행
- 중간 재현군: 서해170, 서해190, 인천
  - 계절별 bias correction 후 투입
  - 겨울/가을 보정 실험 우선
- 주의 지점: 칠발도
  - 별도 정점으로 분리
  - 주변 격자 평균, 다중 격자, 계절별 모델 분리 등을 우선 검토

### 3. Feature set 확장

- 기존 LSTM 입력:
  - 수온, 기온, 풍속, 기압, 유의파고
- 추가 후보:
  - Copernicus `thetao`
  - 표층 해류 `uo`, `vo`
  - 해수면 높이 `zos`
  - 혼합층 두께 `mlotst`
  - 필요 시 염분 `so`

### 4. 모델 실험 순서

1. 현재 LSTM + Copernicus `thetao`만 추가
2. `thetao + uo + vo` 조합 추가
3. 계절별 bias correction 적용 전/후 비교
4. 정점별 성능 차이를 바탕으로 지역별 모델 분리 여부 판단

### 5. 평가 체계 고정

- 공통 평가지표:
  - RMSE
  - MAE
  - Bias
  - Correlation (`R`)
- 공통 분석 축:
  - 전체 기간
  - 월별
  - 계절별
  - 정점군별

### 6. 발표/문서화 방향

- 단순히 “전체 평균 성능”만 제시하지 말고, 정점별/권역별 차이를 핵심 메시지로 정리합니다.
- 특히 다음 메시지를 일관되게 유지하는 것이 좋습니다.
  - Copernicus는 모든 지점에서 시계열 패턴 재현성은 높다.
  - 하지만 지역별 편향 특성이 뚜렷하므로, Physics-informed 입력과 bias correction이 함께 필요하다.
  - 후속 모델은 전국 일괄 구조보다 서해 권역 특성과 정점군 차이를 반영하는 방향이 더 타당하다.

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

- `02.down_numerical_models/s02_compare_results/`
  - 지점별 관측 vs Copernicus 비교 CSV
  - 지점별 시계열 + 산점도 PNG
  - 지점별 월별 성능표 CSV
  - 지점별 계절별 성능표 CSV
  - 전체 요약 CSV
  - 전체 월별/계절별 성능 요약 CSV

## 실행 환경 메모

- 일부 CSV는 `cp949` 인코딩을 사용합니다.
- 스크립트는 한글 파일명과 폴더명을 전제로 작성되어 있습니다.
- 주요 스크립트는 현재 작업 디렉터리가 아니라 스크립트 파일 위치를 기준으로 경로를 해석합니다.
- LSTM 학습 스크립트는 `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `torch`가 필요합니다.
- Copernicus 다운로드 스크립트는 `02.down_numerical_models/requirements_model.txt` 기준 의존성이 필요합니다.

## 주의사항

- 현재 저장소에는 원천 데이터, 중간 산출물, 모델 파일, 최종 결과물이 함께 포함되어 있어 용량이 큽니다.
- 분석 재현성과 결과 검토를 위해 산출물을 그대로 보관하는 구조입니다.
