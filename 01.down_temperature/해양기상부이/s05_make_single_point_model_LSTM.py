# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:07:09 2026

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2026

정점별 LSTM 수온 예측 모델
- 대상 정점: 서해 190, 서해 170, 인천, 외연도, 부산, 칠발도
- 입력 파일: ./merge_data/{지점명}_통합데이터.csv
- 입력 feature: 수온, 기온, 풍속, 기압, 유의파고
- target: 다음 시점의 수온(°C)
- 결측값 처리: 제거
- 데이터 분할: 시간 순서 유지 + 비율 70/15/15
- 스케일링: MinMaxScaler
- inverse transform 적용
- 성능평가: RMSE, MAE
- 예측 결과: 정점별 CSV 저장
"""

import os
import copy
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# =========================
# 0. 기본 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "merge_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "lstm_results")

TARGET_STATIONS = [
    "서해190",
    "서해170",
    "인천",
    "외연도",
    "부안",
    "칠발도"
]

# 파일 내부 컬럼 규칙 (1-based 기준)
COLUMN_CANDIDATES = {
    "datetime": ["일시"],
    "wind_speed": ["풍속(m/s)", "풍속"],
    "pressure": ["현지기압(hPa)", "현지기압"],
    "air_temp": ["기온(°C)", "기온"],
    "sst": ["수온(°C)", "수온"],
    "wave_height": ["유의파고(m)", "유의파고"],
}

# 모델/학습 설정
SEQ_LENGTH = 24 * 7       # 과거 7일 (시간자료 가정)
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 1. 재현성 고정
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# 2. Dataset 정의
# =========================
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# 3. LSTM 모델 정의
# =========================
class SSTPredictor(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


# =========================
# 4. 유틸 함수
# =========================
def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def station_file_path(station_name: str) -> str:
    return os.path.join(INPUT_DIR, f"{station_name}_통합데이터.csv")


def find_first_matching_column(df: pd.DataFrame, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"필수 컬럼을 찾지 못했습니다. 후보={candidates}")


def load_station_data(file_path: str) -> pd.DataFrame:
    """
    파일 컬럼명을 기준으로 필요한 컬럼만 읽어서 표준 컬럼명으로 정리
    """
    try:
        df = pd.read_csv(file_path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="utf-8")

    if df.empty:
        raise ValueError("빈 파일입니다.")

    selected_columns = {
        key: find_first_matching_column(df, candidates)
        for key, candidates in COLUMN_CANDIDATES.items()
    }

    use_df = pd.DataFrame({
        "datetime": pd.to_datetime(df[selected_columns["datetime"]], errors="coerce"),
        "wind_speed": pd.to_numeric(df[selected_columns["wind_speed"]], errors="coerce"),
        "pressure": pd.to_numeric(df[selected_columns["pressure"]], errors="coerce"),
        "air_temp": pd.to_numeric(df[selected_columns["air_temp"]], errors="coerce"),
        "sst": pd.to_numeric(df[selected_columns["sst"]], errors="coerce"),
        "wave_height": pd.to_numeric(df[selected_columns["wave_height"]], errors="coerce"),
    })

    # 결측 제거
    use_df = use_df.dropna().sort_values("datetime").reset_index(drop=True)

    # 중복 시각 제거
    use_df = use_df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    return use_df


def split_by_ratio(df: pd.DataFrame,
                   train_ratio=0.70,
                   val_ratio=0.15,
                   test_ratio=0.15):
    """
    시간 순서를 유지한 채 비율로 분할
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "비율 합은 1이어야 합니다."

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def fit_scalers(train_df: pd.DataFrame):
    feature_cols = ["sst", "air_temp", "wind_speed", "pressure", "wave_height"]
    target_col = ["sst"]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(train_df[feature_cols].values)
    y_scaler.fit(train_df[target_col].values)

    return x_scaler, y_scaler


def transform_df(df: pd.DataFrame, x_scaler: MinMaxScaler, y_scaler: MinMaxScaler):
    feature_cols = ["sst", "air_temp", "wind_speed", "pressure", "wave_height"]

    X_scaled = x_scaler.transform(df[feature_cols].values)
    y_scaled = y_scaler.transform(df[["sst"]].values).reshape(-1)

    dt = df["datetime"].values
    return X_scaled, y_scaled, dt


def create_sequences(X, y, datetimes, seq_length):
    """
    X: [N, n_features]
    y: [N]
    datetimes: [N]
    target = i+seq_length 시점의 sst
    """
    xs, ys, ts = [], [], []

    for i in range(len(X) - seq_length):
        x_seq = X[i:i + seq_length, :]
        y_target = y[i + seq_length]
        t_target = datetimes[i + seq_length]

        xs.append(x_seq)
        ys.append(y_target)
        ts.append(t_target)

    return np.array(xs), np.array(ys), np.array(ts)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(model, loader, criterion, y_scaler):
    model.eval()

    total_loss = 0.0
    preds_scaled = []
    trues_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            output = model(X_batch)
            loss = criterion(output, y_batch)

            total_loss += loss.item() * len(X_batch)

            preds_scaled.append(output.cpu().numpy())
            trues_scaled.append(y_batch.cpu().numpy())

    preds_scaled = np.vstack(preds_scaled)
    trues_scaled = np.vstack(trues_scaled)

    preds = y_scaler.inverse_transform(preds_scaled).reshape(-1)
    trues = y_scaler.inverse_transform(trues_scaled).reshape(-1)

    avg_loss = total_loss / len(loader.dataset)
    score_rmse = rmse(trues, preds)
    score_mae = mean_absolute_error(trues, preds)

    return avg_loss, score_rmse, score_mae, trues, preds


def plot_learning_curve(history, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_test_prediction(result_df, station_name, out_path):
    plt.figure(figsize=(14, 5))
    plt.scatter(result_df["datetime"], result_df["y_true"], s=4, alpha=0.6, label="Observed")
    plt.scatter(result_df["datetime"], result_df["y_pred"], s=4, alpha=0.6, label="Predicted")
    plt.xlabel("Datetime")
    plt.ylabel("SST (°C)")
    plt.title(f"[{station_name}] Test Prediction")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# 5. 정점별 학습 함수
# =========================
def train_for_station(station_name: str):
    print("=" * 80)
    print(f"정점 학습 시작: {station_name}")

    file_path = station_file_path(station_name)
    if not os.path.exists(file_path):
        print(f"파일 없음: {file_path}")
        return None

    # 1) 데이터 로드
    df = load_station_data(file_path)

    if len(df) < 500:
        print(f"데이터가 너무 적어서 제외: {station_name} ({len(df)} rows)")
        return None

    # 2) 분할
    train_df, val_df, test_df = split_by_ratio(
        df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )

    # 시퀀스 생성을 위해 최소 길이 확인
    if min(len(train_df), len(val_df), len(test_df)) <= SEQ_LENGTH:
        print(f"분할 후 데이터가 seq_length보다 작음: {station_name}")
        return None

    # 3) scaler는 train에만 fit
    x_scaler, y_scaler = fit_scalers(train_df)

    # 4) transform
    X_train_raw, y_train_raw, dt_train = transform_df(train_df, x_scaler, y_scaler)
    X_val_raw, y_val_raw, dt_val = transform_df(val_df, x_scaler, y_scaler)
    X_test_raw, y_test_raw, dt_test = transform_df(test_df, x_scaler, y_scaler)

    # 5) sequence 생성
    X_train, y_train, t_train = create_sequences(X_train_raw, y_train_raw, dt_train, SEQ_LENGTH)
    X_val, y_val, t_val = create_sequences(X_val_raw, y_val_raw, dt_val, SEQ_LENGTH)
    X_test, y_test, t_test = create_sequences(X_test_raw, y_test_raw, dt_test, SEQ_LENGTH)

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        print(f"시퀀스 생성 결과가 비어 있음: {station_name}")
        return None

    # 6) DataLoader
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 7) 모델
    model = SSTPredictor(
        n_features=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 8) 학습 루프
    best_val_rmse = np.inf
    best_epoch = 0
    best_model_state = None
    patience_count = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
        "loss_gap": []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(X_batch)

        train_loss = train_loss_sum / len(train_loader.dataset)

        # train 평가
        _, train_rmse_val, _, _, _ = evaluate_model(model, train_loader, criterion, y_scaler)

        # val 평가
        val_loss, val_rmse_val, _, _, _ = evaluate_model(model, val_loader, criterion, y_scaler)

        loss_gap = abs(train_loss - val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse_val)
        history["val_rmse"].append(val_rmse_val)
        history["loss_gap"].append(loss_gap)

        print(
            f"[{station_name}] Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Train RMSE: {train_rmse_val:.4f} | Val RMSE: {val_rmse_val:.4f} | "
            f"Gap: {loss_gap:.6f}"
        )

        # best model 기준: Validation RMSE 최소
        if val_rmse_val < best_val_rmse:
            best_val_rmse = val_rmse_val
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"[{station_name}] Early stopping at epoch {epoch}")
            break

    # 9) best model 복원
    model.load_state_dict(best_model_state)

    # 10) 최종 평가
    train_loss, train_rmse_val, train_mae_val, train_true, train_pred = evaluate_model(
        model, train_loader, criterion, y_scaler
    )
    val_loss, val_rmse_val, val_mae_val, val_true, val_pred = evaluate_model(
        model, val_loader, criterion, y_scaler
    )
    test_loss, test_rmse_val, test_mae_val, test_true, test_pred = evaluate_model(
        model, test_loader, criterion, y_scaler
    )

    # 11) 결과 저장
    station_dir = os.path.join(OUTPUT_DIR, safe_filename(station_name))
    os.makedirs(station_dir, exist_ok=True)

    # a. 예측 결과 CSV (test)
    result_df = pd.DataFrame({
        "datetime": pd.to_datetime(t_test),
        "y_true": test_true,
        "y_pred": test_pred,
        "abs_error": np.abs(test_true - test_pred)
    })
    result_csv = os.path.join(station_dir, f"{safe_filename(station_name)}_test_prediction.csv")
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    # b. 지표 저장
    metrics = {
        "station_name": station_name,
        "n_total": len(df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_train_seq": len(X_train),
        "n_val_seq": len(X_val),
        "n_test_seq": len(X_test),
        "seq_length": SEQ_LENGTH,
        "best_epoch": best_epoch,
        "best_val_rmse": float(best_val_rmse),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "train_rmse": float(train_rmse_val),
        "val_rmse": float(val_rmse_val),
        "test_rmse": float(test_rmse_val),
        "train_mae": float(train_mae_val),
        "val_mae": float(val_mae_val),
        "test_mae": float(test_mae_val),
    }

    metrics_json = os.path.join(station_dir, f"{safe_filename(station_name)}_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # c. history 저장
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(station_dir, f"{safe_filename(station_name)}_train_history.csv")
    history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")

    # d. 모델 저장
    model_path = os.path.join(station_dir, f"{safe_filename(station_name)}_best_model.pth")
    torch.save(model.state_dict(), model_path)

    # e. scaler 저장용 정보
    scaler_info = {
        "x_scaler_min": x_scaler.data_min_.tolist(),
        "x_scaler_max": x_scaler.data_max_.tolist(),
        "y_scaler_min": y_scaler.data_min_.tolist(),
        "y_scaler_max": y_scaler.data_max_.tolist(),
        "feature_order": ["sst", "air_temp", "wind_speed", "pressure", "wave_height"]
    }
    scaler_json = os.path.join(station_dir, f"{safe_filename(station_name)}_scaler_info.json")
    with open(scaler_json, "w", encoding="utf-8") as f:
        json.dump(scaler_info, f, ensure_ascii=False, indent=4)

    # f. plot 저장
    plot_learning_curve(
        history,
        os.path.join(station_dir, f"{safe_filename(station_name)}_learning_curve.png")
    )
    plot_test_prediction(
        result_df,
        station_name,
        os.path.join(station_dir, f"{safe_filename(station_name)}_test_prediction.png")
    )

    print(f"정점 학습 완료: {station_name}")
    print(f"  Best Epoch   : {best_epoch}")
    print(f"  Best Val RMSE: {best_val_rmse:.4f}")
    print(f"  Test RMSE    : {test_rmse_val:.4f}")
    print(f"  Test MAE     : {test_mae_val:.4f}")

    return metrics


# =========================
# 6. 메인 실행
# =========================
def main():
    set_seed(RANDOM_SEED)

    all_metrics = []

    for station_name in TARGET_STATIONS:
        metrics = train_for_station(station_name)
        if metrics is not None:
            all_metrics.append(metrics)

    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_csv = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        print("\n전체 정점 학습 완료")
        print(summary_df[[
            "station_name",
            "best_epoch",
            "best_val_rmse",
            "test_rmse",
            "test_mae"
        ]])
    else:
        print("학습된 정점이 없습니다. 입력 파일과 컬럼 구성을 확인하세요.")


if __name__ == "__main__":
    main()
