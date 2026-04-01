#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FNO baseline for SST field forecasting using Copernicus NetCDF inputs.

Input features (surface): thetao, so, uo, vo, zos, mlotst
Target: multi-lead future thetao fields

Example
-------
python 02.down_numerical_models/s04_train_fno_baseline.py \
  --data-dir 02.down_numerical_models/s01_copernicus_nc_data \
  --lookback 7 --leads 1,3,7,14 --epochs 50 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import xarray as xr


FEATURE_ORDER = ["thetao", "so", "uo", "vo", "zos", "mlotst"]
TARGET_NAME = "thetao"


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + 1e-6) + self.mean


class NumpyWindowDataset:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        lookback: int,
        leads: Sequence[int],
    ) -> None:
        # x shape: [T, C, H, W]
        # y/mask shape: [T, H, W]
        self.x = x
        self.y = y
        self.mask = mask
        self.lookback = lookback
        self.leads = sorted(int(v) for v in leads)
        self.max_lead = max(self.leads)

        # t is the last input time index.
        # targets are t + lead for each lead.
        self.indices = list(range(lookback - 1, x.shape[0] - self.max_lead))
        if not self.indices:
            raise ValueError("샘플을 만들 수 없습니다. lookback/leads 값을 줄이세요.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = self.indices[idx]
        start = t - self.lookback + 1
        end = t + 1

        # [L, C, H, W] -> [L*C, H, W]
        x_seq = self.x[start:end]
        l, c, h, w = x_seq.shape
        x_flat = x_seq.reshape(l * c, h, w)

        y_list = []
        m_list = []
        for lead in self.leads:
            target_t = t + lead
            y_list.append(self.y[target_t])
            m_list.append(self.mask[target_t])

        # [K, H, W]
        y_multi = np.stack(y_list, axis=0)
        m_multi = np.stack(m_list, axis=0)
        return x_flat.astype(np.float32), y_multi.astype(np.float32), m_multi.astype(np.float32)


def _open_surface_dataarray(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    da = ds[var_name]
    if "depth" in da.dims:
        da = da.isel(depth=0)
    if da.dims != ("time", "latitude", "longitude"):
        da = da.transpose("time", "latitude", "longitude")
    return da


def load_feature_stack(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thetao_path = data_dir / "copernicus_phy_thetao_20210101_20251231.nc"
    so_path = data_dir / "copernicus_phy_so_20210101_20251231.nc"
    cur_path = data_dir / "copernicus_phy_cur_20210101_20251231.nc"
    vars2d_path = data_dir / "copernicus_phy_2d_vars_20210101_20251231.nc"

    for p in [thetao_path, so_path, cur_path, vars2d_path]:
        if not p.exists():
            raise FileNotFoundError(f"필수 NetCDF 파일이 없습니다: {p}")

    with xr.open_dataset(thetao_path) as ds_t, xr.open_dataset(so_path) as ds_s, xr.open_dataset(cur_path) as ds_c, xr.open_dataset(vars2d_path) as ds_2:
        da_thetao = _open_surface_dataarray(ds_t, "thetao")
        da_so = _open_surface_dataarray(ds_s, "so")
        da_uo = _open_surface_dataarray(ds_c, "uo")
        da_vo = _open_surface_dataarray(ds_c, "vo")
        da_zos = _open_surface_dataarray(ds_2, "zos")
        da_mlotst = _open_surface_dataarray(ds_2, "mlotst")

        ref_time = da_thetao["time"]
        da_so = da_so.sel(time=ref_time)
        da_uo = da_uo.sel(time=ref_time)
        da_vo = da_vo.sel(time=ref_time)
        da_zos = da_zos.sel(time=ref_time)
        da_mlotst = da_mlotst.sel(time=ref_time)

        times = ref_time.values
        lat = da_thetao["latitude"].values.astype(np.float32)
        lon = da_thetao["longitude"].values.astype(np.float32)

        feats = [
            da_thetao.values,
            da_so.values,
            da_uo.values,
            da_vo.values,
            da_zos.values,
            da_mlotst.values,
        ]
        x = np.stack([np.asarray(f, dtype=np.float32) for f in feats], axis=1)  # [T, C, H, W]
        y = np.asarray(da_thetao.values, dtype=np.float32)  # [T, H, W]

    mask = np.isfinite(y).astype(np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    lat_n = (lat - lat.min()) / (lat.max() - lat.min() + 1e-6)
    lon_n = (lon - lon.min()) / (lon.max() - lon.min() + 1e-6)
    lon_grid, lat_grid = np.meshgrid(lon_n, lat_n)
    coord = np.stack([lat_grid, lon_grid], axis=0).astype(np.float32)  # [2, H, W]
    coord_expand = np.repeat(coord[None, ...], repeats=x.shape[0], axis=0)

    x = np.concatenate([x, coord_expand], axis=1)
    return x, y, mask, times, lat, lon


def split_indices(n_time: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n_time * train_ratio)
    val_end = int(n_time * (train_ratio + val_ratio))
    if train_end < 1 or val_end <= train_end or val_end >= n_time:
        raise ValueError("train/val 비율 설정이 잘못되었습니다.")
    tr = np.arange(0, train_end)
    va = np.arange(train_end, val_end)
    te = np.arange(val_end, n_time)
    return tr, va, te


def fit_scaler_from_train(x: np.ndarray, train_idx: np.ndarray) -> StandardScaler:
    train_x = x[train_idx]
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = train_x.std(axis=(0, 2, 3), keepdims=True)
    return StandardScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


def build_model_and_utils(
    in_channels: int,
    out_channels: int,
    width: int,
    modes1: int,
    modes2: int,
    device: str,
):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "PyTorch가 설치되어 있지 않습니다.\n"
            f"- 현재 Python 버전: {py_ver}\n"
            "- base 환경에서 python=3.13 pin 상태면 conda 충돌이 자주 발생합니다.\n"
            "- 권장: `conda env create -f 02.down_numerical_models/environment_fno.yml`\n"
            "- 이후: `conda activate fno` 후 스크립트를 다시 실행하세요."
        ) from exc

    class SpectralConv2d(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, m1: int, m2: int):
            super().__init__()
            self.out_ch = out_ch
            self.m1 = m1
            self.m2 = m2
            scale = 1.0 / (in_ch * out_ch)
            self.weights = nn.Parameter(scale * torch.randn(in_ch, out_ch, m1, m2, dtype=torch.cfloat))

        def compl_mul2d(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bixy,ioxy->boxy", a, b)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, _, h, w = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros(batch, self.out_ch, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, : self.m1, : self.m2] = self.compl_mul2d(
                x_ft[:, :, : self.m1, : self.m2],
                self.weights,
            )
            return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")

    class FNOBlock(nn.Module):
        def __init__(self, width_: int, m1: int, m2: int):
            super().__init__()
            self.spec = SpectralConv2d(width_, width_, m1, m2)
            self.w = nn.Conv2d(width_, width_, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.gelu(self.spec(x) + self.w(x))

    class FNO2d(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.lift = nn.Conv2d(in_ch, width, kernel_size=1)
            self.block1 = FNOBlock(width, modes1, modes2)
            self.block2 = FNOBlock(width, modes1, modes2)
            self.block3 = FNOBlock(width, modes1, modes2)
            self.proj1 = nn.Conv2d(width, width, kernel_size=1)
            self.proj2 = nn.Conv2d(width, out_ch, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.lift(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = F.gelu(self.proj1(x))
            x = self.proj2(x)
            return x

    model = FNO2d(in_channels, out_channels).to(device)
    return model, torch, nn


def iter_minibatch(dataset: NumpyWindowDataset, batch_size: int, shuffle: bool, seed: int = 42):
    n = len(dataset)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for s in range(0, n, batch_size):
        b = idx[s : s + batch_size]
        xs, ys, ms = [], [], []
        for i in b:
            x_i, y_i, m_i = dataset[i]
            xs.append(x_i)
            ys.append(y_i)
            ms.append(m_i)
        yield np.stack(xs, axis=0), np.stack(ys, axis=0), np.stack(ms, axis=0)


def compute_metrics_by_lead_np(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    leads: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    # pred/true/mask: [N, K, H, W]
    result: Dict[str, Dict[str, float]] = {}
    for k, lead in enumerate(leads):
        valid = mask[:, k, :, :] > 0.5
        if not np.any(valid):
            result[f"lead_{lead}d"] = {"mae": float("nan"), "rmse": float("nan")}
            continue
        err = pred[:, k, :, :][valid] - true[:, k, :, :][valid]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        result[f"lead_{lead}d"] = {"mae": mae, "rmse": rmse}

    valid_all = mask > 0.5
    if np.any(valid_all):
        err_all = pred[valid_all] - true[valid_all]
        result["overall"] = {
            "mae": float(np.mean(np.abs(err_all))),
            "rmse": float(np.sqrt(np.mean(err_all ** 2))),
        }
    else:
        result["overall"] = {"mae": float("nan"), "rmse": float("nan")}
    return result


def compute_error_maps_by_lead_np(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    leads: Sequence[int],
) -> Dict[str, Dict[str, np.ndarray]]:
    # pred/true/mask: [N, K, H, W]
    error_maps: Dict[str, Dict[str, np.ndarray]] = {}
    for k, lead in enumerate(leads):
        diff = pred[:, k, :, :] - true[:, k, :, :]
        valid = mask[:, k, :, :] > 0.5
        valid_count = valid.sum(axis=0).astype(np.float32)  # [H, W]

        abs_sum = np.where(valid, np.abs(diff), 0.0).sum(axis=0)
        sq_sum = np.where(valid, diff ** 2, 0.0).sum(axis=0)

        mae_map = np.full_like(abs_sum, np.nan, dtype=np.float32)
        rmse_map = np.full_like(abs_sum, np.nan, dtype=np.float32)
        positive = valid_count > 0
        mae_map[positive] = (abs_sum[positive] / valid_count[positive]).astype(np.float32)
        rmse_map[positive] = np.sqrt(sq_sum[positive] / valid_count[positive]).astype(np.float32)

        error_maps[f"lead_{lead}d"] = {
            "mae_map": mae_map,
            "rmse_map": rmse_map,
            "valid_count": valid_count,
        }
    return error_maps


def parse_leads(leads_text: str) -> List[int]:
    leads = []
    for token in leads_text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("리드타임은 1 이상 정수여야 합니다.")
        leads.append(value)
    if not leads:
        raise ValueError("유효한 리드타임이 없습니다. 예: --leads 1,3,7,14")
    return sorted(list(dict.fromkeys(leads)))


def print_input_snapshot(
    train_set: NumpyWindowDataset,
    leads: Sequence[int],
    feature_names: Sequence[str],
    lookback: int,
) -> None:
    x0, y0, m0 = train_set[0]
    h, w = x0.shape[-2], x0.shape[-1]
    channels_per_step = x0.shape[0] // lookback

    print("\n=== Input Snapshot ===")
    print(f"lookback: {lookback} days")
    print(f"leads   : {list(leads)} days")
    print(f"grid    : H={h}, W={w}")
    print(f"x shape : {x0.shape} (flattened time-channel)")
    print(f"y shape : {y0.shape} (K, H, W)")
    print(f"m shape : {m0.shape} (K, H, W)")
    print(f"channels per time-step: {channels_per_step}")
    print(f"feature order per step: {list(feature_names)}")
    print(f"x stats: min={x0.min():.4f}, max={x0.max():.4f}, mean={x0.mean():.4f}, std={x0.std():.4f}")

    for k, lead in enumerate(leads):
        valid = m0[k] > 0.5
        ocean_ratio = float(valid.mean())
        if np.any(valid):
            yy = y0[k][valid]
            print(
                f"lead {lead:>2d}d target stats (ocean only): "
                f"min={yy.min():.4f}, max={yy.max():.4f}, mean={yy.mean():.4f}, std={yy.std():.4f}, "
                f"ocean_ratio={ocean_ratio:.3f}"
            )
        else:
            print(f"lead {lead:>2d}d target stats: no valid ocean grid")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    leads = parse_leads(args.leads)
    x, y, mask, times, lat, lon = load_feature_stack(Path(args.data_dir))
    n_time, _, h, w = x.shape

    tr_idx, va_idx, te_idx = split_indices(n_time, args.train_ratio, args.val_ratio)

    scaler = fit_scaler_from_train(x, tr_idx)
    x = scaler.transform(x)

    x = np.concatenate([x, mask[:, None, :, :]], axis=1)

    train_set = NumpyWindowDataset(x[tr_idx], y[tr_idx], mask[tr_idx], args.lookback, leads)
    val_set = NumpyWindowDataset(x[va_idx], y[va_idx], mask[va_idx], args.lookback, leads)
    test_set = NumpyWindowDataset(x[te_idx], y[te_idx], mask[te_idx], args.lookback, leads)

    if args.inspect_input or args.inspect_only:
        print_input_snapshot(
            train_set=train_set,
            leads=leads,
            feature_names=FEATURE_ORDER + ["lat", "lon", "ocean_mask"],
            lookback=args.lookback,
        )
    if args.inspect_only:
        print("inspect-only 모드: 학습을 종료합니다.")
        return

    device = "cuda" if (args.device == "auto" and _torch_cuda_available()) else ("cpu" if args.device == "auto" else args.device)

    model, torch, _ = build_model_and_utils(
        in_channels=x.shape[1] * args.lookback,
        out_channels=len(leads),
        width=args.width,
        modes1=args.modes1,
        modes2=args.modes2,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.lr_scheduler_min_lr,
        )

    def masked_mse(pred: "torch.Tensor", true: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
        # pred/true/mask: [B, K, H, W]
        diff2 = (pred - true) ** 2
        diff2 = diff2 * m
        denom = torch.clamp(m.sum(), min=1.0)
        return diff2.sum() / denom

    best_val = math.inf
    no_improve_count = 0
    history: List[Dict[str, float]] = []
    best_path = out_dir / "fno_baseline_multilead_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, mb in iter_minibatch(train_set, args.batch_size, shuffle=True, seed=args.seed + epoch):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)
            mb_t = torch.from_numpy(mb).to(device)

            optimizer.zero_grad()
            pred = model(xb_t)
            loss = masked_mse(pred, yb_t, mb_t)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, mb in iter_minibatch(val_set, args.batch_size, shuffle=False):
                xb_t = torch.from_numpy(xb).to(device)
                yb_t = torch.from_numpy(yb).to(device)
                mb_t = torch.from_numpy(mb).to(device)
                pred = model(xb_t)
                loss = masked_mse(pred, yb_t, mb_t)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        val_loss = float(np.mean(val_losses)) if val_losses else math.nan
        if scheduler is not None and not math.isnan(val_loss):
            scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.3e}")

        improved = (not math.isnan(val_loss)) and (val_loss < (best_val - args.early_stopping_min_delta))
        if improved:
            best_val = val_loss
            no_improve_count = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_path)
        else:
            no_improve_count += 1

        if args.early_stopping_patience > 0 and no_improve_count >= args.early_stopping_patience:
            print(
                f"[Early Stopping] epoch={epoch}에서 중단 "
                f"(patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta})"
            )
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_pred, all_true, all_mask = [], [], []
    with torch.no_grad():
        for xb, yb, mb in iter_minibatch(test_set, args.batch_size, shuffle=False):
            xb_t = torch.from_numpy(xb).to(device)
            pred = model(xb_t).cpu().numpy()  # [B, K, H, W]
            all_pred.append(pred)
            all_true.append(yb)
            all_mask.append(mb)

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    mask_np = np.concatenate(all_mask, axis=0)

    metrics = compute_metrics_by_lead_np(pred_np, true_np, mask_np, leads)
    error_maps = compute_error_maps_by_lead_np(pred_np, true_np, mask_np, leads)

    print("\n=== Test Metrics (thetao, ocean mask) ===")
    for lead in leads:
        m = metrics[f"lead_{lead}d"]
        print(f"Lead {lead:>2d}d | MAE: {m['mae']:.4f} degC | RMSE: {m['rmse']:.4f} degC")
    print(f"Overall  | MAE: {metrics['overall']['mae']:.4f} degC | RMSE: {metrics['overall']['rmse']:.4f} degC")

    _save_outputs(
        out_dir=out_dir,
        args=args,
        leads=leads,
        history=history,
        metrics=metrics,
        scaler=scaler,
        feature_names=FEATURE_ORDER + ["lat", "lon", "ocean_mask"],
        ckpt_epoch=int(ckpt.get("epoch", -1)),
        n_train=len(train_set),
        n_val=len(val_set),
        n_test=len(test_set),
        grid_shape=(h, w),
        n_time=n_time,
        time_start=str(times[0]),
        time_end=str(times[-1]),
        lat=lat,
        lon=lon,
        error_maps=error_maps,
    )


def _save_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    leads: Sequence[int],
    history: List[Dict[str, float]],
    metrics: Dict[str, Dict[str, float]],
    scaler: StandardScaler,
    feature_names: List[str],
    ckpt_epoch: int,
    n_train: int,
    n_val: int,
    n_test: int,
    grid_shape: Tuple[int, int],
    n_time: int,
    time_start: str,
    time_end: str,
    lat: np.ndarray,
    lon: np.ndarray,
    error_maps: Dict[str, Dict[str, np.ndarray]],
) -> None:
    history_csv = out_dir / "train_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(history)

    report = {
        "target": TARGET_NAME,
        "best_epoch": ckpt_epoch,
        "metrics": metrics,
        "data": {
            "n_time": n_time,
            "time_start": time_start,
            "time_end": time_end,
            "grid_shape": {"lat": grid_shape[0], "lon": grid_shape[1]},
            "feature_names": feature_names,
            "leads_days": list(leads),
            "samples": {"train": n_train, "val": n_val, "test": n_test},
        },
        "config": vars(args),
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    scaler_npz = out_dir / "scaler.npz"
    np.savez(scaler_npz, mean=scaler.mean, std=scaler.std)
    _save_error_maps(out_dir=out_dir, error_maps=error_maps, lat=lat, lon=lon)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["epoch"] for r in history]
        tr = [r["train_loss"] for r in history]
        va = [r["val_loss"] for r in history]
        lr_values = [r["lr"] for r in history]

        plt.figure(figsize=(7, 4))
        plt.plot(xs, tr, label="train")
        plt.plot(xs, va, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Masked MSE")
        plt.title("FNO Multi-Lead Baseline Learning Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "learning_curve.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 3.8))
        plt.plot(xs, lr_values, color="#2f4b7c")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "lr_curve.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"[경고] learning curve 저장 실패: {exc}")


def _save_error_maps(
    out_dir: Path,
    error_maps: Dict[str, Dict[str, np.ndarray]],
    lat: np.ndarray,
    lon: np.ndarray,
) -> None:
    maps_dir = out_dir / "error_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Save raw arrays for later analysis.
    npz_payload: Dict[str, np.ndarray] = {}
    for lead_key, payload in error_maps.items():
        npz_payload[f"{lead_key}_mae"] = payload["mae_map"]
        npz_payload[f"{lead_key}_rmse"] = payload["rmse_map"]
        npz_payload[f"{lead_key}_valid_count"] = payload["valid_count"]
    np.savez(maps_dir / "error_maps.npz", lat=lat, lon=lon, **npz_payload)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        for lead_key, payload in error_maps.items():
            for metric_key in ("mae_map", "rmse_map"):
                arr = payload[metric_key]
                if not np.isfinite(arr).any():
                    continue

                fig, ax = plt.subplots(figsize=(8, 6))
                mesh = ax.pcolormesh(lon_grid, lat_grid, arr, shading="auto", cmap="inferno")
                cbar = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.9)
                cbar.set_label("degC")
                ax.set_title(f"{lead_key.upper()} {metric_key.replace('_map', '').upper()} Error Map")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(maps_dir / f"{lead_key}_{metric_key}.png", dpi=150)
                plt.close(fig)
    except Exception as exc:
        print(f"[경고] 오차맵 PNG 저장 실패: {exc}")


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FNO baseline training for multi-lead SST field prediction")
    parser.add_argument("--data-dir", type=str, default="02.down_numerical_models/s01_copernicus_nc_data")
    parser.add_argument("--outdir", type=str, default="02.down_numerical_models/s04_fno_baseline_multilead")

    parser.add_argument("--lookback", type=int, default=7, help="입력 과거 일수")
    parser.add_argument("--leads", type=str, default="1,3,7,14", help="예측 리드타임(일), 콤마 구분")

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5, help="ReduceLROnPlateau patience (0이면 비활성화)")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5, help="학습률 감소 비율 (new_lr = lr * factor)")
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6, help="학습률 하한")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="val loss 개선 없을 때 기다릴 epoch 수 (0이면 비활성화)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="개선으로 인정할 최소 val loss 감소폭")

    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-input", action="store_true", help="학습 시작 전 첫 샘플 입력/타깃 통계를 출력")
    parser.add_argument("--inspect-only", action="store_true", help="입력 통계만 출력하고 학습은 수행하지 않음")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
