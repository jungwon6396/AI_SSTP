#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FNO 멀티 실험 순차 실행 스크립트.

목표:
1) s04_train_fno_baseline.py를 여러 설정으로 순차 실행
2) 각 실험 결과를 서로 다른 폴더에 저장
3) 실험 요약 CSV 자동 생성
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        v = int(token)
        if v <= 0:
            raise ValueError(f"양수 정수만 허용됩니다: {text}")
        values.append(v)
    if not values:
        raise ValueError(f"유효한 값이 없습니다: {text}")
    return sorted(list(dict.fromkeys(values)))


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        v = float(token)
        if v < 0:
            raise ValueError(f"0 이상 실수만 허용됩니다: {text}")
        values.append(v)
    if not values:
        raise ValueError(f"유효한 값이 없습니다: {text}")
    # order-preserving unique
    out: List[float] = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def _wd_label(wd: float) -> str:
    # filesystem-safe short label
    return format(wd, ".0e").replace("-", "m").replace("+", "p")


def default_experiments(
    lookbacks: List[int],
    batch_sizes: List[int],
    weight_decays: List[float],
) -> List[Dict[str, object]]:
    # 아래 실험들은 lookback + batch size + weight decay 축을 동시에 비교하기 위한 기본 세트.
    # 이유:
    # - lookback: 과거 정보 길이가 예측 정확도에 미치는 영향 확인
    # - batch_size: 업데이트 노이즈/안정성이 일반화 성능에 미치는 영향 확인
    # - weight_decay: 과적합 억제 강도의 영향 확인
    # 같은 데이터/동일 파이프라인에서 세 축을 함께 스윕하면 운영에 맞는 조합을 찾기 쉽다.
    exps: List[Dict[str, object]] = []
    for lb in lookbacks:
        for bs in batch_sizes:
            for wd in weight_decays:
                exps.append(
                    {
                        "name": f"lb{lb:02d}_bs{bs:02d}_wd{_wd_label(wd)}",
                        "lookback": lb,
                        "batch_size": bs,
                        "weight_decay": wd,
                    }
                )
    return exps


def build_command(
    python_exe: str,
    train_script: Path,
    data_dir: Path,
    outdir: Path,
    leads: str,
    exp: Dict[str, object],
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_exe,
        str(train_script),
        "--data-dir",
        str(data_dir),
        "--outdir",
        str(outdir),
        "--lookback",
        str(exp["lookback"]),
        "--leads",
        leads,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(exp["batch_size"]),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(exp["weight_decay"]),
        "--width",
        str(args.width),
        "--modes1",
        str(args.modes1),
        "--modes2",
        str(args.modes2),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--lr-scheduler-patience",
        str(args.lr_scheduler_patience),
        "--lr-scheduler-factor",
        str(args.lr_scheduler_factor),
        "--lr-scheduler-min-lr",
        str(args.lr_scheduler_min_lr),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
    ]
    return cmd


def read_report(report_path: Path) -> Dict[str, object]:
    if not report_path.exists():
        return {}
    try:
        with report_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple FNO experiments sequentially")
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="학습 실행에 사용할 Python 실행파일")
    parser.add_argument("--train-script", type=str, default="02.down_numerical_models/s04_train_fno_baseline.py")
    parser.add_argument("--data-dir", type=str, default="02.down_numerical_models/s01_copernicus_nc_data")
    parser.add_argument("--base-outdir", type=str, default="02.down_numerical_models/s05_fno_runs")
    parser.add_argument("--leads", type=str, default="1,3,7,14")
    parser.add_argument("--lookbacks", type=str, default="3,7,14,30", help="실험할 lookback 목록 (콤마 구분)")
    parser.add_argument("--batch-sizes", type=str, default="4,8,16", help="실험할 batch size 목록 (콤마 구분)")
    parser.add_argument("--weight-decays", type=str, default="0,1e-6,1e-5,1e-4", help="실험할 weight_decay 목록 (콤마 구분)")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8, help="(호환용) 개별 실행 시 기본값")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="(호환용) 개별 실행 시 기본값")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--continue-on-error", action="store_true", help="실험 하나 실패해도 다음 실험 계속 진행")
    args = parser.parse_args()

    train_script = Path(args.train_script).resolve()
    data_dir = Path(args.data_dir).resolve()
    base_outdir = Path(args.base_outdir).resolve()

    if not train_script.exists():
        raise FileNotFoundError(f"학습 스크립트가 없습니다: {train_script}")
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {data_dir}")

    # 순차 실행 이유:
    # 1) GPU/메모리 자원 충돌을 피해서 안정적으로 완료하기 위해
    # 2) 각 실험 로그를 시간 순서대로 읽기 쉽게 만들기 위해
    # 3) 한 실험의 결과 폴더와 체크포인트를 명확히 분리하기 위해
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = base_outdir / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    lookbacks = parse_int_list(args.lookbacks)
    batch_sizes = parse_int_list(args.batch_sizes)
    weight_decays = parse_float_list(args.weight_decays)
    experiments = default_experiments(
        lookbacks=lookbacks,
        batch_sizes=batch_sizes,
        weight_decays=weight_decays,
    )
    summary_rows: List[Dict[str, object]] = []

    print(f"[INFO] run root: {run_root}")
    print(f"[INFO] experiments: {len(experiments)}")

    for i, exp in enumerate(experiments, start=1):
        exp_name = str(exp["name"])
        exp_outdir = run_root / exp_name
        exp_outdir.mkdir(parents=True, exist_ok=True)

        # 설정 이유:
        # - name: 폴더명/리포트 식별용
        # - lookback: 입력 과거 길이 영향 비교
        cmd = build_command(
            python_exe=args.python_exe,
            train_script=train_script,
            data_dir=data_dir,
            outdir=exp_outdir,
            leads=args.leads,
            exp=exp,
            args=args,
        )

        print("\n" + "=" * 72)
        print(f"[RUN {i}/{len(experiments)}] {exp_name}")
        print(" ".join(cmd))
        print("=" * 72)

        completed = subprocess.run(cmd, text=True)
        status = "ok" if completed.returncode == 0 else "failed"

        report = read_report(exp_outdir / "report.json")
        metrics = report.get("metrics", {}) if isinstance(report, dict) else {}

        row: Dict[str, object] = {
            "run_id": run_id,
            "experiment": exp_name,
            "lookback": exp["lookback"],
            "batch_size": exp["batch_size"],
            "weight_decay": exp["weight_decay"],
            "leads": args.leads,
            "status": status,
            "return_code": completed.returncode,
            "outdir": str(exp_outdir),
            "overall_mae": ((metrics.get("overall") or {}).get("mae") if isinstance(metrics, dict) else None),
            "overall_rmse": ((metrics.get("overall") or {}).get("rmse") if isinstance(metrics, dict) else None),
        }

        if isinstance(metrics, dict):
            for lead in [1, 3, 7, 14]:
                key = f"lead_{lead}d"
                row[f"{key}_mae"] = (metrics.get(key) or {}).get("mae")
                row[f"{key}_rmse"] = (metrics.get(key) or {}).get("rmse")

        summary_rows.append(row)

        if completed.returncode != 0 and not args.continue_on_error:
            print(f"[ERROR] {exp_name} 실패. --continue-on-error 옵션이 없어 중단합니다.")
            break

    summary_path = run_root / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\n[INFO] 완료")
    print(f"- run root  : {run_root}")
    print(f"- summary   : {summary_path}")


if __name__ == "__main__":
    main()
