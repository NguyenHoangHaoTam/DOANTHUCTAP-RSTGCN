"""
End-to-end demo pipeline for the rail-delay RSTGCN project.

This script orchestrates three main phases:
  1. Preprocess raw CSV templates into tensors (convert_to_rstgcn)
  2. Train the RSTGCN model on the processed tensors
  3. Run inference on the validation split to produce a CSV of predictions

Usage:
    python -m scripts.workflows.demo_rstgcn_pipeline \
        --templates data/templates_all \
        --processed data/processed \
        --runs runs/demo_rstgcn \
        --slot 60 --window 4 --epochs 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.preprocessing.convert_to_rstgcn import main as convert_main
from scripts.modeling.train_rstgcn import main as train_main
from scripts.modeling.infer_rstgcn import main as infer_main


def _banner(title: str) -> None:
    line = "=" * max(40, len(title) + 8)
    print(f"\n{line}\n▶ {title}\n{line}")


def _time_it(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        dt = time.time() - t0
        print(f"⏱  Done in {dt:.1f}s")
        return result

    return wrapper


@dataclass
class DemoConfig:
    templates: str
    processed: str
    runs: str
    slot: int
    window: int
    target: int
    epochs: int
    batch: int
    hidden: int
    lr: float
    train_ratio: float
    cpu: bool
    skip_preprocess: bool
    skip_train: bool
    skip_infer: bool
    force_preprocess: bool
    metrics: Optional[str]


def _resolve_metrics_path(cfg: DemoConfig) -> Optional[str]:
    if cfg.metrics:
        return cfg.metrics
    return os.path.join(cfg.runs, "metrics.csv")


@_time_it
def preprocess(cfg: DemoConfig) -> None:
    if cfg.skip_preprocess and all(
        os.path.exists(os.path.join(cfg.processed, f))
        for f in ("dataset.npy", "adj.npy", "dist.npy", "freq.npy", "meta.json")
    ):
        print("⚡ Skip preprocess (artifacts already exist)")
        return

    stations = os.path.join(cfg.templates, "stations.csv")
    edges = os.path.join(cfg.templates, "edges.csv")
    stops = os.path.join(cfg.templates, "stop_times_augmented.csv")
    if not os.path.exists(stops):
        stops = os.path.join(cfg.templates, "stop_times.csv")

    for path in (stations, edges, stops):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required CSV: {path}")

    if os.path.isdir(cfg.processed) and os.listdir(cfg.processed) and not cfg.force_preprocess:
        print(f"ℹ️  Processed tensors already in {cfg.processed} (use --force-preprocess to rebuild)")
        return

    ns = argparse.Namespace(
        stations=stations,
        edges=edges,
        stops=stops,
        slot=cfg.slot,
        outdir=cfg.processed,
        **{"st_id_col": None, "lat_col": None, "lon_col": None},
    )
    _banner("Step 1: Preprocess to tensors")
    convert_main(ns)


@_time_it
def train(cfg: DemoConfig) -> str:
    os.makedirs(cfg.runs, exist_ok=True)
    metrics_csv = _resolve_metrics_path(cfg)

    ns = argparse.Namespace(
        data=cfg.processed,
        window=cfg.window,
        hidden=cfg.hidden,
        epochs=cfg.epochs,
        batch=cfg.batch,
        lr=cfg.lr,
        train_ratio=cfg.train_ratio,
        cpu=cfg.cpu,
        outdir=cfg.runs,
        target=cfg.target,
        metrics_csv=metrics_csv,
    )

    _banner("Step 2: Train RSTGCN")
    train_main(ns)
    ckpt = os.path.join(cfg.runs, "rstgcn_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Expected checkpoint not found: {ckpt}")
    return ckpt


@_time_it
def infer(cfg: DemoConfig, ckpt_path: str) -> str:
    out_csv = os.path.join(cfg.runs, "val_predictions.csv")
    ns = argparse.Namespace(
        data=cfg.processed,
        ckpt=ckpt_path,
        out_csv=out_csv,
        hidden=cfg.hidden,
        window=cfg.window,
        target=cfg.target,
        batch=cfg.batch,
        train_ratio=cfg.train_ratio,
    )
    _banner("Step 3: Inference on validation set")
    infer_main(ns)
    return out_csv


def parse_args() -> DemoConfig:
    p = argparse.ArgumentParser(description="One-click RSTGCN demo pipeline")
    p.add_argument("--templates", default="data/templates_all")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--runs", default="runs/demo_rstgcn")
    p.add_argument("--slot", type=int, default=60)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--target", type=int, default=4, help="Feature index to predict (default: headway)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-ratio", type=float, dest="train_ratio", default=0.7)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--metrics", default=None, help="Optional explicit metrics CSV path")

    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-infer", action="store_true")
    p.add_argument("--force-preprocess", action="store_true")

    args = p.parse_args()
    return DemoConfig(
        templates=args.templates,
        processed=args.processed,
        runs=args.runs,
        slot=args.slot,
        window=args.window,
        target=args.target,
        epochs=args.epochs,
        batch=args.batch,
        hidden=args.hidden,
        lr=args.lr,
        train_ratio=args.train_ratio,
        cpu=args.cpu,
        skip_preprocess=args.skip_preprocess,
        skip_train=args.skip_train,
        skip_infer=args.skip_infer,
        force_preprocess=args.force_preprocess,
        metrics=args.metrics,
    )


def main():
    cfg = parse_args()
    os.makedirs(cfg.processed, exist_ok=True)

    preprocess(cfg)

    ckpt = os.path.join(cfg.runs, "rstgcn_best.pt")
    if cfg.skip_train and not os.path.exists(ckpt):
        raise FileNotFoundError("No trained weights found; cannot skip training.")

    if not cfg.skip_train:
        ckpt = train(cfg)

    if cfg.skip_infer:
        print("⚡ Skip inference as requested.")
        return

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    pred_csv = infer(cfg, ckpt)
    print("\n🎉 Demo completed!")
    print(f"   • Processed tensors : {cfg.processed}")
    print(f"   • Checkpoint        : {ckpt}")
    metrics_csv = _resolve_metrics_path(cfg)
    if metrics_csv and os.path.exists(metrics_csv):
        print(f"   • Metrics CSV       : {metrics_csv}")
    print(f"   • Predictions CSV   : {pred_csv}")
    print("Next: `streamlit run app.py` to visualize or open the CSVs for inspection.")


if __name__ == "__main__":
    main()

