"""
control/train_mpc.py
====================
Training pipeline for the Residual Correction Network (RCN).

Usage:
  python -m control.train_mpc --config control/config.yaml
  python -m control.train_mpc --config control/config.yaml --resume weights/control/last.pt

Features:
  - Mixed-precision (AMP) training
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Early stopping (patience=10)
  - Tensorboard + CSV logging
  - ONNX export at end of training
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import yaml

from control.dataset import build_dataloaders
from control.model import ResidualCorrectionNet
from control.loss import ControlLoss

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("control.train")


# ─────────────────────────────────────────────────────────────── utilities ────
def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int,
    max_epochs: int,
    base_lr: float,
) -> None:
    """Apply linear warmup then cosine decay directly."""
    if epoch < warmup_epochs:
        scale = (epoch + 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


# ─────────────────────────────────────────────────────────── train one epoch ──
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: ControlLoss,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    sub_losses: Dict[str, float] = {
        "steering": 0.0, "throttle": 0.0, "brake": 0.0,
        "safety": 0.0, "comfort": 0.0,
    }
    prev_pred: Optional[torch.Tensor] = None
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            pred = model(x)
            loss, parts = criterion(pred, y, prev_pred=prev_pred)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in parts.items():
            sub_losses[k] = sub_losses.get(k, 0.0) + v

        prev_pred = pred.detach()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        **{k: v / n for k, v in sub_losses.items()},
    }


# ─────────────────────────────────────────────────────────── validate ─────────
@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: ControlLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    sub_losses: Dict[str, float] = {}
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)
        loss, parts = criterion(pred, y)
        total_loss += loss.item()
        for k, v in parts.items():
            sub_losses[k] = sub_losses.get(k, 0.0) + v
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        **{k: v / n for k, v in sub_losses.items()},
    }


# ─────────────────────────────────────────────────────────────────── main ─────
def main(config_path: str, resume: Optional[str] = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tr = cfg["training"]
    set_seed(tr["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── dirs ──────────────────────────────────────────────────────────────
    ckpt_dir = Path(tr["checkpoint_dir"])
    log_dir  = Path(tr["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────
    logger.info("Building data loaders …")
    loaders = build_dataloaders(config_path)
    logger.info(
        "Train: %d batches | Val: %d batches | Test: %d batches",
        len(loaders["train"]), len(loaders["val"]), len(loaders["test"]),
    )

    # ── model ─────────────────────────────────────────────────────────────
    model = ResidualCorrectionNet(input_dim=9, hidden_dim=64, dropout=0.1).to(device)
    logger.info("Model: ResidualCorrectionNet | %d parameters", model.count_parameters())

    # ── optimiser + scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
    )
    criterion = ControlLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(log_dir=str(log_dir))

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 10

    # ── resume from checkpoint ────────────────────────────────────────────
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("Resumed from %s | epoch %d", resume, start_epoch)

    # ── CSV log ───────────────────────────────────────────────────────────
    csv_path = log_dir / "train_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "lr", "train_loss", "val_loss",
        "steer", "throttle", "brake", "safety", "comfort",
    ])

    # ── training loop ─────────────────────────────────────────────────────
    logger.info("Starting training for %d epochs …", tr["epochs"])

    for epoch in range(start_epoch, tr["epochs"]):
        cosine_warmup_lr(optimizer, epoch, tr["warmup_epochs"], tr["epochs"], tr["learning_rate"])
        current_lr = optimizer.param_groups[0]["lr"]

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            scaler, device, tr["grad_clip"], epoch,
        )
        val_metrics = validate(model, loaders["val"], criterion, device)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %03d/%03d | LR=%.2e | Train=%.4f | Val=%.4f | "
            "Steer=%.4f | Thr=%.4f | Brk=%.4f | %.1fs",
            epoch + 1, tr["epochs"], current_lr,
            train_metrics["loss"], val_metrics["loss"],
            val_metrics.get("steering", 0), val_metrics.get("throttle", 0),
            val_metrics.get("brake", 0), elapsed,
        )

        # ── Tensorboard ───────────────────────────────────────────────
        writer.add_scalars("Loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        for k in ("steering", "throttle", "brake", "safety", "comfort"):
            if k in val_metrics:
                writer.add_scalar(f"Val/{k}", val_metrics[k], epoch)

        # ── CSV ───────────────────────────────────────────────────────
        csv_writer.writerow([
            epoch + 1, f"{current_lr:.6f}",
            f"{train_metrics['loss']:.6f}", f"{val_metrics['loss']:.6f}",
            f"{val_metrics.get('steering', 0):.6f}",
            f"{val_metrics.get('throttle', 0):.6f}",
            f"{val_metrics.get('brake', 0):.6f}",
            f"{val_metrics.get('safety', 0):.6f}",
            f"{val_metrics.get('comfort', 0):.6f}",
        ])
        csv_file.flush()

        # ── Checkpoint ────────────────────────────────────────────────
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "cfg": cfg,
            }, ckpt_dir / "best.pt")
            logger.info("  ↳ New best model saved (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "cfg": cfg,
        }, ckpt_dir / "last.pt")

        # ── Early stopping ────────────────────────────────────────────
        if patience_counter >= PATIENCE:
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    csv_file.close()
    writer.close()

    # ── Test evaluation ───────────────────────────────────────────────────
    logger.info("Loading best model for test evaluation …")
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    test_metrics = validate(model, loaders["test"], criterion, device)
    logger.info(
        "TEST RESULTS → Loss=%.4f | Steer=%.4f | Thr=%.4f | Brk=%.4f",
        test_metrics["loss"],
        test_metrics.get("steering", 0),
        test_metrics.get("throttle", 0),
        test_metrics.get("brake", 0),
    )

    # ── ONNX export ───────────────────────────────────────────────────────
    onnx_path = str(ckpt_dir / "residual_net.onnx")
    model.export_onnx(onnx_path)
    logger.info("Training complete. Artifacts in %s", ckpt_dir)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroDrive Control RCN")
    parser.add_argument("--config", default="control/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args.config, args.resume)
