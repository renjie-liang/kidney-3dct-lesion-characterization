"""Detection-oriented training script for L3D models.

Key differences from main paper's train.py:
  - Configurable loss assembly (focal / count / hungarian / joint_conf / ...)
  - Multi-seed loop within a single SLURM job (to reduce queue contention)
  - Longer training: default 50 epochs (vs 30 in main)
  - Per-class AP (Method D) tracked as an additional metric during training

Usage:
  python train_detection.py --config configs/focal.yaml

Saves to: experiments/detection_training/{exp_name}/seed{S}/
  - epoch_metrics.csv
  - best_model.pt
  - predictions/epoch_*.npz
  - epoch_metrics_v3.csv (computed at end via recompute_metrics)
"""
import argparse
import csv
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Self-contained: all imports from local detection_training/ subpackage.
DETECTION_ROOT = Path(__file__).parent
sys.path.insert(0, str(DETECTION_ROOT))

from data.dataset import KidneyCTDataset
from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier
from losses.combined import compute_detection_loss

STAGE1_ROOT = DETECTION_ROOT.parent
PROJECT_ROOT = STAGE1_ROOT.parent
OUTPUT_BASE = PROJECT_ROOT / "experiments" / "detection_training"


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(log_file.stem)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt); fh.setLevel(logging.DEBUG)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, encoder_type: str, in_channels: int, device) -> nn.Module:
    """Build encoder + L3 head based on cfg."""
    encoder = SwinUNETREncoder(
        in_channels=in_channels,
        feature_size=48,
        pretrained=encoder_type,
        use_checkpoint=cfg.get("use_checkpoint", False),
    )
    model = KidneyClassifier(
        encoder,
        label_level=cfg.get("label_level", "L3_D"),
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.3),
    )
    return model.to(device)


@torch.no_grad()
def evaluate(model, dataloader, label_level, device, amp_dtype=torch.bfloat16):
    """Collect raw predictions and compute v3-style metrics."""
    model.eval()
    all_preds = []

    for images, labels, study_ids in tqdm(dataloader, desc="eval", leave=False, dynamic_ncols=True):
        images = images.to(device)
        d2_mask = labels.pop("d2_mask", None)
        labels.pop("resize_scale", None)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            l3_output, extra = model(images, mask=d2_mask)

        B = images.shape[0]
        batch = {
            "exists_pred": torch.sigmoid(l3_output.exists).float().cpu().numpy(),
            "exists_gt": torch.cat([labels["l3_left_exists"], labels["l3_right_exists"]], dim=1).view(B, 6).numpy(),
        }
        for feat in ["cyst", "solid"]:
            batch[f"{feat}_pred"] = torch.sigmoid(getattr(l3_output, feat)).float().cpu().numpy()
            batch[f"{feat}_gt"] = torch.cat([labels[f"l3_left_{feat}"], labels[f"l3_right_{feat}"]], dim=1).view(B, 6).numpy()
            batch[f"{feat}_valid"] = torch.cat([labels[f"l3_left_{feat}_valid"], labels[f"l3_right_{feat}_valid"]], dim=1).view(B, 6).numpy()
        batch["size_pred"] = l3_output.size.float().cpu().numpy()
        batch["size_gt"] = torch.cat([labels["l3_left_size"], labels["l3_right_size"]], dim=1).view(B, 6).numpy()
        batch["size_valid"] = torch.cat([labels["l3_left_size_valid"], labels["l3_right_size_valid"]], dim=1).view(B, 6).numpy()
        batch["enh_pred"] = l3_output.enhancement.argmax(dim=-1).cpu().numpy()
        batch["enh_gt"] = torch.cat([labels["l3_left_enhancement"], labels["l3_right_enhancement"]], dim=1).view(B, 6).numpy()
        batch["enh_valid"] = torch.cat([labels["l3_left_enhancement_valid"], labels["l3_right_enhancement_valid"]], dim=1).view(B, 6).numpy()
        batch["att_pred"] = l3_output.attenuation.argmax(dim=-1).cpu().numpy()
        batch["att_gt"] = torch.cat([labels["l3_left_attenuation"], labels["l3_right_attenuation"]], dim=1).view(B, 6).numpy()
        batch["att_valid"] = torch.cat([labels["l3_left_attenuation_valid"], labels["l3_right_attenuation_valid"]], dim=1).view(B, 6).numpy()
        batch["study_ids"] = np.array(study_ids)
        all_preds.append(batch)

    # Concat
    predictions = {}
    for k in all_preds[0].keys():
        if k == "study_ids":
            predictions[k] = np.concatenate([b[k] for b in all_preds])
        else:
            predictions[k] = np.concatenate([b[k] for b in all_preds], axis=0)

    # v3 side-level metrics + per-class AP (delegate to recompute_metrics via in-memory shim)
    metrics = _compute_metrics_dict(predictions)
    # Per-class AP (COCO-greedy protocol)
    ap_metrics = _compute_per_class_AP(predictions, size_tol=1.0)
    metrics.update(ap_metrics)
    return metrics, predictions


def _compute_per_class_AP(predictions: dict, size_tol: float = 1.0) -> dict:
    """Per-class AP via COCO-greedy protocol.

    Returns dict with AP_cyst, AP_solid, mAP.
    mAP excludes NaN classes (n_pos=0) per COCO convention.
    """
    from analysis.lesion_level_eval import compute_class_AP_coco

    aps = {}
    for c in ["cyst", "solid"]:
        result = compute_class_AP_coco(predictions, c, size_tol=size_tol)
        aps[f"AP_{c}"] = result["AP"]
    main_aps = np.array([aps["AP_cyst"], aps["AP_solid"]])
    aps["mAP"] = float(np.nanmean(main_aps)) if not np.all(np.isnan(main_aps)) else float("nan")
    return aps


class _DictNpz:
    """Minimal shim that mimics np.load(npz_path) behavior: supports d[key] and .files."""
    def __init__(self, d: dict):
        self._d = d

    def __getitem__(self, key):
        return self._d[key]

    @property
    def files(self):
        return list(self._d.keys())


def _compute_metrics_dict(predictions: dict) -> dict:
    """Compute v3 side-level metrics from an in-memory prediction dict.

    Reuses the logic in analysis/recompute_metrics.py but without temp-file I/O.
    """
    from analysis.recompute_metrics import compute_metrics_from_predictions
    # recompute_metrics.compute_metrics_from_predictions does np.load(path) then d[key] / d.files.
    # Monkey-patch by temporarily shimming np.load to return our dict-like object.
    # Simpler: call the core logic directly. But that function just does np.load + key access.
    # Cleanest: write a tiny shim file-like object. np.load(path) would need a BytesIO though.
    # Even cleaner: replicate the relevant body here.
    from sklearn.metrics import roc_auc_score

    ex_p = predictions["exists_pred"]
    ex_g = predictions["exists_gt"]
    metrics = {}

    # L1 side-level abnormal
    def side_abn(sl):
        pred = ex_p[:, sl].max(axis=1)
        gt = (ex_g[:, sl].max(axis=1) > 0.5).astype(float)
        if (gt > 0.5).sum() > 0 and (gt < 0.5).sum() > 0:
            return float(roc_auc_score(gt, pred))
        return 0.5
    metrics["auc_left_abnormal"] = side_abn(slice(0, 3))
    metrics["auc_right_abnormal"] = side_abn(slice(3, 6))

    # L2 side-level per-type (v3 fixed: includes normal sides)
    for feat in ["cyst", "solid"]:
        pred = predictions[f"{feat}_pred"]
        gt = predictions[f"{feat}_gt"]
        valid = predictions[f"{feat}_valid"]
        for side_name, sl in [("left", slice(0, 3)), ("right", slice(3, 6))]:
            side_exist = ex_g[:, sl]
            side_valid = valid[:, sl]
            side_gt_vals = gt[:, sl]
            side_pred_vals = pred[:, sl]

            no_lesion = (side_exist.max(axis=1) < 0.5)
            slot_mask = (side_exist > 0.5) & (side_valid > 0.5)
            has_known = slot_mask.any(axis=1)
            include_mask = no_lesion | has_known

            side_gt_agg = np.where(no_lesion, 0.0,
                np.where(has_known,
                    np.where(slot_mask, side_gt_vals, 0).max(axis=1),
                    np.nan))
            side_pred_agg = side_pred_vals.max(axis=1)
            valid_final = include_mask & ~np.isnan(side_gt_agg)
            if valid_final.sum() > 10 and (side_gt_agg[valid_final] > 0.5).sum() > 0 and (side_gt_agg[valid_final] < 0.5).sum() > 0:
                metrics[f"auc_{side_name}_{feat}"] = float(
                    roc_auc_score(side_gt_agg[valid_final], side_pred_agg[valid_final]))
            else:
                metrics[f"auc_{side_name}_{feat}"] = 0.5

    # L3D slot-level
    metrics["count_mae"] = float(np.abs((ex_p > 0.5).sum(axis=1) - ex_g.sum(axis=1)).mean())
    exist_mask = ex_g.flatten() > 0.5
    size_pred = predictions["size_pred"].flatten()
    size_gt = predictions["size_gt"].flatten()
    size_valid = predictions["size_valid"].flatten()
    size_mask = (size_valid > 0.5) & exist_mask
    if size_mask.sum() > 0:
        metrics["size_mae"] = float(np.abs(size_pred[size_mask] - size_gt[size_mask]).mean())

    enh_mask = (predictions["enh_valid"].flatten() > 0.5) & exist_mask
    if enh_mask.sum() > 0:
        metrics["enh_acc"] = float(
            (predictions["enh_pred"].flatten()[enh_mask] ==
             predictions["enh_gt"].flatten()[enh_mask].round()).mean())

    att_mask = (predictions["att_valid"].flatten() > 0.5) & exist_mask
    if att_mask.sum() > 0:
        metrics["att_acc"] = float(
            (predictions["att_pred"].flatten()[att_mask] ==
             predictions["att_gt"].flatten()[att_mask].round()).mean())

    auc_values = [v for k, v in metrics.items() if k.startswith("auc_") and isinstance(v, float)]
    metrics["auc_mean"] = float(np.mean(auc_values)) if auc_values else 0.5
    return metrics


def train_one_seed(cfg: dict, seed: int, exp_dir: Path, logger: logging.Logger):
    """Train one seed of the experiment."""
    seed_dir = exp_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Training seed={seed} in {seed_dir} ===")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_ds = KidneyCTDataset(
        split="train", target_size=(320, 192, 224),
        ct_level=cfg.get("ct_level", "L3"), augment=True,
        mask_strategy=cfg.get("mask_strategy", "binary"),
        mask_dropout_prob=cfg.get("mask_dropout_prob", 0.3),
        flip_x=cfg.get("flip_x", True),
        pad_only=cfg.get("pad_only", 0),
    )
    val_ds = KidneyCTDataset(
        split="valid", target_size=(320, 192, 224),
        ct_level=cfg.get("ct_level", "L3"), augment=False,
        mask_strategy=cfg.get("mask_strategy", "binary"),
        flip_x=cfg.get("flip_x", True),
        pad_only=cfg.get("pad_only", 0),
    )

    # Few-shot / scaling law subset (stratified order-level sampling).
    fewshot_fraction = cfg.get("fewshot_fraction", 0)
    if 0 < fewshot_fraction < 1.0:
        # Import get_fewshot_study_ids from the MAIN paper's data/sampler.py
        # via importlib to avoid colliding with detection_training/data/ package.
        import importlib.util
        sampler_path = STAGE1_ROOT / "data" / "sampler.py"
        spec = importlib.util.spec_from_file_location("main_data_sampler", sampler_path)
        main_sampler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_sampler)
        selected = set(main_sampler.get_fewshot_study_ids(fewshot_fraction, seed=seed))
        keep_idx = [i for i, sid in enumerate(train_ds.study_ids) if sid in selected]
        logger.info(
            f"FEW-SHOT MODE: fraction={fewshot_fraction:.2f}, seed={seed} -> "
            f"{len(keep_idx)}/{len(train_ds)} train samples kept"
        )
        train_ds = torch.utils.data.Subset(train_ds, keep_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=True,
                              num_workers=cfg.get("num_workers", 4), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False,
                            num_workers=cfg.get("num_workers", 4), pin_memory=True)

    # Model
    in_channels = 2 if cfg.get("ct_level", "L3") == "L3" else 1
    model = build_model(cfg, cfg.get("encoder", "suprem"), in_channels, device)

    # Optimizer — layered LR (main paper setup): encoder at lr*0.1, head at lr.
    # Critical for preserving pretrained SuPreM features; uniform lr=5e-4 collapses encoder.
    base_lr = cfg.get("lr", 5e-4)
    encoder_lr_scale = cfg.get("encoder_lr_scale", 0.1)
    params = [
        {"params": list(model.encoder.parameters()), "lr": base_lr * encoder_lr_scale},
        {"params": list(model.head.parameters()), "lr": base_lr},
    ]
    optimizer = torch.optim.AdamW(
        params,
        lr=base_lr,
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    logger.info(f"Optimizer: AdamW head_lr={base_lr:.2e}, encoder_lr={base_lr*encoder_lr_scale:.2e}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.get("epochs", 50))
    # amp_dtype: "bf16" (default) uses bfloat16, no GradScaler needed.
    #            "fp16" uses float16 + GradScaler (main paper setting).
    amp_name = cfg.get("amp_dtype", "bf16")
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    use_scaler = (amp_name == "fp16")
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # Training
    epochs = cfg.get("epochs", 50)
    accumulate_steps = cfg.get("accumulate_steps", 4)
    csv_path = seed_dir / "epoch_metrics.csv"
    pred_dir = seed_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    best_crit = 0.0
    best_mAP = 0.0
    best_crit_epoch = 0
    best_mAP_epoch = 0
    all_rows = []
    csv_columns = None

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        train_loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"seed{seed} ep{epoch+1}/{epochs}",
                    leave=False, dynamic_ncols=True)
        for step, (images, labels, _) in enumerate(pbar):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            d2_mask = labels.pop("d2_mask", None)
            labels.pop("resize_scale", None)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                l3_output, extra = model(images, mask=d2_mask)
                loss, comps = compute_detection_loss(l3_output, labels, cfg)
                loss = loss / accumulate_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_last_batch = (step == len(train_loader) - 1)
            if (step + 1) % accumulate_steps == 0 or is_last_batch:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_loss_sum += loss.item() * accumulate_steps
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item() * accumulate_steps:.3f}",
                              "avg": f"{train_loss_sum/n_batches:.3f}"})

        train_loss = train_loss_sum / max(n_batches, 1)
        scheduler.step()

        # Eval (now includes per-class AP via COCO-greedy)
        metrics, predictions = evaluate(model, val_loader, cfg.get("label_level", "L3_D"), device, amp_dtype=amp_dtype)

        # L3-aware composite crit:
        #   crit = 0.5 * (mean of 6 side-level AUCs: abn/cyst/solid × L/R) + 0.5 * mAP
        # Captures L1/L2 (side aggregation) AND L3 (per-lesion AP) in one number.
        side_aucs = [metrics.get(f"auc_{s}_{f}", 0.5)
                     for s in ("left", "right") for f in ("abnormal", "cyst", "solid")]
        side_auc_mean = sum(side_aucs) / 6
        mAP = metrics.get("mAP", 0.0)
        crit = 0.5 * side_auc_mean + 0.5 * mAP

        elapsed = time.time() - t0
        m = metrics
        logger.info(
            f"[seed{seed}] Ep {epoch+1}/{epochs} ({elapsed:.0f}s) loss={train_loss:.4f} crit={crit:.4f} mAP={mAP:.4f}"
        )
        logger.info(
            f"  L1 AUC abn[L={m.get('auc_left_abnormal', 0):.3f} R={m.get('auc_right_abnormal', 0):.3f}]"
        )
        logger.info(
            f"  L2 AUC "
            f"cyst[L={m.get('auc_left_cyst', 0):.3f} R={m.get('auc_right_cyst', 0):.3f}] "
            f"solid[L={m.get('auc_left_solid', 0):.3f} R={m.get('auc_right_solid', 0):.3f}]"
        )
        logger.info(
            f"  L3 AP[cyst={m.get('AP_cyst', 0):.3f} solid={m.get('AP_solid', 0):.3f}]  "
            f"count_mae={m.get('count_mae', 0):.3f} size_mae={m.get('size_mae', 0):.3f} "
            f"enh_acc={m.get('enh_acc', 0):.3f} att_acc={m.get('att_acc', 0):.3f}"
        )

        # Save predictions for this epoch
        npz_path = pred_dir / f"epoch_{epoch+1}.npz"
        np.savez_compressed(npz_path, **predictions)

        # Log row
        row = {"epoch": epoch + 1, "train_loss": train_loss, **metrics}
        if csv_columns is None:
            csv_columns = ["epoch", "train_loss"] + [k for k in row if k not in ("epoch", "train_loss")]
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=csv_columns)
                w.writeheader()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_columns)
            w.writerow({k: row.get(k, "") for k in csv_columns})
        all_rows.append(row)

        # Save best-by-side-crit checkpoint
        if crit > best_crit:
            best_crit = crit
            best_crit_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "metrics": metrics, "cfg": cfg, "seed": seed, "selection": "crit",
            }, seed_dir / "best_crit_model.pt")

        # Save best-by-mAP checkpoint (our actual KPI)
        if mAP > best_mAP:
            best_mAP = mAP
            best_mAP_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "metrics": metrics, "cfg": cfg, "seed": seed, "selection": "mAP",
            }, seed_dir / "best_mAP_model.pt")

    logger.info(
        f"[seed{seed}] Done. best_crit={best_crit:.4f} (ep{best_crit_epoch}), "
        f"best_mAP={best_mAP:.4f} (ep{best_mAP_epoch})"
    )
    # Free GPU memory before next seed
    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "best_crit": best_crit, "best_crit_epoch": best_crit_epoch,
        "best_mAP": best_mAP, "best_mAP_epoch": best_mAP_epoch,
        "n_epochs": epochs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--exp_name", default=None, help="override exp_name (otherwise derived from config filename)")
    parser.add_argument("--fewshot_fraction", type=float, default=None,
                        help="override cfg.fewshot_fraction (0-1 range; 1.0 or unset uses full data)")
    parser.add_argument("--output_base", default=None,
                        help="override OUTPUT_BASE dir (default: experiments/detection_training)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.fewshot_fraction is not None:
        cfg["fewshot_fraction"] = args.fewshot_fraction

    # Exp name
    exp_name = args.exp_name or Path(args.config).stem
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_base = Path(args.output_base) if args.output_base else OUTPUT_BASE
    exp_dir = output_base / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config + args for reproducibility
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    with open(exp_dir / "run_info.json", "w") as f:
        json.dump({"seeds": args.seeds, "config": args.config, "timestamp": timestamp}, f, indent=2)

    logger = setup_logger(exp_dir / "training.log")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Output: {exp_dir}")
    logger.info(f"Config: {cfg}")
    logger.info(f"Seeds: {args.seeds}")

    # Loop over seeds
    summaries = []
    for seed in args.seeds:
        summary = train_one_seed(cfg, seed, exp_dir, logger)
        summaries.append(summary)

    # Final aggregate
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info(f"All seeds complete: {summaries}")


if __name__ == "__main__":
    main()
