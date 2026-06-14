"""
Training script for kidney CT classification.

Usage:
  python train.py encoder=suprem ct_level=L2 label_level=L1 freeze_encoder=true lr=1e-4
  python train.py encoder=voco ct_level=L3 label_level=L2 freeze_encoder=false lr=5e-4
"""
import logging
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["NUMEXPR_MAX_THREADS"] = "16"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import hydra
import csv
import json

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

from data.dataset import KidneyCTDataset
from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parent))
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"


def compute_l1_loss(logits, labels, pos_weight=None):
    """BCE loss for L1 (left_abnormal, right_abnormal)."""
    targets = torch.stack([labels["left_abnormal"], labels["right_abnormal"]], dim=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion(logits, targets)


def compute_l2_loss(cls_logits, size_pred, labels, cls_pos_weight=None):
    """BCE + MSE for L2."""
    cls_targets = torch.stack([
        labels["left_has_cyst"], labels["right_has_cyst"],
        labels["left_has_solid"], labels["right_has_solid"],
    ], dim=1)

    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)
    cls_loss = cls_criterion(cls_logits, cls_targets)

    # Size regression (only for samples with abnormality)
    size_targets = torch.stack([labels["left_max_size"], labels["right_max_size"]], dim=1)
    # Mask: only compute loss where size > 0
    size_mask = size_targets > 0
    if size_mask.any():
        size_loss = nn.MSELoss()(size_pred[size_mask], size_targets[size_mask])
    else:
        size_loss = torch.tensor(0.0, device=cls_logits.device)

    return cls_loss + 0.1 * size_loss, cls_loss, size_loss


@torch.no_grad()
def evaluate(model, dataloader, label_level, device):
    """Evaluate model on validation set. Returns (metrics, raw_predictions)."""
    model.eval()
    all_preds = []
    all_targets = []
    all_size_preds = []
    all_size_targets = []
    all_study_ids = []
    total_loss = 0
    n_batches = 0

    all_resize_scales = []
    for images, labels, study_ids in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        d2_mask = labels.pop("d2_mask", None)
        resize_scale = labels.pop("resize_scale", None)
        if resize_scale is not None:
            all_resize_scales.append(resize_scale.cpu().numpy())
        all_study_ids.extend(study_ids)

        if label_level == "L1":
            logits = model(images, mask=d2_mask)
            loss = compute_l1_loss(logits, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = torch.stack([labels["left_abnormal"], labels["right_abnormal"]], dim=1).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets)

        elif label_level == "L2":
            cls_logits, size_pred = model(images, mask=d2_mask)
            loss, _, _ = compute_l2_loss(cls_logits, size_pred, labels)
            probs = torch.sigmoid(cls_logits).cpu().numpy()
            targets = torch.stack([
                labels["left_has_cyst"], labels["right_has_cyst"],
                labels["left_has_solid"], labels["right_has_solid"],
            ], dim=1).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets)
            all_size_preds.append(size_pred.cpu().numpy())
            all_size_targets.append(
                torch.stack([labels["left_max_size"], labels["right_max_size"]], dim=1).cpu().numpy()
            )

        elif label_level.startswith("L3"):
            from models.l3_heads import compute_l3_loss
            l3_output, extra = model(images)
            loss, loss_dict = compute_l3_loss(l3_output, extra, labels, label_level.split("_")[-1])
            B = images.shape[0]

            # Collect ALL predictions for full evaluation
            l3_batch = {
                "exists_pred": torch.sigmoid(l3_output.exists).cpu().numpy(),
                "exists_gt": torch.cat([labels["l3_left_exists"], labels["l3_right_exists"]], dim=1).view(B, 6).cpu().numpy(),
            }
            for feat in ["cyst", "mass", "tumor"]:
                pred_logits = getattr(l3_output, feat)
                l3_batch[f"{feat}_pred"] = torch.sigmoid(pred_logits).cpu().numpy()
                l3_batch[f"{feat}_gt"] = torch.cat([
                    labels[f"l3_left_{feat}"], labels[f"l3_right_{feat}"]
                ], dim=1).view(B, 6).cpu().numpy()
                l3_batch[f"{feat}_valid"] = torch.cat([
                    labels[f"l3_left_{feat}_valid"], labels[f"l3_right_{feat}_valid"]
                ], dim=1).view(B, 6).cpu().numpy()

            l3_batch["size_pred"] = l3_output.size.cpu().numpy()
            l3_batch["size_gt"] = torch.cat([labels["l3_left_size"], labels["l3_right_size"]], dim=1).view(B, 6).cpu().numpy()
            l3_batch["size_valid"] = torch.cat([labels["l3_left_size_valid"], labels["l3_right_size_valid"]], dim=1).view(B, 6).cpu().numpy()

            l3_batch["enh_pred"] = l3_output.enhancement.argmax(dim=-1).cpu().numpy()
            l3_batch["enh_gt"] = torch.cat([labels["l3_left_enhancement"], labels["l3_right_enhancement"]], dim=1).view(B, 6).cpu().numpy()
            l3_batch["enh_valid"] = torch.cat([labels["l3_left_enhancement_valid"], labels["l3_right_enhancement_valid"]], dim=1).view(B, 6).cpu().numpy()

            l3_batch["att_pred"] = l3_output.attenuation.argmax(dim=-1).cpu().numpy()
            l3_batch["att_gt"] = torch.cat([labels["l3_left_attenuation"], labels["l3_right_attenuation"]], dim=1).view(B, 6).cpu().numpy()
            l3_batch["att_valid"] = torch.cat([labels["l3_left_attenuation_valid"], labels["l3_right_attenuation_valid"]], dim=1).view(B, 6).cpu().numpy()

            all_preds.append(l3_batch)

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics = {"val_loss": avg_loss}

    if label_level.startswith("L3"):
        # Merge all batches
        def _concat(key):
            return np.concatenate([b[key] for b in all_preds])

        exists_pred = _concat("exists_pred")   # (N, 6)
        exists_gt = _concat("exists_gt")       # (N, 6)

        # ---- Eval as L1: side-level abnormality AUC ----
        # Aggregate slots 0-2 (left) and 3-5 (right) → per-side abnormal score
        left_abn_pred = exists_pred[:, :3].max(axis=1)
        right_abn_pred = exists_pred[:, 3:].max(axis=1)
        left_abn_gt = (exists_gt[:, :3].max(axis=1) > 0.5).astype(float)
        right_abn_gt = (exists_gt[:, 3:].max(axis=1) > 0.5).astype(float)
        try:
            metrics["auc_left_abnormal"] = float(roc_auc_score(left_abn_gt, left_abn_pred))
        except ValueError:
            metrics["auc_left_abnormal"] = 0.5
        try:
            metrics["auc_right_abnormal"] = float(roc_auc_score(right_abn_gt, right_abn_pred))
        except ValueError:
            metrics["auc_right_abnormal"] = 0.5

        # ---- Eval as L2: side-level per-type AUC ----
        # For each type (cyst, mass, tumor): aggregate slots → side-level
        # ALL samples participate (including normal sides with no lesions → GT=0).
        # Only mask out samples where ALL existing slots have unknown labels for this type.
        for feat in ["cyst", "mass", "tumor"]:
            pred = _concat(f"{feat}_pred")    # (N, 6)
            gt = _concat(f"{feat}_gt")        # (N, 6)
            valid = _concat(f"{feat}_valid")  # (N, 6)
            for side, side_name, sl in [("left", "left", slice(0, 3)), ("right", "right", slice(3, 6))]:
                side_exist = exists_gt[:, sl]          # (N, 3)
                side_valid = valid[:, sl]               # (N, 3)
                side_gt_vals = gt[:, sl]                # (N, 3)
                side_pred_vals = pred[:, sl]             # (N, 3)

                # Case 1: normal side (no lesion) → GT=0, pred=max over all 3 slots
                no_lesion = (side_exist.max(axis=1) < 0.5)  # (N,)

                # Case 2: has lesion(s) with known type label
                slot_mask = (side_exist > 0.5) & (side_valid > 0.5)
                has_known = slot_mask.any(axis=1)  # (N,) at least one exist+valid slot

                # Case 3: has lesion(s) but ALL type labels unknown → exclude
                has_lesion = (side_exist.max(axis=1) > 0.5)
                all_unknown = has_lesion & ~has_known

                # Include: normal sides + sides with known type labels
                include_mask = no_lesion | has_known  # excludes all_unknown

                # GT: normal side=0, lesion side=max over valid slots
                side_gt_agg = np.where(no_lesion, 0.0,
                    np.where(has_known,
                        np.where(slot_mask, side_gt_vals, 0).max(axis=1),
                        np.nan))

                # Pred: max over all 3 slots (not restricted to exist/valid)
                side_pred_agg = side_pred_vals.max(axis=1)

                valid_final = include_mask & ~np.isnan(side_gt_agg)
                if valid_final.sum() > 10:
                    try:
                        metrics[f"auc_{side_name}_{feat}"] = float(
                            roc_auc_score(side_gt_agg[valid_final], side_pred_agg[valid_final]))
                    except ValueError:
                        metrics[f"auc_{side_name}_{feat}"] = 0.5
                metrics[f"n_{side_name}_{feat}_valid"] = int(valid_final.sum())

        # ---- Eval as L3D: slot-level metrics ----
        # Count MAE
        metrics["count_mae"] = float(np.abs(
            (exists_pred > 0.5).sum(axis=1) - exists_gt.sum(axis=1)
        ).mean())

        # Size MAE
        exist_mask = exists_gt.flatten() > 0.5
        size_pred = _concat("size_pred").flatten()
        size_gt = _concat("size_gt").flatten()
        size_valid = _concat("size_valid").flatten()
        size_mask = (size_valid > 0.5) & exist_mask
        if size_mask.sum() > 0:
            # Raw size MAE (model predicts cm directly from resized image)
            metrics["size_mae"] = float(np.abs(size_pred[size_mask] - size_gt[size_mask]).mean())
            # Also compute scale-corrected size MAE for comparison
            if all_resize_scales:
                scales = np.concatenate(all_resize_scales)  # (N,)
                scales_expanded = np.repeat(scales, 6)  # 6 slots per sample
                size_pred_corrected = size_pred / np.maximum(scales_expanded, 0.1)
                metrics["size_mae_corrected"] = float(np.abs(size_pred_corrected[size_mask] - size_gt[size_mask]).mean())

        # Enhancement accuracy
        enh_pred = _concat("enh_pred").flatten()
        enh_gt = _concat("enh_gt").flatten()
        enh_valid = _concat("enh_valid").flatten()
        enh_mask = (enh_valid > 0.5) & exist_mask
        if enh_mask.sum() > 0:
            metrics["enh_acc"] = float((enh_pred[enh_mask] == enh_gt[enh_mask].round()).mean())

        # Attenuation accuracy
        att_pred = _concat("att_pred").flatten()
        att_gt = _concat("att_gt").flatten()
        att_valid = _concat("att_valid").flatten()
        att_mask = (att_valid > 0.5) & exist_mask
        if att_mask.sum() > 0:
            metrics["att_acc"] = float((att_pred[att_mask] == att_gt[att_mask].round()).mean())

        # ---- auc_mean: side-level AUCs only ----
        auc_values = [v for k, v in metrics.items()
                      if k.startswith("auc_") and isinstance(v, float)]
        metrics["auc_mean"] = float(np.mean(auc_values)) if auc_values else 0.5

        # Build raw predictions dict
        raw_preds = {
            "study_ids": np.array(all_study_ids),
            "exists_pred": exists_pred,
            "exists_gt": exists_gt,
        }
        for feat in ["cyst", "mass", "tumor"]:
            raw_preds[f"{feat}_pred"] = _concat(f"{feat}_pred")
            raw_preds[f"{feat}_gt"] = _concat(f"{feat}_gt")
            raw_preds[f"{feat}_valid"] = _concat(f"{feat}_valid")
        raw_preds["size_pred"] = _concat("size_pred")
        raw_preds["size_gt"] = _concat("size_gt")
        raw_preds["size_valid"] = _concat("size_valid")
        raw_preds["enh_pred"] = _concat("enh_pred")
        raw_preds["enh_gt"] = _concat("enh_gt")
        raw_preds["enh_valid"] = _concat("enh_valid")
        raw_preds["att_pred"] = _concat("att_pred")
        raw_preds["att_gt"] = _concat("att_gt")
        raw_preds["att_valid"] = _concat("att_valid")

        return metrics, raw_preds

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if label_level == "L1":
        names = ["left_abnormal", "right_abnormal"]
    else:
        names = ["left_cyst", "right_cyst", "left_solid", "right_solid"]

    aucs = []
    for i, name in enumerate(names):
        try:
            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except ValueError:
            auc = 0.5
        metrics[f"auc_{name}"] = auc
        aucs.append(auc)

    metrics["auc_mean"] = np.mean(aucs)

    preds_binary = (all_preds > 0.5).astype(int)
    for i, name in enumerate(names):
        metrics[f"f1_{name}"] = f1_score(all_targets[:, i], preds_binary[:, i], zero_division=0)

    # L2 → Eval as L1: aggregate cyst/solid → side-level abnormal
    if label_level == "L2":
        # preds columns: left_cyst(0), right_cyst(1), left_solid(2), right_solid(3)
        left_abn_pred = np.maximum(all_preds[:, 0], all_preds[:, 2])   # max(left_cyst, left_solid)
        right_abn_pred = np.maximum(all_preds[:, 1], all_preds[:, 3])  # max(right_cyst, right_solid)
        left_abn_gt = np.maximum(all_targets[:, 0], all_targets[:, 2]).astype(float)
        right_abn_gt = np.maximum(all_targets[:, 1], all_targets[:, 3]).astype(float)
        try:
            metrics["auc_left_abnormal"] = float(roc_auc_score(left_abn_gt, left_abn_pred))
        except ValueError:
            metrics["auc_left_abnormal"] = 0.5
        try:
            metrics["auc_right_abnormal"] = float(roc_auc_score(right_abn_gt, right_abn_pred))
        except ValueError:
            metrics["auc_right_abnormal"] = 0.5

    # Build raw predictions dict
    raw_preds = {
        "study_ids": np.array(all_study_ids),
        "preds": all_preds,
        "targets": all_targets,
    }
    if label_level == "L2" and all_size_preds:
        size_preds = np.concatenate(all_size_preds, axis=0)
        size_targets = np.concatenate(all_size_targets, axis=0)
        mask = size_targets > 0
        if mask.any():
            metrics["size_mae"] = float(np.abs(size_preds[mask] - size_targets[mask]).mean())
            # Scale-corrected version for comparison
            if all_resize_scales:
                scales = np.concatenate(all_resize_scales)  # (N,)
                size_preds_corrected = size_preds / np.maximum(scales[:, None], 0.1)
                metrics["size_mae_corrected"] = float(np.abs(size_preds_corrected[mask] - size_targets[mask]).mean())
        raw_preds["size_preds"] = size_preds
        raw_preds["size_targets"] = size_targets

    return metrics, raw_preds


def train_one_epoch(model, dataloader, optimizer, label_level, device, scaler,
                    accumulate_steps: int = 1, hier_weight: float = 1.0, hier_mode: str = "l1l2"):
    """Train one epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0
    n_batches = 0

    optimizer.zero_grad()

    for step, (images, labels, _) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        # D2: extract mask from labels for guided pooling
        d2_mask = labels.pop("d2_mask", None)
        labels.pop("resize_scale", None)  # not needed for training loss

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            if label_level == "L1":
                logits = model(images, mask=d2_mask)
                loss = compute_l1_loss(logits, labels)
            elif label_level == "L2":
                cls_logits, size_pred = model(images, mask=d2_mask)
                loss, _, _ = compute_l2_loss(cls_logits, size_pred, labels)
            elif label_level.startswith("L3"):
                from models.l3_heads import compute_l3_loss
                l3_output, extra = model(images, mask=d2_mask)
                loss, _ = compute_l3_loss(l3_output, extra, labels, label_level.split("_")[-1],
                                         hier_weight=hier_weight, hier_mode=hier_mode)

            loss = loss / accumulate_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulate_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulate_steps
        n_batches += 1

    # Handle remaining gradients
    if n_batches % accumulate_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seeds for reproducibility
    seed = cfg.get("seed", 42)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Experiment directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    mask_strategy = cfg.get("mask_strategy", "none")

    # Build exp_name: {encoder}_{input}_{label}_{preprocess}_{hier}_{timestamp}
    # Input tag
    input_map = {
        ("L1", "none"): "whole",
        ("L2", "none"): "crop",
        ("L3", "none"): "crop",
        ("L3", "binary"): "crop+kmask",
        ("L3", "7class"): "crop+fmask",
        ("D2", "none"): "crop+d2",
    }
    input_tag = input_map.get((cfg.ct_level, mask_strategy), f"{cfg.ct_level}_{mask_strategy}")

    # Label tag
    label_tag = cfg.label_level.replace("L3_", "L3")

    # Preprocess tag
    preproc_parts = []
    if cfg.get("flip_x", False):
        preproc_parts.append("flip")
    pad_only = cfg.get("pad_only", 0)
    if isinstance(pad_only, str) and "," in str(pad_only):
        preproc_parts.append(f"pad{pad_only}")
    elif float(pad_only) > 0:
        preproc_parts.append(f"pad{pad_only}")
    if cfg.get("normalize_mode", "minmax") != "minmax":
        preproc_parts.append(cfg.normalize_mode)
    hu_min, hu_max = cfg.hu_min, cfg.hu_max
    if (hu_min, hu_max) != (-200, 400):
        preproc_parts.append(f"hu{hu_min}_{hu_max}")
    preproc_tag = "_".join(preproc_parts) if preproc_parts else "default"

    # Hier tag
    if cfg.label_level.startswith("L3"):
        hier_w = cfg.get("hier_weight", 1.0)
        hier_m = cfg.get("hier_mode", "l1l2")
        hier_tag = f"hier{hier_w}{hier_m}"
    else:
        hier_tag = ""

    # Combine
    parts = [cfg.encoder, input_tag, label_tag]
    if preproc_tag != "default":
        parts.append(preproc_tag)
    if hier_tag:
        parts.append(hier_tag)
    # Include fewshot fraction tag for scaling law experiments.
    # When fewshot is set, also include seed (multi-seed runs need disambiguation).
    fewshot = cfg.get("fewshot_fraction", 0)
    if 0 < fewshot <= 1.0:
        parts.append(f"frac{int(fewshot*100)}")
        parts.append(f"seed{cfg.get('seed', 42)}")
    elif cfg.get("seed", 42) != 42:
        parts.append(f"seed{cfg.seed}")
    parts.append(timestamp)
    exp_name = "_".join(parts)

    # Exp group
    exp_group = cfg.get("exp_group", "")
    if exp_group:
        exp_dir = EXPERIMENT_DIR / exp_group / exp_name
    else:
        exp_dir = EXPERIMENT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment dir: {exp_dir}")

    # CSV metrics logger
    csv_path = exp_dir / "epoch_metrics.csv"
    csv_columns = None  # will be set on first epoch

    # Data
    target_size = tuple(cfg.target_size)
    mask_strategy = cfg.get("mask_strategy", "none")
    mask_dropout_prob = cfg.get("mask_dropout_prob", 0.3)

    flip_x = cfg.get("flip_x", False)
    pad_only = cfg.get("pad_only", 0)
    normalize_mode = cfg.get("normalize_mode", "minmax")
    swap_lr = cfg.get("swap_lr", False)
    train_ds = KidneyCTDataset(
        split="train", target_size=target_size,
        hu_min=cfg.hu_min, hu_max=cfg.hu_max,
        ct_level=cfg.ct_level, augment=True,
        mask_strategy=mask_strategy,
        mask_dropout_prob=mask_dropout_prob,
        return_mask=False,
        flip_x=flip_x,
        pad_only=pad_only,
        normalize_mode=normalize_mode,
        swap_lr=swap_lr,
    )
    # Val uses same mask_strategy as train (no augmentation/dropout)
    # For dropout/noisy strategies, eval uses the base mask (binary)
    eval_mask_strategy = mask_strategy if mask_strategy in ("none", "binary", "7class") else "binary"
    val_ds = KidneyCTDataset(
        split="valid", target_size=target_size,
        hu_min=cfg.hu_min, hu_max=cfg.hu_max,
        ct_level=cfg.ct_level, augment=False,
        mask_strategy=eval_mask_strategy,
        return_mask=False,
        flip_x=flip_x,
        pad_only=pad_only,
        normalize_mode=normalize_mode,
        swap_lr=swap_lr,
    )

    # Few-shot mode: stratified order-level sampling
    fewshot_fraction = cfg.get("fewshot_fraction", 0)
    if fewshot_fraction > 0 and fewshot_fraction < 1.0:
        from data.sampler import get_fewshot_study_ids
        selected_sids = get_fewshot_study_ids(fewshot_fraction, seed=cfg.get("seed", 42))
        selected_set = set(selected_sids)
        indices = [i for i, sid in enumerate(train_ds.study_ids) if sid in selected_set]
        train_ds.npz_files = [train_ds.npz_files[i] for i in indices]
        train_ds.study_ids = [train_ds.study_ids[i] for i in indices]
        logger.info(f"FEW-SHOT MODE: {fewshot_fraction*100:.0f}% → {len(train_ds)} samples")

    # Legacy overfit mode (for debugging)
    overfit_n = cfg.get("overfit_n", 0)
    if overfit_n > 0:
        train_ds.npz_files = train_ds.npz_files[:overfit_n]
        train_ds.study_ids = train_ds.study_ids[:overfit_n]
        logger.info(f"OVERFIT MODE: using only {overfit_n} training samples")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Model
    in_channels = 2 if cfg.ct_level == "L3" else 1
    is_d2 = cfg.ct_level == "D2"

    if cfg.encoder == "resnet18":
        from models.encoder import ResNet3DEncoder
        encoder = ResNet3DEncoder(in_channels=in_channels)
    elif cfg.encoder in ("ctvit", "ctvit_scratch"):
        from models.ctvit_encoder import CTViTEncoder
        encoder = CTViTEncoder(
            in_channels=in_channels,
            pretrained=(cfg.encoder == "ctvit"),
        )
    elif cfg.encoder in ("vit_scratch", "vit_inflated"):
        from models.encoder import ViT3DEncoder
        pretrained = "inflated_2d" if cfg.encoder == "vit_inflated" else "from_scratch"
        encoder = ViT3DEncoder(
            in_channels=in_channels,
            pretrained=pretrained,
            img_size=tuple(cfg.target_size),
        )
    else:
        encoder = SwinUNETREncoder(
            in_channels=in_channels,
            feature_size=cfg.feature_size,
            pretrained=cfg.encoder,
        )

    if is_d2:
        from models.mask_guided_pooling import MaskGuidedEncoder
        encoder = MaskGuidedEncoder(encoder, feature_dim=encoder.output_dim)

    model = KidneyClassifier(
        encoder=encoder,
        label_level=cfg.label_level,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    )
    model = model.to(device)

    # Freeze encoder if requested
    if cfg.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

    # Optimizer
    if cfg.freeze_encoder:
        params = list(model.head.parameters())
    else:
        encoder_params = list(model.encoder.parameters())
        head_params = list(model.head.parameters())
        params = [
            {"params": encoder_params, "lr": cfg.lr * 0.1},
            {"params": head_params, "lr": cfg.lr},
        ]

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # Training loop
    best_auc = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, cfg.label_level, device, scaler,
            accumulate_steps=cfg.accumulate_steps,
            hier_weight=cfg.get("hier_weight", 1.0),
            hier_mode=cfg.get("hier_mode", "l1l2"),
        )
        scheduler.step()

        val_metrics, raw_preds = evaluate(model, val_loader, cfg.label_level, device)

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[-1]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{cfg.epochs} ({elapsed:.0f}s) — "
            f"train_loss={train_loss:.4f}, val_loss={val_metrics['val_loss']:.4f}, "
            f"auc_mean={val_metrics['auc_mean']:.4f}, lr={lr_current:.6f}"
        )

        # Save predictions
        pred_dir = exp_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        np.savez_compressed(pred_dir / f"epoch_{epoch+1}.npz", **raw_preds)

        # Log to CSV
        row = {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics}
        if csv_columns is None:
            csv_columns = list(row.keys())
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_columns).writeheader()
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_columns).writerow(row)

        # Save every epoch checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "val_metrics": val_metrics,
        }
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(ckpt_data, ckpt_dir / f"epoch_{epoch+1}.pt")

        # Save best model — criterion depends on label level
        # L3D: use stable AUCs (abnormal + cyst, all >260 samples)
        # L1/L2: use auc_mean as before
        if cfg.label_level.startswith("L3"):
            stable_keys = ["auc_left_abnormal", "auc_right_abnormal", "auc_left_cyst", "auc_right_cyst"]
            best_criterion = float(np.mean([val_metrics[k] for k in stable_keys if k in val_metrics]))
        else:
            best_criterion = val_metrics["auc_mean"]

        # Always keep at least one checkpoint as "best" (the first epoch), so the
        # final summary load never fails even if the criterion is degenerate
        # (e.g. NaN on a tiny smoke dataset). On real data the first epoch's AUC
        # already exceeds the initial 0, so this matches the original behaviour.
        if best_criterion > best_auc or not (exp_dir / "best_model.pt").exists():
            best_auc = best_criterion
            ckpt_data["best_auc"] = best_auc
            torch.save(ckpt_data, exp_dir / "best_model.pt")
            logger.info(f"  New best AUC: {best_auc:.4f}")

    # Final results: use metrics saved at best epoch (avoids re-evaluation discrepancy)
    ckpt = torch.load(exp_dir / "best_model.pt", weights_only=False)
    final_metrics = ckpt["val_metrics"]

    logger.info(f"\nFinal results (best epoch {ckpt['epoch']+1}):")
    for k, v in sorted(final_metrics.items()):
        logger.info(f"  {k}: {v:.4f}")

    # Save final metrics
    with open(exp_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"Experiment saved to: {exp_dir}")


if __name__ == "__main__":
    main()
