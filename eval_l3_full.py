"""
Full L3 evaluation — evaluate ALL per-lesion features.

Loads a trained L3D checkpoint and evaluates:
  - exists: AUC (slot-level detection)
  - cyst, mass, tumor: AUC (binary, masked by valid)
  - size: MAE (masked by valid + exists)
  - enhancement: accuracy (masked by valid + exists)
  - attenuation: accuracy (masked by valid + exists)
  - count: MAE per sample

Usage:
  python eval_l3_full.py --checkpoint experiments/.../best_model.pt
"""
import argparse
import os
import warnings

os.environ["NUMEXPR_MAX_THREADS"] = "16"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data.dataset import KidneyCTDataset
from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier


def evaluate_l3_full(model, dataloader, device):
    """Full L3 evaluation with all 7 features."""
    model.eval()

    # Collectors
    all_exists_pred, all_exists_gt = [], []
    all_cyst_pred, all_cyst_gt, all_cyst_valid = [], [], []
    all_mass_pred, all_mass_gt, all_mass_valid = [], [], []
    all_tumor_pred, all_tumor_gt, all_tumor_valid = [], [], []
    all_size_pred, all_size_gt, all_size_valid = [], [], []
    all_enh_pred, all_enh_gt, all_enh_valid = [], [], []
    all_att_pred, all_att_gt, all_att_valid = [], [], []
    all_count_pred, all_count_gt = [], []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            l3_output, extra = model(images)
            B = images.shape[0]

            # Exists
            exists_pred = torch.sigmoid(l3_output.exists).cpu().numpy()
            exists_gt = torch.cat([labels["l3_left_exists"], labels["l3_right_exists"]], dim=1).view(B, 6).cpu().numpy()
            all_exists_pred.append(exists_pred)
            all_exists_gt.append(exists_gt)

            # Count
            pred_count = (exists_pred > 0.5).sum(axis=1)
            gt_count = exists_gt.sum(axis=1)
            all_count_pred.append(pred_count)
            all_count_gt.append(gt_count)

            # Per-feature collection
            for feat_name, pred_logits, gt_key, valid_key, collectors in [
                ("cyst", l3_output.cyst, "l3_{side}_cyst", "l3_{side}_cyst_valid",
                 (all_cyst_pred, all_cyst_gt, all_cyst_valid)),
                ("mass", l3_output.mass, "l3_{side}_mass", "l3_{side}_mass_valid",
                 (all_mass_pred, all_mass_gt, all_mass_valid)),
                ("tumor", l3_output.tumor, "l3_{side}_tumor", "l3_{side}_tumor_valid",
                 (all_tumor_pred, all_tumor_gt, all_tumor_valid)),
            ]:
                pred = torch.sigmoid(pred_logits).cpu().numpy()
                gt = torch.cat([
                    labels[gt_key.format(side="left")],
                    labels[gt_key.format(side="right")]
                ], dim=1).view(B, 6).cpu().numpy()
                valid = torch.cat([
                    labels[valid_key.format(side="left")],
                    labels[valid_key.format(side="right")]
                ], dim=1).view(B, 6).cpu().numpy()
                collectors[0].append(pred)
                collectors[1].append(gt)
                collectors[2].append(valid)

            # Size
            size_pred = l3_output.size.cpu().numpy()
            size_gt = torch.cat([labels["l3_left_size"], labels["l3_right_size"]], dim=1).view(B, 6).cpu().numpy()
            size_valid = torch.cat([labels["l3_left_size_valid"], labels["l3_right_size_valid"]], dim=1).view(B, 6).cpu().numpy()
            all_size_pred.append(size_pred)
            all_size_gt.append(size_gt)
            all_size_valid.append(size_valid)

            # Enhancement (categorical, 2 classes)
            enh_pred = l3_output.enhancement.argmax(dim=-1).cpu().numpy()  # (B, 6)
            enh_gt = torch.cat([labels["l3_left_enhancement"], labels["l3_right_enhancement"]], dim=1).view(B, 6).cpu().numpy()
            enh_valid = torch.cat([labels["l3_left_enhancement_valid"], labels["l3_right_enhancement_valid"]], dim=1).view(B, 6).cpu().numpy()
            all_enh_pred.append(enh_pred)
            all_enh_gt.append(enh_gt)
            all_enh_valid.append(enh_valid)

            # Attenuation (categorical, 4 classes)
            att_pred = l3_output.attenuation.argmax(dim=-1).cpu().numpy()  # (B, 6)
            att_gt = torch.cat([labels["l3_left_attenuation"], labels["l3_right_attenuation"]], dim=1).view(B, 6).cpu().numpy()
            att_valid = torch.cat([labels["l3_left_attenuation_valid"], labels["l3_right_attenuation_valid"]], dim=1).view(B, 6).cpu().numpy()
            all_att_pred.append(att_pred)
            all_att_gt.append(att_gt)
            all_att_valid.append(att_valid)

    # Concatenate
    exists_pred = np.concatenate(all_exists_pred)
    exists_gt = np.concatenate(all_exists_gt)

    metrics = {}

    # 1. Exists AUC
    try:
        metrics["auc_exists"] = float(roc_auc_score(exists_gt.flatten(), exists_pred.flatten()))
    except ValueError:
        metrics["auc_exists"] = 0.5

    # 2. Count MAE
    count_pred = np.concatenate(all_count_pred)
    count_gt = np.concatenate(all_count_gt)
    metrics["count_mae"] = float(np.abs(count_pred - count_gt).mean())

    # 3. Binary features (cyst, mass, tumor)
    exist_mask = exists_gt.flatten() > 0.5
    for feat_name, preds, gts, valids in [
        ("cyst", all_cyst_pred, all_cyst_gt, all_cyst_valid),
        ("mass", all_mass_pred, all_mass_gt, all_mass_valid),
        ("tumor", all_tumor_pred, all_tumor_gt, all_tumor_valid),
    ]:
        pred = np.concatenate(preds).flatten()
        gt = np.concatenate(gts).flatten()
        valid = np.concatenate(valids).flatten()
        mask = (valid > 0.5) & exist_mask
        if mask.sum() > 10:
            try:
                metrics[f"auc_{feat_name}"] = float(roc_auc_score(gt[mask], pred[mask]))
            except ValueError:
                metrics[f"auc_{feat_name}"] = 0.5
        else:
            metrics[f"auc_{feat_name}"] = None
        metrics[f"n_{feat_name}_valid"] = int(mask.sum())

    # 4. Size MAE
    size_pred = np.concatenate(all_size_pred).flatten()
    size_gt = np.concatenate(all_size_gt).flatten()
    size_valid = np.concatenate(all_size_valid).flatten()
    size_mask = (size_valid > 0.5) & exist_mask
    if size_mask.sum() > 0:
        metrics["size_mae"] = float(np.abs(size_pred[size_mask] - size_gt[size_mask]).mean())
        metrics["n_size_valid"] = int(size_mask.sum())
    else:
        metrics["size_mae"] = None

    # 5. Enhancement accuracy
    enh_pred = np.concatenate(all_enh_pred).flatten()
    enh_gt = np.concatenate(all_enh_gt).flatten()
    enh_valid = np.concatenate(all_enh_valid).flatten()
    enh_mask = (enh_valid > 0.5) & exist_mask
    if enh_mask.sum() > 0:
        metrics["enhancement_acc"] = float((enh_pred[enh_mask] == enh_gt[enh_mask].round()).mean())
        metrics["n_enhancement_valid"] = int(enh_mask.sum())
    else:
        metrics["enhancement_acc"] = None

    # 6. Attenuation accuracy
    att_pred = np.concatenate(all_att_pred).flatten()
    att_gt = np.concatenate(all_att_gt).flatten()
    att_valid = np.concatenate(all_att_valid).flatten()
    att_mask = (att_valid > 0.5) & exist_mask
    if att_mask.sum() > 0:
        metrics["attenuation_acc"] = float((att_pred[att_mask] == att_gt[att_mask].round()).mean())
        metrics["n_attenuation_valid"] = int(att_mask.sum())
    else:
        metrics["attenuation_acc"] = None

    # 7. Side-level abnormality AUC (comparable to L1)
    # exists_pred/gt shape: (N, 6), slots [0:3] = left, [3:6] = right
    left_pred = exists_pred[:, :3].max(axis=1)
    right_pred = exists_pred[:, 3:].max(axis=1)
    left_gt = (exists_gt[:, :3].max(axis=1) > 0.5).astype(float)
    right_gt = (exists_gt[:, 3:].max(axis=1) > 0.5).astype(float)
    side_pred = np.concatenate([left_pred, right_pred])
    side_gt = np.concatenate([left_gt, right_gt])
    try:
        metrics["auc_side_abnormal"] = float(roc_auc_score(side_gt, side_pred))
    except ValueError:
        metrics["auc_side_abnormal"] = 0.5

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mask_strategy", default="none", help="none | 7class")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Loaded: {args.checkpoint}")
    print(f"  Saved epoch: {ckpt['epoch']+1}, saved auc_mean: {ckpt['best_auc']:.4f}")

    # Build model
    ct_level = "L3"
    in_channels = 2
    encoder = SwinUNETREncoder(in_channels=in_channels, feature_size=48, pretrained="from_scratch")
    model = KidneyClassifier(encoder, label_level="L3_D", hidden_dim=128, dropout=0.3)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # Data
    val_ds = KidneyCTDataset(
        "valid", target_size=(320, 192, 224),
        ct_level=ct_level, augment=False,
        mask_strategy=args.mask_strategy,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate
    metrics = evaluate_l3_full(model, val_loader, device)

    print("\n=== Full L3 Evaluation ===")
    for k, v in sorted(metrics.items()):
        if v is not None:
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        else:
            print(f"  {k}: N/A (insufficient valid samples)")

    # Save
    out_path = Path(args.checkpoint).parent / "l3_full_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
