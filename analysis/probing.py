"""
Layer-wise linear probing for SwinUNETR encoders.

For each encoder (SuPreM, VoCo, scratch) and each of the 5 SwinUNETR stages,
extract features via global average pooling, then train a logistic regression
to predict L1 abnormality labels.

Measures the discriminative quality of representations at each depth
without the confound of nonlinear head capacity.

Usage:
  python analysis/probing.py --device cuda:0
"""
import os
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import KidneyCTDataset
from models.encoder import SwinUNETREncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parents[1]))
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "analysis" / "probing"


def extract_stage_features(encoder, dataloader, device):
    """Extract features from each SwinUNETR stage via global average pooling.

    Returns:
        features: dict of {stage_idx: np.array (N, C)}
        labels: np.array (N, 2) — [left_abnormal, right_abnormal]
    """
    encoder.eval()
    stage_features = {i: [] for i in range(5)}
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels, study_ids) in enumerate(dataloader):
            images = images.to(device)

            # Get hidden states from all 5 stages
            hidden_states = encoder.swin_unetr.swinViT(images)

            for i, hs in enumerate(hidden_states):
                pooled = nn.functional.adaptive_avg_pool3d(hs, 1).flatten(1)
                stage_features[i].append(pooled.cpu().numpy())

            batch_labels = np.stack([
                labels["left_abnormal"].numpy(),
                labels["right_abnormal"].numpy(),
            ], axis=1)
            all_labels.append(batch_labels)

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Extracted {(batch_idx+1) * images.shape[0]} samples")

    features = {i: np.concatenate(stage_features[i]) for i in range(5)}
    labels = np.concatenate(all_labels)
    return features, labels


def linear_probe(train_features, train_labels, val_features, val_labels):
    """Train logistic regression on train, evaluate AUC on val."""
    target_names = ["left_abnormal", "right_abnormal"]
    aucs = []

    for t, name in enumerate(target_names):
        y_train = train_labels[:, t]
        y_val = val_labels[:, t]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            aucs.append(0.5)
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(train_features, y_train)
        y_prob = clf.predict_proba(val_features)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        aucs.append(auc)

    return {
        "auc_left_abnormal": float(aucs[0]),
        "auc_right_abnormal": float(aucs[1]),
        "auc_mean": float(np.mean(aucs)),
    }


def compute_cka(X, Y):
    """Linear CKA between two feature matrices. X: (n, d1), Y: (n, d2)."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtX = X.T @ X
    YtY = Y.T @ Y
    hsic_xy = np.trace(XtX @ YtY) / ((X.shape[0] - 1) ** 2)
    hsic_xx = np.trace(XtX @ XtX) / ((X.shape[0] - 1) ** 2)
    hsic_yy = np.trace(YtY @ YtY) / ((X.shape[0] - 1) ** 2)
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build datasets — Crop+Kmask (ct_level=L3), flip for SuPreM convention
    logger.info("Building datasets...")
    train_ds = KidneyCTDataset(
        split="train", target_size=(320, 192, 224),
        ct_level="L3", augment=False, mask_strategy="binary",
        flip_x=True, pad_only=0,
    )
    val_ds = KidneyCTDataset(
        split="valid", target_size=(320, 192, 224),
        ct_level="L3", augment=False, mask_strategy="binary",
        flip_x=True, pad_only=0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    encoders = {
        "suprem": "suprem",
        "voco": "voco",
        "scratch": "from_scratch",
    }

    all_results = {}
    all_val_features = {}

    for enc_name, pretrained in encoders.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Encoder: {enc_name} ({pretrained})")
        logger.info(f"{'='*60}")

        encoder = SwinUNETREncoder(in_channels=2, feature_size=48, pretrained=pretrained)
        encoder = encoder.to(device)
        encoder.eval()

        # Extract features
        logger.info("Extracting train features...")
        train_feats, train_labels = extract_stage_features(encoder, train_loader, device)
        logger.info(f"  Train: {train_feats[0].shape[0]} samples")

        logger.info("Extracting val features...")
        val_feats, val_labels = extract_stage_features(encoder, val_loader, device)
        logger.info(f"  Val: {val_feats[0].shape[0]} samples")

        # Store val features for CKA
        all_val_features[enc_name] = val_feats

        # Probe each stage
        enc_results = {}
        for stage in range(5):
            feat_dim = train_feats[stage].shape[1]
            metrics = linear_probe(
                train_feats[stage], train_labels,
                val_feats[stage], val_labels,
            )
            metrics["feature_dim"] = feat_dim
            enc_results[f"stage_{stage}"] = metrics
            logger.info(f"  Stage {stage} (dim={feat_dim}): "
                        f"L={metrics['auc_left_abnormal']:.4f}, "
                        f"R={metrics['auc_right_abnormal']:.4f}, "
                        f"mean={metrics['auc_mean']:.4f}")

        all_results[enc_name] = enc_results

        # Free GPU memory
        del encoder
        torch.cuda.empty_cache()

    # CKA analysis between encoders
    logger.info(f"\n{'='*60}")
    logger.info("CKA Analysis")
    logger.info(f"{'='*60}")
    cka_results = {}
    enc_names = list(encoders.keys())
    for stage in range(5):
        cka_results[f"stage_{stage}"] = {}
        for i in range(len(enc_names)):
            for j in range(i + 1, len(enc_names)):
                n1, n2 = enc_names[i], enc_names[j]
                n = min(all_val_features[n1][stage].shape[0],
                        all_val_features[n2][stage].shape[0])
                cka = compute_cka(all_val_features[n1][stage][:n],
                                  all_val_features[n2][stage][:n])
                pair = f"{n1}_vs_{n2}"
                cka_results[f"stage_{stage}"][pair] = cka
                logger.info(f"  Stage {stage} {pair}: CKA={cka:.4f}")

    # Save
    output = {"probing": all_results, "cka": cka_results}
    out_path = OUTPUT_DIR / "probing_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Encoder':<10} {'Stage 0':>10} {'Stage 1':>10} {'Stage 2':>10} {'Stage 3':>10} {'Stage 4':>10}")
    print(f"{'':<10} {'(48d)':>10} {'(96d)':>10} {'(192d)':>10} {'(384d)':>10} {'(768d)':>10}")
    print(f"{'='*70}")
    for enc_name in encoders:
        row = all_results[enc_name]
        vals = [row[f"stage_{s}"]["auc_mean"] for s in range(5)]
        print(f"{enc_name:<10} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f} {vals[4]:>10.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
