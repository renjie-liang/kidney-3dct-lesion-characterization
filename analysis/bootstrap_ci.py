"""
Bootstrap confidence intervals for AUC and other metrics.

Loads a trained checkpoint, runs inference on validation set,
computes bootstrap 95% CI for all metrics.

Usage:
  python analysis/bootstrap_ci.py --checkpoint experiments/.../best_model.pt --n_bootstrap 1000
"""
import argparse
import json
import os
import sys
import warnings

os.environ["NUMEXPR_MAX_THREADS"] = "16"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import KidneyCTDataset
from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier


def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, seed=42):
    """Compute bootstrap 95% CI for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)
        except ValueError:
            continue

    if not scores:
        return None, None, None

    scores = np.array(scores)
    mean = np.mean(scores)
    ci_low = np.percentile(scores, 2.5)
    ci_high = np.percentile(scores, 97.5)
    return mean, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    label_level = cfg.get("label_level", "L1")
    ct_level = cfg.get("ct_level", "L3")

    print(f"Loaded: {args.checkpoint}")
    print(f"  label_level={label_level}, ct_level={ct_level}")
    print(f"  saved_auc={ckpt['best_auc']:.4f}")

    # Build model
    in_channels = 2 if ct_level == "L3" else 1
    encoder = SwinUNETREncoder(in_channels=in_channels, feature_size=48, pretrained="from_scratch")
    model = KidneyClassifier(encoder, label_level=label_level, hidden_dim=128, dropout=0.3)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Data
    val_ds = KidneyCTDataset("valid", target_size=(320, 192, 224), ct_level=ct_level, augment=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

    # Collect predictions
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            if label_level == "L1":
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                targets = torch.stack([labels["left_abnormal"], labels["right_abnormal"]], dim=1).numpy()
            elif label_level == "L2":
                cls_logits, size_pred = model(images)
                probs = torch.sigmoid(cls_logits).cpu().numpy()
                targets = torch.stack([
                    labels["left_has_cyst"], labels["right_has_cyst"],
                    labels["left_has_solid"], labels["right_has_solid"],
                ], dim=1).numpy()
            all_preds.append(probs)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    print(f"\nSamples: {len(all_preds)}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print()

    # Compute bootstrap CI for each metric
    if label_level == "L1":
        names = ["left_abnormal", "right_abnormal"]
    else:
        names = ["left_cyst", "right_cyst", "left_solid", "right_solid"]

    results = {}

    for i, name in enumerate(names):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        pos_rate = y_true.mean()

        # AUC
        mean, ci_low, ci_high = bootstrap_metric(y_true, y_pred, roc_auc_score, args.n_bootstrap)
        results[f"auc_{name}"] = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high, "pos_rate": pos_rate}
        print(f"  auc_{name}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}] (pos_rate={pos_rate:.3f})")

        # AUPRC
        mean_pr, ci_low_pr, ci_high_pr = bootstrap_metric(y_true, y_pred, average_precision_score, args.n_bootstrap)
        results[f"auprc_{name}"] = {"mean": mean_pr, "ci_low": ci_low_pr, "ci_high": ci_high_pr}
        print(f"  auprc_{name}: {mean_pr:.4f} [{ci_low_pr:.4f}, {ci_high_pr:.4f}]")
        print()

    # Overall AUC
    auc_values = [results[f"auc_{name}"]["mean"] for name in names if results[f"auc_{name}"]["mean"] is not None]
    results["auc_mean"] = float(np.mean(auc_values))
    print(f"  auc_mean: {results['auc_mean']:.4f}")

    # Save
    out_path = Path(args.checkpoint).parent / "bootstrap_ci.json"
    # Convert numpy to float for JSON
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean_results[k] = {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
        else:
            clean_results[k] = float(v)

    with open(out_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
