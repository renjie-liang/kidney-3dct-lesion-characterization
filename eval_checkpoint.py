"""
Standalone evaluation script — load a checkpoint and evaluate on validation set.
Avoids any Hydra working directory issues.

Usage:
  python eval_checkpoint.py --checkpoint experiments/suprem_L3_L1_.../best_model.pt --ct_level L3 --label_level L1
"""
import argparse
import json
import os
import warnings
from pathlib import Path

os.environ["NUMEXPR_MAX_THREADS"] = "16"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

from data.dataset import KidneyCTDataset
from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier


def evaluate(model, dataloader, label_level, device):
    """Evaluate model on validation set."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            d2_mask = labels.pop("d2_mask", None)

            if label_level == "L1":
                logits = model(images, mask=d2_mask)
                targets = torch.stack([labels["left_abnormal"], labels["right_abnormal"]], dim=1)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

            elif label_level == "L2":
                cls_logits, size_pred = model(images, mask=d2_mask)
                targets = torch.stack([
                    labels["left_has_cyst"], labels["right_has_cyst"],
                    labels["left_has_solid"], labels["right_has_solid"],
                ], dim=1)
                probs = torch.sigmoid(cls_logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

            n_batches += 1

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = {}
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

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ct_level", default="L3")
    parser.add_argument("--label_level", default="L1")
    parser.add_argument("--encoder", default="suprem")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    target_size = (320, 192, 224)
    val_ds = KidneyCTDataset("valid", target_size=target_size, ct_level=args.ct_level, augment=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Build model
    in_channels = 2 if args.ct_level == "L3" else 1
    if args.ct_level == "D2":
        from models.mask_guided_pooling import MaskGuidedEncoder
        base_enc = SwinUNETREncoder(in_channels=1, feature_size=48, pretrained="from_scratch")
        encoder = MaskGuidedEncoder(base_enc, feature_dim=768)
    else:
        encoder = SwinUNETREncoder(in_channels=in_channels, feature_size=48, pretrained="from_scratch")

    model = KidneyClassifier(encoder, label_level=args.label_level, hidden_dim=128, dropout=0.3)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Saved at epoch {ckpt['epoch'] + 1}, saved best_auc={ckpt['best_auc']:.4f}")
    print(f"  Saved val_metrics: {ckpt.get('val_metrics', 'N/A')}")

    # Re-evaluate
    print(f"\nRe-evaluating on validation set ({len(val_ds)} samples)...")
    metrics = evaluate(model, val_loader, args.label_level, device)

    print(f"\nResults:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Save
    out_path = Path(args.checkpoint).parent / "reeval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
