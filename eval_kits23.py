"""
KiTS23 external validation — zero-shot transfer from UF-trained models.

Supports L1, L2, and L3D checkpoints. All checkpoints are evaluated on
KiTS23 L1 (side-level abnormality) and L2 (side-level cyst/solid + size) labels.
L3D checkpoints use max-aggregation over slots to produce side-level predictions.

Usage:
  python eval_kits23.py --checkpoint path/to/epoch_11.pt --label_level L1 --mask_strategy binary --flip_x
  python eval_kits23.py --checkpoint path/to/epoch_17.pt --label_level L2 --mask_strategy binary --flip_x
  python eval_kits23.py --checkpoint path/to/epoch_20.pt --label_level L3_D --mask_strategy binary --flip_x
"""
import argparse
import csv
import json
import os
import warnings

os.environ["NUMEXPR_MAX_THREADS"] = "16"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score, f1_score

from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier

KITS_PROCESSED = Path(os.environ.get("KITS23_ROOT", "datasets/KiTS23/processed"))
TARGET_SIZE = (320, 192, 224)

# KiTS23 7-class mask mapping (same semantics as UF):
#   0=background, 1=left_kidney, 2=right_kidney, 3=cyst, 4=tumor, 5=ureter
# Since KiTS23 doesn't have full 7-class, we map available classes:
KITS_TO_7CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6}  # kits tumor→UF tumor(5)


def load_kits_sample(case_id, target_size, flip_x=False, mask_strategy="binary"):
    """Load and preprocess one KiTS23 sample.

    Args:
        case_id: KiTS23 case identifier
        target_size: (X, Y, Z) target volume size
        flip_x: flip x,y axes to match SuPreM convention
        mask_strategy: "binary" (kidney only) or "7class" (multi-label)

    Returns:
        ct_tensor: (C, X, Y, Z) tensor, C=1 (no mask) or C=2 (with mask)
        resize_scale: float, ratio for size correction
    """
    npz_path = KITS_PROCESSED / "step4_cropped" / f"{case_id}.npz"
    data = np.load(str(npz_path))
    image = data["image"].astype(np.float32)
    mask = data["mask"].astype(np.uint8)

    # Flip x,y to match SuPreM/AbdomenAtlas convention
    if flip_x:
        image = np.flip(image, axis=(0, 1)).copy()
        mask = np.flip(mask, axis=(0, 1)).copy()

    # HU window + normalize
    image = np.clip(image, -200, 400)
    image = (image + 200) / 600.0

    # Resize + pad (same as UF pipeline)
    tx, ty, tz = target_size
    sx, sy, sz = image.shape
    scale = min(tx / sx, ty / sy, tz / sz)
    resize_scale = scale
    new_shape = (int(round(sx * scale)), int(round(sy * scale)), int(round(sz * scale)))

    if new_shape != image.shape:
        factors = [n / s for n, s in zip(new_shape, image.shape)]
        image = zoom(image, factors, order=1)
        mask = zoom(mask.astype(np.float32), factors, order=0).astype(np.uint8)

    pad_x = tx - image.shape[0]
    pad_y = ty - image.shape[1]
    pad_z = tz - image.shape[2]
    pad_before = (pad_x // 2, pad_y // 2, pad_z // 2)
    pad_after = (pad_x - pad_before[0], pad_y - pad_before[1], pad_z - pad_before[2])

    image = np.pad(image, [(pad_before[i], pad_after[i]) for i in range(3)], constant_values=0)
    mask = np.pad(mask, [(pad_before[i], pad_after[i]) for i in range(3)], constant_values=0)

    # Build mask channel
    if mask_strategy == "binary":
        mask_channel = ((mask == 1) | (mask == 2)).astype(np.float32)  # kidney parenchyma only
    elif mask_strategy == "7class":
        # Normalize to [0,1] range: divide by max class (6)
        mask_channel = mask.astype(np.float32) / 6.0
    else:
        raise ValueError(f"Unknown mask_strategy: {mask_strategy}")

    ct_tensor = np.stack([image, mask_channel], axis=0)  # (2, X, Y, Z)
    return torch.from_numpy(ct_tensor).float(), resize_scale


def aggregate_l3d_to_side(l3_output):
    """Aggregate L3D slot predictions to side-level predictions.

    Returns dict with side-level probabilities matching L1/L2 format.
    """
    exists_prob = torch.sigmoid(l3_output.exists).cpu().numpy()  # (1, 6)
    cyst_prob = torch.sigmoid(l3_output.cyst).cpu().numpy()      # (1, 6)
    mass_prob = torch.sigmoid(l3_output.mass).cpu().numpy()      # (1, 6)
    tumor_prob = torch.sigmoid(l3_output.tumor).cpu().numpy()    # (1, 6)
    size_pred = l3_output.size.cpu().numpy()                     # (1, 6)

    # Side-level abnormality: max exists prob over 3 slots per side
    left_abn = exists_prob[0, :3].max()
    right_abn = exists_prob[0, 3:].max()

    # Side-level cyst: max(exists * cyst) over slots
    left_cyst = (exists_prob[0, :3] * cyst_prob[0, :3]).max()
    right_cyst = (exists_prob[0, 3:] * cyst_prob[0, 3:]).max()

    # Side-level solid: max(exists * (mass or tumor)) over slots
    solid_prob = 1.0 - (1.0 - mass_prob) * (1.0 - tumor_prob)  # noisy-OR
    left_solid = (exists_prob[0, :3] * solid_prob[0, :3]).max()
    right_solid = (exists_prob[0, 3:] * solid_prob[0, 3:]).max()

    # Size: max size from existing slots
    left_size = size_pred[0, :3].max()
    right_size = size_pred[0, 3:].max()

    return {
        "abn": np.array([left_abn, right_abn]),
        "cls": np.array([left_cyst, right_cyst, left_solid, right_solid]),
        "size": np.array([left_size, right_size]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_level", required=True, help="L1, L2, or L3_D")
    parser.add_argument("--mask_strategy", default="binary", help="binary or 7class")
    parser.add_argument("--flip_x", action="store_true")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: checkpoint parent)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    label_level = args.label_level

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = SwinUNETREncoder(in_channels=2, feature_size=48, pretrained="from_scratch")
    model = KidneyClassifier(encoder, label_level=label_level, hidden_dim=128, dropout=0.3)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    ckpt_name = Path(args.checkpoint).stem
    print(f"Loaded: {args.checkpoint}")
    print(f"  label_level={label_level}, mask_strategy={args.mask_strategy}, flip_x={args.flip_x}")

    # Load KiTS labels
    with open(KITS_PROCESSED / "labels.json") as f:
        kits_labels = json.load(f)

    cases = sorted(kits_labels.keys())
    print(f"KiTS23 cases: {len(cases)}")

    # Collect predictions
    all_abn_preds, all_abn_targets = [], []
    all_cls_preds, all_cls_targets = [], []
    all_size_preds, all_size_targets = [], []
    failed = 0

    with torch.no_grad():
        for i, case_id in enumerate(cases):
            try:
                ct, resize_scale = load_kits_sample(
                    case_id, TARGET_SIZE,
                    flip_x=args.flip_x,
                    mask_strategy=args.mask_strategy,
                )
                ct = ct.unsqueeze(0).to(device)
                lab = kits_labels[case_id]

                # Ground truth
                abn_gt = np.array([
                    float(lab["L1"]["left_abnormal"]),
                    float(lab["L1"]["right_abnormal"]),
                ])
                cls_gt = np.array([
                    float(lab["L2"]["left_has_cyst"]),
                    float(lab["L2"]["right_has_cyst"]),
                    float(lab["L2"]["left_has_solid"]),
                    float(lab["L2"]["right_has_solid"]),
                ])
                size_gt = np.array([
                    float(lab["L2"]["left_max_size_cm"]),
                    float(lab["L2"]["right_max_size_cm"]),
                ])

                # Predict based on label_level
                if label_level == "L1":
                    logits = model(ct)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]  # (2,)
                    all_abn_preds.append(probs)
                    all_abn_targets.append(abn_gt)
                    # L1 cannot produce cyst/solid/size predictions

                elif label_level == "L2":
                    cls_logits, size_pred = model(ct)
                    cls_probs = torch.sigmoid(cls_logits).cpu().numpy()[0]  # (4,)
                    size_vals = size_pred.cpu().numpy()[0]  # (2,)
                    # L2 cls: [left_cyst, right_cyst, left_solid, right_solid]
                    # Abnormality: max(cyst, solid) per side
                    left_abn = max(cls_probs[0], cls_probs[2])
                    right_abn = max(cls_probs[1], cls_probs[3])
                    all_abn_preds.append(np.array([left_abn, right_abn]))
                    all_abn_targets.append(abn_gt)
                    all_cls_preds.append(cls_probs)
                    all_cls_targets.append(cls_gt)
                    # Correct size by resize_scale
                    all_size_preds.append(size_vals / max(resize_scale, 0.1))
                    all_size_targets.append(size_gt)

                elif label_level.startswith("L3"):
                    l3_output, extra = model(ct)
                    side = aggregate_l3d_to_side(l3_output)
                    all_abn_preds.append(side["abn"])
                    all_abn_targets.append(abn_gt)
                    all_cls_preds.append(side["cls"])
                    all_cls_targets.append(cls_gt)
                    # Correct size by resize_scale
                    all_size_preds.append(side["size"] / max(resize_scale, 0.1))
                    all_size_targets.append(size_gt)

                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(cases)}")

            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Failed {case_id}: {e}")

    print(f"\nEvaluated: {len(cases) - failed}, Failed: {failed}")

    # ── Compute metrics ──
    metrics = {}

    # L1: abnormality AUC
    if all_abn_preds:
        abn_preds = np.stack(all_abn_preds)
        abn_targets = np.stack(all_abn_targets)
        for idx, name in enumerate(["left_abnormal", "right_abnormal"]):
            try:
                metrics[f"auc_{name}"] = float(roc_auc_score(abn_targets[:, idx], abn_preds[:, idx]))
            except ValueError:
                metrics[f"auc_{name}"] = 0.5
            metrics[f"pos_rate_{name}"] = float(abn_targets[:, idx].mean())
        metrics["auc_abn_mean"] = np.mean([metrics["auc_left_abnormal"], metrics["auc_right_abnormal"]])

    # L2: cyst/solid AUC
    if all_cls_preds:
        cls_preds = np.stack(all_cls_preds)
        cls_targets = np.stack(all_cls_targets)
        for idx, name in enumerate(["left_cyst", "right_cyst", "left_solid", "right_solid"]):
            try:
                metrics[f"auc_{name}"] = float(roc_auc_score(cls_targets[:, idx], cls_preds[:, idx]))
            except ValueError:
                metrics[f"auc_{name}"] = 0.5
            metrics[f"pos_rate_{name}"] = float(cls_targets[:, idx].mean())
        metrics["auc_cls_mean"] = np.mean([
            metrics["auc_left_cyst"], metrics["auc_right_cyst"],
            metrics["auc_left_solid"], metrics["auc_right_solid"],
        ])

    # Size MAE (only where GT > 0)
    if all_size_preds:
        size_preds = np.stack(all_size_preds)
        size_targets = np.stack(all_size_targets)
        has_lesion = size_targets > 0
        if has_lesion.any():
            metrics["size_mae"] = float(np.abs(size_preds[has_lesion] - size_targets[has_lesion]).mean())

    # Print results
    print(f"\n{'='*50}")
    print(f"KiTS23 Zero-shot Results: {label_level}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"{'='*50}")

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    out_path = output_dir / f"kits23_zeroshot_{label_level}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
