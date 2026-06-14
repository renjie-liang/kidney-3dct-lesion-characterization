"""KiTS23 external validation — zero-shot transfer from detection_training main methods.

Loads checkpoints trained by train_detection.py (HeadB or HeadC + L1L2 hier / focal),
runs inference on all KiTS23 cases, aggregates slot predictions to side-level via
noisy-OR, and reports L1 abn / L2 cyst / L2 solid AUC + size MAE + per-case AP.

Usage:
  python detection_training/eval_kits23_detection.py \
      --checkpoint path/to/best_crit_model.pt \
      --label_level L3_B \
      --mask_strategy none \
      --output_dir experiments/external_validation/kits23_v3_mainmethods/<run_name>/
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["NUMEXPR_MAX_THREADS"] = "16"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score

# Use detection_training's own models (solid-merged, return_spatial=True aware)
DETECTION_ROOT = Path(__file__).parent
sys.path.insert(0, str(DETECTION_ROOT))

from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier

KITS_PROCESSED = Path(os.environ.get("KITS23_ROOT", "datasets/KiTS23/processed"))
TARGET_SIZE = (320, 192, 224)


def load_kits_sample(case_id, target_size, flip_x=True, mask_strategy="none"):
    """Load and preprocess one KiTS23 sample (same pipeline as main eval_kits23.py)."""
    npz_path = KITS_PROCESSED / "step4_cropped" / f"{case_id}.npz"
    data = np.load(str(npz_path))
    image = data["image"].astype(np.float32)
    mask = data["mask"].astype(np.uint8)

    if flip_x:
        image = np.flip(image, axis=(0, 1)).copy()
        mask = np.flip(mask, axis=(0, 1)).copy()

    image = np.clip(image, -200, 400)
    image = (image + 200) / 600.0

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

    # detection_training dataset ALWAYS returns 2-channel input for ct_level=L3.
    # "none" and "binary" both put a binary kidney mask as the second channel;
    # "7class" puts the full multi-class mask normalized to [0, 1].
    if mask_strategy in ("none", "binary"):
        mask_channel = ((mask == 1) | (mask == 2)).astype(np.float32)
    elif mask_strategy == "7class":
        mask_channel = mask.astype(np.float32) / 6.0
    else:
        raise ValueError(f"Unknown mask_strategy: {mask_strategy}")
    ct_tensor = np.stack([image, mask_channel], axis=0)  # (2, X, Y, Z)

    return torch.from_numpy(ct_tensor).float(), resize_scale


def aggregate_l3_to_side(l3_output):
    """Noisy-OR aggregation from per-slot predictions to side-level.

    Uses the SAME formula as training (`losses/combined.py::_noisy_or_hier_loss`):
        P(side has attribute) = 1 − ∏_i (1 − p_i,attribute)
    (not the old max-over-slot heuristic that eval_kits23.py uses).
    """
    eps = 1e-7
    exists = torch.sigmoid(l3_output.exists).cpu().numpy()[0]  # (6,)
    cyst = torch.sigmoid(l3_output.cyst).cpu().numpy()[0]      # (6,)
    solid = torch.sigmoid(l3_output.solid).cpu().numpy()[0]    # (6,)
    size = l3_output.size.cpu().numpy()[0]                     # (6,)

    def noisy_or(p):
        return float(1.0 - np.prod(1.0 - p + eps))

    L, R = slice(0, 3), slice(3, 6)
    left_abn = noisy_or(exists[L])
    right_abn = noisy_or(exists[R])
    left_cyst = noisy_or(exists[L] * cyst[L])
    right_cyst = noisy_or(exists[R] * cyst[R])
    left_solid = noisy_or(exists[L] * solid[L])
    right_solid = noisy_or(exists[R] * solid[R])
    # Size: use max across existing slots as per-side summary
    left_size = float(size[L].max())
    right_size = float(size[R].max())

    return {
        "abn": np.array([left_abn, right_abn]),
        "cls": np.array([left_cyst, right_cyst, left_solid, right_solid]),
        "size": np.array([left_size, right_size]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint. If omitted and --random_init set, model is randomly initialized.")
    parser.add_argument("--random_init", action="store_true",
                        help="Skip checkpoint loading; use freshly initialized model (noise-floor baseline).")
    parser.add_argument("--random_init_seed", type=int, default=42,
                        help="RNG seed for random initialization (only used with --random_init).")
    parser.add_argument("--label_level", required=True, choices=["L3_A", "L3_B", "L3_C", "L3_D"])
    parser.add_argument("--mask_strategy", default="none", choices=["none", "binary", "7class"])
    parser.add_argument("--flip_x", action="store_true", default=True)
    parser.add_argument("--no_flip_x", action="store_false", dest="flip_x")
    parser.add_argument("--output_dir", required=True, help="Directory to save results JSON")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.random_init and args.checkpoint is None:
        parser.error("--checkpoint is required unless --random_init is set")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # detection_training's L3 pipeline always uses 2 input channels
    # (image + mask channel), regardless of mask_strategy="none"/"binary"/"7class".
    in_channels = 2

    if args.random_init:
        # Noise-floor baseline: random init, no SuPreM pretrain, no training.
        torch.manual_seed(args.random_init_seed)
        np.random.seed(args.random_init_seed)
        encoder = SwinUNETREncoder(in_channels=in_channels, feature_size=48, pretrained="from_scratch")
        model = KidneyClassifier(encoder, label_level=args.label_level, hidden_dim=128, dropout=0.3)
        model = model.to(device).eval()
        print(f"Random-init baseline (no checkpoint, seed={args.random_init_seed})")
        print(f"  label_level={args.label_level}, mask_strategy={args.mask_strategy}, "
              f"in_channels={in_channels}, flip_x={args.flip_x}")
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        print(f"  label_level={args.label_level}, mask_strategy={args.mask_strategy}, "
              f"in_channels={in_channels}, flip_x={args.flip_x}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        encoder = SwinUNETREncoder(in_channels=in_channels, feature_size=48, pretrained="from_scratch")
        # hidden_dim=256 matches E0g_l1l2_crop.yaml and E0h_focal_crop.yaml (hidden_dim goes to L3 head)
        model = KidneyClassifier(encoder, label_level=args.label_level, hidden_dim=128, dropout=0.3)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()

    # Load KiTS23 labels.
    with open(KITS_PROCESSED / "labels.json") as f:
        kits_labels = json.load(f)
    cases = sorted(kits_labels.keys())
    print(f"KiTS23 cases: {len(cases)}")

    all_abn_preds, all_abn_targets = [], []
    all_cls_preds, all_cls_targets = [], []
    all_size_preds, all_size_targets = [], []
    all_case_ids = []
    failed = 0

    with torch.no_grad():
        for i, case_id in enumerate(cases):
            try:
                ct, resize_scale = load_kits_sample(
                    case_id, TARGET_SIZE, flip_x=args.flip_x, mask_strategy=args.mask_strategy,
                )
                ct = ct.unsqueeze(0).to(device)
                lab = kits_labels[case_id]

                abn_gt = np.array([float(lab["L1"]["left_abnormal"]),
                                   float(lab["L1"]["right_abnormal"])])
                cls_gt = np.array([float(lab["L2"]["left_has_cyst"]),
                                   float(lab["L2"]["right_has_cyst"]),
                                   float(lab["L2"]["left_has_solid"]),
                                   float(lab["L2"]["right_has_solid"])])
                size_gt = np.array([float(lab["L2"]["left_max_size_cm"]),
                                    float(lab["L2"]["right_max_size_cm"])])

                output = model(ct)
                # detection_training heads return either L3Output alone or (L3Output, extra)
                l3_output = output[0] if isinstance(output, tuple) else output

                side = aggregate_l3_to_side(l3_output)
                all_abn_preds.append(side["abn"])
                all_abn_targets.append(abn_gt)
                all_cls_preds.append(side["cls"])
                all_cls_targets.append(cls_gt)
                # Size correction: predicted in resized-volume units → undo resize.
                all_size_preds.append(side["size"] / max(resize_scale, 0.1))
                all_size_targets.append(size_gt)
                all_case_ids.append(case_id)

                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(cases)}")
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Failed {case_id}: {e}")

    print(f"\nEvaluated: {len(cases) - failed}, Failed: {failed}")

    metrics = {"label_level": args.label_level, "mask_strategy": args.mask_strategy,
               "flip_x": args.flip_x,
               "checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "random_init": args.random_init,
               "random_init_seed": args.random_init_seed if args.random_init else None,
               "n_cases": len(cases) - failed, "n_failed": failed}

    abn_preds = np.stack(all_abn_preds)
    abn_targets = np.stack(all_abn_targets)
    for idx, name in enumerate(["left_abnormal", "right_abnormal"]):
        try:
            metrics[f"auc_{name}"] = float(roc_auc_score(abn_targets[:, idx], abn_preds[:, idx]))
        except ValueError:
            metrics[f"auc_{name}"] = 0.5
        metrics[f"pos_rate_{name}"] = float(abn_targets[:, idx].mean())
    metrics["auc_abn_mean"] = np.mean([metrics["auc_left_abnormal"], metrics["auc_right_abnormal"]])

    cls_preds = np.stack(all_cls_preds)
    cls_targets = np.stack(all_cls_targets)
    for idx, name in enumerate(["left_cyst", "right_cyst", "left_solid", "right_solid"]):
        try:
            metrics[f"auc_{name}"] = float(roc_auc_score(cls_targets[:, idx], cls_preds[:, idx]))
        except ValueError:
            metrics[f"auc_{name}"] = 0.5
        metrics[f"pos_rate_{name}"] = float(cls_targets[:, idx].mean())
    metrics["auc_cls_mean"] = float(np.mean([
        metrics["auc_left_cyst"], metrics["auc_right_cyst"],
        metrics["auc_left_solid"], metrics["auc_right_solid"],
    ]))

    size_preds = np.stack(all_size_preds)
    size_targets = np.stack(all_size_targets)
    has_lesion = size_targets > 0
    if has_lesion.any():
        metrics["size_mae"] = float(np.abs(size_preds[has_lesion] - size_targets[has_lesion]).mean())
    metrics["n_has_lesion_sides"] = int(has_lesion.sum())

    print(f"\n{'='*50}")
    print(f"KiTS23 zero-shot ({args.label_level}) results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "kits23_result.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Per-case predictions for bootstrap significance tests. Side-level only
    # (abn/cls/size), since KiTS23 has no per-slot GT.
    pred_path = output_dir / "predictions.npz"
    np.savez(
        pred_path,
        case_ids=np.array(all_case_ids),
        abn_preds=abn_preds, abn_targets=abn_targets,
        cls_preds=cls_preds, cls_targets=cls_targets,
        size_preds=size_preds, size_targets=size_targets,
    )
    print(f"Saved: {pred_path}")


if __name__ == "__main__":
    main()
