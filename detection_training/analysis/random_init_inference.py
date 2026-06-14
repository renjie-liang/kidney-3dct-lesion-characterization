"""Random-init model baseline: build fully-random weights (SwinUNETR + HeadB)
and run inference on val set. Saves predictions.npz and AP result.

Fresh SuPreM encoder weights are NOT used — the encoder is random-initialized
from scratch. This is the strictest "untrained model" baseline.

Output: <KIDNEY3DCT_ROOT>/experiments/analysis/random_baselines/random_init/
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

STAGE1_ROOT = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parents[2]))
DETECTION_ROOT = STAGE1_ROOT / "detection_training"
sys.path.insert(0, str(DETECTION_ROOT))

from data.dataset import KidneyCTDataset
from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier
from analysis.lesion_level_eval import compute_class_AP_coco

OUT_DIR = STAGE1_ROOT / "experiments" / "analysis" / "random_baselines" / "random_init"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Device: {device}")
    print("Building fully-random KidneyClassifier(SwinUNETR from_scratch + HeadB)...")

    encoder = SwinUNETREncoder(
        in_channels=2, feature_size=48,
        pretrained="from_scratch",   # <- random init
        use_checkpoint=True,
    )
    model = KidneyClassifier(
        encoder, label_level="L3_B",
        hidden_dim=256, dropout=0.3,
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params (all randomly initialized)")

    # Save the random state so the baseline is reproducible
    torch.save({"model_state_dict": model.state_dict(), "seed": 42},
               OUT_DIR / "model_state.pt")
    print(f"Saved random-init weights to {OUT_DIR/'model_state.pt'}")

    # Val dataset
    print("\nLoading val dataset...")
    val_ds = KidneyCTDataset(
        split="valid", target_size=(320, 192, 224), ct_level="L3",
        augment=False, mask_strategy="binary", flip_x=True, pad_only=0,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Inference
    print("Running inference on val set...")
    all_preds = []
    t0 = time.time()
    for batch_idx, (images, labels, sids) in enumerate(val_loader):
        images = images.to(device)
        d2_mask = labels.pop("d2_mask", None)
        labels.pop("resize_scale", None)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(images, mask=d2_mask)

        B = images.shape[0]
        batch = {
            "exists_pred": torch.sigmoid(out.exists).float().cpu().numpy(),
            "exists_gt": torch.cat([labels["l3_left_exists"], labels["l3_right_exists"]], dim=1).view(B, 6).numpy(),
        }
        for feat in ["cyst", "solid"]:
            batch[f"{feat}_pred"] = torch.sigmoid(getattr(out, feat)).float().cpu().numpy()
            batch[f"{feat}_gt"] = torch.cat([labels[f"l3_left_{feat}"], labels[f"l3_right_{feat}"]], dim=1).view(B, 6).numpy()
            batch[f"{feat}_valid"] = torch.cat([labels[f"l3_left_{feat}_valid"], labels[f"l3_right_{feat}_valid"]], dim=1).view(B, 6).numpy()
        batch["size_pred"] = out.size.float().cpu().numpy()
        batch["size_gt"] = torch.cat([labels["l3_left_size"], labels["l3_right_size"]], dim=1).view(B, 6).numpy()
        batch["size_valid"] = torch.cat([labels["l3_left_size_valid"], labels["l3_right_size_valid"]], dim=1).view(B, 6).numpy()
        batch["study_ids"] = np.array(sids)
        all_preds.append(batch)

    print(f"Inference done in {time.time()-t0:.0f}s, {len(all_preds)} batches")

    # Concatenate
    predictions = {}
    for k in all_preds[0].keys():
        if k == "study_ids":
            predictions[k] = np.concatenate([b[k] for b in all_preds])
        else:
            predictions[k] = np.concatenate([b[k] for b in all_preds], axis=0)

    np.savez_compressed(OUT_DIR / "predictions.npz", **predictions)
    print(f"Saved predictions to {OUT_DIR/'predictions.npz'}")

    # Compute AP
    print("\nComputing AP...")
    aps = {}
    for cls in ["cyst", "solid"]:
        r = compute_class_AP_coco(predictions, cls, size_tol=1.0)
        aps[cls] = {"AP": r["AP"], "n_pos": r["n_pos"], "n_pred": r["n_pred"]}
    mAP = (aps["cyst"]["AP"] + aps["solid"]["AP"]) / 2

    result = {
        "method": "random_init",
        "description": "SwinUNETR encoder from_scratch + HeadB with random weights. Single forward pass on val set.",
        "seed": 42,
        "n_params_M": n_params / 1e6,
        "val_size": predictions["exists_pred"].shape[0],
        "AP_cyst": aps["cyst"]["AP"],
        "AP_solid": aps["solid"]["AP"],
        "mAP": mAP,
        "n_pos_cyst": aps["cyst"]["n_pos"],
        "n_pos_solid": aps["solid"]["n_pos"],
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== Random-init baseline ===")
    print(f"  AP_cyst  = {aps['cyst']['AP']:.4f}")
    print(f"  AP_solid = {aps['solid']['AP']:.4f}")
    print(f"  mAP      = {mAP:.4f}")
    print(f"\nSaved to {OUT_DIR/'result.json'}")


if __name__ == "__main__":
    main()
