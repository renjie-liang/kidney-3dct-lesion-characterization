"""
Inference time benchmark for all model configurations.

Measures: preprocessing time + model inference time per sample.

Usage:
  python analysis/inference_time.py --device cuda:0
"""
import os
import sys
import time
import warnings

os.environ["NUMEXPR_MAX_THREADS"] = "16"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import KidneyCTDataset
from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier


def benchmark_preprocessing(ct_level, n_samples=20):
    """Time data loading + preprocessing."""
    ds = KidneyCTDataset("valid", target_size=(320, 192, 224), ct_level=ct_level, augment=False)

    times = []
    for i in range(min(n_samples, len(ds))):
        t0 = time.time()
        img, _, _ = ds[i]
        times.append(time.time() - t0)

    return np.mean(times), np.std(times)


def benchmark_inference(model, ct_level, device, n_samples=20, n_warmup=3):
    """Time model inference (forward pass only)."""
    ds = KidneyCTDataset("valid", target_size=(320, 192, 224), ct_level=ct_level, augment=False)
    model.eval()

    # Warmup
    with torch.no_grad():
        for i in range(n_warmup):
            img, _, _ = ds[i]
            img = img.unsqueeze(0).to(device)
            _ = model(img)

    torch.cuda.synchronize()

    times = []
    for i in range(min(n_samples, len(ds))):
        img, _, _ = ds[i]
        img = img.unsqueeze(0).to(device)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(img)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    return np.mean(times), np.std(times)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Samples per benchmark: {args.n_samples}")
    print()

    results = {}

    configs = [
        ("SuPreM + L2 (no mask)", "L2", 1),
        ("SuPreM + L3 binary mask", "L3", 2),
    ]

    for name, ct_level, in_channels in configs:
        print(f"=== {name} ===")

        # Preprocessing
        prep_mean, prep_std = benchmark_preprocessing(ct_level, args.n_samples)
        print(f"  Preprocessing: {prep_mean:.3f} ± {prep_std:.3f} s")

        # Build model
        encoder = SwinUNETREncoder(in_channels=in_channels, feature_size=48, pretrained="from_scratch")
        model = KidneyClassifier(encoder, label_level="L1", hidden_dim=128, dropout=0.3).to(device)

        # Inference
        inf_mean, inf_std = benchmark_inference(model, ct_level, device, args.n_samples)
        print(f"  Inference: {inf_mean:.3f} ± {inf_std:.3f} s")

        total = prep_mean + inf_mean
        print(f"  Total: {total:.3f} s/sample")
        print()

        results[name] = {
            "preprocessing_s": round(prep_mean, 3),
            "preprocessing_std": round(prep_std, 3),
            "inference_s": round(inf_mean, 3),
            "inference_std": round(inf_std, 3),
            "total_s": round(total, 3),
        }

        del model, encoder
        torch.cuda.empty_cache()

    # Try CTViT if available
    try:
        from models.ctvit_encoder import CTViTEncoder
        print("=== CTViT + L3 binary mask ===")
        prep_mean, prep_std = benchmark_preprocessing("L3", args.n_samples)
        print(f"  Preprocessing (480x480x241): estimating...")

        # CTViT uses different target size
        ds_ctvit = KidneyCTDataset("valid", target_size=(480, 480, 241), ct_level="L3", augment=False)
        prep_times = []
        for i in range(min(5, len(ds_ctvit))):
            t0 = time.time()
            img, _, _ = ds_ctvit[i]
            prep_times.append(time.time() - t0)
        prep_mean_ctvit = np.mean(prep_times)
        print(f"  Preprocessing: {prep_mean_ctvit:.3f} s")

        enc = CTViTEncoder(in_channels=2, pretrained=False)
        model = KidneyClassifier(enc, label_level="L1", hidden_dim=128, dropout=0.3).to(device)

        # Warmup + benchmark
        model.eval()
        with torch.no_grad():
            img, _, _ = ds_ctvit[0]
            img = img.unsqueeze(0).to(device)
            _ = model(img)

        torch.cuda.synchronize()
        inf_times = []
        for i in range(min(5, len(ds_ctvit))):
            img, _, _ = ds_ctvit[i]
            img = img.unsqueeze(0).to(device)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                _ = model(img)
            torch.cuda.synchronize()
            inf_times.append(time.time() - t0)

        inf_mean_ctvit = np.mean(inf_times)
        print(f"  Inference: {inf_mean_ctvit:.3f} s")
        print(f"  Total: {prep_mean_ctvit + inf_mean_ctvit:.3f} s/sample")

        results["CTViT + L3 binary mask"] = {
            "preprocessing_s": round(prep_mean_ctvit, 3),
            "inference_s": round(inf_mean_ctvit, 3),
            "total_s": round(prep_mean_ctvit + inf_mean_ctvit, 3),
        }

        del model, enc
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CTViT benchmark skipped: {e}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Config':<30} {'Preprocess':>10} {'Inference':>10} {'Total':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<30} {r['preprocessing_s']:>9.3f}s {r['inference_s']:>9.3f}s {r['total_s']:>9.3f}s")

    clinical = all(r["total_s"] < 30 for r in results.values())
    print(f"\nClinical feasibility (<30s): {'✅ All pass' if clinical else '❌ Some exceed'}")

    # Save
    out_path = Path(__file__).parent.parent / "experiments" / "inference_time.json"
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
