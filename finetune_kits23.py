"""
Fine-tune UF-pretrained model on KiTS23.

Loads best UF checkpoint, continues training on KiTS23 train split,
evaluates on KiTS23 test split.

Usage:
  python finetune_kits23.py \
    --uf_checkpoint experiments/.../best_model.pt \
    --label_level L2 --lr 1e-5 --epochs 20
"""
import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ["NUMEXPR_MAX_THREADS"] = "16"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, ConcatDataset

from data.kits23_dataset import KiTS23Dataset, kits23_train_test_split
from models.encoder import SwinUNETREncoder
from models.classifier import KidneyClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parent))
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"


def compute_l2_loss(cls_logits, size_pred, labels):
    cls_targets = torch.stack([
        labels["left_has_cyst"], labels["right_has_cyst"],
        labels["left_has_solid"], labels["right_has_solid"],
    ], dim=1)
    cls_loss = nn.BCEWithLogitsLoss()(cls_logits, cls_targets)

    size_targets = torch.stack([labels["left_max_size"], labels["right_max_size"]], dim=1)
    size_mask = size_targets > 0
    if size_mask.any():
        size_loss = nn.MSELoss()(size_pred[size_mask], size_targets[size_mask])
    else:
        size_loss = torch.tensor(0.0, device=cls_logits.device)

    return cls_loss + 0.1 * size_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    all_size_preds, all_size_targets = [], []
    total_loss = 0
    n_batches = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        cls_logits, size_pred = model(images)
        loss = compute_l2_loss(cls_logits, size_pred, labels)

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

        total_loss += loss.item()
        n_batches += 1

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = {"val_loss": total_loss / max(n_batches, 1)}

    names = ["left_cyst", "right_cyst", "left_solid", "right_solid"]
    aucs = []
    for i, name in enumerate(names):
        try:
            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except ValueError:
            auc = 0.5
        metrics[f"auc_{name}"] = auc
        aucs.append(auc)

    metrics["auc_mean"] = float(np.mean(aucs))

    preds_binary = (all_preds > 0.5).astype(int)
    for i, name in enumerate(names):
        metrics[f"f1_{name}"] = float(f1_score(all_targets[:, i], preds_binary[:, i], zero_division=0))

    # Size MAE
    size_preds = np.concatenate(all_size_preds)
    size_targets = np.concatenate(all_size_targets)
    mask = size_targets > 0
    if mask.any():
        metrics["size_mae"] = float(np.abs(size_preds[mask] - size_targets[mask]).mean())

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uf_checkpoint", required=True, help="Path to best UF model checkpoint")
    parser.add_argument("--label_level", default="L2")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulate_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mix_uf", action="store_true",
                        help="Mix UF train data to prevent catastrophic forgetting")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load UF checkpoint
    ckpt = torch.load(args.uf_checkpoint, map_location="cpu", weights_only=False)
    uf_cfg = ckpt.get("config", {})
    logger.info(f"UF checkpoint: {args.uf_checkpoint}")
    logger.info(f"  UF best AUC: {ckpt['best_auc']:.4f}, epoch: {ckpt['epoch']+1}")

    # Build model from UF config
    in_channels = 2 if uf_cfg.get("ct_level", "L3") == "L3" else 1
    encoder = SwinUNETREncoder(
        in_channels=in_channels,
        feature_size=uf_cfg.get("feature_size", 48),
        pretrained="from_scratch",  # weights come from checkpoint
    )
    model = KidneyClassifier(
        encoder, label_level=args.label_level,
        hidden_dim=uf_cfg.get("hidden_dim", 128),
        dropout=uf_cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.info("Loaded UF model weights")

    # Data
    target_size = tuple(uf_cfg.get("target_size", [320, 192, 224]))
    ct_level = uf_cfg.get("ct_level", "L3")

    kits_train_ids, kits_test_ids = kits23_train_test_split(seed=args.seed)
    logger.info(f"KiTS23 split: {len(kits_train_ids)} train, {len(kits_test_ids)} test")

    kits_train_ds = KiTS23Dataset(
        kits_train_ids, target_size=target_size,
        ct_level=ct_level, augment=True,
    )
    kits_test_ds = KiTS23Dataset(
        kits_test_ids, target_size=target_size,
        ct_level=ct_level, augment=False,
    )

    # Optionally mix UF data
    if args.mix_uf:
        from data.dataset import KidneyCTDataset
        uf_train_ds = KidneyCTDataset(
            split="train", target_size=target_size,
            ct_level=ct_level, augment=True,
        )
        train_ds = ConcatDataset([kits_train_ds, uf_train_ds])
        logger.info(f"Mixed training: {len(kits_train_ds)} KiTS + {len(uf_train_ds)} UF = {len(train_ds)}")
    else:
        train_ds = kits_train_ds
        logger.info(f"KiTS-only training: {len(train_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        kits_test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Experiment dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mix_tag = "mix_uf" if args.mix_uf else "kits_only"
    exp_name = f"kits23_ft_{mix_tag}_{args.label_level}_lr{args.lr}_{timestamp}"
    exp_dir = EXPERIMENT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate zero-shot first
    zero_shot_metrics = evaluate(model, test_loader, device)
    logger.info(f"Zero-shot KiTS23: auc_mean={zero_shot_metrics['auc_mean']:.4f}")
    for k, v in sorted(zero_shot_metrics.items()):
        if "auc" in k:
            logger.info(f"  {k}: {v:.4f}")

    # Optimizer — lower lr for fine-tuning
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    # MLflow
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("kits23_finetune")

    with mlflow.start_run(run_name=exp_name):
        mlflow.log_params({
            "uf_checkpoint": args.uf_checkpoint,
            "uf_best_auc": ckpt["best_auc"],
            "label_level": args.label_level,
            "lr": args.lr,
            "epochs": args.epochs,
            "mix_uf": args.mix_uf,
            "kits_train": len(kits_train_ids),
            "kits_test": len(kits_test_ids),
        })
        mlflow.log_metrics({f"zero_shot_{k}": v for k, v in zero_shot_metrics.items()})

        best_auc = 0

        for epoch in range(args.epochs):
            t0 = time.time()
            model.train()
            total_loss = 0
            n_batches = 0
            optimizer.zero_grad()

            for step, (images, labels, _) in enumerate(train_loader):
                images = images.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                with torch.amp.autocast("cuda"):
                    cls_logits, size_pred = model(images)
                    loss = compute_l2_loss(cls_logits, size_pred, labels)
                    loss = loss / args.accumulate_steps

                scaler.scale(loss).backward()

                if (step + 1) % args.accumulate_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * args.accumulate_steps
                n_batches += 1

            if n_batches % args.accumulate_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            scheduler.step()

            train_loss = total_loss / max(n_batches, 1)
            val_metrics = evaluate(model, test_loader, device)
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) — "
                f"train_loss={train_loss:.4f}, val_loss={val_metrics['val_loss']:.4f}, "
                f"auc_mean={val_metrics['auc_mean']:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(k, v, step=epoch)

            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "val_metrics": val_metrics,
                "config": {
                    "uf_checkpoint": args.uf_checkpoint,
                    "label_level": args.label_level,
                    "lr": args.lr,
                    "mix_uf": args.mix_uf,
                    **uf_cfg,
                },
            }
            torch.save(ckpt_data, exp_dir / f"epoch_{epoch+1}.pt")

            if val_metrics["auc_mean"] > best_auc:
                best_auc = val_metrics["auc_mean"]
                ckpt_data["best_auc"] = best_auc
                torch.save(ckpt_data, exp_dir / "best_model.pt")
                logger.info(f"  New best AUC: {best_auc:.4f}")

        # Final summary
        best_ckpt = torch.load(exp_dir / "best_model.pt", weights_only=False)
        final_metrics = best_ckpt["val_metrics"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Zero-shot → Fine-tuned comparison:")
        logger.info(f"  Zero-shot AUC: {zero_shot_metrics['auc_mean']:.4f}")
        logger.info(f"  Fine-tuned AUC: {final_metrics['auc_mean']:.4f} (epoch {best_ckpt['epoch']+1})")
        logger.info(f"  Improvement: {final_metrics['auc_mean'] - zero_shot_metrics['auc_mean']:+.4f}")
        for k in sorted(final_metrics.keys()):
            if "auc" in k:
                zs = zero_shot_metrics.get(k, 0)
                ft = final_metrics[k]
                logger.info(f"  {k}: {zs:.4f} → {ft:.4f} ({ft-zs:+.4f})")

        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})
        mlflow.log_metric("final_improvement", final_metrics["auc_mean"] - zero_shot_metrics["auc_mean"])

        results = {
            "zero_shot": zero_shot_metrics,
            "finetuned": final_metrics,
            "improvement": final_metrics["auc_mean"] - zero_shot_metrics["auc_mean"],
        }
        with open(exp_dir / "final_metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved to: {exp_dir}")


if __name__ == "__main__":
    main()
