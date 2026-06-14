"""
Step 5: Generate L1/L2 labels from cropped NPZ masks.

Input:  step4_cropped/{case_id}.npz
Output: labels.json

Usage:
  python step5_labels.py
"""
import json
import logging
from pathlib import Path

import numpy as np

from config import STEP4_DIR, STEP5_LABELS

logger = logging.getLogger(__name__)


def derive_labels(mask_7class):
    """Derive L1/L2 labels from 7-class mask."""
    left_abnormal = bool(((mask_7class == 3) | (mask_7class == 5)).sum() > 0)
    right_abnormal = bool(((mask_7class == 4) | (mask_7class == 6)).sum() > 0)

    def get_max_size(binary_mask):
        volume_mm3 = binary_mask.sum()  # at 1mm isotropic, 1 voxel = 1mm³
        if volume_mm3 == 0:
            return 0.0
        diameter_mm = 2 * (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3)
        return round(diameter_mm / 10, 1)  # convert to cm

    left_lesion = (mask_7class == 3) | (mask_7class == 5)
    right_lesion = (mask_7class == 4) | (mask_7class == 6)

    return {
        "L1": {
            "abnormality": left_abnormal or right_abnormal,
            "left_abnormal": left_abnormal,
            "right_abnormal": right_abnormal,
        },
        "L2": {
            "left_has_cyst": bool((mask_7class == 5).sum() > 0),
            "right_has_cyst": bool((mask_7class == 6).sum() > 0),
            "left_has_solid": bool((mask_7class == 3).sum() > 0),
            "right_has_solid": bool((mask_7class == 4).sum() > 0),
            "left_max_size_cm": get_max_size(left_lesion),
            "right_max_size_cm": get_max_size(right_lesion),
        },
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    npz_files = sorted(STEP4_DIR.glob("*.npz"))
    logger.info(f"Step 5: Generate labels from {len(npz_files)} NPZ files")

    labels = {}
    for i, npz_path in enumerate(npz_files):
        case_id = npz_path.stem
        data = np.load(str(npz_path))
        mask = data["mask"]
        labels[case_id] = derive_labels(mask)

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(npz_files)}")

    with open(STEP5_LABELS, "w") as f:
        json.dump(labels, f, indent=2)

    # Stats
    n = len(labels)
    n_abn = sum(1 for v in labels.values() if v["L1"]["abnormality"])
    n_ls = sum(1 for v in labels.values() if v["L2"]["left_has_solid"])
    n_rs = sum(1 for v in labels.values() if v["L2"]["right_has_solid"])
    n_lc = sum(1 for v in labels.values() if v["L2"]["left_has_cyst"])
    n_rc = sum(1 for v in labels.values() if v["L2"]["right_has_cyst"])

    logger.info(f"\nLabels saved: {STEP5_LABELS} ({n} cases)")
    logger.info(f"  Abnormal: {n_abn}/{n}")
    logger.info(f"  Left solid: {n_ls}, Right solid: {n_rs}")
    logger.info(f"  Left cyst: {n_lc}, Right cyst: {n_rc}")


if __name__ == "__main__":
    main()
