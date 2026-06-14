"""KiTS23 processing-pipeline configuration.

Paths are read from environment variables so no machine-specific paths are
baked in:

  KITS23_RAW   directory containing the official KiTS23 ``case_XXXXX/`` folders
               (each with ``imaging.nii.gz`` and ``segmentation.nii.gz``).
               Default: ``datasets/KiTS23/raw/dataset``
  KITS23_ROOT  output root for the processed data (NPZ + labels.json).
               Default: ``datasets/KiTS23/processed``

KiTS23 is distributed under CC BY-NC-SA 4.0 by the KiTS challenge organizers
(https://kits-challenge.org/kits23/). This pipeline only transforms a local
copy you downloaded yourself; it does not redistribute the data.
"""
import os
from pathlib import Path

# Raw data: official KiTS23 download (folder of case_XXXXX/ directories).
KITS_RAW = Path(os.environ.get("KITS23_RAW", "datasets/KiTS23/raw/dataset"))

# Processed output root (must match KITS23_ROOT used by eval_kits23.py).
KITS_OUT = Path(os.environ.get("KITS23_ROOT", "datasets/KiTS23/processed"))

# Step outputs
STEP1_DIR = KITS_OUT / "step1_resized"          # resampled 1mm isotropic NIfTI
STEP2_DIR = KITS_OUT / "step2_oriented"          # transposed to match UF (x,y,z)
STEP3_DIR = KITS_OUT / "step3_lateral_mask"       # 7-class lateral mask NIfTI
STEP4_DIR = KITS_OUT / "step4_cropped"            # cropped NPZ + crop_meta.json
STEP5_LABELS = KITS_OUT / "labels.json"           # L1/L2 labels

# Viz directory (unused in the released pipeline)
VIZ_DIR = KITS_OUT / "viz"

# Processing parameters
TARGET_SPACING = (1.0, 1.0, 1.0)
MARGIN_MM = 30
ORIENT_TRANSPOSE = (2, 1, 0)  # KiTS23 raw (z,y,x) -> UF convention (x,y,z)
ORIENT_FLIP_AXES = [2]         # flip z-axis to match UF coronal direction


def get_case_ids():
    """Return sorted case IDs that have a segmentation under KITS_RAW."""
    if not KITS_RAW.exists():
        raise FileNotFoundError(
            f"KITS23_RAW not found: {KITS_RAW}. Download the official KiTS23 "
            f"dataset and set KITS23_RAW to the folder of case_XXXXX/ directories."
        )
    cases = []
    for d in sorted(KITS_RAW.iterdir()):
        if d.is_dir() and d.name.startswith("case_"):
            if (d / "segmentation.nii.gz").exists():
                cases.append(d.name)
    return cases
