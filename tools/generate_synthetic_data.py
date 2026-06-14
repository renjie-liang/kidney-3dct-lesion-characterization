"""Generate a tiny SYNTHETIC kidney-CT dataset that matches the on-disk format
expected by ``data/dataset.py`` (``KidneyCTDataset``).

The real UF Health dataset cannot be released (patient-privacy restrictions).
This generator produces a handful of *fake* cases — random volumes, random
segmentation masks, and randomly sampled lesion labels — solely so that the
training and evaluation code can be run end-to-end.

It does NOT reproduce any paper number. It is a smoke test and a precise,
executable specification of the data contract:

  <out>/UF_Kidney_CT/
  ├── final_dataset/
  │   ├── train/<sid>.npz          # keys: "image" (float32 HU), "mask" (uint8, 0-6)
  │   ├── valid/<sid>.npz
  │   ├── study_id_to_label.json   # {sid: {"order_key": <key>}}
  │   └── labels.jsonl             # one record per order_key (see schema below)
  ├── resized_train_images/<sid>.nii.gz   # full-volume images for the L1 path (optional)
  └── resized_valid_images/<sid>.nii.gz

labels.jsonl record schema (per order_key):
  {
    "DEID_ORDER_KEY": <key>,
    "L1": {"left_abnormal": 0|1, "right_abnormal": 0|1},
    "L2": {"left_has_cyst", "right_has_cyst",
           "left_has_solid", "right_has_solid",
           "left_max_size_cm", "right_max_size_cm"},
    "L3_left_lesions":  [ <lesion>, ... ],
    "L3_right_lesions": [ <lesion>, ... ]
  }
  <lesion> = {"Cyst": "true|false|unknown", "Mass": ..., "Tumor": ...,
              "Size_cm": float, "Enhancement": "enhancement|non-enhancement|unknown",
              "Attenuation": "hypoattenuating|hyperattenuating|isoattenuating|negative|unknown"}

L1 and L2 are derived deterministically from the L3 lesions, exactly as in the
real label pipeline.

Mask label convention (7-class, matching the UF preprocessing):
  0 background, 1 left kidney, 2 right kidney, 3 cyst, 5 solid/tumor.

Usage:
  python tools/generate_synthetic_data.py --out datasets --n-train 8 --n-valid 4
"""
import argparse
import json
from pathlib import Path

import numpy as np

# 7-class mask labels
BG, KIDNEY_L, KIDNEY_R, CYST, _RESERVED4, SOLID, _RESERVED6 = 0, 1, 2, 3, 4, 5, 6

ATTEN_CHOICES = (
    ["unknown"] * 6 + ["hypoattenuating"] * 3 + ["hyperattenuating"] + ["isoattenuating"]
)


def _sphere(shape, center, radius):
    """Boolean sphere mask within a volume of given shape."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return d2 <= radius ** 2


def _sample_lesions(rng, max_lesions=2):
    """Sample 0..max_lesions lesions for one kidney side."""
    n = rng.choice([0, 1, 2], p=[0.45, 0.45, 0.10])
    lesions = []
    for _ in range(n):
        is_cyst = rng.random() < 0.85          # cysts dominate (matches real data)
        size_cm = float(np.round(rng.lognormal(mean=0.2, sigma=0.5), 1))  # ~1-3 cm
        size_cm = max(0.5, min(size_cm, 6.0))
        lesion = {
            "Cyst": "true" if is_cyst else "false",
            "Mass": "false" if is_cyst else ("true" if rng.random() < 0.7 else "false"),
            "Tumor": "false" if is_cyst else ("true" if rng.random() < 0.5 else "false"),
            "Size_cm": size_cm,
            # Enhancement / attenuation are mostly unreported in the real reports.
            "Enhancement": "unknown" if rng.random() < 0.9
            else rng.choice(["enhancement", "non-enhancement"]),
            "Attenuation": str(rng.choice(ATTEN_CHOICES)),
        }
        # Guarantee a solid lesion carries at least one solid flag.
        if not is_cyst and lesion["Mass"] == "false" and lesion["Tumor"] == "false":
            lesion["Mass"] = "true"
        lesions.append(lesion)
    return lesions


def _is_solid(lesion):
    return lesion["Mass"] == "true" or lesion["Tumor"] == "true"


def _derive_side_labels(lesions):
    """Derive L1/L2 side-level labels from L3 lesions (as in the real pipeline)."""
    has_cyst = int(any(l["Cyst"] == "true" for l in lesions))
    has_solid = int(any(_is_solid(l) for l in lesions))
    max_size = float(max([l["Size_cm"] for l in lesions], default=0.0))
    abnormal = int(len(lesions) > 0)
    return abnormal, has_cyst, has_solid, max_size


def _make_volume(rng, lesions_left, lesions_right, shape):
    """Build a fake CT volume (HU) and a 7-class segmentation mask."""
    image = rng.normal(0.0, 20.0, size=shape).astype(np.float32)  # soft-tissue background
    mask = np.zeros(shape, dtype=np.uint8)

    dz, dy, dx = shape
    half = dx // 2
    kidney_r = min(dz, dy, half) // 3

    # Left kidney blob in the left half, right kidney blob in the right half.
    for side, kid_label, lesions, cx in [
        ("left", KIDNEY_L, lesions_left, half // 2),
        ("right", KIDNEY_R, lesions_right, half + half // 2),
    ]:
        center = (dz // 2, dy // 2, cx)
        kidney = _sphere(shape, center, kidney_r)
        mask[kidney] = kid_label
        image[kidney] = rng.normal(30.0, 10.0, size=kidney.sum()).astype(np.float32)

        # Place each lesion as a small sphere inside the kidney blob.
        for li, lesion in enumerate(lesions):
            offset = (li - 0.5) * kidney_r * 0.5
            lc = (int(dz // 2 + offset), dy // 2, cx)
            lrad = max(2, int(kidney_r * 0.4))
            blob = _sphere(shape, lc, lrad)
            if _is_solid(lesion):
                mask[blob] = SOLID
                image[blob] = rng.normal(45.0, 8.0, size=blob.sum()).astype(np.float32)
            else:
                mask[blob] = CYST
                image[blob] = rng.normal(10.0, 5.0, size=blob.sum()).astype(np.float32)

    return image, mask


def generate(out_root: Path, n_train: int, n_valid: int, shape, seed: int, write_nifti: bool):
    rng = np.random.default_rng(seed)
    base = out_root / "UF_Kidney_CT"
    final_ds = base / "final_dataset"
    (final_ds / "train").mkdir(parents=True, exist_ok=True)
    (final_ds / "valid").mkdir(parents=True, exist_ok=True)

    sid_to_meta = {}
    label_records = []

    for split, n in [("train", n_train), ("valid", n_valid)]:
        if write_nifti:
            (base / f"resized_{split}_images").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            sid = f"synthetic_{split}_{i:03d}"
            lesions_left = _sample_lesions(rng)
            lesions_right = _sample_lesions(rng)
            image, mask = _make_volume(rng, lesions_left, lesions_right, shape)

            np.savez_compressed(final_ds / split / f"{sid}.npz", image=image, mask=mask)

            if write_nifti:
                _write_nifti(base / f"resized_{split}_images" / f"{sid}.nii.gz", image)

            l_abn, l_cyst, l_solid, l_size = _derive_side_labels(lesions_left)
            r_abn, r_cyst, r_solid, r_size = _derive_side_labels(lesions_right)

            sid_to_meta[sid] = {"order_key": sid}
            label_records.append({
                "DEID_ORDER_KEY": sid,
                "L1": {"left_abnormal": l_abn, "right_abnormal": r_abn},
                "L2": {
                    "left_has_cyst": l_cyst, "right_has_cyst": r_cyst,
                    "left_has_solid": l_solid, "right_has_solid": r_solid,
                    "left_max_size_cm": l_size, "right_max_size_cm": r_size,
                },
                "L3_left_lesions": lesions_left,
                "L3_right_lesions": lesions_right,
            })

    (final_ds / "study_id_to_label.json").write_text(json.dumps(sid_to_meta, indent=2))
    with open(final_ds / "labels.jsonl", "w") as f:
        for rec in label_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {n_train} train + {n_valid} valid synthetic cases to {final_ds}")
    print(f"  study_id_to_label.json : {len(sid_to_meta)} entries")
    print(f"  labels.jsonl           : {len(label_records)} records")
    if not write_nifti:
        print("  (skipped resized *.nii.gz — install nibabel and pass --nifti for the L1 path)")


def _write_nifti(path: Path, volume: np.ndarray):
    import nibabel as nib  # optional dependency, only for the L1 input path
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4)), str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("datasets"),
                    help="Output root (KIDNEY_DATA_ROOT points at <out>/UF_Kidney_CT/final_dataset).")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-valid", type=int, default=4)
    ap.add_argument("--shape", type=int, nargs=3, default=[64, 64, 64],
                    help="Volume shape (D H W). Keep small for a fast smoke test.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nifti", action="store_true",
                    help="Also write resized *.nii.gz volumes (needs nibabel; enables the L1 path).")
    args = ap.parse_args()

    generate(args.out, args.n_train, args.n_valid, tuple(args.shape), args.seed, args.nifti)


if __name__ == "__main__":
    main()
