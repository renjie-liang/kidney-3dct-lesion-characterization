"""
Step 4: Bilateral kidney crop.

Crops image + 7-class mask using mask bbox + 30mm margin.
Saves as NPZ (same format as UF pipeline) + crop_meta.json.

Input:  step2_oriented/{case_id}_image.nii.gz
        step3_lateral_mask/{case_id}_mask7.nii.gz
Output: step4_cropped/{case_id}.npz
        step4_cropped/crop_meta.json

Usage:
  python step4_crop.py --workers 4
"""
import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np

from config import STEP2_DIR, STEP3_DIR, STEP4_DIR, MARGIN_MM, get_case_ids

logger = logging.getLogger(__name__)


def get_crop_slices(mask_7class, shape, margin):
    kidney = (mask_7class >= 1) & (mask_7class <= 6)
    if kidney.sum() == 0:
        return [0, shape[0], 0, shape[1], 0, shape[2]]
    nz = np.argwhere(kidney)
    mins, maxs = nz.min(axis=0), nz.max(axis=0)
    return [
        int(max(0, mins[0] - margin)), int(min(shape[0], maxs[0] + margin + 1)),
        int(max(0, mins[1] - margin)), int(min(shape[1], maxs[1] + margin + 1)),
        int(max(0, mins[2] - margin)), int(min(shape[2], maxs[2] + margin + 1)),
    ]


def process_single(case_id):
    out_npz = STEP4_DIR / f"{case_id}.npz"

    if out_npz.exists():
        return (case_id, True, "skipped", None)

    in_img = STEP2_DIR / f"{case_id}_image.nii.gz"
    in_mask = STEP3_DIR / f"{case_id}_mask7.nii.gz"

    if not in_img.exists() or not in_mask.exists():
        return (case_id, False, "input not found", None)

    try:
        image = nib.load(str(in_img)).get_fdata(dtype=np.float32)
        mask = nib.load(str(in_mask)).get_fdata(dtype=np.float32).astype(np.uint8)

        crop_slices = get_crop_slices(mask, image.shape, MARGIN_MM)
        s = crop_slices
        cropped_img = image[s[0]:s[1], s[2]:s[3], s[4]:s[5]]
        cropped_mask = mask[s[0]:s[1], s[2]:s[3], s[4]:s[5]]

        np.savez_compressed(
            str(out_npz),
            image=cropped_img.astype(np.int16),
            mask=cropped_mask,
        )

        meta = {
            "orig_shape": list(image.shape),
            "crop_shape": list(cropped_img.shape),
            "crop_slices": crop_slices,
        }
        return (case_id, True, f"crop={cropped_img.shape}", meta)
    except Exception as e:
        return (case_id, False, str(e), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    STEP4_DIR.mkdir(parents=True, exist_ok=True)

    cases = get_case_ids()
    if args.start is not None:
        cases = cases[args.start:]
    if args.end is not None:
        cases = cases[:args.end]

    if args.force:
        for c in cases:
            p = STEP4_DIR / f"{c}.npz"
            if p.exists():
                p.unlink()

    logger.info(f"Step 4: Crop {len(cases)} cases → {STEP4_DIR}")

    t0 = time.time()
    ok, fail, skipped = 0, 0, 0
    crop_meta = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_single, c): c for c in cases}
        for fut in as_completed(futures):
            cid, success, msg, meta = fut.result()
            if msg == "skipped":
                skipped += 1
            elif success:
                ok += 1
                crop_meta[cid] = meta
            else:
                fail += 1
                logger.error(f"FAIL {cid}: {msg}")
            if (ok + skipped + fail) % 50 == 0:
                logger.info(f"  Progress: {ok + skipped + fail}/{len(cases)}")

    # Save crop metadata
    meta_path = STEP4_DIR / "crop_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
        existing.update(crop_meta)
        crop_meta = existing
    with open(meta_path, "w") as f:
        json.dump(crop_meta, f, indent=2)

    logger.info(f"Done: {ok} ok, {skipped} skipped, {fail} failed ({time.time()-t0:.1f}s)")
    logger.info(f"Crop meta: {meta_path} ({len(crop_meta)} entries)")


if __name__ == "__main__":
    main()
