"""
Step 1: Resample KiTS23 raw data to 1mm isotropic.

Input:  raw/{case_id}/imaging.nii.gz, segmentation.nii.gz
Output: step1_resized/{case_id}_image.nii.gz, {case_id}_mask.nii.gz

Usage:
  python step1_resample.py --workers 4
  python step1_resample.py --workers 4 --start 0 --end 50
"""
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from config import KITS_RAW, STEP1_DIR, TARGET_SPACING, get_case_ids

logger = logging.getLogger(__name__)


def resample_volume(data, spacing, target_spacing, order=1):
    factors = [s / t for s, t in zip(spacing, target_spacing)]
    return zoom(data, factors, order=order)


def process_single(case_id):
    out_img = STEP1_DIR / f"{case_id}_image.nii.gz"
    out_mask = STEP1_DIR / f"{case_id}_mask.nii.gz"

    if out_img.exists() and out_mask.exists():
        return (case_id, True, "skipped")

    img_path = KITS_RAW / case_id / "imaging.nii.gz"
    seg_path = KITS_RAW / case_id / "segmentation.nii.gz"

    try:
        img_nii = nib.load(str(img_path))
        seg_nii = nib.load(str(seg_path))

        spacing = img_nii.header.get_zooms()[:3]
        image = img_nii.get_fdata(dtype=np.float32)
        mask = seg_nii.get_fdata(dtype=np.float32).astype(np.uint8)

        # Resample to 1mm isotropic
        image = resample_volume(image, spacing, TARGET_SPACING, order=1)
        mask = resample_volume(mask.astype(np.float32), spacing, TARGET_SPACING, order=0).astype(np.uint8)

        # Save as NIfTI (1mm isotropic affine)
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        nib.save(nib.Nifti1Image(image.astype(np.int16), affine), str(out_img))
        nib.save(nib.Nifti1Image(mask, affine), str(out_mask))

        return (case_id, True, f"shape={image.shape}")
    except Exception as e:
        return (case_id, False, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    STEP1_DIR.mkdir(parents=True, exist_ok=True)

    cases = get_case_ids()
    if args.start is not None:
        cases = cases[args.start:]
    if args.end is not None:
        cases = cases[:args.end]

    if args.force:
        for c in cases:
            for suffix in ["_image.nii.gz", "_mask.nii.gz"]:
                p = STEP1_DIR / f"{c}{suffix}"
                if p.exists():
                    p.unlink()

    logger.info(f"Step 1: Resample {len(cases)} cases → {STEP1_DIR}")

    t0 = time.time()
    ok, fail, skipped = 0, 0, 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_single, c): c for c in cases}
        for fut in as_completed(futures):
            cid, success, msg = fut.result()
            if msg == "skipped":
                skipped += 1
            elif success:
                ok += 1
            else:
                fail += 1
                logger.error(f"FAIL {cid}: {msg}")
            if (ok + skipped + fail) % 50 == 0:
                logger.info(f"  Progress: {ok + skipped + fail}/{len(cases)}")

    logger.info(f"Done: {ok} ok, {skipped} skipped, {fail} failed ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
