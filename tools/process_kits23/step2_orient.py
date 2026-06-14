"""
Step 2: Transpose KiTS23 volumes to match UF orientation.

KiTS23 raw axes are (z, y, x) → transpose(2, 1, 0) → (x, y, z) to match UF.

Input:  step1_resized/{case_id}_image.nii.gz, {case_id}_mask.nii.gz
Output: step2_oriented/{case_id}_image.nii.gz, {case_id}_mask.nii.gz

Usage:
  python step2_orient.py --workers 4
"""
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np

from config import STEP1_DIR, STEP2_DIR, ORIENT_TRANSPOSE, ORIENT_FLIP_AXES, get_case_ids

logger = logging.getLogger(__name__)


def process_single(case_id):
    out_img = STEP2_DIR / f"{case_id}_image.nii.gz"
    out_mask = STEP2_DIR / f"{case_id}_mask.nii.gz"

    if out_img.exists() and out_mask.exists():
        return (case_id, True, "skipped")

    in_img = STEP1_DIR / f"{case_id}_image.nii.gz"
    in_mask = STEP1_DIR / f"{case_id}_mask.nii.gz"

    if not in_img.exists() or not in_mask.exists():
        return (case_id, False, "step1 output not found")

    try:
        image = nib.load(str(in_img)).get_fdata(dtype=np.float32)
        mask = nib.load(str(in_mask)).get_fdata(dtype=np.float32).astype(np.uint8)

        # Transpose to match UF orientation
        image = image.transpose(ORIENT_TRANSPOSE)
        mask = mask.transpose(ORIENT_TRANSPOSE)

        # Flip specified axes
        for axis in ORIENT_FLIP_AXES:
            image = np.flip(image, axis=axis)
            mask = np.flip(mask, axis=axis)

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

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
    STEP2_DIR.mkdir(parents=True, exist_ok=True)

    cases = get_case_ids()
    if args.start is not None:
        cases = cases[args.start:]
    if args.end is not None:
        cases = cases[:args.end]

    if args.force:
        for c in cases:
            for suffix in ["_image.nii.gz", "_mask.nii.gz"]:
                p = STEP2_DIR / f"{c}{suffix}"
                if p.exists():
                    p.unlink()

    logger.info(f"Step 2: Orient {len(cases)} cases → {STEP2_DIR}")

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
