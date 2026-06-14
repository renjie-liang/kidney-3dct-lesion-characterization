"""
Step 3: Assign laterality (left/right) to KiTS23 masks.

Same logic as UF pipeline (step3_assign_laterality._assign_laterality_fast):
  - midline = volume x-center
  - Each connected component assigned by its center_of_mass x vs midline
  - x >= midline → Left, x < midline → Right (RAS convention)

Input:  step2_oriented/{case_id}_mask.nii.gz (3-class: 0=bg, 1=kidney, 2=tumor, 3=cyst)
Output: step3_lateral_mask/{case_id}_mask7.nii.gz (7-class)

Usage:
  python step3_laterality.py --workers 4
"""
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from scipy import ndimage

from config import STEP2_DIR, STEP3_DIR, get_case_ids

logger = logging.getLogger(__name__)

# KiTS 3-class labels
KIDNEY, TUMOR, CYST = 1, 2, 3
# 7-class lateral labels
L_KIDNEY, R_KIDNEY = 1, 2
L_TUMOR, R_TUMOR = 3, 4
L_CYST, R_CYST = 5, 6


def assign_laterality(mask_3class):
    """Convert 3-class mask to 7-class lateral mask.

    Uses volume x-center as midline. Each connected component
    assigned by its center_of_mass. Same as UF pipeline.
    """
    out = np.zeros_like(mask_3class, dtype=np.uint8)
    midline_x = mask_3class.shape[0] / 2.0

    # Kidney
    kidney_mask = (mask_3class == KIDNEY)
    if kidney_mask.any():
        labeled, n_cc = ndimage.label(kidney_mask)
        coms = ndimage.center_of_mass(kidney_mask, labeled, range(1, n_cc + 1))
        for cc_i, com in enumerate(coms, 1):
            out[labeled == cc_i] = L_KIDNEY if com[0] >= midline_x else R_KIDNEY

    # Tumor and cyst — each CC independently
    for src, left_label, right_label in [
        (TUMOR, L_TUMOR, R_TUMOR),
        (CYST, L_CYST, R_CYST),
    ]:
        lesion_mask = (mask_3class == src)
        if not lesion_mask.any():
            continue
        labeled, n_cc = ndimage.label(lesion_mask)
        coms = ndimage.center_of_mass(lesion_mask, labeled, range(1, n_cc + 1))
        for cc_i, com in enumerate(coms, 1):
            out[labeled == cc_i] = left_label if com[0] >= midline_x else right_label

    return out


def process_single(case_id):
    out_mask = STEP3_DIR / f"{case_id}_mask7.nii.gz"

    if out_mask.exists():
        return (case_id, True, "skipped")

    in_mask = STEP2_DIR / f"{case_id}_mask.nii.gz"
    if not in_mask.exists():
        return (case_id, False, "step2 output not found")

    try:
        img = nib.load(str(in_mask))
        mask_3class = img.get_fdata(dtype=np.float32).astype(np.uint8)
        mask_7class = assign_laterality(mask_3class)

        labels_present = sorted(set(np.unique(mask_7class).astype(int)) - {0})

        out_img = nib.Nifti1Image(mask_7class, img.affine, img.header)
        out_img.header.set_data_dtype(np.uint8)
        nib.save(out_img, str(out_mask))

        return (case_id, True, f"labels={labels_present}")
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
    STEP3_DIR.mkdir(parents=True, exist_ok=True)

    cases = get_case_ids()
    if args.start is not None:
        cases = cases[args.start:]
    if args.end is not None:
        cases = cases[:args.end]

    if args.force:
        for c in cases:
            p = STEP3_DIR / f"{c}_mask7.nii.gz"
            if p.exists():
                p.unlink()

    logger.info(f"Step 3: Laterality {len(cases)} cases → {STEP3_DIR}")

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
