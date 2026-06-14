"""Reproduce the KiTS23 external-validation set in our NPZ format.

Runs the five-step preprocessing pipeline (resample -> orient -> laterality ->
crop -> labels) over a local copy of the official KiTS23 dataset and writes the
result to ``<out>/step4_cropped/*.npz`` plus ``<out>/labels.json`` — exactly the
layout ``eval_kits23.py`` expects (``KITS23_ROOT`` should point at ``<out>``).

KiTS23 is distributed under CC BY-NC-SA 4.0 by the KiTS challenge organizers
(https://kits-challenge.org/kits23/). Download it yourself first; this script
only transforms your local copy and does not redistribute the data.

Examples
--------
  # Process the full cohort (489 cases):
  python tools/prepare_kits23.py --raw /path/to/kits23/dataset --out datasets/KiTS23/processed --workers 8

  # Quick check on the first 10 cases:
  python tools/prepare_kits23.py --raw /path/to/kits23/dataset --end 10
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent / "process_kits23"
# Steps run in order; step5 derives labels and takes no --workers/--start/--end.
PARALLEL_STEPS = ["step1_resample.py", "step2_orient.py",
                  "step3_laterality.py", "step4_crop.py"]
LABEL_STEP = "step5_labels.py"


def run_step(script: str, env: dict, workers=None, start=None, end=None, force=False):
    cmd = [sys.executable, script]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    if start is not None:
        cmd += ["--start", str(start)]
    if end is not None:
        cmd += ["--end", str(end)]
    if force:
        cmd += ["--force"]
    print(f"\n=== Running {script} {' '.join(cmd[2:])} ===", flush=True)
    subprocess.run(cmd, cwd=PIPELINE_DIR, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True,
                    help="Folder containing the official KiTS23 case_XXXXX/ directories.")
    ap.add_argument("--out", type=Path, default=Path("datasets/KiTS23/processed"),
                    help="Output root (point KITS23_ROOT here when running eval_kits23.py).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Recompute, ignoring cached step outputs.")
    args = ap.parse_args()

    raw = args.raw.resolve()
    out = args.out.resolve()
    if not raw.exists():
        raise FileNotFoundError(f"--raw not found: {raw}")

    env = dict(os.environ)
    env["KITS23_RAW"] = str(raw)
    env["KITS23_ROOT"] = str(out)
    # The pipeline modules `from config import ...`; ensure their dir is importable.
    env["PYTHONPATH"] = str(PIPELINE_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    for script in PARALLEL_STEPS:
        run_step(script, env, workers=args.workers, start=args.start,
                 end=args.end, force=args.force)
    run_step(LABEL_STEP, env)  # step5 scans step4_cropped/, no slicing args

    print(f"\nDone. Processed KiTS23 written to: {out}")
    print("Set KITS23_ROOT to this path when running eval_kits23.py, e.g.:")
    print(f"  KITS23_ROOT={out} python eval_kits23.py --checkpoint <path/to/best_model.pt>")


if __name__ == "__main__":
    main()
