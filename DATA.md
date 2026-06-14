# Data

This repository contains **no private clinical data**. It is designed to run on
two small public datasets, hosted separately on the companion Hugging Face
dataset:
<https://huggingface.co/datasets/LiangRenjie/kidney-3dct-lesion-characterization>.

## 1. UF Health dataset (not released)

The UF Health kidney-CT dataset described in the paper contains de-identified
clinical CT volumes and radiology-report-derived annotations from a single
academic medical center. It is **not publicly available** because of
patient-privacy restrictions and institutional data-use requirements. No PHI,
reports, identifiers, or private annotations appear anywhere in this repository.

### Data format (the contract)

`KidneyCTDataset` (`data/dataset.py`) expects, under `KIDNEY_DATA_ROOT`:

```
final_dataset/
├── train/<sid>.npz      # keys: "image" (float, HU), "mask" (uint8, 7-class 0-6)
├── valid/<sid>.npz
├── study_id_to_label.json   # {sid: {"order_key": <key>}}
└── labels.jsonl             # one JSON record per order_key
```

7-class mask labels: `0` background, `1` left kidney, `2` right kidney,
`3` left solid, `4` right solid, `5` left cyst, `6` right cyst.

Each `labels.jsonl` record:

```json
{
  "DEID_ORDER_KEY": "<key>",
  "L1": {"left_abnormal": 0, "right_abnormal": 1},
  "L2": {"left_has_cyst": 0, "right_has_cyst": 1,
         "left_has_solid": 0, "right_has_solid": 0,
         "left_max_size_cm": 0.0, "right_max_size_cm": 2.1},
  "L3_left_lesions":  [],
  "L3_right_lesions": [
    {"Cyst": "true", "Mass": "false", "Tumor": "false", "Size_cm": 2.1,
     "Enhancement": "unknown", "Attenuation": "hypoattenuating"}
  ]
}
```

`tools/generate_synthetic_data.py` is an executable specification of this
contract — read it (or its output) to see exactly what the loader expects.

## 2. Synthetic data (runnable substitute)

Because the real data is private, a procedurally generated **synthetic** dataset
lets you run the entire pipeline, including the per-lesion (L3) path:

```bash
python tools/generate_synthetic_data.py --out datasets --n-train 8 --n-valid 4
```

It contains random volumes, random masks, and randomly sampled labels following
the schema above. It exercises the code only — it does **not** reproduce any
paper number. A pre-generated copy is also on the Hugging Face dataset.

## 3. KiTS23 (external validation)

The paper uses [KiTS23](https://kits-challenge.org/kits23/) (489 cases) for
zero-shot external validation. KiTS23 is released by the KiTS challenge
organizers under **CC BY-NC-SA 4.0**.

- **Small subset:** 6 preprocessed cases are provided on the Hugging Face
  dataset (`kits23_subset/`) so the external-validation code path runs out of the
  box. Because they derive from KiTS23, they remain under CC BY-NC-SA 4.0 and
  require attribution (see the subset's `ATTRIBUTION.md`).
- **Full reproduction:** download the official KiTS23 dataset yourself, then
  convert it into our format with:

  ```bash
  python tools/prepare_kits23.py --raw /path/to/kits23/dataset --out datasets/KiTS23/processed --workers 8
  ```

  This runs the five-step pipeline in `tools/process_kits23/`
  (resample → orient → laterality → crop → labels). It only transforms your
  local copy and does not redistribute KiTS23.

KiTS23 provides voxel segmentations but no report-derived attributes, so it
supports the **L1/L2** (side-level) evaluation only — not the L3 enhancement /
attenuation attributes.

If you use the KiTS23 data, cite it as directed by the organizers
(<https://github.com/neheller/kits23>):

```bibtex
@misc{heller2023kits21,
      title={The KiTS21 Challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase CT},
      author={Nicholas Heller and Fabian Isensee and Dasha Trofimova and Resha Tejpaul and Zhongchen Zhao and Huai Chen and Lisheng Wang and others},
      year={2023},
      eprint={2307.01984},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Privacy notice

Do not commit protected health information, radiology reports, patient
identifiers, private annotations, or institution-restricted data to this
repository.
