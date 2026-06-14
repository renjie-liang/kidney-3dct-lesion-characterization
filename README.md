# Multi-Granularity 3D Kidney Lesion Characterization from CT Volumes

Code for the paper *Multi-Granularity 3D Kidney Lesion Characterization from CT
Volumes*. **LesionDETR** is a DETR-style head with size-distance Hungarian
matching. From one 3D kidney CT volume it predicts a variable number of lesions
per kidney, each with four clinical attributes, and a hierarchical loss also
produces side-level outputs: **L1** abnormality, **L2** type and size, and
**L3** per-lesion attributes.

![Overview of the lesion-centric characterization framework](assets/overview.png)

Paper results are in [`RESULTS.md`](RESULTS.md). They require the private UF
Health data and cannot be reproduced from the public data here.

## Data

The UF Health dataset cannot be released for patient privacy. The code instead
runs on two small public datasets on
[Hugging Face](https://huggingface.co/datasets/LiangRenjie/kidney-3dct-lesion-characterization):

| Dataset | Purpose | Granularity | License |
|---|---|---|---|
| Synthetic | full pipeline including per-lesion L3 | L1/L2/L3 | MIT |
| KiTS23 subset, 6 real cases | external-validation path | L1/L2 | CC BY-NC-SA 4.0 |

No PHI, reports, or private annotations are in this repo. See [`DATA.md`](DATA.md).

## Install & quick start

```bash
pip install -r requirements.txt                          # Python >= 3.10
python tools/generate_synthetic_data.py --out datasets   # tiny fake dataset
python train.py --config-name smoke                      # train LesionDETR end-to-end
```

The smoke test runs the whole pipeline with a from-scratch encoder, on GPU or
slowly on CPU. It verifies the code path, not paper numbers.

## Paths & config

Data and weight locations come from environment variables, with runnable defaults:

| Variable | Meaning | Default |
|---|---|---|
| `KIDNEY_DATA_ROOT` | UF-format dataset | `datasets/UF_Kidney_CT/final_dataset` |
| `KITS23_ROOT` | processed KiTS23 | `datasets/KiTS23/processed` |
| `WEIGHTS_ROOT` | pretrained encoder weights | `weights` |

Pretrained encoders go under `WEIGHTS_ROOT`:
[SuPreM](https://github.com/MrGiovanni/SuPreM),
[VoCo](https://github.com/Luffy03/VoCo),
[CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP). SuPreM is the default;
`encoder=from_scratch` needs no weights.

## Layout

```
train.py  eval_kits23.py  eval_l3_full.py     # train + evaluate
models/   data/   analysis/                   # encoders + LesionDETR, datasets, metrics
detection_training/                           # per-lesion head comparison, Table 3
configs/  tools/                              # configs; synthetic + KiTS23 data tools
```

To reproduce the paper with the real dataset, run `train.py` with the `encoder`,
`ct_level`, `label_level`, and `mask_strategy` overrides; see `configs/` and
`detection_training/`. For KiTS23 external validation:
`python tools/prepare_kits23.py --raw <kits23> && python eval_kits23.py --checkpoint <best_model.pt>`.

## Citation & license

Code is under the [MIT License](LICENSE). The KiTS23 data is CC BY-NC-SA 4.0; see
[`DATA.md`](DATA.md). Please cite the paper:

```bibtex
@article{liang2026kidney3dct,
  title  = {Multi-Granularity 3D Kidney Lesion Characterization from CT Volumes},
  author = {Liang, Renjie and Fan, Zhengkang and Pan, Jinqian and Sun, Chenkun
            and Bian, Jiang and Terry, Russell and Xu, Jie},
  year   = {2026}, note = {Manuscript under review}
}
```

Contact: Renjie Liang — liang.renjie@ufl.edu
