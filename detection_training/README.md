# Detection Training (per-lesion head comparison)

Self-contained training stack for the per-lesion detection-head experiments
(the head-architecture comparison reported in the paper). It has its own
`data/`, `models/`, `losses/`, and `analysis/` subpackages and inserts its own
directory on `sys.path`, so it runs independently of the top-level package.

## Quick start

### Submit experiments

Edit the cluster placeholders in `scripts/submit_all.sh` (account, partition,
environment activation) first, then:

```bash
# from the repository root
bash detection_training/scripts/submit_all.sh
```

Each job:
- 1 GPU
- ~8–11 hours per config-seed (50 epochs)
- Output: `experiments/detection_training/{exp_name}/seed{S}/`

### Monitor

```bash
squeue -u "$USER" | grep det_
```

### Aggregate once jobs finish

```bash
python detection_training/scripts/aggregate_results.py
```

Outputs:
- `experiments/detection_training/results_summary.json` — raw per-seed numbers
- `experiments/detection_training/results_summary.md` — markdown table

## Directory structure

```
detection_training/
├── README.md                # this file
├── train_detection.py       # training entry point (self-contained)
├── eval_kits23_detection.py # external validation on KiTS23
├── losses/
│   ├── focal.py             # focal classification loss
│   ├── count_reg.py         # count-regression term (count-conditioned head)
│   ├── hungarian.py         # DETR-style size-distance Hungarian matching
│   ├── joint_conf.py        # joint existence x type confidence
│   ├── label_smooth.py
│   ├── size_weight.py
│   ├── hard_neg.py
│   └── combined.py          # assembles the loss recipe from each config
├── models/                  # encoder + per-lesion heads
├── data/                    # dataset loader (matches top-level data/)
├── analysis/                # AP metric + random baselines
├── configs/
│   ├── base.yaml                       # reference defaults
│   ├── E0g_l1l2_{crop,whole,kmask,fmask}.yaml   # input representation sweep
│   ├── E0g_l1l2_kmask_{voco,from_scratch}.yaml  # encoder comparison
│   ├── E0g_{l1l2,focal_l1l2,nohier_crop}.yaml   # hierarchical-supervision sweep
│   ├── E0g_{count,focal,focal_count,dropout,B_hungarian}.yaml
│   └── E0h_*.yaml                       # count-conditioned head counterparts
└── scripts/
    ├── submit_all.sh        # SLURM submission template
    └── aggregate_results.py # compile per-seed summary table
```

## Config naming

Configs are grouped by head family and ablation axis:

- `E0g_*` — LesionDETR-style head (learnable queries + Hungarian matching).
- `E0h_*` — count-conditioned head (predict count, then fixed slot embeddings).
- Suffixes encode the variable being ablated: input representation
  (`crop` / `whole` / `kmask` / `fmask`), encoder
  (`voco` / `from_scratch`, default is SuPreM), hierarchical supervision
  (`l1l2` / `nohier`), and loss recipe (`focal`, `count`, `dropout`, ...).

These map to the input-representation, encoder, and hierarchical-supervision
results reported in the paper.

## Changing a single experiment

Each YAML is self-contained. To test a new hyperparameter:

```bash
cp configs/E0g_l1l2_crop.yaml configs/E0g_l1l2_crop_g1.5.yaml
# edit focal_gamma (or any field) in the copy
python detection_training/train_detection.py \
    --config detection_training/configs/E0g_l1l2_crop_g1.5.yaml --seeds 42
```

## Aggregating results

After runs finish, `scripts/aggregate_results.py` compiles per-seed metrics
(side-level AUC and per-class AP) across all runs found under
`experiments/detection_training/` into a single summary table.
