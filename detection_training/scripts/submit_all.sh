#!/bin/bash
# Submit detection-training experiments to SLURM (one job per config x seed).
#
# This is a TEMPLATE. Edit the placeholders marked <...> for your cluster
# (account, partition, environment activation), then run:
#
#   bash detection_training/scripts/submit_all.sh
#
# Each job trains one config for one seed. On a single modern GPU, 50 epochs
# takes roughly 8-11 hours depending on input representation.

set -euo pipefail

# Repo root = two levels up from this script (…/detection_training/scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="${ROOT}/detection_training/configs"
SLURM_OUT="${ROOT}/out_slurm"
mkdir -p "$SLURM_OUT"

# --- Cluster-specific placeholders --------------------------------------
ACCOUNT="<YOUR_SLURM_ACCOUNT>"
PARTITION="<YOUR_GPU_PARTITION>"
# Shell commands to activate your Python environment inside the job, e.g.
#   ENV_ACTIVATE='conda activate kidney3dct'
ENV_ACTIVATE='<ACTIVATE_YOUR_ENV>'
# ------------------------------------------------------------------------

# Configs to run. Defaults to the input/encoder/hier families used in the
# paper. Override by passing config base-names as arguments:
#   bash submit_all.sh E0g_l1l2_crop E0h_focal_crop
if [ "$#" -gt 0 ]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=(
        E0g_l1l2_crop      # LesionDETR-style head, L1+L2 hierarchical supervision
        E0g_l1l2_whole
        E0g_l1l2_kmask
        E0g_l1l2_fmask
        E0g_l1l2_kmask_voco
        E0g_l1l2_kmask_from_scratch
        E0h_focal_crop     # count-conditioned head with focal loss
        E0h_focal_kmask
    )
fi
SEEDS=(42 43 44)

for exp in "${EXPERIMENTS[@]}"; do
    config_path="${CONFIG_DIR}/${exp}.yaml"
    if [ ! -f "$config_path" ]; then
        echo "WARNING: config not found, skipping: $config_path" >&2
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        name="det_${exp}_s${seed}"
        echo "Submitting $name"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50gb
#SBATCH --time=12:00:00
#SBATCH --output=${SLURM_OUT}/${name}_%j.out
#SBATCH --error=${SLURM_OUT}/${name}_%j.err

${ENV_ACTIVATE}
export TQDM_DISABLE=1

echo "Job ID: \$SLURM_JOB_ID | Node: \$SLURMD_NODENAME"
nvidia-smi || true

cd ${ROOT}
python detection_training/train_detection.py \\
    --config ${config_path} \\
    --seeds ${seed} \\
    --exp_name ${exp}_s${seed}

EXIT_CODE=\$?
echo "Exit code: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
    done
done

echo "Submitted ${#EXPERIMENTS[@]} configs x ${#SEEDS[@]} seeds."
