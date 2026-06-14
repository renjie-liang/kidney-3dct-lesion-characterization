"""
Stratified order-level sampling for D7 few-shot experiments.

Samples by ORDER_KEY (not study_id) to avoid data leakage,
preserves abnormal/normal ratio, fixed random seed.
"""
import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FINAL_DATASET = Path(os.environ.get("KIDNEY_DATA_ROOT", "datasets/UF_Kidney_CT/final_dataset"))


def get_fewshot_study_ids(fraction: float, seed: int = 42) -> list[str]:
    """Sample a fraction of training study IDs with stratified order-level sampling.

    Args:
        fraction: 0.05, 0.10, 0.25, 0.50, or 1.0
        seed: random seed for reproducibility

    Returns:
        List of study_ids to use for training
    """
    if fraction >= 1.0:
        return None  # use all

    # Load mappings
    with open(FINAL_DATASET / "study_id_to_label.json") as f:
        mapping = json.load(f)

    with open(FINAL_DATASET / "labels.jsonl") as f:
        labels_by_key = {r["DEID_ORDER_KEY"]: r for r in (json.loads(l) for l in f)}

    # Get all train study_ids grouped by order_key
    order_to_sids = {}
    order_to_label = {}
    for sid, meta in mapping.items():
        if meta["split"] != "train":
            continue
        ok = meta["order_key"]
        if ok not in order_to_sids:
            order_to_sids[ok] = []
        order_to_sids[ok].append(sid)

        if ok not in order_to_label and ok in labels_by_key:
            # abnormal = left OR right abnormal
            l1 = labels_by_key[ok]["L1"]
            order_to_label[ok] = int(l1["abnormality"])

    # Split orders into abnormal vs normal
    abnormal_orders = [ok for ok, lbl in order_to_label.items() if lbl == 1]
    normal_orders = [ok for ok, lbl in order_to_label.items() if lbl == 0]

    rng = np.random.RandomState(seed)
    rng.shuffle(abnormal_orders)
    rng.shuffle(normal_orders)

    # Sample same fraction from each stratum
    n_abnormal = max(1, int(len(abnormal_orders) * fraction))
    n_normal = max(1, int(len(normal_orders) * fraction))

    selected_orders = abnormal_orders[:n_abnormal] + normal_orders[:n_normal]

    # Collect all study_ids for selected orders
    selected_sids = []
    for ok in selected_orders:
        selected_sids.extend(order_to_sids[ok])

    selected_sids.sort()

    logger.info(
        f"Few-shot sampling: fraction={fraction}, seed={seed}, "
        f"orders={len(selected_orders)} ({n_abnormal} abnormal + {n_normal} normal), "
        f"study_ids={len(selected_sids)}"
    )

    return selected_sids
