"""
PyTorch Dataset for kidney CT bilateral crops.

Loads NPZ files from final_dataset/, applies HU windowing,
resize (preserve aspect ratio) + pad to target size.
"""
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

FINAL_DATASET = Path(os.environ.get("KIDNEY_DATA_ROOT", "datasets/UF_Kidney_CT/final_dataset"))


class KidneyCTDataset(Dataset):
    """Bilateral kidney crop dataset."""

    def __init__(
        self,
        split: str,
        target_size: tuple[int, int, int],
        hu_min: float = -200.0,
        hu_max: float = 400.0,
        ct_level: str = "L2",
        augment: bool = False,
        mask_strategy: str = "none",
        mask_dropout_prob: float = 0.3,
        return_mask: bool = False,
        flip_x: bool = False,
        pad_only: float = 0,
        normalize_mode: str = "minmax",
        swap_lr: bool = False,
    ):
        self.split = split
        self.target_size = target_size
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.ct_level = ct_level
        self.augment = augment
        self.mask_strategy = mask_strategy      # none | binary | dropout | noisy
        self.mask_dropout_prob = mask_dropout_prob
        self.return_mask = return_mask          # for HMA: return raw 7-class mask
        self.flip_x = flip_x                   # flip x,y axes to match SuPreM convention
        self.pad_only = pad_only               # 0=resize+pad, 1.0=pad only, 1.5=downsample+pad
        self.normalize_mode = normalize_mode   # minmax=[0,1] (SuPreM/VoCo), div1000=[-1,1] (CTViT)
        self.swap_lr = swap_lr                 # diagnostic: swap L/R labels (image unchanged)

        data_dir = FINAL_DATASET / split
        self.npz_files = sorted(data_dir.glob("*.npz"))
        self.study_ids = [p.stem for p in self.npz_files]

        self.resized_image_dir = FINAL_DATASET.parent / f"resized_{split}_images"

        with open(FINAL_DATASET / "study_id_to_label.json") as f:
            self.sid_to_meta = json.load(f)

        with open(FINAL_DATASET / "labels.jsonl") as f:
            labels_list = [json.loads(l) for l in f]
        self.labels_by_key = {r["DEID_ORDER_KEY"]: r for r in labels_list}

        if self.augment:
            self._build_augmentation()

        logger.info(f"KidneyCTDataset({split}): {len(self)} samples, ct_level={ct_level}, "
                     f"target_size={target_size}, hu=[{hu_min},{hu_max}], augment={augment}")

    def _build_augmentation(self):
        """Build augmentation: elastic deformation + intensity transforms."""
        from monai.transforms import (
            Compose,
            RandGaussianNoised,
            RandGaussianSmoothd,
            RandAdjustContrastd,
            RandShiftIntensityd,
            RandScaleIntensityd,
        )

        # Intensity-only augmentation (no elastic — proven to hurt AUC)
        self.aug_transform = Compose([
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.02, prob=0.3),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.0),
                                sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0), prob=0.2),
            RandAdjustContrastd(keys=["image"], gamma=(0.8, 1.2), prob=0.2),
        ])

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        sid = self.study_ids[idx]

        if self.ct_level == "L1" and not self.return_mask:
            import nibabel as nib
            nii_path = self.resized_image_dir / f"{sid}.nii.gz"
            image = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
            mask = None
        else:
            data = np.load(str(npz_path))
            image = data["image"].astype(np.float32)
            mask = data["mask"]

        # Flip x and y axes to match SuPreM/AbdomenAtlas orientation convention
        if self.flip_x:
            image = np.flip(image, axis=(0, 1)).copy()
            if mask is not None:
                mask = np.flip(mask, axis=(0, 1)).copy()

        # HU windowing + normalize
        image = np.clip(image, self.hu_min, self.hu_max)
        if self.normalize_mode == "div1000":
            # CTViT: clip [-1000, 1000] then /1000 → [-1, 1]
            image = (image / 1000.0).astype(np.float32)
        else:
            # Default (minmax): clip then scale to [0, 1]
            image = (image - self.hu_min) / (self.hu_max - self.hu_min)

        # Resample if pad_only specifies target spacing
        # pad_only can be: 0 (resize+pad), scalar (isotropic), or "x,y,z" (anisotropic)
        resample_factors = self._get_resample_factors()
        if resample_factors is not None:
            image = zoom(image, resample_factors, order=1)
            if mask is not None:
                mask = zoom(mask.astype(np.float32), resample_factors, order=0).astype(mask.dtype)

        # Compute resize scale for size correction (before spatial transform)
        if isinstance(self.pad_only, str) or float(self.pad_only) > 0:
            resize_scale = 1.0  # pad_only: no resize, physical space preserved
        else:
            tx, ty, tz = self.target_size
            sx, sy, sz = image.shape
            resize_scale = min(tx / sx, ty / sy, tz / sz)

        # Spatial processing: resize+pad (old) or pad-only (new)
        image = self._spatial_transform(image)

        # Apply intensity augmentation (no elastic, no flip/rotation/scale)
        def _apply_aug(img):
            if self.augment:
                ct = torch.from_numpy(img[np.newaxis]).float()
                img = self.aug_transform({"image": ct})["image"].numpy()[0]
                img = np.clip(img, 0, 1)
            return img

        if self.ct_level == "D2":
            full_mask = mask.astype(np.float32)
            full_mask = self._spatial_transform(full_mask)
            image = _apply_aug(image)
            image = image[np.newaxis]

        elif self.ct_level in ("L3", "L3_mask"):
            if self.mask_strategy == "7class":
                # M6: full 7-class mask as second channel (0-6 normalized to 0-1)
                full_mask = mask.astype(np.float32)
                full_mask = self._spatial_transform(full_mask, order=0)  # nearest for integer labels
                full_mask = np.round(full_mask).clip(0, 6) / 6.0  # normalize to [0, 1]
                kidney_mask = full_mask
            elif self.mask_strategy == "7class_dropout":
                # M7: 7-class + dropout
                full_mask = mask.astype(np.float32)
                full_mask = self._spatial_transform(full_mask, order=0)  # nearest for integer labels
                full_mask = np.round(full_mask).clip(0, 6) / 6.0
                if self.augment and np.random.random() < self.mask_dropout_prob:
                    full_mask = np.zeros_like(full_mask)
                kidney_mask = full_mask
            else:
                kidney_mask = ((mask == 1) | (mask == 2)).astype(np.float32)
                kidney_mask = self._spatial_transform(kidney_mask)
                kidney_mask = self._apply_mask_strategy(kidney_mask)

            image = _apply_aug(image)
            image = np.stack([image, kidney_mask], axis=0)

        else:
            image = _apply_aug(image)
            image = image[np.newaxis]

        # Labels
        order_key = self.sid_to_meta[sid]["order_key"]
        label_record = self.labels_by_key[order_key]
        l1 = label_record["L1"]
        l2 = label_record["L2"]

        labels = {
            "left_abnormal": float(l1["left_abnormal"]),
            "right_abnormal": float(l1["right_abnormal"]),
            "left_has_cyst": float(l2["left_has_cyst"]),
            "right_has_cyst": float(l2["right_has_cyst"]),
            "left_has_solid": float(l2["left_has_solid"]),
            "right_has_solid": float(l2["right_has_solid"]),
            "left_max_size": float(l2["left_max_size_cm"]),
            "right_max_size": float(l2["right_max_size_cm"]),
        }

        # L3: per-lesion features (3 slots per side, padded)
        l3_left = self._parse_lesions(label_record["L3_left_lesions"])
        l3_right = self._parse_lesions(label_record["L3_right_lesions"])
        labels.update({
            "l3_left_exists": l3_left["exists"],
            "l3_left_cyst": l3_left["cyst"],
            "l3_left_cyst_valid": l3_left["cyst_valid"],
            "l3_left_solid": l3_left["solid"],
            "l3_left_solid_valid": l3_left["solid_valid"],
            "l3_left_size": l3_left["size"],
            "l3_left_size_valid": l3_left["size_valid"],
            "l3_left_enhancement": l3_left["enhancement"],
            "l3_left_enhancement_valid": l3_left["enhancement_valid"],
            "l3_left_attenuation": l3_left["attenuation"],
            "l3_left_attenuation_valid": l3_left["attenuation_valid"],
            "l3_left_count": l3_left["count"],
            "l3_right_exists": l3_right["exists"],
            "l3_right_cyst": l3_right["cyst"],
            "l3_right_cyst_valid": l3_right["cyst_valid"],
            "l3_right_solid": l3_right["solid"],
            "l3_right_solid_valid": l3_right["solid_valid"],
            "l3_right_size": l3_right["size"],
            "l3_right_size_valid": l3_right["size_valid"],
            "l3_right_enhancement": l3_right["enhancement"],
            "l3_right_enhancement_valid": l3_right["enhancement_valid"],
            "l3_right_attenuation": l3_right["attenuation"],
            "l3_right_attenuation_valid": l3_right["attenuation_valid"],
            "l3_right_count": l3_right["count"],
        })

        # Resize scale for size correction (1.0 if pad_only mode)
        labels["resize_scale"] = resize_scale

        # Diagnostic: swap L/R labels (image unchanged).
        # Purpose — test if R>L asymmetry comes from label alignment vs image features.
        if self.swap_lr:
            l1_pairs = [("left_abnormal", "right_abnormal")]
            l2_pairs = [
                ("left_has_cyst", "right_has_cyst"),
                ("left_has_solid", "right_has_solid"),
                ("left_max_size", "right_max_size"),
            ]
            l3_prefixes = ["exists", "cyst", "cyst_valid", "solid", "solid_valid",
                           "size", "size_valid",
                           "enhancement", "enhancement_valid",
                           "attenuation", "attenuation_valid", "count"]
            l3_pairs = [(f"l3_left_{k}", f"l3_right_{k}") for k in l3_prefixes]

            for a, b in l1_pairs + l2_pairs + l3_pairs:
                if a in labels and b in labels:
                    labels[a], labels[b] = labels[b], labels[a]

        labels = {k: torch.tensor(v, dtype=torch.float32) for k, v in labels.items()}

        # For D2: include full 7-class mask as separate tensor in labels
        if self.ct_level == "D2":
            labels["d2_mask"] = torch.from_numpy(full_mask[np.newaxis].copy()).float()

        # For HMA: include raw 7-class mask (resized+padded, integer labels)
        if self.return_mask and mask is not None:
            mask_resized = self._spatial_transform(mask.astype(np.float32))
            mask_resized = np.round(mask_resized).clip(0, 6)  # keep integer labels
            labels["d2_mask"] = torch.from_numpy(mask_resized[np.newaxis].copy()).float()

        return torch.from_numpy(image.copy()).float(), labels, sid

    def _apply_mask_strategy(self, kidney_mask: np.ndarray) -> np.ndarray:
        """Apply mask strategy during training.

        Strategies:
          none/binary: return mask as-is
          dropout: randomly zero out entire mask (prob=mask_dropout_prob)
          noisy: random morphological erosion/dilation to simulate seg quality variation
        """
        if not self.augment:
            return kidney_mask  # no augmentation at eval time

        if self.mask_strategy in ("none", "binary"):
            return kidney_mask

        if self.mask_strategy == "dropout":
            if np.random.random() < self.mask_dropout_prob:
                return np.zeros_like(kidney_mask)
            return kidney_mask

        if self.mask_strategy == "noisy":
            from scipy.ndimage import binary_erosion, binary_dilation
            binary_mask = kidney_mask > 0.5
            if np.random.random() < 0.5:
                iters = np.random.randint(1, 4)
                binary_mask = binary_erosion(binary_mask, iterations=iters)
            else:
                iters = np.random.randint(1, 4)
                binary_mask = binary_dilation(binary_mask, iterations=iters)
            return binary_mask.astype(np.float32)

        return kidney_mask

    MAX_LESIONS = 3  # max slots per side

    @staticmethod
    def _encode_bool(val) -> tuple[float, float]:
        """Encode bool feature → (value, valid_mask)."""
        s = str(val).lower().strip()
        if s in ("true", "1"):
            return 1.0, 1.0
        elif s in ("false", "0"):
            return 0.0, 1.0
        return 0.0, 0.0  # unknown

    @staticmethod
    def _encode_enhancement(val) -> tuple[float, float]:
        """Encode enhancement → (class_id, valid). 0=enhancement, 1=non-enhancement."""
        s = str(val).lower().strip()
        if s == "enhancement":
            return 0.0, 1.0
        elif s == "non-enhancement":
            return 1.0, 1.0
        return 0.0, 0.0

    @staticmethod
    def _encode_attenuation(val) -> tuple[float, float]:
        """Encode attenuation → (class_id, valid). 0=hypo, 1=hyper, 2=negative, 3=iso."""
        s = str(val).lower().strip()
        mapping = {"hypoattenuating": 0, "hyperattenuating": 1, "negative": 2, "isoattenuating": 3}
        if s in mapping:
            return float(mapping[s]), 1.0
        return 0.0, 0.0

    def _parse_lesions(self, lesion_list: list) -> dict:
        """Parse lesion list into padded tensors (max 3 slots)."""
        # Sort by size descending (unknown size = 0, goes last)
        sorted_lesions = sorted(lesion_list, key=lambda x: x.get("Size_cm", 0) or 0, reverse=True)
        sorted_lesions = sorted_lesions[:self.MAX_LESIONS]

        n = len(sorted_lesions)
        exists = [1.0] * n + [0.0] * (self.MAX_LESIONS - n)
        cyst, cyst_v = [], []
        solid, solid_v = [], []     # solid = mass OR tumor (merged at L3)
        size, size_v = [], []
        enh, enh_v = [], []
        att, att_v = [], []

        for lesion in sorted_lesions:
            c, cv = self._encode_bool(lesion.get("Cyst", "unknown"))
            cyst.append(c); cyst_v.append(cv)
            # Merge mass + tumor -> solid. Lenient valid: any determinable side counts.
            m, mv = self._encode_bool(lesion.get("Mass", "unknown"))
            t, tv = self._encode_bool(lesion.get("Tumor", "unknown"))
            s_val = 1.0 if (m > 0.5 or t > 0.5) else 0.0
            s_valid = 1.0 if (mv > 0.5 or tv > 0.5) else 0.0
            solid.append(s_val); solid_v.append(s_valid)

            sz = lesion.get("Size_cm", 0) or 0
            size.append(float(sz))
            size_v.append(1.0 if sz > 0 else 0.0)

            e, ev = self._encode_enhancement(lesion.get("Enhancement", "unknown"))
            enh.append(e); enh_v.append(ev)
            a, av = self._encode_attenuation(lesion.get("Attenuation", "unknown"))
            att.append(a); att_v.append(av)

        # Pad to MAX_LESIONS
        pad = self.MAX_LESIONS - n
        cyst += [0.0] * pad; cyst_v += [0.0] * pad
        solid += [0.0] * pad; solid_v += [0.0] * pad
        size += [0.0] * pad; size_v += [0.0] * pad
        enh += [0.0] * pad; enh_v += [0.0] * pad
        att += [0.0] * pad; att_v += [0.0] * pad

        return {
            "exists": exists, "count": float(n),
            "cyst": cyst, "cyst_valid": cyst_v,
            "solid": solid, "solid_valid": solid_v,
            "size": size, "size_valid": size_v,
            "enhancement": enh, "enhancement_valid": enh_v,
            "attenuation": att, "attenuation_valid": att_v,
        }

    def _get_resample_factors(self):
        """Compute zoom factors from pad_only setting. Returns None if no resample needed.

        pad_only=0: no resample (resize+pad mode)
        pad_only=1.0: no resample (pad only at original 1.0mm)
        pad_only=1.5: isotropic, factor=1.0/1.5 per axis
        pad_only="0.75,0.75,1.5": anisotropic, factor=1.0/spacing per axis
        """
        if isinstance(self.pad_only, str) and "," in str(self.pad_only):
            # Anisotropic: "0.75,0.75,1.5"
            target_spacing = [float(x) for x in str(self.pad_only).split(",")]
            return tuple(1.0 / s for s in target_spacing)
        val = float(self.pad_only)
        if val <= 0 or val == 1.0:
            return None
        return (1.0 / val, 1.0 / val, 1.0 / val)

    def _spatial_transform(self, volume: np.ndarray, order: int = 1) -> np.ndarray:
        """Dispatch to resize+pad or center_crop+pad based on pad_only setting."""
        if isinstance(self.pad_only, str) or float(self.pad_only) > 0:
            return self._center_crop_and_pad(volume)
        return self._resize_and_pad(volume, order=order)

    def _resize_and_pad(self, volume: np.ndarray, order: int = 1) -> np.ndarray:
        """Resize preserving aspect ratio, then zero-pad to target_size."""
        tx, ty, tz = self.target_size
        sx, sy, sz = volume.shape

        scale = min(tx / sx, ty / sy, tz / sz)
        new_shape = (
            int(round(sx * scale)),
            int(round(sy * scale)),
            int(round(sz * scale)),
        )

        if new_shape != volume.shape:
            zoom_factors = [n / s for n, s in zip(new_shape, volume.shape)]
            volume = zoom(volume, zoom_factors, order=order)

        pad_x = tx - volume.shape[0]
        pad_y = ty - volume.shape[1]
        pad_z = tz - volume.shape[2]

        pad_before = (pad_x // 2, pad_y // 2, pad_z // 2)
        pad_after = (pad_x - pad_before[0], pad_y - pad_before[1], pad_z - pad_before[2])

        volume = np.pad(
            volume,
            [(pad_before[i], pad_after[i]) for i in range(3)],
            mode="constant",
            constant_values=0,
        )
        return volume

    def _center_crop_and_pad(self, volume: np.ndarray) -> np.ndarray:
        """Center crop (if larger) + zero-pad (if smaller) to target_size. No resize."""
        tx, ty, tz = self.target_size

        # Center crop if any dim exceeds target
        for d, t in enumerate((tx, ty, tz)):
            if volume.shape[d] > t:
                start = (volume.shape[d] - t) // 2
                slices = [slice(None)] * 3
                slices[d] = slice(start, start + t)
                volume = volume[tuple(slices)]

        # Zero-pad if any dim is smaller than target
        pad_widths = []
        for d, t in enumerate((tx, ty, tz)):
            diff = t - volume.shape[d]
            if diff > 0:
                pad_widths.append((diff // 2, diff - diff // 2))
            else:
                pad_widths.append((0, 0))

        volume = np.pad(volume, pad_widths, mode="constant", constant_values=0)
        return volume
