# Multi-Granularity 3D Kidney Lesion Characterization from CT Volumes

This repository will host code and documentation for multi-granularity 3D kidney lesion characterization from CT volumes.

The project formulates kidney CT characterization as a lesion-centric prediction task. Given a 3D kidney CT volume, the framework predicts structured outputs at multiple granularities, including side-level abnormality, side-level lesion type and size, and per-lesion attributes.

## Repository Status

This repository is being prepared for public release. Source code, configuration files, and reproducible evaluation utilities will be added after final manuscript submission cleanup.

## Data Availability

The UF Health dataset used in the manuscript is not publicly available due to patient privacy restrictions. The KiTS23 dataset is publicly available at <https://kits-challenge.org/kits23/>.

No private patient data, radiology reports, protected health information, or private annotations are included in this repository.

## Planned Contents

- Training and evaluation code for lesion-centric kidney CT characterization
- Model configuration files for 3D encoder and lesion prediction heads
- Scripts for KiTS23 preprocessing and external validation
- Evaluation utilities for side-level metrics and per-lesion average precision
- Documentation for reproducing experiments where data access permits

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{liang2026kidney3dct,
  title = {Multi-Granularity 3D Kidney Lesion Characterization from CT Volumes},
  author = {Liang, Renjie and Fan, Zhengkang and Pan, Jinqian and Sun, Chenkun and Bian, Jiang and Terry, Russell and Xu, Jie},
  year = {2026},
  note = {Manuscript under review}
}
```

## Contact

For questions, please contact Jie Xu at xujie@ufl.edu.
