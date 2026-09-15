# AURORA Reproducibility and Execution SOP

## Detector-Plane Dynamical Observability Framework

---

# 1. Repository Structure

Canonical repository structure:

```text
docs/
examples/
outputs/
src/
src_canonical/
archive_experimental/
```

---

# 2. Environment Setup

Recommended Python version:

```text
Python 3.9+
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

* numpy
* matplotlib
* opencv-python
* nibabel
* scipy

---

# 3. Canonical Detector Lineage

Canonical detector implementations:

```text
src_canonical/
├── aurora_v6_core.py
├── aurora_final_pipeline.py
├── v17_2_multidim_pipeline.py
├── v17_3_normalized_detector.py
├── v18_phase_aware_detector.py
├── v19_optical_flow_detector.py
└── v20_topology_detector.py
```

These represent the stabilized detector evolution hierarchy used for validation.

---

# 4. Validation Dataset and Provenance

Canonical cardiac examples originate from the Automated Cardiac Diagnosis
Challenge (ACDC) dataset.

Required ACDC citation:

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.,
"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures
Segmentation and Diagnosis: Is the Problem Solved?",
IEEE Transactions on Medical Imaging, vol. 37, no. 11,
pp. 2514-2525, Nov. 2018.

DOI: 10.1109/TMI.2018.2837502

The ACDC distribution used during development states that the dataset is
provided under CC BY-NC-SA 4.0 and subject to its supplied non-commercial
scientific-research terms.

The active AURORA repository does not require redistribution of the source
clinical imaging files. Obtain ACDC separately under its original terms.

For canonical figure reproduction, place the required files at:

```text
examples/
├── stable/
│   └── patient029_4d.nii.gz
├── irregular/
│   └── patient094_4d.nii.gz
└── low_contraction/
    └── patient008_4d.nii.gz
```

Representative cases:

| AURORA descriptor | ACDC case | ACDC group |
|---|---|---|
| Stable | patient029 | HCM |
| Irregular | patient094 | RV |
| Low Contraction | patient008 | DCM |

The AURORA descriptors are phenomenological dynamical labels. They are not
ACDC diagnostic labels and are not clinically validated disease classes.

The representative cases can be traced to an earlier exploratory AURORA
Motion Strength Index (MSI) workflow:

| ACDC case | Archived MSI | Exploratory label |
|---|---:|---|
| patient008 | 0.2752293578 | LOW_CONTRACTION |
| patient029 | 0.4049756304 | NORMAL |
| patient094 | 0.4772727273 | HIGH_MOTION |

The historical MSI thresholds were exploratory engineering heuristics and
were not clinically calibrated diagnostic cut-offs.

The later canonical terminology:

- LOW_CONTRACTION -> Low Contraction
- NORMAL -> Stable
- HIGH_MOTION -> Irregular

is a descriptive presentation mapping for representative dynamical behavior,
not a validated clinical classification mapping.

For complete provenance and evidence boundaries, see:

`docs/scientific-integrity/DATA_PROVENANCE.md`

---

# 5. Validation Workflow

The canonical validation workflow consists of:

1. Detector activity extraction
2. Observable construction
3. Observable normalization
4. State-space trajectory embedding
5. Differential manifold comparison
6. Phenomenological interpretation

---

# 6. Generate Validation Videos

Validation visualization videos are generated using:

```bash
python3 src/aurora_validation_visualizer.py
```

Generated outputs:

```text
outputs/validation_videos/
```

Generated videos include:

* stable validation trajectory
* irregular validation trajectory
* low-contraction validation trajectory

---

# 7. Generate Figure Suite

## Figure 1 — Differential Regime Comparison

Run:

```bash
python3 src/build_validation_figures.py
```

Output:

```text
outputs/figure_suite/figure_1_regime_comparison.png
```

---

## Figure 2 — Differential Manifold Trajectory Overlay

Run:

```bash
python3 src/build_overlay_figure.py
```

Output:

```text
outputs/figure_suite/figure_2_trajectory_overlay.png
```

---

# 8. Validation Interpretation Framework

Validation analysis focuses on:

* observable synchronization,
* manifold geometry,
* relational coupling structure,
* oscillatory structure,
* dynamical coupling,
* and differential regime morphology.

The framework does NOT rely on:

* segmentation,
* supervised learning,
* latent embeddings,
* or diagnostic classifiers.

---

# 9. Canonical Validation Regimes

Three primary dynamical regimes are used throughout validation:

| Regime          | Dynamical Signature            |
| --------------- | ------------------------------ |
| Stable          | Coherent damped manifold       |
| Irregular       | Fragmented angular geometry    |
| Low Contraction | Compressed low-energy manifold |

---

# 10. Canonical Outputs

Canonical outputs include:

```text
outputs/
├── figure_suite/
├── validation/
├── validation_videos/
└── plots/
```

---

# 11. Scientific Positioning

AURORA should be interpreted as:

# a detector-plane dynamical observability framework

rather than:

* a clinical diagnostic system,
* segmentation pipeline,
* or supervised machine learning model.

The primary object of analysis is the geometry of observable evolution in detector-conditioned state space.

---

# 12. Current Limitations

Current limitations include:

* exploratory validation scale,
* phenomenological rather than statistical validation,
* limited cohort size,
* absence of formal attractor proofs,
* and absence of clinical outcome validation.

The present work establishes observability structure rather than clinical efficacy.

