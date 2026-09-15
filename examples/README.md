# AURORA Canonical Example Data

The canonical cardiac validation figures use three cine MRI cases from the
Automated Cardiac Diagnosis Challenge (ACDC) dataset.

The source clinical imaging data is not intended to be distributed in the
active AURORA repository. Obtain ACDC from its original source subject to the
dataset license and terms, then place the required 4-D NIfTI files as follows:

examples/
├── stable/
│   └── patient029_4d.nii.gz
├── irregular/
│   └── patient094_4d.nii.gz
└── low_contraction/
    └── patient008_4d.nii.gz

## Representative Cases

| AURORA descriptor | ACDC case | ACDC clinical group |
|---|---|---|
| Stable | patient029 | HCM |
| Irregular | patient094 | RV |
| Low Contraction | patient008 | DCM |

The AURORA descriptors are phenomenological dynamical labels. They are not
ACDC diagnostic labels and should not be interpreted as clinical diagnoses.

## Selection Provenance

These representative cases can be traced to an earlier exploratory AURORA
Motion Strength Index (MSI) pipeline:

- patient008: MSI 0.2752293578 -> LOW_CONTRACTION
- patient029: MSI 0.4049756304 -> NORMAL
- patient094: MSI 0.4772727273 -> HIGH_MOTION

The historical MSI thresholds were exploratory engineering heuristics and
were not clinically calibrated diagnostic cut-offs.

The later canonical presentation uses the descriptive terms:

- LOW_CONTRACTION -> Low Contraction
- NORMAL -> Stable
- HIGH_MOTION -> Irregular

This is a presentation mapping for representative dynamical regimes, not a
validated clinical classification mapping.

## Integrity Checks

Local preserved source copies used during the provenance audit had these
SHA-256 hashes:

- patient008_4d.nii.gz:
  e37a103c52e37cbebf27defc4156e0f4f45994efeeab16c0f197521244eefc52
- patient029_4d.nii.gz:
  6e1d7d7cb69e4879095355d14fab2ea600a901f4ddc294bb60f12174fdbdd83c
- patient094_4d.nii.gz:
  78095576196193ae7c40aabde1917e478c3a5a2b8b0ae69b9fa36ff7c0eb4534

These hashes identify the files used for the canonical example setup; they do
not replace the original dataset licensing or provenance requirements.

## Required ACDC Citation

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.,
"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures
Segmentation and Diagnosis: Is the Problem Solved?",
IEEE Transactions on Medical Imaging, vol. 37, no. 11,
pp. 2514-2525, Nov. 2018.

DOI: 10.1109/TMI.2018.2837502

See also:

docs/scientific-integrity/DATA_PROVENANCE.md
