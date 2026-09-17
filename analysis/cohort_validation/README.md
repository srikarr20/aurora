# AURORA Cohort Validation

Reproducible research analyses supporting the current AURORA evidence audit.

No clinical imaging datasets are included in this repository.

## Core evidence pipeline

1. canonical_cohort.py — build C/K/E cohort features.
2. canonical_unsupervised_structure.py — diagnosis-blind clustering and stability.
3. canonical_clinical_reveal.py — post-hoc ACDC phenotype comparison.
4. canonical_heldout_permutation.py — repeated held-out replication and permutation control.

## Reproduced evidence

Using the local 150-subject ACDC cohort, the portable pipeline reproduced:

- full-cohort Cramers V ≈ 0.3751
- chi-square p ≈ 0.000301
- median held-out Cramers V ≈ 0.4421
- permutation null mean ≈ 0.2825
- permutation null 95th percentile ≈ 0.3363
- permutation null 99th percentile ≈ 0.3616
- empirical permutation p ≈ 0.000999

## Scientific boundary

These are research-level detector-observability analyses.

They do not establish clinical diagnostic accuracy, validated disease classification, patient-level risk prediction, superiority to ejection fraction, or prospective external clinical validation.

See docs/scientific-integrity/CURRENT_EVIDENCE_STATUS.md
