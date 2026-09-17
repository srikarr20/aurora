# AURORA — Current Evidence Status

## Scope

This document records the current scientific evidence boundary for AURORA.

AURORA is presently supported as a detector-conditioned dynamical observability framework. It is not established as a clinically validated diagnostic or risk-prediction system.

## 1. Canonical C/K/E cohort result

A full-cohort analysis was performed on 150 ACDC cine MRI subjects using diagnosis-blind detector-derived trajectory geometry.

Five geometry descriptors were studied:

- compactness
- mean turning angle
- sharp-turn fraction
- radial variability
- mean absolute C/K/E observable synchronization

Acquisition effects from native temporal phase count, slice count, and spatial size were explicitly examined.

After acquisition residualization, a two-mode unsupervised representation showed the strongest overall combination of internal separation and resampling stability.

The two modes are described phenomenologically as:

- smoother / more synchronized detector trajectories
- more angular / less synchronized detector trajectories

These are measurement-space modes, not clinical states.

## 2. Post-hoc association with ACDC phenotypes

ACDC diagnostic labels were not used to create the detector modes.

After the unsupervised model was frozen, the two-mode partition showed a moderate association with independently supplied ACDC diagnostic groups.

Full-cohort exploratory association:

- N = 150
- Cramer's V ≈ 0.375
- chi-square p ≈ 0.00030

The association was not equivalent to healthy-versus-disease separation. HCM and NOR subjects were both enriched in the smoother / more synchronized mode, while DCM, MINF, and RV were more mixed.

## 3. Held-out internal replication

Repeated 70/30 train/test analyses were performed without using diagnosis during:

- acquisition residualization
- feature scaling
- clustering
- cluster identity alignment
- held-out subject assignment

Across 200 held-out repetitions:

- median held-out Cramer's V ≈ 0.442
- mean held-out Cramer's V ≈ 0.440
- 10th percentile ≈ 0.304
- 90th percentile ≈ 0.565
- 92% of repetitions had V >= 0.30

This is internal replication using repeated splits of the same 150-subject cohort, not external clinical validation.

## 4. Permutation validation

The held-out phenotype association was tested against randomized ACDC diagnostic labels.

Using 1,000 label permutations:

- observed median held-out Cramer's V = 0.4421
- null mean median V = 0.2825
- null 95th percentile = 0.3363
- null 99th percentile = 0.3616
- empirical p ≈ 0.000999
- observed/null ratio ≈ 1.565

The observed diagnosis association therefore exceeded the randomized-label null under the tested procedure.

## 5. Temporal-resolution limitation

The current canonical trajectory geometry is not temporally invariant.

Changing temporal sampling caused substantial changes in two-mode assignments.

Uniform image-level temporal resampling produced approximately:

- 75% temporal resolution: 70.7% same-mode retention, ARI ≈ 0.162
- 50% temporal resolution: 68.0% same-mode retention, ARI ≈ 0.119

Turning-angle and sharp-turn descriptors were particularly sensitive to temporal sampling.

This is an active limitation and must be disclosed.

## 6. Recovered historical temporal logic

Earlier AURORA code contains C/K/E cycle resampling logic:

- C/K/E peak-to-peak detector segments
- normalization to a common temporal coordinate
- resampling to N = 30 points

A cohort audit found:

- 148/150 subjects with >=2 detected C peaks
- 141/150 with >=1 valid historical peak-to-peak segment
- 96/150 with >=2 valid segments

However, same-patient segment waveforms were not more similar than between-patient segments in C/K/E correlation. Geometry distance showed only a modest same-patient advantage.

Therefore the historical V9.3 segmentation should not currently be interpreted as validated physiological cardiac-cycle segmentation.

## 7. Fixed-length C/K/E reconstruction

Whole-sequence C/K/E trajectories were resampled to fixed lengths N = 20, 30, and 50.

N = 30 best preserved the native representation:

- same-mode retention ≈ 80%
- ARI ≈ 0.342
- compactness correlation ≈ 0.989
- radial-variability correlation ≈ 0.990
- observable-synchronization correlation ≈ 0.993

However, turning-angle and sharp-turn descriptors remained dependent on native temporal phase count.

Fixed-length interpolation therefore did not eliminate the temporal-sampling limitation.

## 8. V18 phase-aware detector

The historical V18 detector was audited across the same 150 ACDC subjects.

Its robust-core features were:

- Deviation
- SpatialVar
- TemporalInstability
- ContractionBias

These showed relatively low acquisition dependence.

The historical PhaseConsistency feature showed strong dependence on native phase count and was therefore excluded from the robust-core clustering audit.

V18 robust-core k=2 showed weaker post-hoc association with ACDC diagnostic groups:

- Cramer's V ≈ 0.201
- p ≈ 0.195

Thus V18 did not reproduce the canonical C/K/E phenotype association.

## 9. Cross-detector observability

Canonical C/K/E and V18 two-mode partitions were compared directly.

They showed:

- best binary overlap ≈ 67.3%
- Adjusted Rand Index ≈ 0.114
- normalized mutual information ≈ 0.081
- Cramer's V between detector partitions ≈ 0.316

The two detectors therefore produced partially overlapping but non-equivalent organizations.

Repeated held-out analysis reproduced approximately one-third disagreement between the detector formulations.

## 10. Detector disagreement

Cross-detector disagreement was not explained by:

- native phase count
- slice count
- spatial voxel count
- ACDC diagnostic group

High-confidence disagreement persisted even when both detectors had substantial cluster margins.

Among subjects with canonical and V18 margins >= 0.20:

- 69 high-confidence subjects
- 18 remained detector-discordant
- disagreement rate ≈ 26%

High-confidence disagreement was characterized by:

- higher V18 TemporalInstability
- lower V18 ContractionBias
- moderately higher canonical turning angle

These are detector-space observations, not clinical phenotypes.

## 11. Signed-versus-absolute V18 ablation

Replacing signed temporal differences with absolute differences substantially altered the V18 observable space.

Signed and absolute V18 k=2 partitions showed approximately:

- best overlap ≈ 53.3%
- ARI ≈ -0.004

Removing sign did not make V18 more similar to canonical C/K/E:

- signed V18 vs canonical overlap ≈ 67.3%
- absolute V18 vs canonical overlap ≈ 54.0%

Therefore cross-detector divergence cannot be attributed solely to signed versus absolute temporal change.

Detector architecture as a whole matters.

## 12. DII/SII five-state branch

The historical DII/SII five-state branch was numerically reproduced from stored inputs through state/risk assignment.

However, its external DICOM input construction concatenated multiple spatial slice-level cine sequences into a single long vector.

Slice-boundary jumps materially increased DII in the external cases.

Therefore the historical five-state labels and risk categories should not currently be presented as clinically validated or externally validated physiological states.

## Current scientific positioning

The strongest currently supportable interpretation is:

AURORA is a detector-conditioned dynamical observability framework in which different detector operators expose partially overlapping but non-equivalent organizations of spatiotemporal cardiac imaging signals.

Current evidence supports:

- deterministic detector-derived observables
- reproducible cohort-level dynamical organization
- internal held-out replication
- permutation-controlled association with independently supplied cardiac phenotypes
- detector-conditioned differences across measurement operators

Current evidence does not establish:

- clinical diagnostic accuracy
- validated disease classification
- patient-level risk prediction
- superiority to ejection fraction
- temporally invariant trajectory geometry
- external prospective clinical validation

## Next scientific priorities

1. Controlled detector-component ablation.
2. Improve or replace temporally fragile angular trajectory descriptors.
3. Robustness to temporal, intensity, noise, and spatial perturbation.
4. Freeze the detector formulation before external validation.
5. Evaluate on an independently processed external cine MRI cohort.
