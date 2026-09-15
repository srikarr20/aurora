# AURORA Data Provenance and Validation Boundaries

## 1. Canonical Cardiac Imaging Dataset

The canonical cardiac cine MRI examples used by AURORA originate from the
Automated Cardiac Diagnosis Challenge (ACDC) dataset.

Required dataset citation:

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.,
"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures
Segmentation and Diagnosis: Is the Problem Solved?",
IEEE Transactions on Medical Imaging,
vol. 37, no. 11, pp. 2514-2525, Nov. 2018.

DOI: 10.1109/TMI.2018.2837502

The local ACDC distribution used during AURORA development states that the
dataset is provided under the CC BY-NC-SA 4.0 license and is restricted by
the supplied dataset terms to non-commercial scientific research use.

AURORA does not claim ownership of the ACDC data.

Source clinical imaging data should be obtained from the original ACDC
distribution subject to its licensing and access terms rather than being
redistributed as part of AURORA.

---

## 2. Canonical AURORA Analysis Path

The canonical detector-plane validation path operates directly on 4-D cine
MRI signal fields.

The canonical figure-generation workflow is:

4-D cine MRI

→ temporal detector field ΔV

→ detector-conditioned observables C(t), K(t), E(t)

→ observable normalization

→ low-dimensional state-space trajectory

→ differential manifold visualization

The canonical analysis path does not consume anatomical segmentation masks,
supervised disease labels, or trained diagnostic classifiers.

This segmentation-free statement applies specifically to the canonical
AURORA detector/observable analysis.

Archived experimental branches may contain segmentation-derived reference
measurements, EF experiments, machine-learning experiments, and other
non-canonical methods. Those experiments should not be interpreted as part
of the canonical segmentation-free computation.

---

## 3. Canonical Representative Cases

The canonical validation figures use three ACDC cine MRI sequences:

| Canonical regime | ACDC case |
|---|---|
| Stable | patient029 |
| Irregular | patient094 |
| Low Contraction | patient008 |

Their ACDC clinical groups are:

| ACDC case | ACDC group | AURORA canonical descriptor |
|---|---|---|
| patient029 | HCM | Stable |
| patient094 | RV | Irregular |
| patient008 | DCM | Low Contraction |

The AURORA descriptors are not ACDC diagnostic labels.

In particular:

- "Stable" does not mean clinically healthy.
- "Irregular" is not a cardiac diagnosis.
- "Low Contraction" is a phenomenological motion descriptor, not a disease class.

The canonical comparison is intended to demonstrate differences in
detector-conditioned dynamical geometry rather than diagnostic
classification performance.

---

## 4. Representative-Case Selection Provenance

The three canonical examples can be traced to an earlier exploratory AURORA
validation pipeline.

That pipeline calculated an experimental Motion Strength Index (MSI) and
applied the following heuristic categories:

- MSI > 0.45: HIGH_MOTION
- 0.30 < MSI <= 0.45: NORMAL
- MSI <= 0.30: LOW_CONTRACTION

For the three later canonical examples, the archived results were:

| ACDC case | Archived MSI | Exploratory V2 label |
|---|---:|---|
| patient008 | 0.2752293578 | LOW_CONTRACTION |
| patient029 | 0.4049756304 | NORMAL |
| patient094 | 0.4772727273 | HIGH_MOTION |

These cases were subsequently used as representative examples in the
canonical detector-plane visualization:

- LOW_CONTRACTION → Low Contraction
- NORMAL → Stable
- HIGH_MOTION → Irregular

The later descriptive terminology should not be interpreted as a formally
validated one-to-one clinical classification mapping.

The MSI thresholds were exploratory engineering heuristics. No recovered
AURORA documentation establishes them as clinically calibrated or validated
diagnostic thresholds.

---

## 5. Validation Scope

The canonical AURORA validation asks whether different spatiotemporal signal
behaviors produce distinguishable detector-conditioned observable geometry.

It does not establish:

- diagnostic sensitivity or specificity,
- disease classification accuracy,
- equivalence to clinical cardiac measurements,
- superiority to segmentation-based analysis,
- patient-outcome prediction,
- or clinical decision support performance.

The current evidence supports phenomenological and dynamical observability
claims only.

---

## 6. Legacy EF Validation

A separate archived EF-validation workflow should be treated as historical
experimental provenance rather than canonical evidence.

Two reproducibility/design issues have been identified.

First, the archived validation script imports an `optimize_operator(V)`
implementation that is not available in the corresponding packaged source
tree with the exact interface expected by that script.

Second, the legacy script hard-coded:

- ED = frame01
- ES = frame12

instead of reading the patient-specific ED and ES values from ACDC `Info.cfg`.

An audit of the 100 ACDC training cases found:

- 100 total patients,
- 1 patient with ED != 1,
- 80 patients with ES != 12,
- all 100 patients had their correct patient-specific ED/ES GT files,
- only 20 patients contained both the hard-coded frame01 and frame12 GT files.

Accordingly, legacy EF correlation outputs should not be cited as validated
evidence for the canonical AURORA framework.

---

## 7. Separation of Evidence

For interpretation of this repository, maintain the following separation:

DATA SOURCE
→ ACDC cardiac cine MRI

CANONICAL METHOD
→ segmentation-free detector-plane dynamical observability

REPRESENTATIVE-CASE SELECTION
→ exploratory AURORA MSI heuristic

CANONICAL EVIDENCE
→ observable evolution and manifold morphology

ARCHIVED EXPERIMENTS
→ exploratory EF, classification, ML, risk, and other historical branches

These evidence classes should not be merged into a single clinical-validation
claim.
