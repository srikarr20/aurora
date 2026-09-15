# AURORA Methodological References and Terminology Boundaries

## 1. Purpose

This document records external methods used by the canonical AURORA detector
lineage and distinguishes them from AURORA-defined exploratory metrics.

Only methods actually present in the implementation are attributed to external
methodological literature.

---

## 2. Dense Optical Flow

The canonical V19 and V20 detector implementations use OpenCV dense Farneback
optical flow through:

`cv2.calcOpticalFlowFarneback`

Relevant methodological reference:

G. Farneback,
"Two-Frame Motion Estimation Based on Polynomial Expansion,"
in Scandinavian Conference on Image Analysis (SCIA),
Lecture Notes in Computer Science, vol. 2749,
pp. 363-370, 2003.

DOI: 10.1007/3-540-45103-X_50

AURORA uses the resulting dense flow field as an intermediate representation
from which regional motion magnitude, directional consistency, coupling, and
divergence-related observables are derived.

---

## 3. V18 Historical "Phase-Aware" Terminology

The file:

`src_canonical/v18_phase_aware_detector.py`

has historically been described as a phase-aware detector.

However, the implemented quantity does not extract or compare the complex phase
angle of the Fourier transform.

The implementation computes:

- an FFT of a global signal,
- the magnitude spectrum using `abs(fft)`,
- the dominant spectral magnitude,
- and a dominant-to-total spectral ratio.

Accordingly, the implemented quantity is more precisely interpreted as a
spectral concentration or dominant-frequency-strength metric.

The historical filename is retained for lineage and reproducibility, but the
implementation should not be presented as a formal phase-synchronization,
Hilbert-phase, or phase-locking method.

No external phase-synchronization citation is therefore claimed for this
implementation.

---

## 4. V20 Historical "Topology-Aware" Terminology

The file:

`src_canonical/v20_topology_detector.py`

has historically been described as topology-aware.

The implementation does not compute established topological-data-analysis
objects such as:

- persistent homology,
- persistence diagrams,
- Betti numbers,
- simplicial complexes,
- or homology groups.

Instead, its additional observables include:

- inter-regional correlation of optical-flow magnitude, represented as
  `Coupling`,
- and spatial optical-flow divergence variance.

The implementation is therefore more precisely described as
relational-coupling and flow-structure observability.

The historical filename is retained for lineage and reproducibility.

No persistent-homology or topological-data-analysis citation is claimed for
the current V20 implementation.

---

## 5. Clustering

Several detector generations use:

- `sklearn.preprocessing.StandardScaler`
- `sklearn.cluster.KMeans`

with three exploratory clusters.

These cluster identifiers are unsupervised numerical groupings and should not
be interpreted as clinical disease labels.

If software attribution is required, cite scikit-learn:

F. Pedregosa et al.,
"Scikit-learn: Machine Learning in Python,"
Journal of Machine Learning Research,
12, pp. 2825-2830, 2011.

The clustering stages are exploratory analysis components rather than evidence
of supervised diagnostic classification.

---

## 6. Canonical C-K-E Observable Framework

The canonical figure-generation path computes temporal detector differences and
AURORA-defined observables:

- C: spatial dispersion relative to mean detector activity,
- K: temporal change in detector-field structure,
- E: total detector activity.

These quantities are AURORA-defined operational observables in the current
implementation.

They should not be presented as standardized clinical biomarkers or as
established quantities imported from another medical-imaging framework.

---

## 7. Terminology Rule

For current scientific communication:

Historical name:
`Phase-Aware Detector`

Preferred implementation description:
`spectral-concentration detector extension`

Historical name:
`Topology-Aware Detector`

Preferred implementation description:
`relational-coupling / flow-structure detector extension`

Historical filenames may remain unchanged for provenance, but papers, README
text, outreach, and validation claims should describe what the code actually
computes.

---

## 8. Evidence Boundary

External methodological attribution currently required:

- ACDC dataset provenance and citation
- Farneback dense optical flow

Useful software attribution:

- scikit-learn
- OpenCV
- NiBabel
- SciPy

AURORA-defined exploratory quantities do not acquire scientific validity merely
through external citations. Their validity must come from explicit definition,
reproducibility, and independent validation.
