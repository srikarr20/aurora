# AURORA

## Detector-Plane Dynamical Observability Framework for Spatiotemporal Signal Fields

AURORA is a detector-plane dynamical observability framework that transforms spatiotemporal signal evolution into low-dimensional observable manifolds.

Rather than relying on segmentation, supervised learning, or diagnostic classification, AURORA analyzes the geometry of detector-conditioned observable evolution.

The framework was validated using cardiac cine MRI as a dynamical signal field testbed.

---

# Core Concept

AURORA treats observable evolution as the primary object of analysis.

The framework transforms:

V(x,y,z,t)

into detector-conditioned observable trajectories:

X(t)=[C(t),K(t),E(t)]

where:
- C represents coherence-related structure,
- K represents instability-sensitive dynamics,
- E represents detector energy/activity.

Distinct dynamical regimes produce distinct manifold geometries in observable state space.


---

# Canonical Validation Findings

Validation experiments demonstrated:

- coherent low-dimensional manifold emergence,
- observable synchronization,
- topology-sensitive dynamics,
- bounded instability,
- and differential regime geometry.

Three primary dynamical regimes emerged:

| Regime | Dynamical Signature |
|---|---|
| Stable | Coherent damped manifold |
| Irregular | Fragmented angular geometry |
| Low Contraction | Compressed low-energy manifold |


---

# Figure 1 — Differential Regime Comparison

![Figure 1](docs/figures/figure_1_regime_comparison.png)

Top row:
Detector accumulation topology.

Middle row:
Observable evolution for:
- coherence (C)
- instability sensitivity (K)
- detector energy (E)

Bottom row:
Low-dimensional observable trajectories embedded in detector state space.

Distinct dynamical regimes produce distinct manifold morphologies.


---

# Figure 2 — Differential Manifold Trajectory Overlay

![Figure 2](docs/figures/figure_2_trajectory_overlay.png)

Overlay of detector-conditioned observable trajectories across:
- stable
- irregular
- low-contraction regimes

The trajectories occupy distinct geometric regions while remaining bounded and dynamically structured.

---

# Detector Evolution Lineage

Canonical detector hierarchy:

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

Detector evolution progression:

| Generation | Primary Contribution |
|---|---|
| Accumulation Detector | Motion persistence |
| Regional Detector | Spatial observability |
| Normalized Detector | Invariant manifold geometry |
| Phase-Aware Detector | Temporal directional structure |
| Optical Flow Detector | Vector dynamics |
| Topology-Aware Detector | Relational coupling |

---

# Repository Structure

```text
docs/
examples/
outputs/
src/
src_canonical/
archive_experimental/
```

---

# Validation Workflow

Canonical workflow:

1. Detector activity extraction
2. Observable construction
3. Observable normalization
4. State-space trajectory embedding
5. Differential manifold comparison
6. Phenomenological interpretation

The framework does NOT rely on:

- segmentation
- supervised learning
- latent embeddings
- diagnostic classifiers

---

# Generate Validation Videos

```bash
python3 src/aurora_validation_visualizer.py
```

Outputs:

```text
outputs/validation_videos/
```

---

# Generate Figure Suite

## Figure 1

```bash
python3 src/build_validation_figures.py
```

## Figure 2

```bash
python3 src/build_overlay_figure.py
```

Outputs:

```text
outputs/figure_suite/
```

---

# Semiconductor Translation

AURORA generalizes naturally beyond cardiac imaging.

Equivalent mappings include:

| Cardiac Dynamics | Semiconductor Dynamics |
|---|---|
| Coherent manifold | Stable process attractor |
| Fragmented geometry | Plasma instability |
| Compressed manifold | Under-driven process |
| Topology drift | Chamber conditioning drift |
| Observable desynchronization | Coupling breakdown |

Potential applications include:

- plasma etch monitoring
- ALD/CVD observability
- CMP topology drift detection
- industrial process excursion analysis

---

# Scientific Positioning

AURORA should be interpreted as:

## a detector-plane dynamical observability framework

rather than:

- a clinical diagnostic system
- segmentation pipeline
- supervised machine learning model

The primary object of analysis is the geometry of observable evolution in detector-conditioned state space.

---

# Reproducibility

See:

```text
docs/reproducibility.md
```

and:

```text
docs/validation_report.md
```

So the final section becomes:

docs/validation_report.md

for:
- methodology
- validation workflow
- canonical outputs
- execution SOP

---

# Current Limitations

Current limitations include:

- exploratory validation scale
- phenomenological rather than statistical validation
- limited cohort size
- absence of formal attractor proofs
- absence of clinical outcome validation

The present work establishes observability structure rather than clinical efficacy.
