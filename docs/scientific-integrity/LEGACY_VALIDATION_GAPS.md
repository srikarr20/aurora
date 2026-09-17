# Legacy Validation Gaps

Classification: VALIDATION_BLOCKED -- MISSING_ORIGINAL_OPERATOR

## Original Source

Original local source path:

```text
/Users/rallabandisailesh/Desktop/SRIKAR'S STUFF/Medical Device's/AURORA_AMTZ_DEMO/aurora_instrument/aurora_project/run_validation.py
```

## Intended Flow

The legacy validation script intended to execute:

```python
cx, ct = optimize_operator(V)
rho = compute_rho(V, cx, ct)
```

The intended behavior was therefore a two-step flow:

1. Select detector parameters with `optimize_operator(V)`.
2. Generate the validation signal with `compute_rho(V, cx, ct)`.

## Missing Operator

No definition of `optimize_operator` was found in the available local source
tree or repository source tree during the cleanup audit. The only references
found were the import and call site inside the original legacy validation
script.

Because the parameter-selection method is missing, the original validation
workflow cannot currently be reproduced from the available source.

## `compute_rho` Versus `compute_rho_raw`

The original `compute_rho(V, cx, ct)` implementation computes a masked
frame-difference signal and normalizes the complete resulting sequence.

The available `compute_rho_raw(V, cx=70, ct=2)` implementation in the packaged
pipeline shares the same basic masked frame-difference calculation, but it also
trims boundary samples before normalization:

```python
if len(rho) > 2:
    rho = rho[1:-1]
```

It also uses default `cx` and `ct` values when no optimizer is supplied.

These differences affect:

- signal length
- normalization range
- `FVC`
- `Instability`

Therefore `compute_rho_raw` is not a scientifically neutral substitute for the
legacy `optimize_operator(V) -> compute_rho(V, cx, ct)` workflow.

## Decision

Executable validation is disabled in
`src_instrument/aurora_project/run_validation.py`.

The module is retained only for provenance. It raises
`MissingValidationOperatorError` if validation execution is requested. This
prevents silent substitution of a different detector algorithm and avoids
claiming that legacy validation results are reproducible.

## Recovery Requirements

Before this workflow can be re-enabled, one of the following must happen:

- Recover the original `optimize_operator` implementation and restore the
  original `optimize_operator(V) -> compute_rho(V, cx, ct)` behavior.
- Independently validate and document a replacement operator as a new
  scientific method, without presenting it as reproduction of the original
  legacy validation.

Until then, the legacy validation workflow remains blocked.
