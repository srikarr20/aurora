"""Legacy validation entry point retained for provenance only.

The original validation script intended to execute the following workflow:

    cx, ct = optimize_operator(V)
    rho = compute_rho(V, cx, ct)

The required ``optimize_operator`` implementation is missing from the available
local and repository source history. Validation execution is disabled here to
avoid silently changing scientific results by substituting a different detector
or signal algorithm. This module is retained only for provenance until the
original method is recovered or a replacement is independently validated.
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


VALIDATION_BLOCKED_REASON = "VALIDATION_BLOCKED -- MISSING_ORIGINAL_OPERATOR"


class MissingValidationOperatorError(RuntimeError):
    """Raised when legacy validation is requested without optimize_operator."""


def _disabled_message(data_root=None, output_dir=None):
    details = [
        VALIDATION_BLOCKED_REASON,
        "Legacy validation requires the original optimize_operator(V) -> "
        "compute_rho(V, cx, ct) workflow.",
        "No optimize_operator definition is present in the available source "
        "history, so execution is disabled to avoid substituting a different "
        "scientific algorithm or changing validation metrics.",
    ]
    if data_root is not None:
        details.append(f"data_root={Path(data_root)}")
    if output_dir is not None:
        details.append(f"output_dir={Path(output_dir)}")
    return " ".join(details)


# -----------------------------
# EF FROM GT
# -----------------------------
def compute_ef(ed_gt_path, es_gt_path):
    ed = nib.load(ed_gt_path).get_fdata()
    es = nib.load(es_gt_path).get_fdata()

    ed_area = np.sum(ed == 3)
    es_area = np.sum(es == 3)

    return (ed_area - es_area) / (ed_area + 1e-8)


def run_validation(data_root, output_dir):
    raise MissingValidationOperatorError(_disabled_message(data_root, output_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    raise MissingValidationOperatorError(
        _disabled_message(args.data_root, args.output_dir)
    )


if __name__ == "__main__":
    main()
