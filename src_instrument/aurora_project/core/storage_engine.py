import json
import pandas as pd
import os


# -----------------------------
# SAVE METRICS TABLE
# -----------------------------
def save_metrics_table(metrics_list, patient_names, out_path):

    df = pd.DataFrame(metrics_list)
    df.insert(0, "patient", patient_names)

    df.to_csv(out_path, index=False)


# -----------------------------
# SAVE BASELINE
# -----------------------------
def save_baseline(baseline, out_path):

    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=4)


# -----------------------------
# LOAD BASELINE
# -----------------------------
def load_baseline(path):

    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# SAVE Z-SCORES
# -----------------------------
def save_zscores(zscores, label, out_path):

    data = {
        "zscores": zscores,
        "classification": label
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
