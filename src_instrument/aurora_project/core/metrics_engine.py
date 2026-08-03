import numpy as np


# -----------------------------
# EF (SHAPE-BASED, NOT AMPLITUDE)
# -----------------------------
def compute_ef(x):

    x = np.array(x)

    # normalize safely
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    ED = np.max(x)
    ES = np.min(x)

    EF = ED - ES  # shape-based contraction

    return float(EF)


# -----------------------------
# INSTABILITY
# -----------------------------
def compute_instability_index(instability):
    return float(np.mean(instability))


# -----------------------------
# RISK SCORE (ROBUST)
# -----------------------------
def compute_risk_score(R_total):

    flat = R_total.flatten()

    # top 10% (less aggressive than 5%)
    threshold = np.percentile(flat, 90)
    top_values = flat[flat >= threshold]

    return float(np.mean(top_values))


# -----------------------------
# LOCALIZATION (SPREAD-AWARE)
# -----------------------------
def compute_localization_index(R_total):

    flat = R_total.flatten()

    # normalize
    flat = flat / (np.max(flat) + 1e-8)

    # entropy-like measure
    p = flat / (np.sum(flat) + 1e-8)

    entropy = -np.sum(p * np.log(p + 1e-8))

    # normalize entropy → invert
    loc = 1 - entropy / np.log(len(p))

    return float(loc)


# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify_patient(EF, inst, risk, loc):

    if EF > 0.6:
        ef_state = "NORMAL"
    elif EF > 0.4:
        ef_state = "MILD"
    else:
        ef_state = "SEVERE"

    if inst < 0.15:
        dyn_state = "STABLE"
    elif inst < 0.30:
        dyn_state = "MODERATE"
    else:
        dyn_state = "UNSTABLE"

    if risk < 0.05:
        risk_state = "LOW"
    elif risk < 0.15:
        risk_state = "MODERATE"
    else:
        risk_state = "HIGH"

    if loc < 0.3:
        loc_state = "DIFFUSE"
    elif loc < 0.7:
        loc_state = "REGIONAL"
    else:
        loc_state = "FOCAL"

    return {
        "EF_state": ef_state,
        "Dynamics": dyn_state,
        "Risk": risk_state,
        "Localization": loc_state
    }
