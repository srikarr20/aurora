import numpy as np


# =============================
# TRANSITION MATRIX
# =============================
def build_transition_matrix(labels, k=3):

    T = np.zeros((k, k))

    for i in range(len(labels) - 1):
        curr = labels[i]
        nxt = labels[i + 1]
        T[curr, nxt] += 1

    # normalize rows → probabilities
    row_sums = T.sum(axis=1, keepdims=True) + 1e-8
    T = T / row_sums

    return T


# =============================
# NEXT STATE PREDICTION
# =============================
def predict_next_state(current_label, T):

    probs = T[current_label]

    next_label = np.argmax(probs)
    confidence = probs[next_label]

    return next_label, confidence, probs


# =============================
# FULL SEQUENCE PREDICTION
# =============================
def predict_sequence(labels, T):

    preds = []
    confs = []

    for i in range(len(labels)):
        nxt, conf, _ = predict_next_state(labels[i], T)
        preds.append(nxt)
        confs.append(conf)

    return np.array(preds), np.array(confs)
