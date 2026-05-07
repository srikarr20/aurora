import os
import numpy as np
import nibabel as nib
import csv
import cv2

# =========================
# LOAD MRI
# =========================
def load_data(path):
    V = nib.load(path).get_fdata()
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)

    if len(V.shape) == 4:
        z = V.shape[2] // 2
        V = V[:, :, z, :]

    return V


# =========================
# ACCUMULATION
# =========================
def compute_accumulation(V):

    ref = V[:, :, 0]
    A = np.zeros_like(ref)

    for t in range(V.shape[2]):
        diff = np.abs(V[:, :, t] - ref)

        if diff.max() > 0:
            diff = diff / diff.max()

        A += diff

    A = A / V.shape[2]

    if A.max() > 0:
        A = A / A.max()

    return A


# =========================
# ADAPTIVE ROI
# =========================
def apply_adaptive_roi(A):

    h, w = A.shape

    mask = A > 0.3
    coords = np.argwhere(mask)

    if len(coords) < 50:
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        return A[y1:y2, x1:x2], (x1, y1, x2, y2)

    cy, cx = coords.mean(axis=0).astype(int)

    size = min(h, w) // 4

    y1 = max(cy - size, 0)
    y2 = min(cy + size, h)
    x1 = max(cx - size, 0)
    x2 = min(cx + size, w)

    return A[y1:y2, x1:x2], (x1, y1, x2, y2)


# =========================
# ACI
# =========================
def compute_aci(A_roi):
    return float(np.std(A_roi) - 0.3 * np.mean(A_roi))


# =========================
# UPDATED TCI (FIXED)
# =========================
def compute_tci(V):

    energy = []

    for t in range(1, V.shape[2]):
        diff = np.abs(V[:, :, t] - V[:, :, t - 1])

        # capture variability instead of mean
        energy.append(np.std(diff))

    # scale up for observability
    return float(np.std(energy) * 10)


# =========================
# UPDATED CLASSIFICATION
# =========================
def classify(aci, tci):

    if aci < 0.045:
        return "LOW_CONTRACTION"

    elif tci > 0.002:
        return "IRREGULAR"

    else:
        return "NORMAL"


# =========================
# VISUALIZATION
# =========================
def save_visual(V, A, roi_box, out_path):

    os.makedirs(out_path, exist_ok=True)

    h, w = V.shape[:2]

    raw = (V[:, :, 0] * 255).astype(np.uint8)
    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    heat = (A * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    x1, y1, x2, y2 = roi_box
    cv2.rectangle(heat, (x1, y1), (x2, y2), (255,255,255), 2)

    combined = np.hstack([raw, heat])

    cv2.putText(combined, "MRI", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(combined, "AURORA V6.1 (TCI FIX)", (w + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imwrite(os.path.join(out_path, "aurora.png"), combined)


# =========================
# SINGLE RUN
# =========================
def run_single(path, out_dir):

    V = load_data(path)
    A = compute_accumulation(V)

    A_roi, roi_box = apply_adaptive_roi(A)

    aci = compute_aci(A_roi)
    tci = compute_tci(V)
    label = classify(aci, tci)

    print("\n=== AURORA V6.1 (TCI FIX) ===")
    print(f"ACI: {aci:.4f}")
    print(f"TCI: {tci:.4f}")
    print(f"Label: {label}")

    save_visual(V, A, roi_box, out_dir)


# =========================
# BATCH RUN
# =========================
def run_batch(input_dir, output_csv):

    results = []

    for patient in sorted(os.listdir(input_dir)):

        p_path = os.path.join(input_dir, patient)

        if not os.path.isdir(p_path):
            continue

        nii_files = [f for f in os.listdir(p_path) if f.endswith("_4d.nii.gz")]

        if len(nii_files) == 0:
            continue

        nii_path = os.path.join(p_path, nii_files[0])

        V = load_data(nii_path)
        A = compute_accumulation(V)

        A_roi, _ = apply_adaptive_roi(A)

        aci = compute_aci(A_roi)
        tci = compute_tci(V)
        label = classify(aci, tci)

        print(f"{patient} → ACI:{aci:.4f} TCI:{tci:.4f} → {label}")

        results.append([patient, aci, tci, label])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Patient", "ACI", "TCI", "Label"])
        writer.writerows(results)

    print(f"\nSaved: {output_csv}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    MODE = "batch"

    if MODE == "single":

        run_single(
            path="examples/stable/patient029_4d.nii.gz",
            out_dir="viz/sample"
        )

    elif MODE == "batch":

        run_batch(
            input_dir="data",
            output_csv="outputs/aci_results.csv"
        )
