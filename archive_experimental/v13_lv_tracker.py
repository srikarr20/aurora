import numpy as np
import pandas as pd
import nibabel as nib
import os
import cv2

# =========================
# LOAD MRI
# =========================
def load_4d(path):
    data = np.squeeze(nib.load(path).get_fdata())
    if data.ndim != 4:
        raise ValueError(f"Expected (H,W,Z,T), got {data.shape}")
    return data


# =========================
# DETECT CANDIDATES (PER FRAME)
# =========================
def detect_candidates(frame):

    img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv = 255 - img

    _, mask = cv2.threshold(inv, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    regions = []

    for i in range(1, num_labels):
        region = (labels == i)
        area = np.sum(region)

        if area < 50:
            continue

        ys, xs = np.where(region)
        cy, cx = np.mean(ys), np.mean(xs)

        regions.append({
            "mask": region,
            "area": area,
            "centroid": (cy, cx)
        })

    return regions


# =========================
# MATCH REGIONS (TRACKING)
# =========================
def match_regions(prev_regions, curr_regions):

    matches = []

    for pr in prev_regions:

        py, px = pr["centroid"]

        best = None
        best_dist = 1e9

        for cr in curr_regions:

            cy, cx = cr["centroid"]

            d = np.sqrt((py-cy)**2 + (px-cx)**2)

            if d < best_dist:
                best_dist = d
                best = cr

        matches.append(best if best_dist < 20 else None)

    return matches


# =========================
# BUILD TRACKS (PER SLICE)
# =========================
def build_tracks(slice_data):

    H, W, T = slice_data.shape

    tracks = []

    # initialize from frame 0
    regions0 = detect_candidates(slice_data[:,:,0])

    for r in regions0:
        tracks.append([r])

    prev_regions = regions0

    for t in range(1, T):

        curr_regions = detect_candidates(slice_data[:,:,t])

        matches = match_regions(prev_regions, curr_regions)

        for i in range(len(tracks)):
            tracks[i].append(matches[i] if i < len(matches) else None)

        prev_regions = curr_regions

    return tracks


# =========================
# CONVERT TRACK → VOLUME
# =========================
def track_to_volume(track):

    vols = []

    for r in track:
        if r is None:
            vols.append(np.nan)
        else:
            vols.append(r["area"])

    vols = np.array(vols)

    if np.sum(~np.isnan(vols)) < 5:
        return None

    # interpolate missing
    inds = np.arange(len(vols))
    good = ~np.isnan(vols)

    vols[~good] = np.interp(inds[~good], inds[good], vols[good])

    return vols


# =========================
# SCORE TRACK (CRITICAL)
# =========================
def score_track(vol):

    if vol is None:
        return -np.inf

    amplitude = np.max(vol) - np.min(vol)

    fft = np.fft.fft(vol)
    power = np.abs(fft)
    power[0] = 0

    periodicity = np.max(power) / (np.sum(power) + 1e-6)

    diff = np.diff(vol)
    instability = np.std(diff)
    drift = np.abs(np.mean(diff))

    score = (
        2.0 * amplitude +
        3.0 * periodicity -
        1.5 * instability -
        1.0 * drift
    )

    return score


# =========================
# PROCESS PATIENT
# =========================
def process_patient(path):

    data = load_4d(path)
    H, W, Z, T = data.shape

    all_tracks = []

    for z in range(Z):
        tracks = build_tracks(data[:,:,z,:])

        for t in tracks:
            vol = track_to_volume(t)
            if vol is not None:
                all_tracks.append(vol)

    if len(all_tracks) == 0:
        return None

    # score all tracks
    scored = [(score_track(v), v) for v in all_tracks]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_track = scored[0][1]

    v_max = np.max(best_track)
    v_min = np.min(best_track)

    ef = (v_max - v_min) / (v_max + 1e-6)

    return {
        "EF_est": ef,
        "V_max": v_max,
        "V_min": v_min
    }


# =========================
# PROCESS DATASET
# =========================
def process(data_dir):

    rows = []

    for patient in sorted(os.listdir(data_dir)):

        p_dir = os.path.join(data_dir, patient)
        if not os.path.isdir(p_dir):
            continue

        nii_path = os.path.join(p_dir, f"{patient}_4d.nii.gz")

        if not os.path.exists(nii_path):
            continue

        try:
            feat = process_patient(nii_path)
            if feat is None:
                continue

        except Exception as e:
            print("Skipping:", patient, "|", e)
            continue

        row = {"Patient": patient}
        row.update(feat)
        rows.append(row)

        print("Processed:", patient)

    df = pd.DataFrame(rows)

    os.makedirs("outputs_v13", exist_ok=True)
    df.to_csv("outputs_v13/results.csv", index=False)

    print("\nSaved: outputs_v13/results.csv")
    print("Total processed:", len(df))


if __name__ == "__main__":
    process("data")
