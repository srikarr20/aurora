import cv2
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path


def build_visual_demo(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "aurora_vs_mri.mp4"

    print("Loading MRI from:", input_path)

    # =========================
    # LOAD MRI
    # =========================
    V = nib.load(input_path).get_fdata()

    # pick middle slice
    z = V.shape[2] // 2
    T = V.shape[3]

    # normalize entire volume
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)

    h, w = V.shape[0], V.shape[1]

    # =========================
    # VIDEO WRITER
    # =========================
    fps = 5
    out = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w * 2, h)
    )

    prev = None

    print("Generating video...")

    for t in range(T):
        frame = V[:, :, z, t]

        # =========================
        # LEFT: RAW MRI
        # =========================
        left = (frame * 255).astype(np.uint8)
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)

        cv2.putText(left, "MRI (Raw)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # =========================
        # RIGHT: AURORA (motion)
        # =========================
        right = left.copy()

        if prev is not None:
            diff = np.abs(frame - prev)

            if np.max(diff) > 0:
                diff = diff / np.max(diff)

            heat = cv2.applyColorMap(
                (diff * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            right = cv2.addWeighted(right, 0.7, heat, 0.6, 0)

        cv2.putText(right, "AURORA (Motion)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        prev = frame

        # =========================
        # COMBINE
        # =========================
        combined = np.hstack([left, right])

        out.write(combined)

    out.release()

    print("\nDEMO VIDEO CREATED")
    print("Saved to:", video_path)
    return video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    build_visual_demo(args.input, args.output_dir)


if __name__ == "__main__":
    main()
