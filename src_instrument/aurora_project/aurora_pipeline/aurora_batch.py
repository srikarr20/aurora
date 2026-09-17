import argparse
import subprocess
import sys
from pathlib import Path


def run_batch(data_root, output_root):
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for patient_dir in sorted(data_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        try:
            vol = [path for path in patient_dir.iterdir() if "_4d.nii.gz" in path.name][0]
            out_dir = output_root / patient_dir.name

            cmd = [
                sys.executable,
                "-m",
                "src_instrument.aurora_project.aurora_pipeline.aurora_run",
                "--input",
                str(vol),
                "--output",
                str(out_dir),
            ]
            print("Running", patient_dir.name)
            subprocess.run(cmd, check=False)

        except Exception as e:
            print("Skip:", patient_dir.name, e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    run_batch(args.data_root, args.output_root)


if __name__ == "__main__":
    main()
