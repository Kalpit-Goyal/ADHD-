"""Prepare the IEEE DataPort 121-child EEG dataset for the pipeline.

Converts each subject .mat file to a float32 .npy recording and writes a
manifest.csv. Expected input layout (after extracting the four archives):

    data/real/raw/ADHD_part1/*.mat
    data/real/raw/ADHD_part2/*.mat
    data/real/raw/Control_part1/*.mat
    data/real/raw/Control_part2/*.mat

Channel order in this dataset is:
    Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2,
    F7, F8, T7, T8, P7, P8, Fz, Cz, Pz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_recording

GROUPS = [
    ("ADHD_part1", 1),
    ("ADHD_part2", 1),
    ("Control_part1", 0),
    ("Control_part2", 0),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/real/raw")
    parser.add_argument("--out-root", default="data/real")
    parser.add_argument("--fs", type=int, default=128)
    parser.add_argument("--n-channels", type=int, default=19)
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-convert existing .npy files"
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    recordings_dir = out_root / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen: set[str] = set()

    for folder, label in GROUPS:
        folder_path = raw_root / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"Missing folder: {folder_path}")

        files = sorted(folder_path.glob("*.mat"))
        print(f"{folder}: {len(files)} files (label {label})")

        for path in files:
            subject_id = path.stem
            if subject_id in seen:
                raise ValueError(f"Duplicate subject ID: {subject_id}")
            seen.add(subject_id)

            destination = recordings_dir / f"{subject_id}.npy"
            if destination.exists() and not args.overwrite:
                recording = np.load(destination)
            else:
                recording, _ = load_recording(
                    path, n_channels=args.n_channels, fs=args.fs
                )
                np.save(destination, recording.astype(np.float32))

            if recording.shape[1] != args.n_channels:
                raise ValueError(
                    f"{path.name}: expected {args.n_channels} channels, "
                    f"got {recording.shape}"
                )

            rows.append(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "site": "ieee-121",
                    "file": f"recordings/{subject_id}.npy",
                    "n_samples": int(recording.shape[0]),
                    "duration_seconds": round(
                        recording.shape[0] / float(args.fs), 2
                    ),
                }
            )

    manifest = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    manifest_path = out_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    counts = manifest["label"].value_counts().to_dict()
    durations = manifest["duration_seconds"]
    print("\nConverted recordings")
    print(f"  subjects : {len(manifest)}")
    print(f"  ADHD     : {int(counts.get(1, 0))}")
    print(f"  control  : {int(counts.get(0, 0))}")
    print(
        f"  duration : {durations.min():.1f}-{durations.max():.1f} s "
        f"(median {durations.median():.1f} s)"
    )
    print(f"  manifest : {manifest_path}")
    print(f"  data     : {recordings_dir}")


if __name__ == "__main__":
    main()
