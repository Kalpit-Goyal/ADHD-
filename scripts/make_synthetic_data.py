"""Generate a synthetic 19-channel EEG dataset for pipeline smoke tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic", help="Output directory")
    parser.add_argument("--subjects", type=int, default=60)
    parser.add_argument("--duration", type=float, default=120.0, help="Seconds per subject")
    parser.add_argument("--fs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--identity-strength",
        type=float,
        default=0.45,
        help="Subject-identity signal; higher values make epoch-level leakage worse",
    )
    parser.add_argument(
        "--class-effect",
        type=float,
        default=1.0,
        help="Strength of the ADHD vs control spectral difference (0 = none)",
    )
    args = parser.parse_args()

    generate_synthetic_dataset(
        output_dir=args.out,
        n_subjects=args.subjects,
        duration=args.duration,
        fs=args.fs,
        seed=args.seed,
        identity_strength=args.identity_strength,
        class_effect=args.class_effect,
    )


if __name__ == "__main__":
    main()
