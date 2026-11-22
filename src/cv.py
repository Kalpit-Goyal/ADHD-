"""Subject-wise cross-validation utilities and leakage checks.

All experiments in this repository use participant-disjoint folds.  The
functions here create, serialise, load, and validate those folds.  A validation
failure raises :class:`LeakageError` rather than only printing a warning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


class LeakageError(ValueError):
    """Raised when a cross-validation partition is not participant-disjoint."""


@dataclass(frozen=True)
class Fold:
    """One train/validation/test partition of subject identifiers."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    repeat: int = 0
    fold: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def read_subject_labels(manifest: str | Path) -> dict[str, int]:
    """Read subject_id -> binary label from a manifest CSV."""
    frame = pd.read_csv(manifest)
    required = {"subject_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    labels = {
        str(row.subject_id): int(row.label)
        for row in frame[["subject_id", "label"]].itertuples(index=False)
    }
    if len(labels) != len(frame):
        raise ValueError("Duplicate subject_id values in manifest")
    if set(labels.values()) - {0, 1}:
        raise ValueError("Labels must be binary (0/1)")
    return labels


def _require_subjects(fold: Fold, labels: Mapping[str, int]) -> None:
    missing = (set(fold.train) | set(fold.val) | set(fold.test)) - set(labels)
    if missing:
        raise LeakageError(f"Fold references unknown participants: {sorted(missing)}")


def validate_folds(
    folds: Sequence[Fold],
    labels: Mapping[str, int],
    *,
    require_full_coverage: bool = True,
    require_balanced_test: bool = False,
) -> dict:
    """Validate participant disjointness and fold coverage.

    Checks
    ------
    * train/validation/test are pairwise disjoint within every fold;
    * a participant is tested at most once per repeat;
    * every fold has both classes in train/validation/test;
    * optionally, each repeat covers every participant exactly once in test;
    * optionally, each test fold is class-balanced.
    """
    if not folds:
        raise LeakageError("No folds supplied")

    repeats: dict[int, list[Fold]] = {}
    summary: dict[str, object] = {"n_folds": len(folds), "repeats": {}}

    for fold in folds:
        _require_subjects(fold, labels)
        train, val, test = set(fold.train), set(fold.val), set(fold.test)
        if not train or not val or not test:
            raise LeakageError(f"Empty split in repeat {fold.repeat} fold {fold.fold}")
        if train & val:
            raise LeakageError(
                f"Train/validation participant overlap in repeat {fold.repeat} "
                f"fold {fold.fold}: {sorted(train & val)}"
            )
        if train & test:
            raise LeakageError(
                f"Train/test participant overlap in repeat {fold.repeat} "
                f"fold {fold.fold}: {sorted(train & test)}"
            )
        if val & test:
            raise LeakageError(
                f"Validation/test participant overlap in repeat {fold.repeat} "
                f"fold {fold.fold}: {sorted(val & test)}"
            )
        for split_name, split in (("train", train), ("val", val), ("test", test)):
            split_labels = {labels[s] for s in split}
            if len(split_labels) < 2:
                raise LeakageError(
                    f"{split_name} contains one class in repeat {fold.repeat} "
                    f"fold {fold.fold}"
                )
        if require_balanced_test:
            n_positive = sum(labels[s] for s in test)
            if n_positive * 2 != len(test):
                raise LeakageError(
                    f"Unbalanced test fold {fold.fold}: "
                    f"{n_positive}/{len(test)} positive"
                )
        repeats.setdefault(fold.repeat, []).append(fold)

    for repeat, repeat_folds in sorted(repeats.items()):
        test_lists = [set(fold.test) for fold in repeat_folds]
        union = set().union(*test_lists)
        total_tests = sum(len(test) for test in test_lists)
        if total_tests != len(union):
            raise LeakageError(
                f"A participant is tested more than once in repeat {repeat}"
            )
        if require_full_coverage and union != set(labels):
            missing = set(labels) - union
            extra = union - set(labels)
            raise LeakageError(
                f"Repeat {repeat} does not cover all participants exactly once; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        summary["repeats"][str(repeat)] = {
            "n_folds": len(repeat_folds),
            "n_subjects": len(union),
            "n_test_total": total_tests,
            "n_positive": int(sum(labels[s] for s in union)),
            "fold_sizes": [
                {
                    "fold": fold.fold,
                    "train": len(fold.train),
                    "val": len(fold.val),
                    "test": len(fold.test),
                    "test_positive": int(sum(labels[s] for s in fold.test)),
                }
                for fold in repeat_folds
            ],
        }
    return summary


def load_official_folds(
    path: str | Path,
    labels: Mapping[str, int],
    *,
    require_full_coverage: bool = True,
) -> list[Fold]:
    """Load the CTE-Net paper's fixed 5-fold partitions.

    The official file is a pickle containing five ``(train, val, test)``
    subject lists.  This project stores the same content as JSON.  The official
    cohort has 120 participants (61 ADHD minus ``v36p`` = 60).
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        partitions = raw.get("folds", raw)
    else:
        import pickle

        with path.open("rb") as handle:
            partitions = pickle.load(handle)

    folds: list[Fold] = []
    for fold_index, partition in enumerate(partitions):
        if isinstance(partition, dict):
            train, val, test = partition["train"], partition["val"], partition["test"]
        else:
            train, val, test = partition
        folds.append(
            Fold(
                train=tuple(map(str, train)),
                val=tuple(map(str, val)),
                test=tuple(map(str, test)),
                repeat=0,
                fold=fold_index,
            )
        )
    validate_folds(
        folds,
        labels,
        require_full_coverage=require_full_coverage,
        require_balanced_test=True,
    )
    return folds


def make_repeated_folds(
    labels: Mapping[str, int],
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    val_fraction: float = 0.2,
    seed: int = 42,
    existing_first_repeat: Sequence[Fold] | None = None,
    require_balanced_test: bool = False,
) -> list[Fold]:
    """Create repeated stratified participant-wise 5-fold partitions.

    The first repeat can be an existing partition (for example the official
    CTE-Net folds); later repeats use different random seeds.  Validation is a
    stratified subset of each training partition and is never used for fitting
    except for threshold selection / early stopping / meta-learning.
    """
    subjects = np.asarray(sorted(labels), dtype=object)
    y = np.asarray([labels[str(s)] for s in subjects], dtype=int)

    folds: list[Fold] = []
    if existing_first_repeat is not None:
        folds.extend(existing_first_repeat)

    first_generated = 1 if existing_first_repeat is not None else 0
    for repeat in range(first_generated, n_repeats):
        repeat_seed = seed + 1000 * repeat
        outer = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=repeat_seed
        )
        repeat_folds: list[Fold] = []
        for fold_index, (train_index, test_index) in enumerate(outer.split(subjects, y)):
            train_subjects = subjects[train_index]
            train_labels = y[train_index]
            inner_train, inner_val = train_test_split(
                train_subjects,
                test_size=val_fraction,
                stratify=train_labels,
                random_state=repeat_seed,
            )
            fold = Fold(
                train=tuple(sorted(map(str, inner_train))),
                val=tuple(sorted(map(str, inner_val))),
                test=tuple(sorted(map(str, subjects[test_index]))),
                repeat=repeat,
                fold=fold_index,
            )
            repeat_folds.append(fold)
        folds.extend(repeat_folds)

    validate_folds(
        folds,
        labels,
        require_full_coverage=True,
        require_balanced_test=require_balanced_test,
    )
    return folds


def save_folds(folds: Sequence[Fold], path: str | Path, metadata: dict | None = None) -> None:
    """Save folds as human-readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "folds": [
            {
                "repeat": fold.repeat,
                "fold": fold.fold,
                "train": list(fold.train),
                "val": list(fold.val),
                "test": list(fold.test),
            }
            for fold in folds
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_folds(path: str | Path) -> list[Fold]:
    """Load folds previously written by :func:`save_folds`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = []
    for item in payload["folds"]:
        folds.append(
            Fold(
                train=tuple(map(str, item["train"])),
                val=tuple(map(str, item["val"])),
                test=tuple(map(str, item["test"])),
                repeat=int(item.get("repeat", 0)),
                fold=int(item.get("fold", 0)),
            )
        )
    return folds


def split_indices(
    subjects: np.ndarray | Sequence[str], split_subjects: Iterable[str]
) -> np.ndarray:
    """Indices of rows whose subject identifier is in ``split_subjects``."""
    wanted = {str(s) for s in split_subjects}
    subjects = np.asarray(subjects)
    mask = np.asarray([str(s) in wanted for s in subjects], dtype=bool)
    return np.flatnonzero(mask)


def assert_no_subject_overlap(*splits: Iterable[str]) -> None:
    """Raise if any two subject collections intersect."""
    sets = [set(map(str, split)) for split in splits]
    for index in range(len(sets)):
        for other in range(index + 1, len(sets)):
            overlap = sets[index] & sets[other]
            if overlap:
                raise LeakageError(
                    f"Participant overlap between splits {index} and {other}: "
                    f"{sorted(overlap)}"
                )
