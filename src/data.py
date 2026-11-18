"""Data loading, preprocessing, epoching, and subject-wise splitting."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from .config import DataConfig, FeatureConfig
from .features import extract_epoch_features

try:  # tqdm is optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


LABEL_ALIASES = {
    "adhd": 1,
    "case": 1,
    "patient": 1,
    "patients": 1,
    "yes": 1,
    "true": 1,
    "1": 1,
    "hc": 0,
    "control": 0,
    "controls": 0,
    "td": 0,
    "typical": 0,
    "healthy": 0,
    "no": 0,
    "false": 0,
    "0": 0,
}


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    label: int
    site: str
    path: Path


@dataclass
class FeatureBundle:
    raw: np.ndarray          # (N, 1, C, T)
    node_features: np.ndarray  # (N, C, F)
    adjacency: np.ndarray    # (N, C, C)
    labels: np.ndarray       # (N,)
    subject_ids: np.ndarray  # (N,)
    epoch_index: np.ndarray  # (N,)
    subject_labels: dict[str, int]
    fs: int
    epoch_seconds: float

    @property
    def n_channels(self) -> int:
        return int(self.raw.shape[2])

    @property
    def n_times(self) -> int:
        return int(self.raw.shape[3])

    @property
    def n_node_features(self) -> int:
        return int(self.node_features.shape[2])


def parse_label(value: object) -> int:
    """Parse common binary label encodings."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            raise ValueError("Label is NaN")
        return int(value)

    text = str(value).strip().lower()
    if text in LABEL_ALIASES:
        return LABEL_ALIASES[text]
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Cannot parse label: {value!r}") from exc


def read_manifest(root: str | Path, manifest: str = "manifest.csv") -> list[SubjectRecord]:
    """Read a manifest with columns subject_id, label, and optional site/file.

    If ``file`` is omitted, the loader looks for ``<subject_id>.npy``,
    ``<subject_id>.csv``, ``<subject_id>.txt``, or ``<subject_id>.set`` under
    ``root``.
    """
    root = Path(root)
    manifest_path = root / manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    frame.columns = [str(c).strip() for c in frame.columns]

    required = {"subject_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    records: list[SubjectRecord] = []
    for row_index, row in frame.iterrows():
        subject_id = str(row["subject_id"]).strip()
        label = parse_label(row["label"])
        if label not in (0, 1):
            raise ValueError(
                f"Subject {subject_id}: binary label expected, got {label}"
            )

        site = "unknown"
        if "site" in frame.columns and not pd.isna(row["site"]):
            site = str(row["site"]).strip()

        if "file" in frame.columns and not pd.isna(row["file"]):
            path = Path(str(row["file"]).strip())
            if not path.is_absolute():
                path = root / path
        else:
            candidates = [
                root / f"{subject_id}{suffix}"
                for suffix in (".npy", ".csv", ".txt", ".set")
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                raise FileNotFoundError(
                    f"No recording found for subject {subject_id} under {root}"
                )

        if not path.exists():
            raise FileNotFoundError(f"Recording not found for subject {subject_id}: {path}")

        records.append(SubjectRecord(subject_id, label, site, path))

    if not records:
        raise ValueError("Manifest is empty")

    return records


def _mat_arrays(node):
    """Yield numeric 2-D/3-D arrays from a loaded MATLAB file, recursively."""
    if isinstance(node, np.ndarray):
        if node.dtype == object:
            for item in node.ravel():
                yield from _mat_arrays(item)
        elif node.ndim in (2, 3) and np.issubdtype(node.dtype, np.number):
            yield node
    elif isinstance(node, np.void):
        for name in node.dtype.names or ():
            yield from _mat_arrays(node[name])
    elif isinstance(node, dict):
        for key, item in node.items():
            if isinstance(key, str) and key.startswith("__"):
                continue
            yield from _mat_arrays(item)
    elif hasattr(node, "_fieldnames"):
        for name in node._fieldnames:
            yield from _mat_arrays(getattr(node, name))


def load_recording(
    path: str | Path,
    n_channels: int | None = None,
    fs: int | None = None,
) -> tuple[np.ndarray, int | None]:
    """Load a recording as (n_times, n_channels).

    Supports .npy, .csv/.txt, .mat, and .set (if MNE is installed).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    source_fs: int | None = None

    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
        drop = [
            column
            for column in frame.columns
            if str(column).strip().lower() in {"time", "timestamp", "index", "unnamed: 0"}
        ]
        frame = frame.drop(columns=drop, errors="ignore")
        frame = frame.select_dtypes(include=[np.number])
        if frame.shape[1] == 0:
            raise ValueError(f"No numeric columns in {path}")
        array = frame.to_numpy()
    elif suffix == ".mat":
        from scipy.io import loadmat

        contents = loadmat(path, squeeze_me=True, struct_as_record=False)
        candidates = list(_mat_arrays(contents))
        if not candidates:
            raise ValueError(f"No numeric signal array found in {path}")
        if n_channels is not None:
            channel_arrays = [
                item for item in candidates if n_channels in item.shape
            ]
            if channel_arrays:
                candidates = channel_arrays
        array = max(candidates, key=lambda item: item.size).astype(np.float64)

        if array.ndim == 3:
            # (channels, time, trials) or (trials, channels, time).
            if n_channels is not None and array.shape[0] == n_channels:
                array = np.concatenate(
                    [array[..., trial] for trial in range(array.shape[2])], axis=1
                )
            elif n_channels is not None and array.shape[1] == n_channels:
                array = np.concatenate(
                    [array[trial] for trial in range(array.shape[0])], axis=0
                )
            else:
                array = array.reshape(array.shape[0], -1)
    elif suffix == ".set":
        try:
            import mne
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Reading .set files requires MNE: pip install mne"
            ) from exc
        raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
        array = raw.get_data().T
        source_fs = int(round(float(raw.info["sfreq"])))
    else:
        raise ValueError(f"Unsupported recording format: {suffix}")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {array.shape} for {path}")

    # Drop an accidental time column if it is monotonic.
    if n_channels is not None and array.shape[1] == n_channels + 1:
        if np.all(np.diff(array[:, 0]) > 0):
            array = array[:, 1:]

    if n_channels is not None:
        if array.shape[1] == n_channels:
            pass
        elif array.shape[0] == n_channels:
            array = array.T
        else:
            raise ValueError(
                f"Expected {n_channels} channels, got shape {array.shape} for {path}"
            )
    elif array.shape[0] < array.shape[1]:
        # Assume the long axis is time.
        array = array.T

    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    if fs is not None and source_fs is None:
        source_fs = fs

    return array, source_fs


def to_microvolts(array: np.ndarray) -> np.ndarray:
    """Heuristically convert volts or millivolts to microvolts."""
    scale = float(np.median(np.abs(array)))
    if scale == 0.0:
        return array.astype(np.float32)
    if scale < 1e-3:
        return (array * 1e6).astype(np.float32)
    if scale < 0.5:
        return (array * 1e3).astype(np.float32)
    return array.astype(np.float32)


def preprocess_recording(
    array: np.ndarray,
    fs: int,
    bandpass: tuple[float, float] = (0.5, 45.0),
    notch: float | None = 50.0,
    source_fs: int | None = None,
) -> np.ndarray:
    """Resample if needed, convert to microvolts, notch/band-pass filter."""
    if source_fs is not None and int(source_fs) != int(fs):
        divisor = math.gcd(int(source_fs), int(fs))
        array = resample_poly(
            array, int(fs) // divisor, int(source_fs) // divisor, axis=0
        )

    array = to_microvolts(np.asarray(array, dtype=np.float32))
    array = array - array.mean(axis=0, keepdims=True)

    nyquist = fs / 2.0
    if notch is not None and 0 < notch < nyquist:
        b, a = iirnotch(notch, Q=30.0, fs=fs)
        array = np.asarray(filtfilt(b, a, array, axis=0), dtype=np.float32)

    low, high = float(bandpass[0]), float(bandpass[1])
    high = min(high, nyquist - 1.0)
    if low <= 0:
        low = 0.1
    if high <= low:
        raise ValueError(f"Invalid band-pass: {bandpass} at fs={fs}")

    sos = butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
    array = np.asarray(sosfiltfilt(sos, array, axis=0), dtype=np.float32)

    return array


def remove_blink_components(
    array: np.ndarray,
    fs: int,
    blink_channels: tuple[int, ...] = (0, 1),
    max_components: int = 2,
    min_corr: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Remove eye-blink components with FastICA.

    A component is treated as ocular if it correlates with the mean frontal
    channel (Fp1/Fp2 in this montage) and is dominated by delta power. At most
    ``max_components`` are removed. Falls back to the input if ICA fails.
    """
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning
    from scipy.signal import welch

    x = np.asarray(array, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    x = x / scale

    try:
        ica = FastICA(
            n_components=x.shape[1],
            random_state=seed,
            max_iter=1000,
            tol=1e-2,
            whiten="unit-variance",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            sources = ica.fit_transform(x)
    except Exception:  # pragma: no cover - convergence failure
        return np.asarray(array, dtype=np.float32)

    frontal = x[:, list(blink_channels)].mean(axis=1)
    freqs, psd = welch(sources, fs=fs, axis=0, nperseg=min(256, len(sources)))
    delta = (freqs >= 0.5) & (freqs <= 4.0)
    total = (freqs >= 0.5) & (freqs <= 45.0)

    ranked: list[tuple[float, int]] = []
    for component in range(sources.shape[1]):
        corr = abs(float(np.corrcoef(sources[:, component], frontal)[0, 1]))
        if not np.isfinite(corr) or corr < min_corr:
            continue
        ratio = float(psd[delta, component].sum()) / (
            float(psd[total, component].sum()) + 1e-12
        )
        ranked.append((corr * ratio, component))

    selected = [component for _, component in sorted(ranked, reverse=True)[:max_components]]
    if not selected:
        return np.asarray(array, dtype=np.float32)

    cleaned = sources.copy()
    cleaned[:, selected] = 0.0
    reconstructed = ica.inverse_transform(cleaned) * scale
    return reconstructed.astype(np.float32)


def epoch_recording(
    array: np.ndarray,
    fs: int,
    epoch_seconds: float = 5.0,
    overlap: float = 0.0,
    max_abs_uv: float | None = None,
) -> np.ndarray:
    """Split a recording into non-overlapping (or overlapping) epochs.

    Returns (n_epochs, n_channels, n_times).
    """
    n_times = int(round(epoch_seconds * fs))
    if n_times <= 0:
        raise ValueError("epoch_seconds must be positive")
    if array.shape[0] < n_times:
        return np.empty((0, array.shape[1], n_times), dtype=np.float32)

    step = max(1, int(round(n_times * (1.0 - float(overlap)))))
    epochs: list[np.ndarray] = []

    for start in range(0, array.shape[0] - n_times + 1, step):
        epoch = array[start : start + n_times].T
        if not np.isfinite(epoch).all():
            continue
        if max_abs_uv is not None and float(np.max(np.abs(epoch))) > max_abs_uv:
            continue
        epochs.append(epoch.astype(np.float32))

    if not epochs:
        return np.empty((0, array.shape[1], n_times), dtype=np.float32)
    return np.stack(epochs, axis=0)


def normalize_epochs(epochs: np.ndarray) -> np.ndarray:
    """Per-channel z-score using statistics across all epochs of a recording.

    Normalising at the recording level (rather than per epoch) preserves
    relative amplitude differences between epochs while removing electrode
    gain differences between subjects.
    """
    if epochs.size == 0:
        return epochs.astype(np.float32)
    mean = epochs.mean(axis=(0, 2), keepdims=True)
    std = epochs.std(axis=(0, 2), keepdims=True)
    return ((epochs - mean) / (std + 1e-8)).astype(np.float32)


def build_feature_bundle(
    records: Iterable[SubjectRecord],
    data_cfg: DataConfig,
    feature_cfg: FeatureConfig,
    verbose: bool = True,
) -> FeatureBundle:
    """Load every recording and build raw epochs, node features, adjacencies."""
    records = list(records)
    raw_list: list[np.ndarray] = []
    node_list: list[np.ndarray] = []
    adj_list: list[np.ndarray] = []
    label_list: list[float] = []
    subject_list: list[str] = []
    epoch_list: list[int] = []
    seen_labels: dict[str, int] = {}
    subject_labels: dict[str, int] = {}

    iterator = tqdm(records, desc="Loading EEG", unit="subject") if (verbose and tqdm) else records

    for record in iterator:
        previous = seen_labels.get(record.subject_id)
        if previous is not None and previous != record.label:
            raise ValueError(f"Inconsistent labels for subject {record.subject_id}")
        seen_labels[record.subject_id] = record.label

        array, source_fs = load_recording(
            record.path, n_channels=data_cfg.n_channels, fs=data_cfg.fs
        )
        array = preprocess_recording(
            array,
            fs=data_cfg.fs,
            bandpass=data_cfg.bandpass,
            notch=data_cfg.notch,
            source_fs=source_fs,
        )
        if data_cfg.artifact_removal.lower() == "ica":
            array = remove_blink_components(
                array,
                fs=data_cfg.fs,
                blink_channels=tuple(data_cfg.blink_channels),
                max_components=data_cfg.ica_max_components,
                min_corr=data_cfg.ica_blink_corr,
            )
        epochs = epoch_recording(
            array,
            fs=data_cfg.fs,
            epoch_seconds=data_cfg.epoch_seconds,
            overlap=data_cfg.overlap,
            max_abs_uv=data_cfg.max_abs_uv,
        )
        if epochs.shape[0] == 0:
            if verbose:
                print(f"  skipping {record.subject_id}: no clean epochs")
            continue

        subject_labels[record.subject_id] = record.label
        epochs = normalize_epochs(epochs)

        if data_cfg.max_abs_z is not None:
            peaks = np.abs(epochs).max(axis=(1, 2))
            keep = peaks <= float(data_cfg.max_abs_z)
            if not np.any(keep):
                if verbose:
                    print(
                        f"  skipping {record.subject_id}: "
                        f"no epochs below z={data_cfg.max_abs_z}"
                    )
                subject_labels.pop(record.subject_id, None)
                continue
            epochs = epochs[keep]

        for epoch_index, epoch in enumerate(epochs):
            node, adjacency = extract_epoch_features(
                epoch,
                fs=data_cfg.fs,
                bands=feature_cfg.bands,
                connectivity=data_cfg.connectivity,
                var_order=data_cfg.var_order,
            )
            raw_list.append(epoch[None, :, :])
            node_list.append(node)
            adj_list.append(adjacency)
            label_list.append(float(record.label))
            subject_list.append(record.subject_id)
            epoch_list.append(epoch_index)

    if not raw_list:
        raise ValueError("No usable epochs were produced from the manifest")

    return FeatureBundle(
        raw=np.stack(raw_list, axis=0).astype(np.float32),
        node_features=np.stack(node_list, axis=0).astype(np.float32),
        adjacency=np.stack(adj_list, axis=0).astype(np.float32),
        labels=np.asarray(label_list, dtype=np.float32),
        subject_ids=np.asarray(subject_list),
        epoch_index=np.asarray(epoch_list, dtype=np.int32),
        subject_labels=subject_labels,
        fs=data_cfg.fs,
        epoch_seconds=data_cfg.epoch_seconds,
    )


def subject_folds(
    bundle: FeatureBundle,
    n_splits: int = 5,
    val_size: float = 0.2,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (train_subjects, val_subjects, test_subjects) per outer fold.

    Splits are made at the subject level, so no subject appears in more than
    one of train/validation/test.
    """
    subjects = np.asarray(sorted(bundle.subject_labels), dtype=object)
    labels = np.asarray([bundle.subject_labels[str(s)] for s in subjects], dtype=int)

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(subjects) < n_splits:
        raise ValueError(
            f"Only {len(subjects)} subjects for {n_splits}-fold CV"
        )
    if len(np.unique(labels)) < 2:
        raise ValueError("Both classes are required")

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for train_index, test_index in outer.split(subjects, labels):
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]
        train_labels = labels[train_index]

        if len(np.unique(train_labels)) < 2:
            raise ValueError("A training fold contains only one class")

        try:
            inner_train, inner_val = train_test_split(
                train_subjects,
                test_size=val_size,
                stratify=train_labels,
                random_state=seed,
            )
        except ValueError as exc:
            raise ValueError(
                "Could not create a stratified validation split. "
                "Use more subjects or a larger val_size."
            ) from exc

        folds.append(
            (
                np.asarray(inner_train, dtype=object),
                np.asarray(inner_val, dtype=object),
                np.asarray(test_subjects, dtype=object),
            )
        )

    return folds


def indices_for_subjects(bundle: FeatureBundle, subjects: Iterable[str]) -> np.ndarray:
    """Epoch indices belonging to the given subject IDs."""
    wanted = {str(subject) for subject in subjects}
    mask = np.asarray([str(s) in wanted for s in bundle.subject_ids])
    return np.flatnonzero(mask)


class EpochDataset(Dataset):
    """Epoch-level dataset that returns raw windows or graph features."""

    def __init__(self, bundle: FeatureBundle, indices: np.ndarray, model_name: str):
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=int)
        self.model_name = model_name

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        label = torch.tensor(self.bundle.labels[index], dtype=torch.float32)

        if self.model_name == "eegnet":
            raw = torch.from_numpy(self.bundle.raw[index]).float()
            return raw, label

        node = torch.from_numpy(self.bundle.node_features[index]).float()
        adjacency = torch.from_numpy(self.bundle.adjacency[index]).float()
        return node, adjacency, label
