"""Synthetic EEG generator for smoke-testing the pipeline without real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

FRONTAL = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"}
POSTERIOR = {"P7", "P3", "Pz", "P4", "P8", "O1", "O2"}
TEMPORAL = {"T7", "T8"}


def _one_over_f(
    rng: np.random.Generator,
    n_channels: int,
    n_samples: int,
    fs: int,
    exponent: float,
) -> np.ndarray:
    """Coloured background noise with a 1/f^exponent spectrum."""
    white = rng.normal(size=(n_channels, n_samples))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    spectrum = np.fft.rfft(white, axis=1)
    spectrum /= np.maximum(freqs, 0.5) ** (exponent / 2.0)
    signal = np.fft.irfft(spectrum, n=n_samples, axis=1)
    signal /= signal.std(axis=1, keepdims=True) + 1e-8
    return signal


def generate_recording(
    rng: np.random.Generator,
    label: int,
    fs: int = 128,
    duration: float = 120.0,
    identity_strength: float = 0.45,
    class_effect: float = 1.0,
) -> np.ndarray:
    """Generate one synthetic 19-channel EEG recording as (n_times, n_channels).

    The class effect is a higher theta/beta ratio and lower alpha power in the
    ADHD group, scaled by ``class_effect``. Subject identity is injected through
    channel gains, a random mixing matrix, and subject-specific peak
    frequencies. The identity signal is what makes epoch-level splitting leak.
    """
    n_samples = int(round(fs * duration))
    if n_samples < fs:
        raise ValueError("duration is too short")
    if class_effect < 0:
        raise ValueError("class_effect must be non-negative")

    n_channels = len(CHANNELS)
    time = np.arange(n_samples) / fs

    exponent = rng.uniform(0.7, 1.2)
    background = _one_over_f(rng, n_channels, n_samples, fs, exponent) * 0.9

    frontal_profile = np.array(
        [1.0 if channel in FRONTAL else 0.25 for channel in CHANNELS]
    )
    posterior_profile = np.array(
        [1.0 if channel in POSTERIOR else 0.20 for channel in CHANNELS]
    )

    # Subject-specific peak frequencies are an identity cue.
    theta_frequency = rng.uniform(4.5, 6.5)
    alpha_frequency = rng.uniform(8.5, 11.5)
    beta_frequency = rng.uniform(18.0, 24.0)

    common_theta_phase = rng.uniform(0.0, 2.0 * np.pi)
    common_alpha_phase = rng.uniform(0.0, 2.0 * np.pi)
    common_beta_phase = rng.uniform(0.0, 2.0 * np.pi)

    theta_phase = common_theta_phase + rng.normal(0.0, 0.25, size=(n_channels, 1))
    alpha_phase = common_alpha_phase + rng.normal(0.0, 0.25, size=(n_channels, 1))
    beta_phase = common_beta_phase + rng.normal(0.0, 0.25, size=(n_channels, 1))

    if label == 1:
        theta_amplitude = 0.55 + 0.50 * class_effect
        alpha_amplitude = max(0.05, 1.05 - 0.50 * class_effect)
        beta_amplitude = 0.30 + 0.25 * class_effect
    else:
        theta_amplitude = 0.55
        alpha_amplitude = 1.05
        beta_amplitude = 0.30

    # Subject-level variability in the class effect.
    theta_amplitude *= rng.uniform(0.85, 1.15)
    alpha_amplitude *= rng.uniform(0.85, 1.15)

    theta = (
        theta_amplitude
        * frontal_profile[:, None]
        * np.sin(2.0 * np.pi * theta_frequency * time[None, :] + theta_phase)
    )
    alpha = (
        alpha_amplitude
        * posterior_profile[:, None]
        * np.sin(2.0 * np.pi * alpha_frequency * time[None, :] + alpha_phase)
    )
    beta = (
        beta_amplitude
        * np.sin(2.0 * np.pi * beta_frequency * time[None, :] + beta_phase)
    )

    signal = background + theta + alpha + beta

    # Subject identity: random mixing and channel gains.
    mixing = np.eye(n_channels) + identity_strength * 0.20 * rng.normal(
        size=(n_channels, n_channels)
    )
    gains = np.exp(identity_strength * rng.normal(0.0, 0.35, size=n_channels))
    signal = mixing @ signal
    signal *= gains[:, None]

    # Eyeblinks in frontal channels.
    frontal_indices = [
        index for index, channel in enumerate(CHANNELS) if channel in FRONTAL
    ]
    n_blinks = int(rng.poisson(max(0.5, duration / 40.0)))
    blink_length = max(2, int(0.4 * fs))
    for _ in range(n_blinks):
        start = int(rng.integers(0, max(1, n_samples - blink_length)))
        end = min(n_samples, start + blink_length)
        envelope = np.hanning(blink_length)[: end - start]
        amplitude = rng.uniform(2.0, 4.5)
        spatial = rng.uniform(0.7, 1.3, size=(len(frontal_indices), 1))
        signal[frontal_indices, start:end] += (
            amplitude * spatial * envelope[None, :]
        )

    # Short muscle bursts in temporal channels.
    temporal_indices = [
        index for index, channel in enumerate(CHANNELS) if channel in TEMPORAL
    ]
    n_bursts = int(rng.poisson(max(0.3, duration / 90.0)))
    burst_length = max(2, int(0.2 * fs))
    for _ in range(n_bursts):
        start = int(rng.integers(0, max(1, n_samples - burst_length)))
        end = min(n_samples, start + burst_length)
        envelope = np.hanning(burst_length)[: end - start]
        burst = rng.normal(0.0, 1.0, size=(len(temporal_indices), end - start))
        signal[temporal_indices, start:end] += 0.8 * burst * envelope[None, :]

    # Scale to microvolts and add a constant electrode offset.
    signal -= signal.mean(axis=1, keepdims=True)
    signal /= signal.std() + 1e-8
    signal *= 15.0
    signal += rng.normal(0.0, 2.0, size=(n_channels, 1))

    return signal.T.astype(np.float32)


def generate_synthetic_dataset(
    output_dir: str | Path,
    n_subjects: int = 60,
    duration: float = 120.0,
    fs: int = 128,
    seed: int = 42,
    identity_strength: float = 0.45,
    class_effect: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Write .npy recordings plus a manifest and return the manifest frame."""
    if n_subjects < 4:
        raise ValueError("n_subjects must be at least 4")
    if n_subjects % 2 != 0:
        raise ValueError("n_subjects must be even for a balanced dataset")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n_subjects // 2) + [1] * (n_subjects // 2))
    rng.shuffle(labels)

    rows = []
    for index, label in enumerate(labels, start=1):
        subject_id = f"sub-{index:03d}"
        recording = generate_recording(
            rng,
            label=int(label),
            fs=fs,
            duration=duration,
            identity_strength=identity_strength,
            class_effect=class_effect,
        )
        filename = f"{subject_id}.npy"
        np.save(output_dir / filename, recording)
        rows.append(
            {
                "subject_id": subject_id,
                "label": int(label),
                "site": "synthetic",
                "file": filename,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    if verbose:
        n_epochs = max(1, int(duration // 5))
        print(
            f"Synthetic data: {n_subjects} subjects "
            f"({int(labels.sum())} ADHD / {int((1 - labels).sum())} control), "
            f"{int(duration)} s each, ~{n_epochs} epochs/subject, {fs} Hz"
        )
        print(
            f"Class effect: {class_effect:.2f} | "
            f"identity strength: {identity_strength:.2f}"
        )
        print(f"Manifest: {output_dir / 'manifest.csv'}")

    return manifest
