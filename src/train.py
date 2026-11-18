"""Training and subject-wise cross-validation."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import Config
from .data import (
    EpochDataset,
    FeatureBundle,
    indices_for_subjects,
    subject_folds,
)
from .evaluate import binary_metrics, youden_threshold
from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def _forward(
    model: nn.Module,
    batch,
    model_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_name == "eegnet":
        features, labels = batch
        logits = model(features.to(device))
    else:
        features, adjacency, labels = batch
        logits = model(features.to(device), adjacency.to(device))
    return logits, labels.to(device)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return epoch probabilities, subject IDs, and epoch indices."""
    model.eval()
    dataset: EpochDataset = loader.dataset
    probabilities: list[np.ndarray] = []
    subject_ids: list[str] = []
    epoch_indices: list[int] = []
    offset = 0

    for batch in loader:
        logits, _ = _forward(model, batch, model_name, device)
        batch_probabilities = torch.sigmoid(logits).cpu().numpy()
        batch_size = len(batch_probabilities)
        indices = dataset.indices[offset : offset + batch_size]

        probabilities.append(batch_probabilities)
        subject_ids.extend(dataset.bundle.subject_ids[indices].tolist())
        epoch_indices.extend(indices.tolist())
        offset += batch_size

    if not probabilities:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(probabilities),
        np.asarray(subject_ids),
        np.asarray(epoch_indices, dtype=int),
    )


def aggregate_subject_probs(
    subject_ids: np.ndarray,
    probabilities: np.ndarray,
) -> pd.Series:
    """Average epoch probabilities per subject."""
    frame = pd.DataFrame(
        {"subject_id": subject_ids, "probability": probabilities}
    )
    return frame.groupby("subject_id", sort=False)["probability"].mean()


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
) -> float:
    """Mean loss over a loader."""
    model.eval()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        logits, labels = _forward(model, batch, model_name, device)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * len(labels)
        total_items += len(labels)

    return total_loss / max(total_items, 1)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def train_one_fold(
    bundle: FeatureBundle,
    train_subjects: np.ndarray,
    val_subjects: np.ndarray,
    test_subjects: np.ndarray,
    config: Config,
    device: torch.device,
    fold: int,
) -> dict:
    """Train one outer fold with subject-disjoint train/val/test sets."""
    model_name = config.model.name.lower()
    train_indices = indices_for_subjects(bundle, train_subjects)
    val_indices = indices_for_subjects(bundle, val_subjects)
    test_indices = indices_for_subjects(bundle, test_subjects)

    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        raise ValueError(f"Fold {fold} has an empty split")

    train_dataset = EpochDataset(bundle, train_indices, model_name)
    val_dataset = EpochDataset(bundle, val_indices, model_name)
    test_dataset = EpochDataset(bundle, test_indices, model_name)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    model = build_model(
        config.model,
        n_channels=bundle.n_channels,
        n_features=bundle.n_node_features,
        n_times=bundle.n_times,
    ).to(device)

    train_labels = bundle.labels[train_indices]
    n_positive = float(train_labels.sum())
    n_negative = float(len(train_labels) - n_positive)
    pos_weight = torch.tensor(
        [n_negative / max(n_positive, 1.0)], dtype=torch.float32, device=device
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, config.training.patience // 3),
    )

    best_auc = -float("inf")
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, labels = _forward(model, batch, model_name, device)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)

        train_loss = running_loss / max(len(train_dataset), 1)

        val_loss = evaluate_loss(
            model, val_loader, criterion, device, model_name
        )
        val_probs, val_subject_ids, _ = predict(
            model, val_loader, device, model_name
        )
        val_subject_probs = aggregate_subject_probs(val_subject_ids, val_probs)
        val_y = np.asarray(
            [bundle.subject_labels[str(s)] for s in val_subject_probs.index],
            dtype=int,
        )
        val_auc = _safe_auc(val_y, val_subject_probs.to_numpy())

        scheduler.step(val_loss)

        # Prefer higher AUROC; among equally good epochs, prefer lower loss.
        if best_state is None:
            improved = True
        elif np.isfinite(val_auc) and val_auc > best_auc + 1e-4:
            improved = True
        elif (
            np.isfinite(val_auc)
            and abs(val_auc - best_auc) <= 1e-4
            and val_loss < best_loss - 1e-4
        ):
            improved = True
        else:
            improved = False

        if improved:
            if np.isfinite(val_auc):
                best_auc = val_auc
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_auc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if epoch == 1 or epoch % 5 == 0 or bad_epochs == 0:
            print(
                f"    epoch {epoch:03d} | loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | val AUROC {val_auc:.3f}"
            )

        if bad_epochs >= config.training.patience:
            print(f"    early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold} failed to produce a checkpoint")
    model.load_state_dict(best_state)

    val_probs, val_subject_ids, _ = predict(
        model, val_loader, device, model_name
    )
    val_subject_probs = aggregate_subject_probs(val_subject_ids, val_probs)
    val_y = np.asarray(
        [bundle.subject_labels[str(s)] for s in val_subject_probs.index],
        dtype=int,
    )
    threshold = float(youden_threshold(val_y, val_subject_probs.to_numpy()))

    test_probs, test_subject_ids, test_epoch_indices = predict(
        model, test_loader, device, model_name
    )

    return {
        "test_probs": test_probs,
        "test_subject_ids": test_subject_ids,
        "test_epoch_indices": test_epoch_indices,
        "history": history,
        "best_val_auc": best_auc,
        "threshold": threshold,
    }


def run_cross_validation(bundle: FeatureBundle, config: Config) -> dict[str, pd.DataFrame]:
    """Run subject-wise CV and return out-of-fold predictions and metrics."""
    set_seed(config.training.seed)
    device = resolve_device(config.training.device)
    print(f"Device: {device}")
    print(
        f"Subjects: {len(bundle.subject_labels)} | epochs: {len(bundle.labels)} | "
        f"channels: {bundle.n_channels} | samples/epoch: {bundle.n_times}"
    )

    folds = subject_folds(
        bundle,
        n_splits=config.training.n_splits,
        val_size=config.training.val_size,
        seed=config.training.seed,
    )

    oof_rows: list[dict] = []
    epoch_rows: list[dict] = []
    fold_rows: list[dict] = []
    history_rows: list[dict] = []

    for fold, (train_subjects, val_subjects, test_subjects) in enumerate(folds, 1):
        print(
            f"\nFold {fold}/{len(folds)} | "
            f"train {len(train_subjects)} | val {len(val_subjects)} | "
            f"test {len(test_subjects)}"
        )
        result = train_one_fold(
            bundle,
            train_subjects,
            val_subjects,
            test_subjects,
            config,
            device,
            fold,
        )

        subject_probs = aggregate_subject_probs(
            result["test_subject_ids"], result["test_probs"]
        )

        for subject_id, probability in subject_probs.items():
            oof_rows.append(
                {
                    "subject_id": str(subject_id),
                    "fold": fold,
                    "y_true": int(bundle.subject_labels[str(subject_id)]),
                    "y_prob": float(probability),
                    "y_pred": int(probability >= result["threshold"]),
                    "threshold": float(result["threshold"]),
                }
            )

        for subject_id, epoch_index, probability in zip(
            result["test_subject_ids"],
            result["test_epoch_indices"],
            result["test_probs"],
        ):
            epoch_rows.append(
                {
                    "subject_id": str(subject_id),
                    "fold": fold,
                    "epoch_index": int(epoch_index),
                    "y_true": int(bundle.subject_labels[str(subject_id)]),
                    "y_prob": float(probability),
                }
            )

        fold_y = np.asarray(
            [bundle.subject_labels[str(s)] for s in subject_probs.index], dtype=int
        )
        fold_metrics = binary_metrics(
            fold_y,
            subject_probs.to_numpy(),
            threshold=result["threshold"],
            n_boot=500,
            seed=config.training.seed + fold,
        )
        fold_metrics.update(
            {
                "fold": fold,
                "threshold": float(result["threshold"]),
                "n_train_subjects": int(len(train_subjects)),
                "n_val_subjects": int(len(val_subjects)),
                "n_test_subjects": int(len(test_subjects)),
                "best_val_auroc": float(result["best_val_auc"]),
            }
        )
        fold_rows.append(fold_metrics)
        history_rows.extend(result["history"])

        print(
            f"  fold {fold}: AUROC {fold_metrics['auroc']:.3f} | "
            f"accuracy {fold_metrics['accuracy']:.3f} | "
            f"threshold {result['threshold']:.2f} | "
            f"best val AUROC {result['best_val_auc']:.3f}"
        )

    oof = pd.DataFrame(oof_rows).sort_values("subject_id").reset_index(drop=True)
    epoch_predictions = pd.DataFrame(epoch_rows)
    folds_frame = pd.DataFrame(fold_rows)
    history = pd.DataFrame(history_rows)

    if not oof["subject_id"].is_unique:
        duplicates = oof.loc[oof["subject_id"].duplicated(), "subject_id"].tolist()
        raise RuntimeError(f"Subjects appear in multiple test folds: {duplicates}")

    return {
        "oof": oof,
        "epoch": epoch_predictions,
        "folds": folds_frame,
        "history": history,
    }
