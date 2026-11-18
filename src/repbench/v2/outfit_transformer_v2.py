"""Image-only OutfitTransformer baseline on frozen item embeddings (HANDOFF_V2 §6.2).

Reuses the fixed v1 architecture and focal loss
(`repbench.models.outfit_transformer`, unchanged) with the v2 training protocol:
learning rate as a parameter (grid {1e-4, 3e-4}), at most 100 epochs, early
stopping on validation CP ROC-AUC with patience 10, fp32 training (no autocast),
deterministic algorithms and seeded data order.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from repbench.models.outfit_transformer import OutfitTransformer, focal_binary_cross_entropy

LR_GRID = (1e-4, 3e-4)


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    seed: int
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 50
    weight_decay: float = 1e-4
    model_dim: int = 64
    min_delta: float = 1e-7
    loss: str = "focal BCE gamma=2 alpha=0.25 (v1)"
    optimizer: str = "AdamW"
    precision: str = "float32"


class EmbeddingTable:
    """Dense view of an item_id -> vector store with a trailing all-zero padding row."""

    def __init__(self, store):
        self.ids = sorted(store)
        if not self.ids:
            raise ValueError("Empty embedding store")
        self.index = {item: i for i, item in enumerate(self.ids)}
        matrix = np.stack([np.asarray(store[i], dtype=np.float32) for i in self.ids])
        self.dim = matrix.shape[1]
        self.pad = len(self.ids)
        self.values = torch.from_numpy(
            np.concatenate([matrix, np.zeros((1, self.dim), np.float32)])
        )

    def encode(self, item_lists, max_length):
        indices = np.full((len(item_lists), max_length), self.pad, dtype=np.int64)
        mask = np.ones((len(item_lists), max_length), dtype=bool)
        missing = set()
        for row, items in enumerate(item_lists):
            if not 1 <= len(items) <= max_length:
                raise ValueError(f"Unsupported outfit length {len(items)} (max {max_length})")
            for col, item in enumerate(items):
                if item not in self.index:
                    missing.add(item)
                    continue
                indices[row, col] = self.index[item]
            mask[row, : len(items)] = False
        if missing:
            raise KeyError(f"{len(missing)} items lack embeddings, e.g. {sorted(missing)[:3]}")
        return torch.from_numpy(indices), torch.from_numpy(mask)


def set_determinism():
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _forward(model, table, indices, mask, device):
    values = table.values[indices].to(device)
    return model(values, mask.to(device))


def predict(model, table, encoded, *, batch_size=512, device="cpu"):
    indices, mask = encoded
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            logits = _forward(
                model,
                table,
                indices[start : start + batch_size],
                mask[start : start + batch_size],
                device,
            )
            outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros(0, np.float32)


def max_length_of(*item_lists_groups):
    return max(len(items) for group in item_lists_groups for items in group)


@dataclass
class TrainResult:
    model: torch.nn.Module
    best_state: dict
    history: list
    best_epoch: int
    best_valid_auc: float
    epochs_run: int
    early_stopped: bool
    training_seconds: float
    config: dict


def train(config, table, train_examples, valid_examples, *, max_length, device="cpu", log=None):
    """Train on CP examples; select the epoch with the best validation ROC-AUC."""
    device = torch.device(device)
    set_determinism()
    torch.manual_seed(config.seed)
    np_rng = np.random.default_rng(config.seed)
    train_encoded = table.encode([e.item_ids for e in train_examples], max_length)
    valid_encoded = table.encode([e.item_ids for e in valid_examples], max_length)
    y_train = torch.tensor([e.label for e in train_examples], dtype=torch.float32)
    y_valid = np.asarray([e.label for e in valid_examples])
    if len(set(y_train.tolist())) != 2 or len(set(y_valid.tolist())) != 2:
        raise ValueError("Train and valid CP examples must contain both labels")
    model = OutfitTransformer(input_dim=table.dim, model_dim=config.model_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    best_auc, best_epoch, stale = -np.inf, 0, 0
    best_state = None
    history = []
    early_stopped = False
    started = time.perf_counter()
    epoch = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.from_numpy(np_rng.permutation(len(y_train)))
        losses = []
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, table, train_encoded[0][rows], train_encoded[1][rows], device)
            loss = focal_binary_cross_entropy(logits, y_train[rows].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        valid_scores = predict(model, table, valid_encoded, device=device)
        valid_auc = float(roc_auc_score(y_valid, valid_scores))
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_auc": valid_auc}
        )
        if valid_auc > best_auc + config.min_delta:
            best_auc, best_epoch, stale = valid_auc, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if log:
            log(f"lr={config.lr} seed={config.seed} epoch={epoch} val_auc={valid_auc:.6f}")
        if stale >= config.patience:
            early_stopped = True
            break
    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        best_state=best_state,
        history=history,
        best_epoch=best_epoch,
        best_valid_auc=float(best_auc),
        epochs_run=epoch,
        early_stopped=early_stopped,
        training_seconds=time.perf_counter() - started,
        config=asdict(config),
    )


def score_cp(model, table, examples, *, max_length, device="cpu"):
    encoded = table.encode([e.item_ids for e in examples], max_length)
    return predict(model, table, encoded, device=device)


def score_fitb(model, table, questions, *, max_length, device="cpu"):
    """Score each candidate as partial outfit + candidate; returns (scores [Q, K], correct)."""
    if not questions:
        return np.zeros((0, 0)), np.zeros(0, bool)
    k = len(questions[0].candidate_item_ids)
    if any(len(q.candidate_item_ids) != k for q in questions):
        raise ValueError("FITB questions must share the candidate count")
    sets = [
        tuple(q.question_item_ids) + (candidate,)
        for q in questions
        for candidate in q.candidate_item_ids
    ]
    scores = predict(model, table, table.encode(sets, max_length), device=device)
    scores = scores.reshape(len(questions), k)
    truth = np.asarray([q.correct_index for q in questions])
    return scores, np.argmax(scores, axis=1) == truth
