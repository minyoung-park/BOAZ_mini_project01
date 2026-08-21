"""Mini-batch PyTorch/CUDA BPR-MF baseline for Spotify's MPD."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bpr_mf import load_mpd


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int):
        super().__init__()
        # Sparse gradients avoid allocating a dense gradient for every MPD embedding.
        self.user_embedding = nn.Embedding(n_users, factors, sparse=True)
        self.item_embedding = nn.Embedding(n_items, factors, sparse=True)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):
        user = self.user_embedding(users)
        positive = self.item_embedding(positives)
        negative = self.item_embedding(negatives)
        margin = (user * (positive - negative)).sum(dim=1)
        return margin, user, positive, negative


def build_interactions(histories: list[np.ndarray]):
    lengths = np.fromiter((len(x) for x in histories), dtype=np.int64)
    users = np.repeat(np.arange(len(histories), dtype=np.int64), lengths)
    items = np.concatenate(histories).astype(np.int64, copy=False)
    return users, items


def build_seen_keys(histories: list[np.ndarray], n_items: int, device: torch.device):
    """Encode sorted (user, item) pairs as user * n_items + item."""
    keys = np.concatenate([
        np.sort(items.astype(np.int64, copy=False)) + user * n_items
        for user, items in enumerate(histories)
    ])
    # Histories are emitted in user order and sorted within each user.
    return torch.from_numpy(keys).to(device=device, non_blocking=True)


def in_seen(query: torch.Tensor, seen_keys: torch.Tensor):
    positions = torch.searchsorted(seen_keys, query)
    safe_positions = positions.clamp_max(len(seen_keys) - 1)
    return (positions < len(seen_keys)) & (seen_keys[safe_positions] == query)


@torch.no_grad()
def sample_unseen(
    users: torch.Tensor,
    n_samples: int,
    n_items: int,
    seen_keys: torch.Tensor,
    generator: torch.Generator,
    targets: torch.Tensor | None = None,
):
    shape = (len(users), n_samples)
    negatives = torch.randint(n_items, shape, device=users.device, generator=generator)
    user_grid = users[:, None].expand(shape)
    invalid = in_seen(user_grid * n_items + negatives, seen_keys)
    if targets is not None:
        invalid |= negatives.eq(targets[:, None])

    while invalid.any().item():
        negatives[invalid] = torch.randint(
            n_items, (int(invalid.sum().item()),), device=users.device, generator=generator
        )
        invalid = in_seen(user_grid * n_items + negatives, seen_keys)
        if targets is not None:
            invalid |= negatives.eq(targets[:, None])
    return negatives


def train(
    model: BPRMF,
    histories: list[np.ndarray],
    n_items: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    reg: float,
    samples_per_epoch: int | None,
    seed: int,
    cpu_interactions: bool,
):
    users_np, items_np = build_interactions(histories)
    storage_device = torch.device("cpu") if cpu_interactions else device
    interaction_users = torch.from_numpy(users_np).to(storage_device)
    interaction_items = torch.from_numpy(items_np).to(storage_device)
    seen_keys = build_seen_keys(histories, n_items, device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
    generator = torch.Generator(device=device).manual_seed(seed)
    cpu_generator = torch.Generator().manual_seed(seed)
    epoch_samples = len(items_np) if samples_per_epoch is None else samples_per_epoch
    n_batches = math.ceil(epoch_samples / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        processed = 0
        for batch_idx in range(n_batches):
            size = min(batch_size, epoch_samples - batch_idx * batch_size)
            if cpu_interactions:
                indices = torch.randint(len(items_np), (size,), generator=cpu_generator)
                users = interaction_users[indices].to(device, non_blocking=True)
                positives = interaction_items[indices].to(device, non_blocking=True)
            else:
                indices = torch.randint(len(items_np), (size,), device=device, generator=generator)
                users = interaction_users[indices]
                positives = interaction_items[indices]
            negatives = sample_unseen(users, 1, n_items, seen_keys, generator).squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            margin, user, positive, negative = model(users, positives, negatives)
            ranking_loss = -F.logsigmoid(margin).mean()
            l2 = (user.square().sum(1) + positive.square().sum(1) + negative.square().sum(1)).mean()
            loss = ranking_loss + reg * l2
            loss.backward()
            optimizer.step()
            total_loss += float(ranking_loss.detach()) * size
            processed += size
        print(f"epoch={epoch:02d} bpr_loss={total_loss / processed:.5f}")
    return seen_keys


@torch.no_grad()
def evaluate(
    model: BPRMF,
    histories: list[np.ndarray],
    targets_np: np.ndarray,
    n_items: int,
    device: torch.device,
    k: int,
    n_negatives: int,
    batch_size: int,
    seed: int,
):
    model.eval()
    seen_keys = build_seen_keys(histories, n_items, device)
    targets_all = torch.from_numpy(targets_np.astype(np.int64, copy=False))
    generator = torch.Generator(device=device).manual_seed(seed)
    hits = ndcg = auc = 0.0
    count = len(histories)

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        users = torch.arange(start, end, device=device)
        targets = targets_all[start:end].to(device, non_blocking=True)
        negatives = sample_unseen(users, n_negatives, n_items, seen_keys, generator, targets)
        user = model.user_embedding(users)
        positive_score = (user * model.item_embedding(targets)).sum(1)
        negative_score = (user[:, None, :] * model.item_embedding(negatives)).sum(2)
        rank = 1 + (negative_score > positive_score[:, None]).sum(1)
        hits += float((rank <= k).sum())
        ndcg += float(torch.where(rank <= k, 1.0 / torch.log2(rank.float() + 1), 0.0).sum())
        auc += float((negative_score < positive_score[:, None]).float().mean(1).sum())

    return {
        f"hit_rate@{k}": hits / count,
        f"ndcg@{k}": ndcg / count,
        "sampled_auc": auc / count,
        "users": count,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train mini-batch BPR-MF on MPD with PyTorch/CUDA")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bpr_mf_cuda.pt"))
    parser.add_argument("--max-slices", type=int, default=10, help="Use 0 for all slices")
    parser.add_argument("--max-playlists", type=int, default=None)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.0025)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-negatives", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument(
        "--cpu-interactions", action="store_true",
        help="Keep sampled training pairs on CPU to save VRAM (recommended for the full MPD)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(name)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    else:
        print(f"device={device} torch={torch.__version__}")

    train_history, validation, test, playlist_ids, track_uris = load_mpd(
        args.data_dir, args.max_slices or None, args.max_playlists
    )
    if not train_history:
        raise ValueError("No eligible playlists found (each needs at least three unique tracks).")
    n_users, n_items = len(train_history), len(track_uris)
    print(f"users={n_users:,} items={n_items:,} train_interactions={sum(map(len, train_history)):,}")

    model = BPRMF(n_users, n_items, args.factors).to(device)
    train(
        model, train_history, n_items, device, args.epochs, args.batch_size, args.lr, args.reg,
        args.samples_per_epoch, args.seed, args.cpu_interactions,
    )
    validation_metrics = evaluate(
        model, train_history, validation, n_items, device, args.top_k, args.eval_negatives,
        args.eval_batch_size, args.seed,
    )
    print("validation", validation_metrics)
    test_history = [np.append(items, val).astype(np.int32) for items, val in zip(train_history, validation)]
    test_metrics = evaluate(
        model, test_history, test, n_items, device, args.top_k, args.eval_negatives,
        args.eval_batch_size, args.seed + 1,
    )
    print("test", test_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({
        "model_state_dict": state,
        "playlist_ids": playlist_ids,
        "track_uris": track_uris,
        "config": {**vars(args), "data_dir": str(args.data_dir), "output": str(args.output)},
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
