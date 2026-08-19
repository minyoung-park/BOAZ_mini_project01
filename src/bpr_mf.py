"""Spotify MPD를 위한 미니배치 PyTorch/CUDA BPR-MF 베이스라인."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def load_mpd(data_dir: Path, max_slices: int | None, max_playlists: int | None):
    """MPD JSON slice를 읽고 플레이리스트별 학습·검증·테스트 데이터를 만든다."""
    files = sorted(
        data_dir.glob("mpd.slice.*.json"),
        key=lambda path: int(path.name.split(".")[2].split("-")[0]),
    )
    if max_slices is not None:
        files = files[:max_slices]

    track_to_id: dict[str, int] = {}
    id_to_track: list[str] = []
    train_history: list[np.ndarray] = []
    validation: list[int] = []
    test: list[int] = []
    playlist_ids: list[int] = []

    for path in files:
        with path.open(encoding="utf-8") as file:
            playlists = json.load(file)["playlists"]

        for playlist in playlists:
            # BPR의 암시적 피드백을 이진 상호작용으로 보기 위해 같은 트랙은 한 번만 남긴다.
            ordered_tracks = sorted(playlist["tracks"], key=lambda track: track["pos"])
            track_uris = list(dict.fromkeys(track["track_uri"] for track in ordered_tracks))
            if len(track_uris) < 3:
                continue

            item_ids = []
            for track_uri in track_uris:
                if track_uri not in track_to_id:
                    track_to_id[track_uri] = len(id_to_track)
                    id_to_track.append(track_uri)
                item_ids.append(track_to_id[track_uri])

            # 마지막에서 두 번째 트랙은 검증, 마지막 트랙은 테스트 정답으로 남겨둔다.
            train_history.append(np.asarray(item_ids[:-2], dtype=np.int32))
            validation.append(item_ids[-2])
            test.append(item_ids[-1])
            playlist_ids.append(int(playlist["pid"]))

            if max_playlists is not None and len(train_history) >= max_playlists:
                return (
                    train_history,
                    np.asarray(validation),
                    np.asarray(test),
                    playlist_ids,
                    id_to_track,
                )

    return (
        train_history,
        np.asarray(validation),
        np.asarray(test),
        playlist_ids,
        id_to_track,
    )


class BPRMF(nn.Module):
    """플레이리스트와 트랙 임베딩의 내적으로 선호도를 계산하는 BPR-MF 모델."""

    def __init__(self, n_users: int, n_items: int, factors: int):
        super().__init__()
        # 희소 기울기를 사용해 MPD 전체 임베딩에 대한 밀집 기울기 할당을 피한다.
        self.user_embedding = nn.Embedding(n_users, factors, sparse=True)
        self.item_embedding = nn.Embedding(n_items, factors, sparse=True)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ):
        user = self.user_embedding(users)
        positive = self.item_embedding(positives)
        negative = self.item_embedding(negatives)
        margin = (user * (positive - negative)).sum(dim=1)
        return margin, user, positive, negative


def build_interactions(histories: list[np.ndarray]):
    """플레이리스트별 이력을 평탄한 (사용자, 아이템) 배열로 변환한다."""
    lengths = np.fromiter((len(items) for items in histories), dtype=np.int64)
    users = np.repeat(np.arange(len(histories), dtype=np.int64), lengths)
    items = np.concatenate(histories).astype(np.int64, copy=False)
    return users, items


def build_seen_keys(histories: list[np.ndarray], n_items: int, device: torch.device):
    """관찰된 (사용자, 아이템) 쌍을 정렬된 정수 키로 변환한다."""
    keys = np.concatenate(
        [
            np.sort(items.astype(np.int64, copy=False)) + user * n_items
            for user, items in enumerate(histories)
        ]
    )
    return torch.from_numpy(keys).to(device=device, non_blocking=True)


def in_seen(query: torch.Tensor, seen_keys: torch.Tensor):
    """정수 키가 관찰된 상호작용에 포함되는지 이진 탐색으로 검사한다."""
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
    """사용자가 관찰하지 않은 트랙을 GPU에서 negative로 추출한다."""
    shape = (len(users), n_samples)
    negatives = torch.randint(
        n_items, shape, device=users.device, generator=generator
    )
    user_grid = users[:, None].expand(shape)
    invalid = in_seen(user_grid * n_items + negatives, seen_keys)
    if targets is not None:
        invalid |= negatives.eq(targets[:, None])

    # 실제 positive나 평가 정답이 뽑힌 위치만 다시 추출한다.
    while invalid.any().item():
        negatives[invalid] = torch.randint(
            n_items,
            (int(invalid.sum().item()),),
            device=users.device,
            generator=generator,
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
    """미니배치 BPR 손실로 모델을 학습한다."""
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
                indices = torch.randint(
                    len(items_np), (size,), generator=cpu_generator
                )
                users = interaction_users[indices].to(device, non_blocking=True)
                positives = interaction_items[indices].to(device, non_blocking=True)
            else:
                indices = torch.randint(
                    len(items_np), (size,), device=device, generator=generator
                )
                users = interaction_users[indices]
                positives = interaction_items[indices]

            negatives = sample_unseen(
                users, 1, n_items, seen_keys, generator
            ).squeeze(1)
            optimizer.zero_grad(set_to_none=True)
            margin, user, positive, negative = model(users, positives, negatives)
            ranking_loss = -F.logsigmoid(margin).mean()
            l2_loss = (
                user.square().sum(1)
                + positive.square().sum(1)
                + negative.square().sum(1)
            ).mean()
            loss = ranking_loss + reg * l2_loss
            loss.backward()
            optimizer.step()

            total_loss += float(ranking_loss.detach()) * size
            processed += size

        print(f"epoch={epoch:02d} bpr_loss={total_loss / processed:.5f}")


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
    """샘플링한 negative를 이용해 HitRate, NDCG, AUC를 계산한다."""
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
        negatives = sample_unseen(
            users, n_negatives, n_items, seen_keys, generator, targets
        )

        user = model.user_embedding(users)
        positive_score = (user * model.item_embedding(targets)).sum(1)
        negative_score = (
            user[:, None, :] * model.item_embedding(negatives)
        ).sum(2)
        rank = 1 + (negative_score > positive_score[:, None]).sum(1)

        hits += float((rank <= k).sum())
        ndcg += float(
            torch.where(
                rank <= k,
                1.0 / torch.log2(rank.float() + 1),
                0.0,
            ).sum()
        )
        auc += float(
            (negative_score < positive_score[:, None]).float().mean(1).sum()
        )

    return {
        f"hit_rate@{k}": hits / count,
        f"ndcg@{k}": ndcg / count,
        "sampled_auc": auc / count,
        "users": count,
    }


def parse_args():
    """명령행 인자를 정의하고 읽는다."""
    parser = argparse.ArgumentParser(
        description="PyTorch/CUDA로 MPD BPR-MF를 학습합니다."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/bpr_mf_cuda.pt")
    )
    parser.add_argument(
        "--max-slices", type=int, default=10,
        help="사용할 slice 수입니다. 전체 데이터는 0을 지정합니다.",
    )
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
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu", "auto"]
    )
    parser.add_argument(
        "--cpu-interactions",
        action="store_true",
        help="VRAM 절약을 위해 학습 상호작용을 CPU에 보관합니다.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(name: str):
    """요청한 실행 장치를 확인하고 torch 장치 객체를 반환한다."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 PyTorch가 CUDA 장치를 찾지 못했습니다.")
    return torch.device(name)


def main():
    """데이터 로드부터 학습, 평가, 체크포인트 저장까지 실행한다."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        print(
            f"device={torch.cuda.get_device_name(0)} "
            f"torch={torch.__version__} cuda={torch.version.cuda}"
        )
    else:
        print(f"device={device} torch={torch.__version__}")

    train_history, validation, test, playlist_ids, track_uris = load_mpd(
        args.data_dir, args.max_slices or None, args.max_playlists
    )
    if not train_history:
        raise ValueError(
            "학습 가능한 플레이리스트가 없습니다. 데이터 경로와 JSON 파일을 확인하세요."
        )

    n_users = len(train_history)
    n_items = len(track_uris)
    print(
        f"users={n_users:,} items={n_items:,} "
        f"train_interactions={sum(map(len, train_history)):,}"
    )

    model = BPRMF(n_users, n_items, args.factors).to(device)
    train(
        model,
        train_history,
        n_items,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.reg,
        args.samples_per_epoch,
        args.seed,
        args.cpu_interactions,
    )

    validation_metrics = evaluate(
        model,
        train_history,
        validation,
        n_items,
        device,
        args.top_k,
        args.eval_negatives,
        args.eval_batch_size,
        args.seed,
    )
    print("validation", validation_metrics)

    # 테스트 시점에는 검증 트랙도 이미 관찰한 이력으로 간주해 추천 후보에서 제외한다.
    test_history = [
        np.append(items, val).astype(np.int32)
        for items, val in zip(train_history, validation)
    ]
    test_metrics = evaluate(
        model,
        test_history,
        test,
        n_items,
        device,
        args.top_k,
        args.eval_negatives,
        args.eval_batch_size,
        args.seed + 1,
    )
    print("test", test_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 다른 장치에서도 불러올 수 있도록 체크포인트의 가중치는 CPU 텐서로 저장한다.
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": state,
            "playlist_ids": playlist_ids,
            "track_uris": track_uris,
            "config": {
                **vars(args),
                "data_dir": str(args.data_dir),
                "output": str(args.output),
            },
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        args.output,
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
