import random
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# 使用 dirichlet分布模拟非独立同分布
# alpha用来决定异构程度，等于1时为均匀分布
def _dirichlet_split(labels, num_clients: int, alpha: float, seed: int):
    rng = random.Random(seed)
    label2idx = {}
    for idx, y in enumerate(labels):
        label2idx.setdefault(int(y), []).append(idx)
    for idxs in label2idx.values():
        rng.shuffle(idxs)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for _, idxs in label2idx.items():
        proportions = (
            torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha] * num_clients))
            .sample()
            .tolist()
        )
        split_points = [0]
        for p in proportions:
            split_points.append(split_points[-1] + int(p * len(idxs)))
        split_points[-1] = len(idxs)
        for cid in range(num_clients):
            client_indices[cid].extend(idxs[split_points[cid] : split_points[cid + 1]])
    return client_indices


def get_partitions(cfg: Dict):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        root=cfg["data"]["data_path"],
        train=True,
        download=cfg["data"].get("download", True),
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root=cfg["data"]["data_path"],
        train=False,
        download=cfg["data"].get("download", True),
        transform=transform,
    )
    num_clients = cfg["clients"]["total_clients"]
    seed = cfg["clients"]["random_seed"]
    if cfg["data"]["sampler"] == "iid":
        indices = list(range(len(train_ds)))
        random.Random(seed).shuffle(indices)
        chunk = len(indices) // num_clients
        client_indices = [
            indices[i * chunk : (i + 1) * chunk] for i in range(num_clients)
        ]
    else:
        alpha = cfg["data"].get("dirichlet_alpha", 0.5)
        client_indices = _dirichlet_split(train_ds.targets, num_clients, alpha, seed)
    client_loaders = {}
    for cid, idxs in enumerate(client_indices):
        subset = Subset(train_ds, idxs)
        client_loaders[cid] = DataLoader(
            subset,
            batch_size=cfg["trainer"]["batch_size"],
            shuffle=True,
            num_workers=0,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"].get("test_batch_size", 256),
        shuffle=False,
        num_workers=0,
    )
    return client_loaders, test_loader
