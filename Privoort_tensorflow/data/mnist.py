import random
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import numpy as np
import tensorflow as tf




# 使用 dirichlet分布模拟非独立同分布
# alpha用来决定异构程度，等于1时为均匀分布
def _dirichlet_split(labels, num_clients: int, alpha: float, seed: int):
    rng = torch.Generator().manual_seed(seed)
    label2idx = {}
    for idx, y in enumerate(labels):
        label2idx.setdefault(int(y), []).append(idx)
    for v in label2idx.values():
        random.Random(seed).shuffle(v)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for _, idxs in label2idx.items():
        proportions = (
            torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha] * num_clients))
        ).sample().tolist()
        split_points = [0]
        for p in proportions:
            split_points.append(split_points[-1] + int(p * len(idxs)))
        split_points[-1] = len(idxs)
        for cid in range(num_clients):
            client_indices[cid].extend(idxs[split_points[cid] : split_points[cid + 1]])
    return client_indices

def make_ds(x, y, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle and len(x) > 0:
        ds = ds.shuffle(buffer_size = max(1, len(x)), reshuffle_each_iteration = True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_partitions(cfg: Dict):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype("float32")[..., None]
    x_test = (x_test / 255.0).astype("float32")[..., None]
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    num_clients = cfg["clients"]["total_clients"]
    seed = cfg["clients"]["random_seed"]
    if cfg["data"]["sampler"] == "iid":
        indices = list(range(len(x_train)))
        random.Random(seed).shuffle(indices)
        chunk = len(indices) // num_clients
        client_indices = [
            indices[i * chunk : (i + 1) * chunk] for i in range(num_clients)
        ]
    else:
        alpha = cfg["data"].get("dirichlet_alpha", 0.5)
        client_indices = _dirichlet_split(y_train, num_clients, alpha, seed)
    client_datasets = {}
    bs = cfg["trainer"]["batch_size"]
    for cid, idxs in enumerate(client_indices):
        client_datasets[cid] = make_ds(x_train[idxs], y_train[idxs], batch_size = bs, shuffle=True)

    test_bs = cfg["data"].get("test_batch_size", 256)
    test_ds = make_ds(x_test, y_test, batch_size = test_bs, shuffle=False)
    return client_datasets, test_ds
