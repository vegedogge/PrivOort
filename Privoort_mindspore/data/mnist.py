import random
from typing import Dict, List, Tuple

# import torch
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets, transforms

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import transforms as C
from paddle.vision.datasets import MNIST

# import tensorflow as tf
# import paddle
# from paddle.io import DataLoader, TensorDataset
# from paddle.vision.datasets import MNIST


# 使用 dirichlet分布模拟非独立同分布
# alpha用来决定异构程度，等于1时为均匀分布
def _dirichlet_split(labels, num_clients: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    # rng = torch.Generator().manual_seed(seed)
    label2idx = {}
    for idx, y in enumerate(labels):
        label2idx.setdefault(int(y), []).append(idx)
    for v in label2idx.values():
        rng.shuffle(v)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for _, idxs in label2idx.items():
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = rng.multinomial(len(idxs), proportions)
        offset = 0
        for cid, c in enumerate(counts):
            if c > 0:
                client_indices[cid].extend(idxs[offset : offset + c])
            offset += c
    # 当出现空分片时增加一条数据
    n_total = len(labels)
    for idxs in client_indices:
        if len(idxs) == 0:
            idxs.append(rng.integers(low=0, high=n_total))
    return client_indices


def _to_nchw(x):
    if x.ndim == 2:
        x = x.reshape(-1, 28, 28)
    if x.ndim == 3:
        x = x[:, None, :, :]
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = np.transpose(x, (0, 3, 1, 2))
    return x

def make_ds(x, y, batch_size, shuffle=True):
    ds_set = ds.NumpySlicesDataset((x, y), column_names = ["image", "label"], shuffle = shuffle)
    ds_set = ds_set.batch(batch_size, drop_remainder = False)
    return ds_set


def get_partitions(cfg: Dict):
    train_ds = MNIST(mode="train", backend="cv2")
    test_ds = MNIST(mode="test", backend="cv2")
    x_train = np.array(train_ds.images, dtype="float32") / 255.0
    y_train = np.array(train_ds.labels, dtype="int64").reshape(-1)
    x_test = np.array(test_ds.images, dtype="float32") / 255.0
    y_test = np.array(test_ds.labels, dtype="int64").reshape(-1)
    if x_train.ndim == 2:  # 扁平 784
        x_train = x_train.reshape(-1, 28, 28)
        x_test = x_test.reshape(-1, 28, 28)
    if x_train.ndim == 3:  # (N,28,28)
        x_train = x_train[:, None, :, :]
        x_test = x_test[:, None, :, :]
    elif x_train.ndim == 4 and x_train.shape[-1] == 1:  # (N,28,28,1)
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        x_test = np.transpose(x_test, (0, 3, 1, 2))

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

    # 防止出现空分片
    rng = random.Random(seed)
    for idxs in client_indices:
        if len(idxs) == 0:
            idxs.append(rng.randrange(len(x_train)))

    bs = cfg["trainer"]["batch_size"]
    client_datasets = {
        cid: make_ds(x_train[idxs], y_train[idxs], batch_size=bs, shuffle=True)
        for cid, idxs in enumerate(client_indices)
    }

    test_bs = cfg["data"].get("test_batch_size", 256)
    test_ds = make_ds(x_test, y_test, batch_size=test_bs, shuffle=False)
    return client_datasets, test_ds
