import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.mnist import get_partitions

cfg = {
    "data": {
        "data_path": "./data",      # MNIST 下载/缓存目录
        "download": True,
        "sampler": "dirichlet",           # 换成 "dirichlet" 并设置 dirichlet_alpha 测试非 IID
        "dirichlet_alpha": 0.1,
        "test_batch_size": 256,
    },
    "clients": {"total_clients": 3, "random_seed": 42},
    "trainer": {"batch_size": 32},
}

train_loaders, test_loader = get_partitions(cfg)
print("train sizes:", {cid: len(dl.dataset) for cid, dl in train_loaders.items()})
xb, yb = next(iter(train_loaders[0]))
print("first train batch shapes:", xb.shape, yb.shape)
xb_t, yb_t = next(iter(test_loader))
print("first test batch shapes:", xb_t.shape, yb_t.shape)
