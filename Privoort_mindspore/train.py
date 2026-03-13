import argparse
import random
import numpy as np
import os, warnings, logging

warnings.filterwarnings("ignore")              # 全局忽略 Python warnings
os.environ["GLOG_minloglevel"] = "3"           # MindSpore 日志只输出 ERROR
os.environ["MS_SUPPRESS_LOG"] = "1"            # 抑制 MindSpore 日志输出
logging.disable(logging.CRITICAL)              # 禁用所有 logging 输出
#import torch
import toml
#import paddle
#import tensorflow as tf
import mindspore as ms

ms.set_device("CPU")                
ms.set_context(mode=ms.GRAPH_MODE)  # 或 PYNATIVE_MODE

from data.mnist import get_partitions
from federated.client import Client
from federated.server import Server
from models.lenet5 import Model
from utils.metrics import MetricsWriter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def load_config(path: str):
    return toml.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["clients"]["random_seed"])
    ms.set_context(mode = ms.GRAPH_MODE, device_target = "CPU")

    model_fn = lambda: Model(num_classes=cfg["model"]["num_classes"])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Model(num_classes=cfg["model"]["num_classes"])

    client_dataset, test_dataset = get_partitions(cfg)
    clients = {cid: Client(cid, ds, cfg, model_fn = model_fn) for cid, ds in client_dataset.items()}
    writer = MetricsWriter(cfg["results"]["metrics_csv"])

    server = Server(model_fn, clients, test_dataset, cfg)
    server.run(writer)


if __name__ == "__main__":
    main()
