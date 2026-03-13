import argparse
import random
import numpy as np
#import torch
import toml
import tensorflow as tf

from data.mnist import get_partitions
from federated.client import Client
from federated.server import Server
from models.lenet5 import Model
from utils.metrics import MetricsWriter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(path: str):
    return toml.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["clients"]["random_seed"])

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
