import unittest
import torch

from federated.client import Client
from models.lenet5 import Model
from data.mnist import get_partitions


class ClientTrainTest(unittest.TestCase):
    def test_train_returns_metrics_with_lenet5(self):
        torch.manual_seed(0)
        cfg = {
            "data": {
                "data_path": "./data",
                "download": True,
                "sampler": "dirichlet",          # 设置非 IID 
                "dirichlet_alpha": 0.1,          # 设置 dirichlet_alpha
                "test_batch_size": 256,
            },
            "clients": {"total_clients": 3, "random_seed": 42},
            "trainer": {"batch_size": 32, "epochs": 1},
            "parameters": {"optimizer": {"lr": 0.1, "momentum": 0.0, "weight_decay": 0.0}},
        }
        train_loaders, _ = get_partitions(cfg)
        client_sizes = {cid: len(dl.dataset) for cid, dl in train_loaders.items()}
        print("client sample sizes: ", client_sizes)
        loader = train_loaders[0]
        dataset = loader.dataset  # 用于后面 len(dataset)

        device = torch.device("cpu")
        client = Client(cid=0, train_loader=loader, cfg=cfg, device=device)

        model = Model()
        state_dict, sample_count, statistical_utility , train_time = client.train(model)
        print(f"statistical_utility={statistical_utility:.4f}, train_time={train_time:.3f}s, sample_count={sample_count}")

        """ self.assertEqual(sample_count, len(dataset))
        self.assertIsInstance(state_dict, dict)
        self.assertSetEqual(set(state_dict.keys()), set(model.state_dict().keys()))
        self.assertGreaterEqual(statistical_utility, 0.0)
        self.assertGreaterEqual(train_time, 0.0)
 """

#if __name__ == "__main__":
    #unittest.main()