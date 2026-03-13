from typing import Dict, List, Tuple, Callable, Optional, Any

import torch
from torch import nn
import tenseal as ts
from federated.client import Client
from federated.selection import OortSelector
from utils import he


class Server:
    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        clients: Dict[int, Client],
        test_loader,
        cfg: Dict,
    ):
        self.model: nn.Module = model_fn()
        self.clients = clients
        self.test_loader = test_loader
        self.cfg = cfg
        self.selector = OortSelector(
            exploration_factor=cfg["server"]["exploration_factor"],
            desired_duration=cfg["server"]["desired_duration"],
            step_window=cfg["server"]["step_window"],
            penalty=cfg["server"]["penalty"],
            cut_off=cfg["server"].get("cut_off", 0.95),
            blacklist_num=cfg["server"].get("blacklist_num", 10),  # 默认设置为 10
            seed=cfg["clients"]["random_seed"],
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        he_cfg = cfg["he"]
        self.ctx = he.get_ckks_context(
            dir_path = he_cfg["ckks_context_dir"],
            name = he_cfg.get("context_name", "context"),
            poly_modulus_degree = he_cfg["poly_modulus_degree"],
            coeff_mod_bit_sizes = tuple(he_cfg["coeff_mod_bit_sizes"]),
            global_scale = he_cfg["global_scale"],
        )
        flat, self.shapes, self.sizes = he.flatten_weights(self.model.state_dict())
        self.encrypted_global = he.encrypt_vector(flat, self.ctx)
        self.selector.setup(total_clients=len(clients))

    def aggregate(self, enc_updates: List[Dict]):
        if not enc_updates:
            raise ValueError("No client updates to aggregate") 
        total_samples = sum(u["num_samples"] for u in enc_updates)  # 第一个代表的是模型参数，第二个代表的是样本数量
        # new_state = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        # for state, samples in updates:
        #     weight = samples / total_samples
        #     for k, v in state.items():
        #         new_state[k] += v * weight  # 基于样本量的加权平均聚合
        # self.model.load_state_dict(self.model.state_dict())  # 加载当前模型参数
        first = enc_updates[0]
        acc: ts.CKKSVector = he.load_encrypted_vector(first["encrypted_vector"], self.ctx) * (first["num_samples"] / total_samples)
        for u in enc_updates:
            w = u["num_samples"] / total_samples
            vec = he.load_encrypted_vector(u["encrypted_vector"], self.ctx)
            acc = acc + vec * w
        self.encrypted_global = acc.serialize()

    @torch.no_grad()
    def evaluate(self):
        vec = he.decrypt_vector(self.encrypted_global, self.ctx)
        state = he.rebuild_weights(vec, self.shapes, self.sizes)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(self.device)
        correct, total = 0, 0
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        self.model.cpu()
        return correct / total if total else 0.0

    def run(self, writer):
        client_ids = list(self.clients.keys())
        rounds = self.cfg["trainer"]["rounds"]
        target = self.cfg["trainer"]["target_accuracy"]

        for r in range(1, rounds + 1):
            selected = self.selector.select(
                client_ids, self.cfg["clients"]["per_round"], r
            )

            enc_updates = []
            util_updates = []
            for cid in selected:
                result: Dict[str, Any] = self.clients[cid].train(self.encrypted_global, self.shapes, self.sizes)
                enc_updates.append({"encrypted_vector": result["encrypted_vector"], "num_samples": result["num_samples"]})
                # state, samples, stat_util, train_time = self.clients[cid].train(
                #     self.model
                # )
                util_updates.append(
                    {
                        "client_id": cid,
                        "statistical_utility": result["statistical_utility"],
                        "training_time": result["training_time"],
                    }
                )

            self.aggregate(enc_updates)
            acc = self.evaluate()
            self.selector.update(util_updates, r)

            writer.write(
                round=r, accuracy=acc, clients=selected, utilities=util_updates
            )
            if acc >= target:
                break
