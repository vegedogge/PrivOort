import copy
import math
import time
from typing import Dict, Tuple, Sized, Callable, cast
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import he

class Client:
    def __init__(self, cid: int, train_loader, cfg: Dict, device: torch.device, model_fn: Callable[[], nn.Module]):
        self.cid = cid
        self.train_loader = train_loader
        self.cfg = cfg
        self.device = device
        self.model: nn.Module = model_fn().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.ctx = he.get_ckks_context(
            dir_path = cfg["he"]["ckks_context_dir"],
            name = cfg["he"].get("context_name", "context"),
            poly_modulus_degree = cfg["he"]["poly_modulus_degree"],
            coeff_mod_bit_sizes = cfg["he"]["coeff_mod_bit_sizes"],
            global_scale = cfg["he"]["global_scale"]
        )
        self.shapes = None
        self.sizes = None

    def load_encrypted_global(self, enc_blob):
        vec = he.decrypt_vector(enc_blob, self.ctx)
        state = he.rebuild_weights(vec, self.shapes, self.sizes)
        self.model.load_state_dict(state)

    def train(self, enc_global_blob, shapes, sizes):
        self.shapes = shapes
        self.sizes = sizes
        self.load_encrypted_global(enc_global_blob)
        model = self.model
        opt_cfg = self.cfg["parameters"]["optimizer"]
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.0),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
        model.train()
        num_samples = len(cast(Sized, self.train_loader.dataset))
        sum_sq_loss = 0.0
        start = time.time()

        for _ in range(self.cfg["trainer"]["epochs"]):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                per_sample_loss = self.criterion(outputs, target)
                loss = per_sample_loss.mean()
                loss.backward()
                optimizer.step()
                sum_sq_loss += (per_sample_loss.detach() ** 2).sum().item()
                
        train_time = time.time() - start
        mean_sq_loss = sum_sq_loss / num_samples if num_samples else 0.0
        statistical_utility = (num_samples * math.sqrt(mean_sq_loss) if num_samples and mean_sq_loss > 0 else 0.0)
        
        flat, _, _ =he.flatten_weights(model.state_dict())
        env_vec = he.encrypt_vector(flat, self.ctx)
        #return env_vec, num_samples, statistical_utility, train_time
        return {
            "encrypted_vector": env_vec,
            "num_samples": num_samples,
            "statistical_utility": statistical_utility,
            "training_time": train_time,
        }
