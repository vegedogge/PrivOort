import copy
import math
import time
from typing import Dict, Tuple, Sized, Callable, cast
#import torch
#from torch import nn, optim
#from torch.utils.data import DataLoader
#import tensorflow as tf
import paddle
import paddle.nn.functional as F
from utils import he

class Client:
    def __init__(self, cid: int, train_loader, cfg: Dict, model_fn: Callable[[], paddle.nn.Layer]):
        self.cid = cid
        self.train_loader = train_loader
        self.cfg = cfg
        self.model = model_fn()
        self.criterion = paddle.nn.CrossEntropyLoss(reduction = "none")
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
        weights = he.rebuild_weights(vec, self.shapes, self.sizes)
        for p, w in zip(self.model.parameters(), weights):
            p.set_value(paddle.to_tensor(w, dtype = 'float32'))

    def train(self, enc_global_blob, shapes, sizes):
        self.shapes = shapes
        self.sizes = sizes
        self.load_encrypted_global(enc_global_blob)

        opt_cfg = self.cfg["parameters"]["optimizer"]
        if opt_cfg.get("momentum", 0.0) > 0:
            optimizer = paddle.optimizer.Momentum(
                learning_rate=opt_cfg["lr"],
                momentum=opt_cfg["momentum"],
                weight_decay=opt_cfg.get("weight_decay", 0.0),
                parameters=self.model.parameters(),
            )
        else:
            optimizer = paddle.optimizer.SGD(
                learning_rate=opt_cfg["lr"],
                weight_decay=opt_cfg.get("weight_decay", 0.0),
                parameters=self.model.parameters(),
            )


        num_samples = 0
        sum_sq_loss = 0.0
        start = time.time()

        for _ in range(self.cfg["trainer"]["epochs"]):
            for x, y in self.train_loader:
                y = paddle.reshape(y, [-1])
                logits = self.model(x)
                per_sample_loss = self.criterion(logits, y)
                loss = per_sample_loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                num_samples += x.shape[0]
                sum_sq_loss += float((per_sample_loss ** 2).detach().sum().numpy())
            # for data, target in self.train_loader:
            #     data, target = data.to(self.device), target.to(self.device)
            #     optimizer.zero_grad()
            #     outputs = model(data)
            #     per_sample_loss = self.criterion(outputs, target)
            #     loss = per_sample_loss.mean()
            #     loss.backward()
            #     optimizer.step()
            #     sum_sq_loss += (per_sample_loss.detach() ** 2).sum().item()
                
        train_time = time.time() - start
        mean_sq_loss = sum_sq_loss / num_samples if num_samples else 0.0
        statistical_utility = (num_samples * math.sqrt(mean_sq_loss) if num_samples and mean_sq_loss > 0 else 0.0)
        
        flat, shapes, sizes = he.flatten_weights(self.model)
        env_vec = he.encrypt_vector(flat, self.ctx)
        #return env_vec, num_samples, statistical_utility, train_time
        return {
            "encrypted_vector": env_vec,
            "num_samples": num_samples,
            "statistical_utility": statistical_utility,
            "training_time": train_time,
            "shapes": shapes,
            "sizes": sizes
        }
