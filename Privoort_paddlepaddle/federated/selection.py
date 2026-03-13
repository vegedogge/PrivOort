import random
import logging
import math
from typing import Dict, List, Tuple, Sized, cast
import numpy as np

class OortSelector:
    def __init__(self, *, exploration_factor: float, desired_duration: float, step_window: int, penalty: float, cut_off: float, blacklist_num: int, seed: int = 1):
        self.exploration_factor = exploration_factor
        self.desired_duration = desired_duration
        self.step_window = step_window
        self.penalty = penalty
        self.cut_off = cut_off
        self.blacklist_num = blacklist_num
        self.rng = random.Random(seed)
        self.blacklist: List[int] = []
        self.client_utilities: Dict[int, float] = {}
        self.client_durations: Dict[int, float] = {}
        self.client_last_rounds: Dict[int, int] = {}
        self.client_selected_times: Dict[int, int] = {}
        self.explored_clients: List[int] = []
        self.unexplored_clients: List[int] = []
        self.util_history: List[float] = []
        self.pacer_step = desired_duration


    def setup(self, total_clients: int):
        self.blacklist = []
        self.client_utilities = {cid: 0.0 for cid in range(total_clients)}  # 每个客户端的效用初始化为0
        self.client_durations = {cid: 0.0 for cid in range(total_clients)}  
        self.client_last_rounds = {cid: 0 for cid in range(total_clients)}
        self.client_selected_times = {cid: 0 for cid in range(total_clients)}
        self.explored_clients = []
        self.unexplored_clients = list(range(total_clients))
        self.util_history = []
        self.pacer_step = self.desired_duration


    def select(self, clients_pool: List[int], clients_count: int, current_round:int) -> List[int]:
        assert clients_count <= len(clients_pool)
        selected: List[int] = []

        # exploitation
        exploited_cnt = max(
            math.ceil((1.0 -self.exploration_factor) * clients_count),
            clients_count - len(self.unexplored_clients)
        )

        if current_round > 1 and exploited_cnt > 0:
            sorted_by_util = sorted(self.client_utilities, key = lambda cid: self.client_utilities[cid], reverse=True)
            sorted_by_util = [c for c in sorted_by_util if c in clients_pool]

            if sorted_by_util and exploited_cnt <= len(sorted_by_util):
                cut_off_util = self.client_utilities[sorted_by_util[exploited_cnt - 1]] * self.cut_off
            else:
                cut_off_util = 0.0

            exploited_candidates = [c for c in sorted_by_util if self.client_utilities[c] > cut_off_util and c not in self.blacklist]
            total_util = float(sum(self.client_utilities[c] for c in exploited_candidates))
            if exploited_candidates and total_util > 0.0:
                probs = np.array([self.client_utilities[c] /total_util for c in exploited_candidates])
                probs = probs / probs.sum()
                exploited_sample = np.random.choice(exploited_candidates, min(len(exploited_candidates), exploited_cnt), p = probs, replace=False)
                selected.extend(exploited_sample.tolist())


            
            last_idx = -1 if not selected else sorted_by_util.index(selected[-1])
            while len(selected) < exploited_cnt:
                last_idx += 1
                if last_idx >= len(sorted_by_util):
                    break
                cand = sorted_by_util[last_idx]
                if cand in self.blacklist or cand in selected:
                    continue
                selected.append(cand)

            
        # 优先选择未探索
        remaining = clients_count - len(selected)
        if remaining > 0:
            available_unexplored = [c for c in self.unexplored_clients if c in clients_pool and c not in selected]
            explore_cnt = min(remaining, len(available_unexplored))
            if explore_cnt > 0:
                chosen = self.rng.sample(available_unexplored, explore_cnt)
                self.explored_clients.extend(chosen)
                for c in chosen:
                    if c in self.unexplored_clients:
                        self.unexplored_clients.remove(c)
                selected.extend(chosen)


        # 随机
        if len(selected) < clients_count:
            remaining_pool = [c for c in clients_pool if c not in selected and c not in self.blacklist]
            need = min(clients_count - len(selected), len(remaining_pool))
            if need > 0:
                selected.extend(self.rng.sample(remaining_pool, need))

        for c in selected:
            self.client_selected_times[c] += 1

        logging.info("Oort selected clients: %s", selected)
        return selected
    
    def update(self, updates: List[Dict], current_round: int):
        # updates: [{client_id, statistical_utility, training_time}]
        for upd in updates:
            cid = upd["client_id"]
            stat_util = upd.get("statistical_utility", 0.0)
            train_time = upd.get("training_time", 0.0)

            self.client_utilities[cid] = stat_util
            self.client_durations[cid] = train_time
            self.client_last_rounds[cid] = max(current_round, 1)
            self.client_utilities[cid] = self.calc_client_util(cid, current_round)

        if updates:
            self.util_history.append(sum(u.get("statistical_utility", 0.0) for u in updates))

        if self.step_window > 0 and len(self.util_history) >= 2 * self.step_window:
            last_window = sum(self.util_history[-2 * self.step_window : -self.step_window])
            current_window = sum(self.util_history[-self.step_window : ])
            if current_window < last_window:
                self.desired_duration += self.pacer_step

        for upd in updates:
            cid = upd["client_id"]
            if self.client_selected_times.get(cid, 0) > self.blacklist_num and cid not in self.blacklist:
                self.blacklist.append(cid)

    # 计算出每个客户端的综合效用
    def calc_client_util(self, cid: int, current_round: int):
        base_util = self.client_utilities.get(cid, 0.0)
        last_round = self.client_last_rounds.get(cid, 1)
        exploration_bonus = 0.0
        if last_round > 0 and current_round > 1:
            exploration_bonus = math.sqrt(max(0.0, 0.1 * math.log(current_round) / last_round))
        util = base_util + exploration_bonus
        duration = self.client_durations.get(cid, 0.0)
        if duration > 0 and self.desired_duration > duration:
            util *= (self.desired_duration / duration) ** self.penalty
        return util
        

    




    
