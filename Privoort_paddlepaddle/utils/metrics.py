import csv
import os
from typing import List


class MetricsWriter:
    def __init__(self, csv_path: str):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "accuracy", "clients", "utilities"])

        # if not os.path.exists(csv_path):
        #     with open(csv_path, "w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["round", "accuracy", "clients", "utilities"])

    def write(self, round: int, accuracy: float, clients: List[int], utilities: List[dict]):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    round,
                    f"{accuracy:.4f}",
                    "|".join(map(str, clients)),
                    "|".join(
                        f"{u['client_id']}:{u.get('statistical_utility',0):.4f}/"
                        f"{u.get('training_time',0):.2f}s"
                        for u in utilities
                    ),
                ]
            )
