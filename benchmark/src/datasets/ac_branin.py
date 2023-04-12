from __future__ import annotations

import pandas as pd
from pathlib import Path
from src.datasets.dataset import Dataset


class ACBraninDataset(Dataset):
    def __init__(self) -> None:
        self._data = pd.read_csv(Path(__file__).parent / "ac_branin_instances_features.csv", skipinitialspace=True)
        self.data["instance"] = self.data["instance"].apply(str)

    def get_instances(self, n: int | None = None) -> list[str]:
        """Create instances from the dataset which include two classes only."""
        instances = self.data["instance"].to_list()

        if n is not None:
            instances = instances[:n]

        return instances

    def get_instance_features(self, n: int | None = None) -> dict[str, list[int | float]]:
        """Returns the mean and variance of all instances as features."""
        features = {}
        for instance in self.get_instances(n):
            features[instance] = [float(self.data["feature"][self.data["instance"] == instance])]

        return features
