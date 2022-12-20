import numpy as np
from sklearn import datasets
from src.datasets.dataset import Dataset


class BreastCancerDataset(Dataset):
    def __init__(self) -> None:
        self._data = datasets.load_breast_cancer()

    def get_X(self) -> np.ndarray:
        return self.data.data

    def get_Y(self) -> np.ndarray:
        return self.data.target
