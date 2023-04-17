from sklearn import datasets
from src.datasets.dataset import Dataset


class IrisDataset(Dataset):
    def __init__(self) -> None:
        self._data = datasets.load_iris()
