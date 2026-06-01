import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
