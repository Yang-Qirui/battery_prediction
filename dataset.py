import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, seqs, targets) -> None:
        super().__init__()
        self.seqs = seqs
        self.targets = targets

    def __getitem__(self, index):
        return self.seqs[index], self.targets[index]

    def __len__(self):
        return len(self.targets)