from torch.utils.data import Dataset, DataLoader

class ContrastiveDataset(Dataset):
    def __init__(self, features, labels, batchsize):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels
        self.batch = batchsize

    def __getitem__(self, index):
        start = index
        end = index + self.batch
        next_start = index + 1
        next_end = index + 1 + self.batch
        if next_end > len(self.features):  # 超过tensor边界
            return None
        current_features = self.features[start:end]
        next_features = self.features[next_start:next_end]
        current_labels = self.labels[start:end]
        next_labels = self.labels[next_start:next_end]
        return (current_features, current_labels), (next_features, next_labels)

    def __len__(self):
        return len(self.features) - self.batch - 1
