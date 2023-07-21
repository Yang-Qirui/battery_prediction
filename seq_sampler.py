from torch.utils.data import Sampler, TensorDataset

class SeqSampler(Sampler):
    def __init__(self, data_source: TensorDataset, type) -> None:
        super().__init__(data_source)
        self.data = data_source
        self.type = type

    def __iter__(self):
        """tensors:[features,label]
        features: seq,seq_len,feature_num
        label: seq, feas [rul,len,num]
        """
        indices_map = {}
        features = self.data.tensors[0]
        labels = self.data.tensors[1]
        for i in range(features.shape[0]):
            tail_dq = features[i][-1][-1]
            origin_dq = features[i][-1][0]
            # rul = labels[i][0]  # tail rul
            # tot_seq_len = labels[i][1]
            # pos = int((tot_seq_len - rul).item())
            # pos = "%.4f" % (tail_dq / origin_dq)
            pos = labels[i][-1]
            if pos in indices_map.keys():
                indices_map[pos].append(i)
            else:
                indices_map[pos] = [i]
        indices = []
        keys = list(indices_map.keys())
        if self.type == "train":
            nei_keys = [keys[i + 1] for i in range(len(keys) - 1)]
            nei_keys.append(keys[-2])
            assert len(keys) == len(nei_keys)
            for i in range(len(keys)):
                indices += indices_map[keys[i]]
                indices += indices_map[nei_keys[i]]
        else:
            for i in range(len(keys)):
                indices += indices_map[keys[i]]
        return iter(indices)

    def __len__(self):
        if self.type == "train":
            return self.data.tensors[0].shape[0] * 2
        else:
            return self.data.tensors[0].shape[0]