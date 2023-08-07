from lyc_data_loader import BatteryDataset
from torch.utils.data import DataLoader, TensorDataset
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from models import lstm_encoder, MLP
import math
from dataset import ContrastiveDataset
from dtw import dtw
import numpy as np
from utils import score_weight_loss
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import os
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


def find_closest_k_dim_vectors(dataset, Y, m):
    # 假设dataset是一个已经按第一维度排序的二维张量
    values = dataset.samples
    
    # 使用二分查找找到最接近Y的索引
    left, right = 0, len(values) - 1
    while left <= right:
        mid = (left + right) // 2
        if values[mid][1][0][0] < Y:
            left = mid + 1
        elif values[mid][1][0][0] > Y:
            right = mid - 1
        else:
            break

    # 找到m个最接近的索引
    start_idx = max(mid - m // 2, 0)
    end_idx = min(mid + m // 2 + 1, len(values))

    return DataLoader(dataset[start_idx:end_idx], batch_size=m, shuffle=False)

class RUL_MLP(nn.Module):
    def __init__(self, fea_num, seq_len, hid_size=128):
        super().__init__()
        inp_dim = fea_num * seq_len
        self.linear1 = nn.Linear(inp_dim, hid_size*2)
        self.linear2 = nn.Linear(hid_size*2, hid_size)
        self.linear3 = nn.Linear(hid_size, 1)
    
    def forward(self, x):
        batch_sz = x.size(0)
        xx = x.view(batch_sz, -1)
        xx = F.relu(self.linear1(xx))
        xx = F.relu(self.linear2(xx))
        y = self.linear3(xx)
        return y

class RUL_RNN(nn.Module):
    def __init__(self, fea_num, seq_len, hid_size=128):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=fea_num, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(hid_size, 1)
    
    def forward(self, x):
        batch_sz = x.size(0)
        xx, hn = self.rnn1(x)
        # print(xx.shape)
        xx = xx[:, -1, :]
        y = self.linear1(xx)
        return y


def baseline_run(train_loader, test_loader, args, fea_num, seq_len):
    device = args.device
    net = RUL_MLP(fea_num, seq_len).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # for sample in train_loader.dataset:
    #     print(sample[0].shape, sample[1].shape, sample[2].shape)
    min_error = math.inf

    for epoch in range(args.epoch):
        total_num = 0
        total_error = 0
        total_loss = 0
        net.train()
        with tqdm(total=len(train_loader.dataset)) as t:
            for battery_ids, features, labels in train_loader:
                battery_ids, features, labels = battery_ids.to(device), features.to(device), labels.to(device)
                batch_sz = labels.size(0)
                net.zero_grad()
                predictions = net(features).squeeze()
                # predictions = torch.clamp(predictions, min=0.0001)
                errors = torch.abs(predictions - labels) / torch.max(predictions, labels)
                error = errors.mean()
                # loss = error
                loss = F.mse_loss(predictions, labels)
                loss.backward()
                optimizer.step()
                t.set_description(f"Epoch {epoch}")
                t.set_postfix(loss=loss.item(), error=error.item())
                t.update(batch_sz)
                total_error += errors.sum().item()
                total_loss += loss * batch_sz
                total_num += batch_sz
        train_loss = total_loss / total_num
        train_error = total_error / total_num

        total_num = 0
        total_error = 0
        total_loss = 0
        net.eval()
        for battery_ids, features, labels in test_loader:
            battery_ids, features, labels = battery_ids.to(device), features.to(device), labels.to(device)
            predictions = net(features).squeeze()
            # predictions = torch.clamp(predictions, min=0.0001)
            errors = torch.abs(predictions - labels) / torch.max(predictions, labels)
            loss = F.mse_loss(predictions, labels, reduce='sum')
            # print(f'predictions:{predictions.shape},\nlabels:{labels.shape},\nerrors:{errors.shape}')
            total_error += errors.sum().item()
            total_loss += loss
            total_num += errors.size(0)
        test_loss = total_loss / total_num
        test_error = total_error / total_num
        min_error = test_error if test_error < min_error else min_error
        print(f'Epoch {epoch}: train_loss {train_loss:.4f}, train_error {train_error:.4f}, test_loss {test_loss:.4f}, test_error {test_error:.4f}, min_error {min_error:.4f} ({total_num} samples)')


class RUL_RetrieveNet(nn.Module):
    def __init__(self, fea_num, seq_len, hid_size=128, n_refs=3):
        super().__init__()
        self.n_refs = n_refs

        encoder_inp_dim = fea_num * seq_len
        self.encoder_linear1 = nn.Linear(encoder_inp_dim, hid_size*2)
        self.encoder_linear2 = nn.Linear(hid_size*2, hid_size)

        relation_inp_dim = hid_size * 2
        self.relation_linear1 = nn.Linear(relation_inp_dim, hid_size)
        self.relation_linear2 = nn.Linear(hid_size, 1)
        self.relation_parameter_free = True

        aggregator_inp_dim = hid_size * n_refs + hid_size + n_refs
        self.aggregator_linear1 = nn.Linear(aggregator_inp_dim, hid_size)
        self.aggregator_linear2 = nn.Linear(hid_size, 3) # 1
        self.aggregator_parameter_free = False
    
    def encode(self, x):
        batch_sz = x.size(0)
        xx = x.view(batch_sz, -1)
        xx = F.relu(self.encoder_linear1(xx))
        xx = F.relu(self.encoder_linear2(xx))
        y = F.normalize(xx, dim=1)
        # return xx # miss normalization. A fake cosine sim.
        return y

    def relation(self, enc, enc_):
        if self.relation_parameter_free:
            # parameter free version
            enc_sim_mat = torch.mm(enc, enc_.t())
            enc_sim_mat = torch.clamp(enc_sim_mat, min=0)
            return enc_sim_mat
        else:
            # TODO
            batch_sz = x.size(0)
            xx = F.relu(self.relation_linear1(xx))
            y = F.sigmoid(self.relation_linear2(xx))
            return y
    
    def aggregate(self, self_enc, ref_weight, ref_enc, ref_rul):
        if self.aggregator_parameter_free:
            # parameter free version, directly use the weights to synthsize RUL
            ref_weight = ref_weight / ref_weight.sum(dim=1, keepdim=True)
            weighted_ruls = ref_weight * ref_rul
            predictions = weighted_ruls.sum(dim=1, keepdim=True)
            return predictions
        else:
            batch_sz = self_enc.size(0)
            xx = torch.concatenate([self_enc, ref_enc.view(batch_sz, -1), ref_rul], dim=1)
            print(xx.shape)
            xx = F.relu(self.aggregator_linear1(xx))
            predictions = self.aggregator_linear2(xx)
            pred = torch.mul(predictions, ref_rul).sum(dim=1)
            print(pred)
            _pred = F.sigmoid(pred)
            print(_pred)
            return pred


def my_run(train_loader, test_loader, args, fea_num, seq_len):
    device = args.device
    net = RUL_RetrieveNet(fea_num, seq_len, n_refs=args.top_k).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # for sample in train_loader.dataset:
    #     print(sample[0].shape, sample[1].shape, sample[2].shape)
    min_error = math.inf

    for epoch in range(args.epoch):
        total_num = 0
        total_error = 0
        total_loss = 0
        net.train()

        with tqdm(total=len(train_loader.dataset)) as t:
            for battery_ids, features, labels, _ in train_loader:
                battery_ids, features, labels = battery_ids.to(device), features.to(device), labels.to(device)
                batch_sz = labels.size(0)
                net.zero_grad()
                enc = net.encode(features)
                enc_sim_mat = net.relation(enc, enc)

                # mask out the same-battery references
                battery_id_mat = ((battery_ids.unsqueeze(-1) - battery_ids.unsqueeze(0)) != 0)
                ref_weight_mat = enc_sim_mat * battery_id_mat

                ref_weight, ref_idx = ref_weight_mat.topk(k=args.top_k, dim=1)
                ref_enc = enc[ref_idx]
                ref_rul = labels[ref_idx]
                predictions = net.aggregate(enc, ref_weight, ref_enc, ref_rul).squeeze()

                errors = torch.abs(predictions - labels) / torch.max(predictions, labels)
                error = errors.mean()
                # loss = error
                loss = F.mse_loss(predictions, labels)
                loss.backward()
                optimizer.step()
                t.set_description(f"Epoch {epoch}")
                t.set_postfix(loss=loss.item(), error=error.item())
                t.update(batch_sz)
                total_error += errors.sum().item()
                total_loss += loss * batch_sz
                total_num += batch_sz
        train_loss = total_loss / total_num
        train_error = total_error / total_num

        total_num = 0
        total_error = 0
        total_loss = 0
        net.eval()
        assert 0
        # prepare a reference set
        # refset_train_loader = DataLoader(train_loader.dataset, batch_size=1000, shuffle=False)
        for battery_ids, features, labels, _ in test_loader:
            refset_train_loader = find_closest_k_dim_vectors(train_loader.dataset, features[0, 0, 0], 50)
            refset_battery_ids, refset_features, refset_labels, _ = next(iter(refset_train_loader))
            refset_battery_ids = refset_battery_ids.to(device)
            refset_features = refset_features.to(device)
            refset_labels = refset_labels.to(device)
            refset_enc = net.encode(refset_features)

            battery_ids, features, labels = battery_ids.to(device), features.to(device), labels.to(device)
            batch_sz = labels.size(0)
            enc = net.encode(features)
            enc_sim_mat = net.relation(enc, refset_enc)
            print(enc_sim_mat)

            # mask out the same-battery references
            battery_id_mat = ((battery_ids.unsqueeze(-1) - refset_battery_ids.unsqueeze(0)) != 0)
            ref_weight_mat = enc_sim_mat * battery_id_mat

            ref_weight, ref_idx = ref_weight_mat.topk(k=args.top_k, dim=1)
            print(ref_weight_mat)
            assert 0
            ref_enc = refset_enc[ref_idx]
            ref_rul = refset_labels[ref_idx]
            predictions = net.aggregate(enc, ref_weight, ref_enc, ref_rul).squeeze()

            errors = torch.abs(predictions - labels) / torch.max(predictions, labels)
            loss = F.mse_loss(predictions, labels, reduce='sum')

            del refset_train_loader

            # print(f'predictions:{predictions.shape},\nlabels:{labels.shape},\nerrors:{errors.shape}')
            # print(enc_sim_mat.shape)
            # print(battery_id_mat.shape)
            # print(ref_weight_mat.shape)
            # print(ref_enc.shape)
            # print(ref_rul.shape)
            # print('predictions', predictions)
            # input()
            
            total_error += errors.sum().item()
            total_loss += loss
            total_num += errors.size(0)
        test_loss = total_loss / total_num
        test_error = total_error / total_num
        min_error = test_error if test_error < min_error else min_error

        print(f'Epoch {epoch}: train_loss {train_loss:.4f}, train_error {train_error:.4f}, test_loss {test_loss:.4f}, test_error {test_error:.4f}, min_error {min_error:.4f} ({total_num} samples)')


def contrastive_loss(source, pos_sample, tao):
    assert source.shape[0] == pos_sample.shape[0]
    N = source.shape[0]

    def sim(tensor1, tensor2):
        if tensor1.shape != tensor2.shape:
            tensor1 = tensor1.reshape(1, -1)
            return torch.cosine_similarity(tensor1, tensor2)
        else:
            return torch.cosine_similarity(tensor1, tensor2, dim=0)

    def _l(i, type):
        denominator = 0
        if type == "src":
            denominator += torch.sum(torch.exp(sim(source[i], source) / tao))
            denominator += torch.sum(torch.exp(sim(source[i], pos_sample) / tao))
        else:
            denominator += torch.sum(torch.exp(sim(pos_sample[i], pos_sample) / tao))
            denominator += torch.sum(torch.exp(sim(pos_sample[i], source) / tao))
        denominator -= math.exp(1 / tao)
        numerator = torch.exp(sim(pos_sample[i], source[i]) / tao)
        return -torch.log(numerator / denominator).item()

    L = 0
    for i in range(N):
        L += _l(i, "src") + _l(i, "pos")
    # print((e-s).microseconds / 10**6)
    return L / (2 * N)


def main(args):
    train_ds = BatteryDataset(train=True)
    test_ds = BatteryDataset(train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

    # ==== Train =====
    # baseline_run(train_loader=train_loader, test_loader=test_loader, args=args, fea_num=train_ds[0][1].size(-1), seq_len=args.seq_len)
    my_run(train_loader=train_loader, test_loader=test_loader, args=args, fea_num=train_ds[0][1].size(-1), seq_len=args.seq_len)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-device", type=str, default='cuda:0')
    argparser.add_argument("-batch", type=int, default=64)
    argparser.add_argument("-valid_batch", type=int, default=1)
    argparser.add_argument("-epoch", type=int, default=100)#100
    argparser.add_argument("-lr", help="initial learning rate", type=float, default=1e-3)
    argparser.add_argument("-gamma", help="learning rate decay rate", type=float, default=0.9)
    argparser.add_argument("--seq-len", type=int, help="number of cycles as the feature", default=100)  # 100
    argparser.add_argument("--lstm-hidden", type=int, help="lstm hidden layer number", default=128)  # 128
    argparser.add_argument("--fc-hidden", type=int, help="fully connect layer hidden dimension", default=98)  # 128
    argparser.add_argument("--fc-out", type=int, help="embedded sequence dimmension", default=64)  # 128
    argparser.add_argument("--dropout", type=float, default=0.3)  # 0.1
    argparser.add_argument("--lstm-layer", type=int, default=1)  # 0.1
    argparser.add_argument("-top_k", help="use top k curves to retrieve", type=int, default=3)
    argparser.add_argument("-tao", help="tao in contrastive loss calculation ", type=float, default=0.5)
    argparser.add_argument("-alpha", help="zoom factor of contrastive loss", type=float, default=0.1)
    
    args = argparser.parse_args()
    main(args)

