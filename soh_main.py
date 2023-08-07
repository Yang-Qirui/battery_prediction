from data_aug import data_aug, interpolate
import numpy as np
import argparse
from scipy.optimize import root
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SeqDataset
from dtw import *
from models import MLP
import matplotlib.pyplot as plt
import os
from utils import score_weight_loss

class NegativeMSELoss(nn.Module):
    def __init__(self):
        super(NegativeMSELoss, self).__init__()

    def forward(self, input, target):
        return -torch.mean((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

# def custom_loss(y_pred, x):
#     mean_centered_pred = y_pred - torch.mean(y_pred)
#     mean_centered_input = x - torch.mean(x)
#     corr = torch.sum(mean_centered_pred * mean_centered_input) / (torch.sqrt(torch.sum(mean_centered_pred ** 2)) * torch.sqrt(torch.sum(mean_centered_input ** 2)))
#     return corr



def build_retrieval_set(curve_funcs, curve_lens, soh, seq_len):
    retrieval_set = []
    for j, curve_func in enumerate(curve_funcs):
        point = root(get_root, x0=0, args=(curve_func, soh))
        curve_len = curve_lens[j]
        start_cycle = round(point.x.item() * curve_len)
        retrieval_seq = []
        for k in range(seq_len):
            retrieval_seq.append(curve_func((start_cycle - seq_len + k) / curve_len))
        retrieval_set.append(np.array(retrieval_seq))
    return retrieval_set

def build_dataset(raw, seq_len, N):
    seqs = []
    targets = []
    for i in range(raw.shape[0] - seq_len - N):
        seq = raw[i: i + seq_len]
        target = raw[i + seq_len + N - 1]
        seqs.append(seq)
        targets.append([N, *target])
    return np.array(seqs), np.array(targets)

def get_root(x, spline, y_target):
    return spline(x) - y_target 

def run(seq_len, N, curve_lens, curve_funcs, args):
    train_data = np.load(f'{args.data_path}train/soh.npy')
    test_data = np.load(f'{args.data_path}test/soh.npy')
    _, smooth_train_data, _, _, _, _ = interpolate(train_data)
    _, smooth_test_data, _, _, _, _ = interpolate(test_data)
    smooth_train_data = smooth_train_data.reshape(-1, 1)
    smooth_test_data = smooth_test_data.reshape(-1, 1)
    # for i, curve in enumerate(curves):
        # plt.plot([i for i in range(len(curve))], curve, label="i")
    # plt.plot([i for i in range(len(smooth_test_data))], smooth_test_data, label='test')
    # plt.legend()
    # plt.show()
    train_seqs, train_targets = build_dataset(smooth_train_data, seq_len, N)
    test_seqs, test_targets = build_dataset(smooth_test_data, seq_len, N)
    print("Train Dataset:", train_seqs.shape, train_targets.shape, "Test Dataset:", test_seqs.shape, test_targets.shape)

    train_set = SeqDataset(train_seqs, train_targets)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.num_worker, shuffle=False)

    test_set = SeqDataset(test_seqs, test_targets)
    test_loader = DataLoader(test_set, batch_size=args.valid_batch, num_workers=args.num_worker, shuffle=False)

    criterion = nn.MSELoss()
    score_model = MLP(args.top_k, args.top_k)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=args.lr)

    score_model.train()
    min_valid_loss = float('inf')
    min_valid_epoch = 0
    for epoch in range(args.epoch):
        print("Epoch", epoch)
        # Train
        print("Start training")
        sum_loss, batch_num = 0, 0
        self_curve_len = len(smooth_train_data)
        for _, (seq, target) in enumerate(train_loader):
            loss = 0
            start_soh = seq[:, -1, :]
            for i in range(len(start_soh)):
                retrieval_set = build_retrieval_set(curve_funcs, curve_lens, start_soh[i].item(), seq_len)
                x = seq[i].numpy().reshape(-1)
                scores = []
                for retrieval_seq in retrieval_set:
                    alignment = dtw(x, retrieval_seq, keep_internals=True) # DTW算法
                    scores.append(alignment.distance)
                scores = np.array(scores)
                chosen_scores, indices = torch.topk(torch.Tensor(-scores), args.top_k) # 乘 -1 找最小的
                chosen_scores = -chosen_scores
                # print(chosen_scores)
                max_ = torch.max(chosen_scores)
                min_ = torch.min(chosen_scores)
                chosen_scores = (chosen_scores - min_) / (max_ - min_)
                output = score_model(chosen_scores)
                # weight = 1 / (1 + output)
                # print(chosen_scores, output, indices)
                error, diff = score_weight_loss(output, chosen_scores)
                loss_0 = score_model.loss_weights[0] * error + score_model.loss_weights[1] * diff

                target_cycle = target[i][0].float()
                target_soh = target[i][1]
                # TODO：这里的权重到最后权重的映射不是线性的
                output_clone = output.clone()
                for n in range(len(indices)):
                    curve_id = indices[n]
                    start_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], start_soh[i].data.item()))
                    predict_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], target_soh.item()))
                    output_clone[n] *= (predict_cycle.x.item() - start_cycle.x.item()) * curve_lens[curve_id]
                output = output_clone
                # loss += criterion(output.sum() / self_curve_len, target_cycle / self_curve_len)
                loss_1 = criterion(output.sum(), target_cycle) 
                loss += loss_0 + loss_1

            loss /= args.batch
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sum_loss += loss.item()
            batch_num += 1
        print("train_average_loss", sum_loss / batch_num)
        # Eval
        print("Start validating")
        score_model.eval()
        loss = 0
        batch_num = 0
        predict_list, target_list = [], []
        self_curve_len = len(smooth_test_data)
        for step, (seq, target) in enumerate(test_loader):
            if step == 0:
                for i in range(len(seq[0])):
                    predict_list.append(i)
                    target_list.append((i, seq[0][i].item()))
            start_soh = seq[:, -1, :]
            retrieval_set =  build_retrieval_set(curve_funcs, curve_lens, start_soh.item(), seq_len)
            x = seq.numpy().reshape(-1)
            scores = []
            for retrieval_seq in retrieval_set:
                alignment = dtw(x, retrieval_seq, keep_internals=True) # DTW算法
                scores.append(alignment.distance)
            scores = np.array(scores)
            chosen_scores, indices = torch.topk(torch.Tensor(-scores), args.top_k) # 乘 -1 找最小的
            chosen_scores = -chosen_scores
            max_ = torch.max(chosen_scores)
            min_ = torch.min(chosen_scores)
            chosen_scores = (chosen_scores - min_) / (max_ - min_)
            output = score_model(chosen_scores)
            # print(chosen_scores, output, indices)
            target = target.flatten()
            target_cycle = target[0].float()
            target_soh = target[1]
            #TODO：这里的权重到最后权重的映射不是线性的
            output_clone = output.clone()
            for n in range(len(indices)):
                curve_id = indices[n]
                start_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], start_soh.item()))
                predict_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], target_soh.item()))
                output_clone[n] *= (predict_cycle.x.item() - start_cycle.x.item()) * curve_lens[curve_id]
            if step % 40 == 0:
                print(output_clone.sum(), target_cycle)
            output = output_clone
            # print(output_clone.sum().detach().item() * self_curve_len, target_cycle.detach().item())
            target_list.append((target_cycle.detach().numpy(), target_soh.detach().numpy()))
            predict_list.append(step + seq_len + output_clone.sum().detach().item())
            loss += (output_clone.sum().detach().item() - target_cycle.detach().item())
            batch_num += 1
        print("test_average_loss", loss / batch_num)
        # for i in range(len(target_list)):
        #     print(target_list[i], predict_list[i])
        # assert 0
        if abs(loss / batch_num) < min_valid_loss:
            min_valid_loss = abs(loss / batch_num)
            min_valid_epoch = epoch
            plt.plot([i for i in range(len(smooth_test_data))], smooth_test_data)
            plt.plot([i for i in predict_list], [soh for cycle, soh in target_list])
            save_path = f"./figures/seq_len={seq_len}&N={N}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig('/'.join([save_path, f'epoch{epoch}_{min_valid_loss}.png']))
            plt.close()
    with open(f"./reports/seq_len={seq_len}&N={N}.txt", 'w') as f:
        f.write(f"Summary: min_valid_loss: {min_valid_loss} in epoch: {min_valid_epoch}, relative_min_loss: {min_valid_loss / N * 100}%")
        f.close()

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-seq_len", help="The sequence length", default=100, type=int)
    arg_parser.add_argument("-N", help="Predict the next N cycle's SoH", default=10, type=int)
    arg_parser.add_argument("-batch", help="batch_size", default=16, type=int)
    arg_parser.add_argument("-valid_batch", help="batch_size", default=1, type=int)
    arg_parser.add_argument("-num_worker", help="number of worker", default=0, type=int)
    arg_parser.add_argument("-epoch", help="num of epoch", default=10, type=int)
    arg_parser.add_argument("-lr", help="learning rate", default=1e-3, type=float)
    arg_parser.add_argument("-top_k", help="choose top_k to retrieval", default=3, type=int)
    arg_parser.add_argument("-aug_path", help="the path of train set's soh.npy for data augmentation", default='./data/wanguo/train', type=str)
    arg_parser.add_argument("-data_path", help="the path of train set and test set", default='./data/wanguo/', type=str)


    args = arg_parser.parse_args()

    curves, curve_funcs, curve_lens = data_aug(args.aug_path)
    # print(curve_lens)
    # assert 0
    print("Curves number after augmentation:", len(curves))
    
    run(args.seq_len, args.N, curve_lens, curve_funcs, args)

            

if __name__ == "__main__":
    main()