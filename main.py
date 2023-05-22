from data_aug import data_aug
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
from mlp import MLP

def build_retrieval_set(curve_funcs, curve_lens, soh, seq_len):
    retrieval_set = []
    for j, curve_func in enumerate(curve_funcs):
        point = root(get_root, x0=0, args=(curve_func, soh))
        curve_len = curve_lens[j]
        start_cycle = round(point.x.item() * curve_len)
        retrieval_seq = []
        for k in range(seq_len):
            retrieval_seq.append(curve_func((start_cycle + k) / curve_len))
        retrieval_set.append(np.array(retrieval_seq))
    return retrieval_set

def build_dataset(raw, seq_len, N):
    seqs = []
    targets = []
    for i in range(raw.shape[0] - seq_len - N):
        seq = raw[i: i + seq_len]
        target = raw[i + seq_len + N - 1]
        seqs.append(seq)
        targets.append([i + seq_len + N - 1, *target])
    return np.array(seqs), np.array(targets)

def get_root(x, spline, y_target):
    return spline(x) - y_target 

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-seq_len", help="The sequence length", default=100)
    arg_parser.add_argument("-N", help="Predict the next N cycle's SoH", default=1)
    arg_parser.add_argument("-batch", help="batch_size", default=16)
    arg_parser.add_argument("-valid_batch", help="batch_size", default=1)
    arg_parser.add_argument("-num_worker", help="number of worker", default=0)
    arg_parser.add_argument("-epoch", help="num of epoch", default=100)
    arg_parser.add_argument("-lr", help="learning rate", default=1e-3)
    arg_parser.add_argument("-top_k", help="choose top_k to retrieval", default=3)

    args = arg_parser.parse_args()

    curves, curve_funcs, curve_lens = data_aug()
    print("Curves number after augmentation:", len(curves))
    
    seq_len = args.seq_len

    train_data = np.load('./dataset/train/soh.npy')
    test_data = np.load('./dataset/test/soh.npy')
    train_seqs, train_targets = build_dataset(train_data, seq_len, args.N)
    test_seqs, test_targets = build_dataset(test_data, seq_len, args.N)
    print("Train Dataset:", train_seqs.shape, train_targets.shape, "Test Dataset:", test_seqs.shape, test_targets.shape)

    train_set = SeqDataset(train_seqs, train_targets)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.num_worker, shuffle=False)

    test_set = SeqDataset(test_seqs, test_targets)
    test_loader = DataLoader(test_set, batch_size=args.valid_batch, num_workers=args.num_worker, shuffle=False)

    criterion = nn.MSELoss()
    score_model = MLP(args.top_k, args.top_k)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=args.lr)

    score_model.train()
    for epoch in range(args.epoch):
        print("Epoch", epoch)
        # Train
        print("Start training")
        sum_loss, batch_num = 0, 0
        for _, (seq, target) in enumerate(train_loader):
            loss = 0
            start_soh = seq[:, 0, :]
            self_curve_len = len(train_loader) + seq_len
            for i in range(len(start_soh)):
                retrieval_set =  build_retrieval_set(curve_funcs, curve_lens, start_soh[i].data.item(), seq_len)
                x = seq[i].numpy().reshape(-1)
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
                
                target_cycle = target[i][0].float()
                target_soh = target[i][1]
                #TODO：这里的权重到最后权重的映射不是线性的
                output_clone = output.clone()
                for n in range(len(indices)):
                    curve_id = indices[n]
                    predict_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], target_soh.item()))
                    output_clone[n] *= predict_cycle.x.item() * curve_lens[curve_id] / self_curve_len
                output = output_clone
                loss += criterion(output.sum(), target_cycle / self_curve_len)
            
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
        for step, (seq, target) in enumerate(test_loader):
            start_soh = seq[:, 0, :]
            self_curve_len = len(test_loader) + seq_len
            retrieval_set =  build_retrieval_set(curve_funcs, curve_lens, start_soh.data.item(), seq_len)
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
            # print(output, indices)
            target = target.flatten()
            target_cycle = target[0].float()
            target_soh = target[1]
            #TODO：这里的权重到最后权重的映射不是线性的
            output_clone = output.clone()
            for n in range(len(indices)):
                curve_id = indices[n]
                predict_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], target_soh.item()))
                # print(predict_cycle.x)
                output_clone[n] *= predict_cycle.x.item() * curve_lens[curve_id] / self_curve_len
            if step % 40 == 0:
                print(output_clone.sum() * self_curve_len, target_cycle)
            output = output_clone
            loss += criterion(output.sum(), target_cycle / self_curve_len)
            batch_num += 1
        print("test_average_loss", loss / batch_num)

if __name__ == "__main__":
    main()