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
        print(f'Epoch {epoch}: train_loss {train_loss:.4f}, train_error {train_error:.4f}, test_loss {test_loss:.4f}, test_error {test_error:.4f} ({total_num} samples)')


class RetrieveRUL_Net(nn.Module):
    def __init__(self, fea_num, seq_len, hid_size=128, n_refs=3):
        super().__init__()
        self.n_refs = n_refs

        encoder_inp_dim = fea_num * seq_len
        self.encoder_linear1 = nn.Linear(encoder_inp_dim, hid_size*2)
        self.encoder_linear2 = nn.Linear(hid_size*2, hid_size)

        relation_inp_dim = hid_size * 2
        self.relation_linear1 = nn.Linear(relation_inp_dim, hid_size)
        self.relation_linear2 = nn.Linear(hid_size, 1)

        aggregator_inp_dim = hid_size * n_refs + hid_size + n_refs
        self.aggregator_linear1 = nn.Linear(aggregator_inp_dim, hid_size)
        self.aggregator_linear2 = nn.Linear(hid_size, 1)
    
    def encode(self, x):
        batch_sz = x.size(0)
        xx = x.view(batch_sz, -1)
        xx = F.relu(self.encoder_linear1(xx))
        xx = F.relu(self.encoder_linear2(xx))
        y = F.normalize(xx, dim=1)
        return xx

    def relation(self, enc1, enc2):
        batch_sz = x.size(0)
        xx = F.relu(self.relation_linear1(xx))
        y = F.sigmoid(self.relation_linear2(xx))
        return y
    
    def aggregate(self, x):
        batch_sz = x.size(0)
        xx = F.relu(self.aggregator_linear1(xx))
        y = self.aggregator_linear2(xx)
        return y


def my_run(train_loader, test_loader, args, fea_num, seq_len):
    device = args.device
    net = RetrieveRUL_Net(fea_num, seq_len).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # for sample in train_loader.dataset:
    #     print(sample[0].shape, sample[1].shape, sample[2].shape)

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
                enc = net.encode(features)
                enc_sim_mat = torch.mm(enc, enc.t())
                enc_sim_mat = torch.clamp(enc_sim_mat, min=1e-5)
                battery_id_mat = ((battery_ids.unsqueeze(-1) - battery_ids.unsqueeze(0)) != 0)

                ref_weight_mat = enc_sim_mat * battery_id_mat
                ref_weight_mat = ref_weight_mat / ref_weight_mat.sum(dim=1, keepdim=True)

                # directly use the weights to synthsize RUL
                weighted_ruls = ref_weight_mat * labels.unsqueeze(0)
                predictions = weighted_ruls.sum(dim=1, keepdim=False)

                # print(enc_sim_mat)
                # print(battery_id_mat)
                # print(ref_weight_mat)
                # print(weighted_ruls)
                # print('predictions', predictions)
                # input()

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

        # total_num = 0
        # total_error = 0
        # total_loss = 0
        # net.eval()
        # for battery_ids, features, labels in test_loader:
        #     battery_ids, features, labels = battery_ids.to(device), features.to(device), labels.to(device)
        #     predictions = net(features).squeeze()
        #     # predictions = torch.clamp(predictions, min=0.0001)
        #     errors = torch.abs(predictions - labels) / torch.max(predictions, labels)
        #     loss = F.mse_loss(predictions, labels, reduce='sum')
        #     # print(f'predictions:{predictions.shape},\nlabels:{labels.shape},\nerrors:{errors.shape}')
        #     total_error += errors.sum().item()
        #     total_loss += loss
        #     total_num += errors.size(0)
        test_loss = total_loss / total_num
        test_error = total_error / total_num
        print(f'Epoch {epoch}: train_loss {train_loss:.4f}, train_error {train_error:.4f}, test_loss {test_loss:.4f}, test_error {test_error:.4f} ({total_num} samples)')


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

def run(train_loader, test_loader, args, fea_num):
    encoder = lstm_encoder(indim=fea_num, hiddendim=args.lstm_hidden, fcdim=args.fc_hidden, outdim=args.fc_out, n_layers=args.lstm_layer, dropout=args.dropout)
    encoder_optimizer = Adam(encoder.parameters(), lr=args.lr)
    encoder_lr_scheduler = StepLR(encoder_optimizer, step_size=1, gamma=args.gamma)
    relation_model = MLP(in_channel=args.top_k, out_channel=args.top_k)
    relation_model_optimizer = Adam(
            relation_model.parameters(),
            lr=args.lr,
        )
    relation_lr_scheduler = StepLR(
            relation_model_optimizer, step_size=100, gamma=args.gamma
        )

    train_loss = []
    train_loss = []
    for e in range(args.epoch):
        all_retrieval_feas = []
        all_retrieval_lbls = []   


        encoder_lr_scheduler.step(e)
        relation_lr_scheduler.step(e)

            
        print(
            "training epoch:",
            e,
            "learning rate:",
            encoder_optimizer.param_groups[0]["lr"],
        )


        encoder.train().cuda()
        relation_model.train().cuda()


        for step, ((features, labels), (nei_features, _)) in enumerate(train_loader):
            features = features.squeeze(0).cuda()
            nei_features = nei_features.squeeze(0).cuda()
            labels = labels.squeeze(0).cuda()
            encoded_source = encoder(features)
            all_retrieval_feas.append(encoded_source.cpu())
            all_retrieval_lbls.append(labels.cpu())
            encoded_neigh = encoder(nei_features)
            assert encoded_source.shape == encoded_neigh.shape
            loss = 0
            # contrastive_l = contrastive_loss(encoded_source, encoded_neigh, args.tao)
            for i in range(len(features)):
                '''
                    generate retrieval set
                '''
                retreival_lbls = torch.cat((labels[:i], labels[i + 1 :]), dim=0)
                retreival_ruls = retreival_lbls[:, -1]
                target_rul = labels[i, -1]

                encoded_retrieval_feas = torch.cat((encoded_source[:i], encoded_source[i + 1:]), dim=0)

                relation_scores = []
                        

                # for retrieval_tensor in encoded_retrieval_feas:
                #     s = F.cosine_similarity(encoded_source[i], retrieval_tensor, dim=0).cpu().detach().numpy()
                #     relation_scores.append(s)
                # relation_scores = np.array(relation_scores)

                        
                # import pdb;pdb.set_trace()
                relation_scores = F.cosine_similarity(encoded_source[i].unsqueeze(0), encoded_retrieval_feas,dim=1)
                chosen_scores, indices = torch.topk(torch.Tensor(relation_scores), args.top_k,dim=0) # 乘 -1 找最小的


                # relation_scores = F.cosine_similarity(encoded_source, encoded_retrieval_feas)
                # chosen_scores, indices = torch.topk(relation_scores, args.top_k)
                        
                # relation_scores = F.cosine_similarity(encoded_source[:-1], encoded_retrieval_feas, dim=1)
                # relation_scores = relation_scores.t() # 转置操作，使得relation_scores的形状与原来代码中的一致
                # chosen_scores, indices = torch.topk(relation_scores, args.top_k)


                # chosen_scores = -chosen_scores
                max_ = torch.max(chosen_scores)
                min_ = torch.min(chosen_scores)
                chosen_scores = (chosen_scores - min_) / (max_ - min_)
                chosen_scores=chosen_scores.cuda()

                output = relation_model(chosen_scores)
                
                error, diff = score_weight_loss(output, chosen_scores)
                # TODO: Here meet a bug. the loss_0 will become negative
                # loss_0 = relation_model.loss_weights[0] * error + relation_model.loss_weights[1] * diff
                loss_0 = error + diff

                predict_rul = torch.sum(chosen_scores * retreival_ruls[indices])
                loss_1 = nn.MSELoss()(target_rul, predict_rul)
                loss += loss_0 + loss_1

            loss /= len(features)
            # loss += contrastive_l * args.alpha
            encoder_optimizer.zero_grad()
            relation_model_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            relation_model_optimizer.step()
            train_loss.append(loss.cpu().detach().numpy())

            if step % 40 == 0:
                print(
                    "step:",
                    step,
                    "train loss:",
                    train_loss[-1],
                    "avg train loss:",
                    np.average(train_loss),
                )
            
        print("Start validating")
        encoder.eval()
        relation_model.eval()
        all_retrieval_feas = torch.vstack(all_retrieval_feas)
        all_retrieval_lbls = torch.vstack(all_retrieval_lbls).cuda()
        print(all_retrieval_feas.shape, all_retrieval_lbls.shape)
        loss = 0
        percent_rul=0
        # percent = 0
        batch_num = 0
        with torch.no_grad():
            for step, (seq, target) in enumerate(test_loader):
                target_rul = target[:, -1].cuda()
                x = encoder(seq.cuda())
                #scores = []
                #import pdb;pdb.set_trace()
                scores = F.cosine_similarity(x, all_retrieval_feas.cuda(),dim=1)
                # for retrieval_seq in all_retrieval_feas:
                #     alignment = dtw(x.detach().cpu().numpy(), retrieval_seq.detach().cpu().numpy(), keep_internals=True) # DTW算法
                #     scores.append(alignment.distance)
                # scores = np.array(scores)
                chosen_scores, indices = torch.topk(torch.Tensor(scores), args.top_k,dim=0) # 乘 -1 找最小的
                # chosen_scores = -chosen_scores
                max_ = torch.max(chosen_scores)
                min_ = torch.min(chosen_scores)
                chosen_scores = (chosen_scores - min_) / (max_ - min_)
                output = relation_model(chosen_scores)

                predict_rul = torch.sum(chosen_scores * all_retrieval_lbls[indices, -1])

                if step % 40 == 0:
                    print("step:",
                            step,
                            "predict_rul",
                            predict_rul * 3000, 
                            "target_rul",
                            target_rul * 3000)
                # print(output_clone.sum().detach().item() * self_curve_len, target_cycle.detach().item())
               
                loss_rul = nn.MSELoss()(predict_rul, target_rul)
                percent_rul+=torch.abs(torch.sqrt(loss_rul)/target_rul)
                
                # percent_rul = loss_rul/target_rul
                loss = loss+loss_rul
                # percent = percent+percent_rul
                batch_num += 1
            print("test_average_loss", 
                  loss / batch_num,
                   "test_loss_percent",
                   percent_rul / batch_num,
                  )
 

def main(args):
    train_ds = BatteryDataset(train=True)
    test_ds = BatteryDataset(train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

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
    argparser.add_argument("-top_k", help="use top k curves to retrieve", type=int, default=5)
    argparser.add_argument("-tao", help="tao in contrastive loss calculation ", type=float, default=0.5)
    argparser.add_argument("-alpha", help="zoom factor of contrastive loss", type=float, default=0.1)
    
    args = argparser.parse_args()
    main(args)

