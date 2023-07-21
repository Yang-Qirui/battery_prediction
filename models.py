import torch
from torch import nn
import math
from einops import rearrange, repeat

# from einops.layers.torch import Rearrange
import torch.nn.functional as F

# helpers

# 定义MLP网络
class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channel, 64),  # 输入层，假设有100个神经元
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 64),  # 隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(64, out_channel), # 输出层，假设有1个神经元
            nn.Softmax(dim=0)
        )
        self.loss_weights = nn.Parameter(torch.ones(2))

    def forward(self, x):
        output = self.layers(x)
        # output = self.std_layer(output)
        # output = nn.ReLU()(output)
        return output


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head=64, mlp_dim=64, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class lstm_encoder(nn.Module):
    def __init__(self, indim, hiddendim, fcdim, outdim, n_layers, dropout=0.4):
        super(lstm_encoder, self).__init__()
        self.lstm1 = torch.nn.LSTM(
            input_size=indim,
            hidden_size=hiddendim,
            batch_first=True,
            bidirectional=False,
            num_layers=n_layers,
        )
        # self.lstm2 = torch.nn.LSTM(input_size=hiddendim, hidden_size=hiddendim, batch_first=True, bidirectional=False, num_layers=n_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hiddendim * n_layers, fcdim)
        self.bn1 = torch.nn.LayerNorm(normalized_shape=fcdim)
        self.fc2 = torch.nn.Linear(fcdim, outdim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        out, (h, c) = self.lstm1(x)
        # out, (h, c) = self.lstm2(h)
        h = h.reshape(x.size(0), -1)
        h = self.dropout(h)
        # h = h.squeeze()
        x = self.fc1(h)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.sigmoid(x)
        return x