import torch
from torch import nn

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