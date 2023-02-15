"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-02-13 20:56:51
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # B, C   x(B,C,128,128)
        y = self.avg_pool(x).view(b, c)  # (B,C,1,1) -> (B,C)
        y = self.fc(y).view(b, c, 1, 1)  # (B,C) -> (B,C,1,1)
        return x * y


# model = SEBlock(256)
# print(model)
# inputs = torch.ones([2, 256, 128, 128])
# outputs = model(inputs)
# print(outputs.shape)


# ******************** MobileNetV3 ********************
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:  # min_ch限制channels的下限值
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)  # '//'取整除
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(ch=input_c // squeeze_factor, divisor=8)
        self.fc1 = nn.Conv2d(
            input_c, squeeze_c, kernel_size=1
        )  # fc1: expand_channel // 4 (Conv2d 1x1代替全连接层)
        self.fc2 = nn.Conv2d(
            squeeze_c, input_c, kernel_size=1
        )  # fc2: expand_channel (Conv2d 1x1代替全连接层)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(
            x, output_size=(1, 1)
        )  # 自适应全局平均池化处理 x(B,C,128,128) -> scale(B,C,1,1)
        scale = self.fc1(scale)  # (B,C',1,1)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)  # (B,C,1,1)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x  # SE通道注意力机制与Conv3x3主分支结果相乘


# model = SqueezeExcitation(input_c=256, squeeze_factor=4)
# print(model)
# inputs = torch.ones([2, 256, 128, 128])
# outputs = model(inputs)
# print(outputs.shape)


# ******************** RepVGGPlus ********************
class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.up = nn.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(
            inputs, kernel_size=inputs.size(3)
        )  # input(B,C,128,128)   x(B,C,1,1)
        x = self.down(x)  # (B,C',1,1)
        x = F.relu(x)
        x = self.up(x)  # x(B,C,1,1)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)  # x(B,C,1,1)
        return inputs * x


model = SEBlock(input_channels=256, internal_neurons=64)
print(model)
inputs = torch.ones([2, 256, 128, 128])
outputs = model(inputs)
print(outputs.shape)
