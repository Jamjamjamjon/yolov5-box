"""
for CPU
虽然有许多轻量级网络在基于ARM的设备上的推断速度很快，但很少有网络考虑到Intel CPU上的速度，特别是在启用了MKLDNN之类的加速策略时。
许多提高模型精度的方法在ARM设备上不会增加太多的推理时间，但是当切换到Intel CPU设备时，情况会有所不同。

1. 使用MobileNetV1提到的DepthSepConv作为基本块。该块没有shortcut方式之类的操作，
因此没有concat或elementwise-add之类的附加操作，这些操作不仅会降低类的推理速度模型，而且对小模型也不会提高精度。
2. 该块经过Intel CPU加速库的深度优化，推理速度可以超过其他轻量级块，如 inverted-block或shufflenet-block。


3.激活函数
EfficientNet => Swish
MobileNetV3 => HSwish，从而避免了大量的指数运算。
作者还将BaseNet中的激活函数从ReLU替换为HSwish，性能有了很大的提高，而推理时间几乎没有改变。

4. SE block
将SE模块添加到网络尾部附近的模块中。这带来了一个更好的精度-速度平衡。
与MobileNetV3一样，SE模块的2层激活函数分别为ReLU和HSigmoid。

"""


import math
import torch
import torch.nn as nn
import numpy as np


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



NET_CONFIG = {
    # k, in_c, out_c, s, use_se
    "blocks2":[[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}
BLOCK_LIST = ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]

def make_divisible_LC(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# h_swish
class HardSwish(nn.Module):         
    def __init__(self, inplace=True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x+3) / 6

# h_ sigmoid
class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        # self.relu6 = nn.ReLU6(inplace=inplace)
        self.HardSwish = HardSwish(inplace=inplace)

    def forward(self, x):
        # return (self.relu6(x+3)) / 6
        return x * self.HardSwish(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HardSigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        return x * y


class DepthwiseSeparable(nn.Module):
    # depth-wise + point-wise
    def __init__(self, inp, oup, dw_size, stride, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.stride = stride
        self.inp = inp
        self.oup = oup
        self.dw_size = dw_size
        self.depthwise_pointwise = nn.Sequential(
            nn.Conv2d(self.inp, self.inp, kernel_size=self.dw_size, stride=self.stride,
                      padding=autopad(self.dw_size, None), groups=self.inp, bias=False),
            nn.BatchNorm2d(self.inp),
            HardSwish(),

            nn.Conv2d(self.inp, self.oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.oup),
            HardSwish(),
        )
        self.se = SELayer(self.oup)

    def forward(self, x):
        x = self.depthwise_pointwise(x)
        if self.use_se:
            x = self.se(x)
        return x

class PPLC_Conv(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(3, out_channels=make_divisible_LC(16 * self.scale),
                               kernel_size=3, stride=2, padding=1, bias=False)
    def forward(self, x):
        return self.conv(x)

class PPLC_Block(nn.Module):
    def __init__(self, scale, block_num):
        super().__init__()
        self.scale = scale
        self.block_num = BLOCK_LIST[block_num]
        self.block = nn.Sequential(*[
            DepthwiseSeparable(inp=make_divisible_LC(in_c * self.scale),
                               oup=make_divisible_LC(out_c * self.scale),
                               dw_size=k, stride=s, use_se=use_se)
            for i, (k, in_c, out_c, s, use_se) in enumerate(NET_CONFIG[self.block_num])
        ])
    def forward(self, x):
        return self.block(x)


