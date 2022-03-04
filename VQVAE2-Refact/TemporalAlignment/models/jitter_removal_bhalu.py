#imports
import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn import functional as F

class Conv3d(nn.Module):
    """Extended nn.Conv1d for incremental dilated convolutions
    """

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv3d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm3d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            try:
                out += x
            except:
                print(out.size())
                print(x.size())
        return self.act(out)

class PostnetVQ1(nn.Module):

    def __init__(self):
        super(PostnetVQ1, self).__init__()

        self.channels = 3

        self.conv1 = nn.Sequential(
            Conv3d(self.channels, self.channels, kernel_size=5, stride=1, padding=2, residual=True),
            Conv3d(self.channels, self.channels, kernel_size=5, stride=1, padding=2, residual=True),
            Conv3d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, residual=True),
            )
        self.conv2 = nn.Conv3d(self.channels, self.channels, kernel_size=1, stride=1, padding=0)
        # self.act = nn.Tanh()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        o1 = self.conv1(x)
        # out = self.act(x + self.conv2(o1))
        out = x + self.conv2(o1)
        out = out.permute(0, 2, 1, 3, 4)
        # return out
        return nn.Tanh()(out)