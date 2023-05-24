__author__ = "DeathSprout"

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGnet(nn.Module):

    def __init__(self,in_chans=64,data_class = 2):  #
        super(EEGnet, self).__init__()

        in_filter = 1  #
        self.conv1 = nn.Conv2d(in_filter, 16, (1, in_chans), padding=0)  #
        self.batchnorm1 = nn.BatchNorm2d(16, False)  #
        # Layer 2   #
        self.conv2 = nn.Conv2d(16, 32, (64, 1))  #
        self.batchnorm2 = nn.BatchNorm2d(32, False)
        self.pooling1 = nn.AvgPool2d((1, 4))

        # Layer 3
        self.conv3 = nn.Conv2d(32, 32,(1,32))
        self.batchnorm3 = nn.BatchNorm2d(32, False)
        self.pooling2 = nn.AvgPool2d((2, 3))

        self.line1 = nn.Linear(384, 64)
        self.line2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        # Layer 2
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 2, 1, 3)
        x = self.pooling1(x)
        x = x.permute(0, 2, 1, 3)
        # Layer 3
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 2, 1, 3)
        x = self.pooling2(x)

        x = x.flatten(start_dim=1)
        x = self.line1(x)
        x = self.line2(x)

        return x