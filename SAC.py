#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/5 21:33
@author: phil
"""

import torch.nn as nn
import torch
from torch.nn import Conv1d, Conv2d, BatchNorm2d
import torch.nn.functional as F
# import tensorflow as tf
import numpy as np


class SAC(nn.Module):
    def __init__(self, feature_dim, tag_fearure_dim, tag_num):
        super(SAC, self).__init__()
        self.Wn = nn.Parameter(torch.rand(tag_fearure_dim, feature_dim))
        self.Wa = nn.Parameter(torch.rand(feature_dim, (feature_dim+tag_fearure_dim)))

        self.tag_vecs = nn.Parameter(torch.rand(tag_num, tag_fearure_dim))

        self.dense = nn.Linear(feature_dim+tag_fearure_dim, feature_dim)

    def forward(self, x):
        # x shape (batch_size, seq_len, feature_dim)
        # print(x.shape, self.Wn.t().shape)
        # print(torch.matmul(x, self.Wn.t()).shape)
        e = torch.matmul(torch.matmul(x, self.Wn.t()), self.tag_vecs.t())
        a = F.softmax(e, dim=2)
        s = torch.matmul(a, self.tag_vecs)
        cated = torch.cat([x, s], dim=2)
        h = self.dense(cated)
        output = torch.tanh(h)  # shape (seq_len, n) 和输入维度相同

        return output, e


if __name__ == "__main__":
    # model = Cnn_extractor(feature_dim=32, filter_num=64)
    # x = torch.rand(8, 1, 128, 32)  # (batch_size, max_len, feature_num)
    # print(model(x).shape)
    x = torch.rand(32, 10, 20)
    mask = torch.rand(32, 10) > 0.5
    model = SAC(feature_dim=20, tag_fearure_dim=50, tag_num=2)
    output, e = model(x)
    loss = nn.CrossEntropyLoss()(e.reshape(-1, e.shape[-1]), mask.long().reshape(-1))
    print(output.shape, e.shape, mask.shape)
