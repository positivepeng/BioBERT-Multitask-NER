#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/4/30 9:38
@author: phil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, kernel_num=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super(CharCNN, self).__init__()

        V = vocab_size
        D = embed_dim

        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        # print("shape  after embed", x.shape)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # print("shape  after unsqueeze", x.shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        # print("shape  after relu", [xx.shape for xx in x])
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # print("shape  after pool", [xx.shape for xx in x])
        x = torch.cat(x, 1)
        # print("shape  after cat", x.shape)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        # print("shape  after dropout", x.shape)

        return x


if __name__ == "__main__":
    a = torch.tensor([[1, 2, 1, 2, 1, 3, 2, 4, 2, 1, 2, 1, 1] for i in range(10)])
    # (batch，max_word_len)
    model = CharCNN(vocab_size=5, embed_dim=10)
    output = model(a)
    print(output.shape)
