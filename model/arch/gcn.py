import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import numpy as np
class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, kernel_sizes, embed_dim, last_kernel_num, drop_out, **params):
        super(CNN_Gate_Aspect_Text, self).__init__()

        self.embed_dim= embed_dim  # 就是最后一维size D

        self.kernel_sizes = kernel_sizes
        self.last_kernel_num = last_kernel_num
        self.drop_out_num = drop_out

        self.convs1 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.last_kernel_num, kernel_size=K,
                      padding=int((K - 1) / 2))
            for K in self.kernel_sizes])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.last_kernel_num, kernel_size=K,
                      padding=int((K - 1) / 2))#(in_channels=self.embed_dim, out_channels=self.last_kernel_num, kernel_size=K,padding=int((K - 1) / 2))
            for K in self.kernel_sizes])

        self.dropout = nn.Dropout(self.drop_out_num)
        self.fc_aspect = nn.Linear(self.embed_dim, self.last_kernel_num)
        self.fc_cla = nn.Linear(len(self.kernel_sizes)*self.last_kernel_num, 1)
        self.fc_reg = nn.Linear(len(self.kernel_sizes)*self.last_kernel_num, 1)

    def forward(self, feature):

        feature = feature.to(torch.float32)


        #print(f"Initial feature shape: {feature.shape}")
        transposed_feature = feature.transpose(1,2)
        #print(f"Transposed feature shape: {transposed_feature.shape}")

        # logger.info(feature.shape)
        aspect_v = feature.sum(1) / feature.size(1)
        # logger.info(f"aspect_v.shape:{aspect_v.shape}")
        x = [torch.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]

        #print(f"shape after conv1: {[i.shape for i in x]}")
        
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        # logger.info(f"x.shape:{x[0].shape}")
        # logger.info(f"y.shape:{y[0].shape}")
        x = [i*j for i, j in zip(x, y)]
        x0 = [F.max_pool1d(i, int(i.size(2))).squeeze(2) for i in x]
        #print(f"x0 shape pooling: {[i.shape for i in x0]}")
        x0 = [i.view(i.size(0), -1) for i in x0]
        x0 = torch.cat(x0, 1)
        #print(f"x0 shape final: {x0.shape}")
        #logger.info(x0.shape)
        # logger.info(x0.shape)
        logit = self.fc_reg(x0)  # (N,C)

        return logit    
             