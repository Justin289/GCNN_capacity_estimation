import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc2(torch.relu(self.fc1(out[:, -1, :])))
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class GatedCNN(nn.Module):
    def __init__(self,
                 seq_len,
                 n_layers,
                 kernel,
                 out_chs,
                 res_block_count,
                 ans_size):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count


        self.conv_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0))
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0))
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))

        self.conv = nn.ModuleList(
            [nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        self.conv_gate = nn.ModuleList(
            [nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])

        self.fc = nn.Linear(out_chs * seq_len * 4, ans_size)

    def forward(self, x):
        # x: (N, seq_len)

        bs, seq_len = x.shape[0], x.shape[1]  # batch size

        # CNN
        x = x.unsqueeze(1)  # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x)  # (bs, Cout, seq_len, 1)
        A += self.b_0.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_0(x)  # (bs, Cout, seq_len, 1)
        B += self.c_0.repeat(1, 1, seq_len, 1)
        h = A * F.sigmoid(B)  # (bs, Cout, seq_len, 1)
        res_input = h  #

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            A = conv(h) + self.b[i].repeat(1, 1, seq_len, 1)
            B = conv_gate(h) + self.c[i].repeat(1, 1, seq_len, 1)
            h = A * F.sigmoid(B)  # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0:  # size of each residual block
                h += res_input
                res_input = h

        h = h.view(bs, -1)  # (bs, Cout*seq_len)
        out = self.fc(h)  # (bs, ans_size)

        return out
