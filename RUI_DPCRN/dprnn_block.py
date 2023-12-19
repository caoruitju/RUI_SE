# coding: utf-8
# Author：Rui Cao, TianRui Wang
# Date ：2023/12/19 
import torch


class DPRnn(torch.nn.Module):
    def __init__(self, input_ch, F_dim, hidden_ch):
        super(DPRnn, self).__init__()
        self.F_dim = F_dim
        self.input_size = input_ch
        self.hidden = hidden_ch
        self.intra_rnn = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden // 2, bidirectional=True,
                                       batch_first=True)
        self.intra_fc = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.intra_ln = torch.nn.LayerNorm([F_dim, hidden_ch])

        self.inter_rnn = torch.nn.LSTM(input_size=self.hidden, hidden_size=self.hidden, batch_first=True)
        self.inter_fc = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.inter_ln = torch.nn.LayerNorm([F_dim, hidden_ch])

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        B, C, F, T = x.size()
        x = x.permute(0, 3, 2, 1)  # B,T,F,C
        intra_in = torch.reshape(x, [B * T, F, C])
        intra_rnn_out, _ = self.intra_rnn(intra_in)
        intra_out = self.intra_ln(torch.reshape(self.intra_fc(intra_rnn_out), [B, T, F, C]))  # B,T,F,C
        intra_out = x + intra_out  # B,T,F,C
        inter_in = intra_out.permute(0, 2, 1, 3)  # B,F,T,C
        inter_in = torch.reshape(inter_in, [B * F, T, C])
        inter_rnn_out, _ = self.inter_rnn(inter_in)
        inter_out = self.inter_ln(
            torch.reshape(self.inter_fc(inter_rnn_out), [B, F, T, C]).permute(0, 2, 1, 3))  # B,T,F,C
        out = (intra_out + inter_out).permute(0, 3, 2, 1)
        return out


if __name__ == '__main__':
    test_inp = torch.randn(2, 128, 4, 34)
    test_model = DPRnn(input_ch=128, hidden_ch=128, F_dim=4)
    print(test_model(test_inp).size())
