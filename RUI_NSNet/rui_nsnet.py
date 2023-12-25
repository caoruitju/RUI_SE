# coding: utf-8
# Author：Rui Cao, TianRui Wang
# Date ：2023/12/19 

import torch.nn as nn
import torch
import math
from utils.stft import *
import torch.nn.functional as torchF
from torch.nn.modules.activation import MultiheadAttention

class CausalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(CausalConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.left_pad = kernel_size[1] - 1
        # padding = (kernel_size[0] // 2, 0)
        padding = (kernel_size[0] // 2, self.left_pad)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding)

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        B, C, F, T = x.size()
        # x = F.pad(x, [self.left_pad, 0])
        return self.conv(x)[..., :T]


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding):
        super(CausalTransConvBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                                             stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        T = x.size(-1)
        conv_out = self.trans_conv(x)[..., :T]
        return conv_out

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

class IntegralAttention(nn.Module):
    def __init__(self, in_ch, u_path, n_head, freq_dim):
        super(IntegralAttention, self).__init__()
        self.in_ch = in_ch
        self.n_head = n_head
        self.freq_dim = freq_dim
        # 0.65R 79
        # 1R 59
        if u_path.find("nfft_1R.npy") != -1:
            temp = 59
        elif u_path.find("nfft.npy") != -1:
            temp = 79
        elif u_path.find("nfft_2R.npy") != -1:
            temp = 29
        if not os.path.exists(u_path):
            print(u_path)
            u_path = r"./U_512nfft_1R.npy"
            if not os.path.exists(u_path):
                u_path = r"./U_512nfft_1R.npy"
        self.register_buffer("u", torch.tensor([[np.load(u_path)[temp:, :].T]], dtype=torch.float)[:,:,:-1,:])  # 1x1x161x(672-79)
        # drawer.plot_mesh(self.u[0][0].data.T, "u")
        self.v_convs = nn.Sequential(
            nn.LayerNorm(freq_dim),
            nn.Conv2d(in_ch, self.in_ch * n_head, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.k_convs = nn.Sequential(
            nn.LayerNorm(freq_dim),
            nn.Conv2d(in_ch, self.in_ch * n_head, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.choosed_convs = nn.Sequential(
            nn.Conv2d(in_ch * n_head, self.in_ch * n_head, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.out_convs = nn.Sequential(
            nn.Conv2d(in_ch * n_head, self.in_ch, kernel_size=(1, 3), padding=(0, 1)),
        )

    def forward(self, x):
        # Harmonic integration
        v = self.v_convs(x)  # B,C*n_head,T,F -> V
        k = self.k_convs(x ** 2)  # B,C*n_head,T,F -> K
        atten = torch.matmul(k, self.u)  # B,C*n_head,T,candidates
        atten = torchF.softmax(atten, dim=-1) 
        H = torch.matmul(atten, self.u.permute(0, 1, 3, 2))
        choosed = self.choosed_convs(H)  # B,C*n_head,F,T
        v = choosed * v 
        return self.out_convs(v)

class HarmonicAttention(torch.nn.Module):
    def __init__(self, in_ch, out_ch, conv_ker, u_path, n_head, integral_atten=True, CFFusion=True, freq_dim=256):
        super(HarmonicAttention, self).__init__()

        self.conv_res = bool(in_ch == out_ch)
        self.out_ch = out_ch
        self.n_head = n_head
        self.integral_atten = integral_atten
        self.CFFusion = CFFusion
        self.in_ch = in_ch

        self.in_norm = nn.LayerNorm([in_ch, freq_dim])

        self.in_conv = nn.Sequential(
            CausalConv(in_ch, out_ch, kernel_size=conv_ker, stride=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )

        if self.integral_atten:
            self.ln0 = nn.LayerNorm(freq_dim)
            self.integral_attention = IntegralAttention(in_ch=out_ch, u_path=u_path, n_head=n_head, freq_dim=freq_dim)

        if self.CFFusion:
            self.ln1 = nn.LayerNorm(freq_dim)
            self.channel_atten = MultiheadAttention(embed_dim=freq_dim, num_heads=8)

            self.ln2 = nn.LayerNorm(self.out_ch)
            self.f_atten = MultiheadAttention(embed_dim=self.out_ch, num_heads=2 if self.out_ch >= 8 else 1)

        self.dprnn = DPRnn(input_ch=out_ch, F_dim=freq_dim, hidden_ch=out_ch)
        # self.t_module = LSTM(in_dim=freq_dim * in_ch, hidden_ch=freq_dim, binary=False)

    def forward(self, s):
        """
        s: B,C,F,T
        """
        s = self.in_norm(s.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # BCFT->BTCF->BCFT

        if self.conv_res:
            s = self.in_conv(s) + s
        else:
            s = self.in_conv(s)
        B, C, F, T = s.size()
        s_ = s.permute(0, 1, 3, 2)  # B,C,T,F

        if self.integral_atten:
            ia = self.ln0(s_) 
            s_ = s_ + self.integral_attention(ia)  # B,C,T,F 

        if self.CFFusion:
            # channel attention
            ch_atten = self.ln1(s_).permute(1, 0, 2, 3).reshape(self.out_ch, -1, F)  # C,B*T,F
            ch_atten = self.channel_atten(ch_atten, ch_atten, ch_atten)[0]
            ch_atten = ch_atten.reshape(self.out_ch, B, T, F).permute(1, 0, 2, 3)
            s_ = s_ + ch_atten

            # frequency attention
            f_atten = self.ln2(s_.permute(3, 0, 2, 1).reshape(F, -1, self.out_ch))  # F,B*T,C
            f_atten = self.f_atten(f_atten, f_atten, f_atten)[0]
            f_atten = f_atten.reshape(F, B, T, self.out_ch).permute(1, 3, 2, 0)
            s_ = s_ + f_atten

        # temporal modeling
        # out = self.t_module(s_).permute(0, 1, 3, 2)  # BCTF->BCFT
        out = self.dprnn(s_.permute(0, 1, 3, 2))
        return out

class RUI_NSNet(nn.Module):
    def __init__(self, nfft=512, hop_len=128):
        super(RUI_NSNet, self).__init__()
        self.nfft = nfft
        self.hop_len = hop_len
        self.fft_len = 512

        '''Pre_enhancement module(PEM)'''
        self.encoder = nn.Sequential(
            nn.Linear(257, 400),
            nn.PReLU(),
            nn.LayerNorm(400)
        )
        self.rnn = nn.Sequential(
            nn.GRU(input_size=400, hidden_size=400, batch_first=True, num_layers=2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(400, 600),
            nn.LayerNorm(600),
            nn.PReLU(),
            nn.Linear(600, 600),
            nn.LayerNorm(600),
            nn.PReLU(),
            nn.Linear(600, 257),
            nn.Sigmoid()
        )

        '''Underlying information extractor(UIE)'''
        self.extractor = nn.Sequential(
            HarmonicAttention(in_ch=2, out_ch=6, conv_ker=self.conv_ker, u_path=r"./U_512nfft_1R.npy", n_head=n_head_num, freq_dim=self.fft_len//2,
                              integral_atten=True, CFFusion=False),
            HarmonicAttention(in_ch=6, out_ch=12, conv_ker=self.conv_ker, u_path=r"./U_512nfft_1R.npy", n_head=n_head_num, freq_dim=self.fft_len//2,
                              integral_atten=True, CFFusion=False)
        )

        ''' Refinement'''
        self.refinement = nn.ModuleList()
        iter_num = 4  ## total number of refinement iterations
        self.iter_num = iter_num
        n_head_num = 4 
        self.conv_ker = (5, 2)

        for i in range(iter_num):
            self.refinement.append(nn.Sequential(
                HarmonicAttention(in_ch=14, out_ch=6, conv_ker=self.conv_ker, u_path=r"./U_512nfft_1R.npy", n_head=n_head_num, freq_dim=self.fft_len//2,
                                  integral_atten=True, CFFusion=True),
                CausalConv(6, 2, kernel_size=self.conv_ker, stride=(1, 1))
            ))


    def forward(self, x):
        # x : B, T
        real, imag = stft_splitter(x, n_fft=self.nfft, hop_len=self.hop_len)
        spec_complex = torch.stack([real, imag], dim=1)[:, :, :-1]  # B,2,F,T
        out = spec_complex

        feature_head = self.extractor(out) # B,12,F,T

        real = real.permute(0, 2, 1)
        imag = imag.permute(0, 2, 1)
        log_pow = torch.log((real ** 2 + imag ** 2).clamp_(min=1e-12)) / torch.log(torch.tensor(10.0)) # B,F,T

        ff_result = self.encoder(log_pow)
        rnn_result, _ = self.rnn(ff_result)
        mask = self.decoder(rnn_result)

        real_result = (real * mask).permute(0, 2, 1)
        imag_result = (imag * mask).permute(0, 2, 1)

        '''multiple refinement iterator'''
        refinement_out = torch.stack([real_result[:,:-1,:], imag_result[:,:-1,:]], dim= 1)
        residual = torch.stack([real_result[:,:-1,:], imag_result[:,:-1,:]], dim= 1) # B,2,F,T
        for idx in range(self.iter_num):
            feature_input = torch.cat((feature_head, residual),dim = 1)
            refinement = self.refinement[idx](feature_input)
            '''S-path'''
            residual = residual - refinement.detach()
            '''A-path'''
            refinement_out = refinement_out + refinement # B,2,F,T
        refinement_out = F.pad(refinement_out, [0, 0, 0, 1], value=1e-8)

        return stft_mixer(refinement_out[:,0], refinement_out[:,1], n_fft=self.fft_len, hop_len=self.hop_len)

