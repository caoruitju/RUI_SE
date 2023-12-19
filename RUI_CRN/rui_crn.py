# coding: utf-8
# Author：Rui Cao, TianRui Wang
# Date ：2023/12/19 
import math
from utils.stft import *
import torch.nn.functional as torchF
from torch.nn.modules.activation import MultiheadAttention


def complex_cat(x1, x2):
    x1_real, x1_imag = torch.chunk(x1, 2, dim=1)
    x2_real, x2_imag = torch.chunk(x2, 2, dim=1)
    return torch.cat(
        [torch.cat([x1_real, x2_real], dim=1), torch.cat([x1_imag, x2_imag], dim=1)], dim=1
    )


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


class RUI_CRN(nn.Module):
    def __init__(self, rnn_hidden=128, win_len=512, hop_len=128, fft_len=512, win_type='hanning',
                 kernel_size=5, kernel_num=(16, 32, 64, 128, 128, 128)):
        super(RUI_CRN, self).__init__()
        self.rnn_hidden = rnn_hidden
        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.win_type = win_type

        self.kernel_size = kernel_size
        self.kernel_num = (2,) + kernel_num

        ''' Pre_enhancement module(PEM)'''
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    CausalConv(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        self.enhance = nn.LSTM(
            input_size=hidden_dim * self.kernel_num[-1],
            hidden_size=self.rnn_hidden,
            num_layers=1,
            dropout=0.0,
            batch_first=False
        )
        self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        CausalTransConvBlock(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        CausalTransConvBlock(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        )
                    )
                )
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

        '''Underlying information extractor'''
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
        real, imag = stft_splitter(x, n_fft=self.fft_len, hop_len=self.hop_len)
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan(imag / (real + 1e-8))
        phase_adjust = (real < 0).to(torch.int) * torch.sign(imag) * math.pi
        spec_phase = spec_phase + phase_adjust
        spec_complex = torch.stack([real, imag], dim=1)[:, :, :-1]  # B,2,F,T
        out = spec_complex

        feature_head = self.extractor(out) # B,12,F,T

        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            encoder_out.append(out)

        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        out = torch.reshape(out, [T, B, C * D])
        out, _ = self.enhance(out)
        out = self.transform(out)
        out = torch.reshape(out, [T, B, C, D])
        out = out.permute(1, 2, 3, 0)

        for idx in range(len(self.decoder)):
            out = torch.cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 0, 1], value=1e-8)
        mask_imag = F.pad(mask_imag, [0, 0, 0, 1], value=1e-8)
        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan(
            imag_phase / (real_phase + 1e-8)
        )
        phase_adjust = (real_phase < 0).to(torch.int) * torch.sign(imag_phase) * math.pi
        mask_phase = mask_phase + phase_adjust
        mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        real = est_mags * torch.cos(est_phase)
        imag = est_mags * torch.sin(est_phase)

        '''multiple refinement iterations'''
        refinement_out = torch.stack([real[:,:-1,:], imag[:,:-1,:]], dim= 1)
        residual = torch.stack([real[:,:-1,:], imag[:,:-1,:]], dim= 1) # B,2,F,T
        for idx in range(self.iter_num):
            feature_input = torch.cat((feature_head, residual),dim = 1)
            refinement = self.refinement[idx](feature_input)
            '''S-path'''
            residual = residual - refinement.detach()
            '''A-path'''
            refinement_out = refinement_out + refinement # B,2,F,T
        refinement_out = F.pad(refinement_out, [0, 0, 0, 1], value=1e-8)

        return stft_mixer(refinement_out[:,0], refinement_out[:,1], n_fft=self.fft_len, hop_len=self.hop_len)
    