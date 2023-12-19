import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
import os
import soundfile as sf


def stft_splitter(audio, n_fft=512, hop_len=128):
    with torch.no_grad():
        audio_stft = torch.stft(audio,
                                n_fft=n_fft,
                                hop_length=hop_len,
                                onesided=True,
                                return_complex=False)
        # print(audio_stft.size()) 
        return audio_stft[..., 0], audio_stft[..., 1] # 返回实部和虚部 B,F,T


def stft_mixer(real, imag, n_fft=512, hop_len=128):
    """
    real: B, F, T
    imag: B, F, T
    """
    # print(real.size())
    return torch.istft(
        torch.complex(real, imag),
        n_fft=n_fft, hop_length=hop_len, onesided=True
    )


if __name__ == "__main__":
    test_inp = torch.randn(1, 16000)
    real, imag = stft_splitter(test_inp)
    print(real.size(), imag.size())
    print(stft_mixer(real, imag).size())