# coding: utf-8
# Author：WangTianRui
# Date ：2020/10/23 19:31
import matplotlib.pyplot as plt
import numpy as np
import os, torch
from pathlib import Path
import soundfile as sf
import matplotlib


def set_agg():
    matplotlib.use('Agg')


plt.figure(dpi=500)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# def load_best_param(log_file, model, index, gpu=False, test=True):
#     print(model)
#     noisy_abs, noisy_arg = stft_splitter(torch.randn(1, 16000))
#     denoised_abs = model(noisy_abs)
#     model_path = os.path.join(log_file, "_ckpt_epoch_%d.ckpt" % index)
#     if not os.path.exists(model_path):
#         exit("log path error:" + model_path)
#     if gpu:
#         ckpt = torch.load(model_path, map_location=torch.device("cuda:0"))
#     else:
#         ckpt = torch.load(model_path, map_location="cpu")
#     model.load_state_dict(ckpt)
#     # if test:
#     #     model.eval()
#     print("load param ok", model_path)
#     return model


def plot_mesh(img, title="", save_home="", cmap=None):
    img = img
    # print(img.shape[1])
    # img = img[:, int((150 / 626) * img.shape[1]):int((300 / 626) * img.shape[1])]
    # print(img.shape)
    fig, ax = plt.subplots()
    plt.title(title)
    fig.colorbar(plt.pcolormesh(range(img.shape[1]), range(img.shape[0]), img, cmap=cmap))
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
        plt.close()
        return
    plt.show()


def plot_spec_mesh(img, title="", save_home="", cmap=None):
    img = np.log(abs(img))
    # print(img.shape[1])
    # img = img[:, int((150 / 626) * img.shape[1]):int((300 / 626) * img.shape[1])]
    # print(img.shape)
    fig, ax = plt.subplots()
    plt.title(title)
    fig.colorbar(plt.pcolormesh(range(img.shape[1]), range(img.shape[0]), img, cmap=cmap))
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
        plt.close()
        return
    plt.show()


def plot_scatter(array, title="1", save_home=""):
    xs = np.arange(len(array))
    plt.scatter(xs, array)
    plt.title(title)
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
        plt.close()
        return
    plt.show()


def plot(array, title, save_home=""):
    plt.plot(array)
    plt.title(title)
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
        plt.close()
        return
    plt.show()


# def plot_3D(array,title, save_home=""):


def numParams(net):
    count = sum([int(np.prod(param.shape)) for param in net.parameters()])
    # print('Trainable parameter count: {:,d} -> {:.2f} MB'.format(count, count * 32 / 8 / (2 ** 20)))
    print('Trainable parameter count: {:,f} M'.format(count / 1e6))
    return count


def get_all_wavs(root):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".wav"):
            files.append(str(p))
        for s in p.rglob('*.wav'):
            files.append(str(s))
    return list(set(files))


def get_all_pics(root):
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".png"):
            files.append(str(p))
        for s in p.rglob('*.png'):
            files.append(str(s))
    return list(set(files))


def audiowrite(destpath, audio, sample_rate=16000):
    '''Function to write audio'''
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    sf.write(destpath, audio, sample_rate)
    return


class DoNoThing(torch.nn.Module):
    def __init__(self):
        super(DoNoThing, self).__init__()
        self.hop_len = 1

    def forward(self, x):
        return x
