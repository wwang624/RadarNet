import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from numpy.core import roll


class CFEL(nn.Module):
    def __init__(self, range_out_channels, angle_out_channels):
        super(CFEL, self).__init__()
        self.range_out_channels = range_out_channels
        self.angle_out_channels = angle_out_channels

        self.range_conv = nn.Conv2d(in_channels=1, out_channels=range_out_channels, kernel_size=(1, 800), stride=1,
                                    padding=0, dtype=torch.complex128, bias=False)
        self.angle_conv = nn.Conv2d(in_channels=1, out_channels=angle_out_channels, kernel_size=(1, 8), stride=1,
                                    padding=0, dtype=torch.complex128, bias=False)
        # nn.init.xavier_uniform_(self.range_conv.weight)
        # nn.init.xavier_uniform_(self.angle_conv.weight)
        range_weight = self.range_conv.weight
        angle_weight = self.angle_conv.weight
        perturb_range = np.random.normal(0, 0.1, (self.range_out_channels, 800))
        perturb_range = np.reshape(perturb_range, (self.range_out_channels, 1, 1, 800))
        perturb_angle = np.random.normal(0, 0.1, (self.angle_out_channels, 8))
        perturb_angle = np.reshape(perturb_angle, (self.angle_out_channels, 1, 1, 8))
        with torch.no_grad():
            f_samp_fast = 857.14e3
            f_samp_slow = (1 / 70) * 1e3
            q = 0
            for self.f_parm_fast in np.linspace(0, f_samp_fast, self.range_out_channels):
                for n in range(800):
                    range_weight[q, :, :, n] = np.exp(-1j * 2 * np.pi * (n + 1) * self.f_parm_fast / f_samp_fast)
                q = q + 1
            range_weight = range_weight + perturb_range
            range_weight = range_weight[0:64, :, :, :]

            p = 0
            for self.f_parm_slow in np.linspace(0, f_samp_slow, self.angle_out_channels):
                for m in range(8):
                    self.f_parm_slow = self.f_parm_slow + f_samp_slow / 2
                    if p >= 32:
                        self.f_parm_slow = self.f_parm_slow - f_samp_slow
                    angle_weight[p, :, :, m] = np.exp(-1j * 2 * np.pi * (m + 1) * self.f_parm_slow / f_samp_slow)
                p = p + 1
            angle_weight = angle_weight + perturb_angle

        self.range_conv.weight = nn.Parameter(range_weight, requires_grad=True)
        self.angle_conv.weight = nn.Parameter(angle_weight, requires_grad=True)

    def forward(self, x):
        batch_size, channel, w, h = x.shape
        #        x_out = torch.zeros((w, self.range_out_channels)).cuda()
        # x_win = F.conv3d(x, self.weight1, self.bias1, stride=(2, 1, 1), padding=(0, 0,0))
        # x_win = x_win.view(batch_size, self.range_out_channels, w)
        # x_win = F.conv3d(x_win, self.weight2, self.bias2, stride=(1,1), padding=(0,0))
        x_win = self.range_conv(x)
        # x_win = x_win.view(batch_size, 1, self.range_out_channels, w)
        x_win = x_win.view(batch_size, 1, 64, w)
        x_win = self.angle_conv(x_win)
        x_win = x_win[0, :, :, 0].T
        # x_out = x_win.reshape(batch_size, 1, self.range_out_channels, self.angle_out_channels)
        x_out = x_win.reshape(batch_size, 1, 64, self.angle_out_channels)
        return x_out


if __name__ == '__main__':
    batch_size = 1
    w = 8
    h = 800
    range_out_channels = 256
    angle_out_channels = 64
    path = r'F:\RadarNet\RadarNet\radar_adc4.mat'
    matdata = io.loadmat(path)
    radar_adc = matdata['radar_adc'].astype(np.complex128)
    input = torch.tensor(radar_adc).T
    input = input.reshape(batch_size, 1, w, h)
    # input = torch.rand(batch_size, 1, w, h, dtype=torch.complex128)
    cfel = CFEL(range_out_channels=range_out_channels, angle_out_channels=angle_out_channels)
    output = cfel(input)
    fig = plt.figure(1)
    img = output.detach().numpy()
    # img = roll(img[0, 0, :, :], 32, 1)
    plt.imshow(np.real(img[0, 0, :, :]) ** 2 + np.imag(img[0, 0, :, :]) ** 2, origin='lower', aspect='auto')
    fig1 = plt.figure(2)
    range_kernel = cfel.range_conv.weight.detach().numpy()
    range_kernel = range_kernel[:, 0, 0, :]
    plt.imshow(np.real(range_kernel[:, :]))
    # plt.imshow(np.real(range_kernel[:, :]) ** 2 + np.imag(range_kernel[:, :]) ** 2, origin='lower', aspect='auto')
    fig2 = plt.figure(3)
    angle_kernel = cfel.angle_conv.weight.detach().numpy()
    angle_kernel = angle_kernel[:, 0, 0, :]
    plt.imshow(np.real(angle_kernel[:, :].T))
    plt.show()

