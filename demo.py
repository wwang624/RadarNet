import pickle
import scipy.io as io
import torch
import torch.nn as nn
# from RadarNet.t import CFEL
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io


class CFEL(nn.Module):
    def __init__(self, range_out_channels, angle_out_channels):
        super(CFEL, self).__init__()
        self.range_out_channels = range_out_channels
        self.angle_out_channels = angle_out_channels

        self.range_conv = nn.Conv2d(in_channels=1, out_channels=range_out_channels, kernel_size=(1, 128), stride=1,
                                    padding=0, dtype=torch.complex128, bias=False)
        self.angle_conv = nn.Conv2d(in_channels=1, out_channels=angle_out_channels, kernel_size=(1, 8), stride=1,
                                    padding=0, dtype=torch.complex128, bias=False)
        # nn.init.xavier_uniform_(self.range_conv.weight)
        # nn.init.xavier_uniform_(self.angle_conv.weight)
        range_weight = self.range_conv.weight
        angle_weight = self.angle_conv.weight
        perturb_range = np.random.normal(0, 0.1, (self.range_out_channels, 128))
        perturb_range = np.reshape(perturb_range, (self.range_out_channels, 1, 1, 128))
        perturb_angle = np.random.normal(0, 0.1, (self.angle_out_channels, 8))
        perturb_angle = np.reshape(perturb_angle, (self.angle_out_channels, 1, 1, 8))
        with torch.no_grad():
            f_samp_fast = 4e6
            f_samp_slow = 3.134e4
            q = 0
            for self.f_parm_fast in np.linspace(0, f_samp_fast, self.range_out_channels):
                for n in range(128):
                    range_weight[q, :, :, n] = np.exp(-1j * 2 * np.pi * (n + 1) * self.f_parm_fast / f_samp_fast)
                q = q + 1
            range_weight = range_weight + perturb_range
            # range_weight = range_weight[0:128, :, :, :]

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
        x_win = x_win.view(batch_size, 1, 128, w)
        x_win = self.angle_conv(x_win)
        x_win = x_win[0, :, :, 0].T
        # x_out = x_win.reshape(batch_size, 1, self.range_out_channels, self.angle_out_channels)
        x_out = x_win.reshape(batch_size, 1, 128, self.angle_out_channels)
        return x_out


path = r'F:\RadarNet\ADC-data\AWR1843 Automotive\2019_04_30_pcms001\radar_raw_frame\000041.mat'
matdata = io.loadmat(path)
rawdata = np.zeros([128,8,255],dtype=np.complex128)
radar_adc = matdata['adcData']
ww = 0
for pp in range(2):
    for kk in range(4):
        rawdata[:,ww,:] = radar_adc[:, :, kk, pp]
        ww=ww+1

# radar_adc = np.reshape(radar_adc,(128,255,8))
input = torch.tensor(rawdata[:, :, 240]).T
input = input.reshape(1, 1, 8, 128)
# input = torch.rand(batch_size, 1, w, h, dtype=torch.complex128)
cfel = CFEL(range_out_channels=128, angle_out_channels=128)
output = cfel(input)
fig = plt.figure(1)
img = output.detach().numpy()
# img = roll(img[0, 0, :, :], 32, 1)
plt.imshow(np.sqrt(np.real(img[0, 0, :, :]) ** 2 + np.imag(img[0, 0, :, :]) ** 2), origin='lower')
plt.show()
