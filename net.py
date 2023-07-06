import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io


class vanille_net(nn.Module):
    def __init__(self, range_channel, angle_channel):
        super(vanille_net, self).__init__()
        self.range_channel = range_channel
        self.angle_channel = angle_channel
        self.cfel = CFEL(self.range_channel, self.angle_channel)
        self.encode = RODEncode_RA()
        self.decode = RODDecode_RA()

    def forward(self, x):
        x_cfel = self.cfel(x)
        x_encode = self.encode(x_cfel)
        x_decode = self.decode(x_encode)
        return x_decode, x_cfel


class CFEL(nn.Module):
    def __init__(self, range_out_channels, angle_out_channels):
        super(CFEL, self).__init__()
        self.range_out_channels = range_out_channels
        self.angle_out_channels = angle_out_channels

        self.range_conv = nn.Conv3d(in_channels=1, out_channels=range_out_channels, kernel_size=(1, 1, 128), stride=(1, 1, 1),
                                    padding=0, dtype=torch.complex128, bias=False)
        self.angle_conv = nn.Conv3d(in_channels=1, out_channels=angle_out_channels, kernel_size=(1, 1, 8), stride=(1, 1, 1),
                                    padding=0, dtype=torch.complex128, bias=False)
        range_weight = self.range_conv.weight
        angle_weight = self.angle_conv.weight
        perturb_range = np.random.normal(0, 0.1, (self.range_out_channels, 128))
        perturb_range = np.reshape(perturb_range, (self.range_out_channels, 1, 1, 1, 128))
        perturb_angle = np.random.normal(0, 0.1, (self.angle_out_channels, 8))
        perturb_angle = np.reshape(perturb_angle, (self.angle_out_channels, 1, 1, 1, 8))
        with torch.no_grad():
            f_samp_fast = 4e6
            f_samp_slow = 1.307e-4
            q = 0
            for self.f_parm_fast in np.linspace(0, f_samp_fast, self.range_out_channels):
                for n in range(128):
                    range_weight[q, :, :, :, n] = np.exp(-1j * 2 * np.pi * (n + 1) * self.f_parm_fast / f_samp_fast)
                q = q + 1
            range_weight = range_weight + perturb_range

            p = 0
            for self.f_parm_slow in np.linspace(0, f_samp_slow, self.angle_out_channels):
                for m in range(8):
                    self.f_parm_slow = self.f_parm_slow + f_samp_slow / 2
                    if p >= 32:
                        self.f_parm_slow = self.f_parm_slow - f_samp_slow
                    angle_weight[p, :, :, :, m] = np.exp(-1j * 2 * np.pi * (m + 1) * self.f_parm_slow / f_samp_slow)
                p = p + 1
            angle_weight = angle_weight + perturb_angle

        self.range_conv.weight = nn.Parameter(range_weight, requires_grad=True)
        self.angle_conv.weight = nn.Parameter(angle_weight, requires_grad=True)

    def forward(self, x):
        batch_size, channel, d, w, h = x.shape
        x_out = torch.zeros((batch_size, 1, d, 128, h),dtype=torch.float).cuda()
        # x_win = F.conv3d(x, self.weight1, self.bias1, stride=(2, 1, 1), padding=(0, 0,0))
        # x_win = x_win.view(batch_size, self.range_out_channels, w)
        # x_win = F.conv3d(x_win, self.weight2, self.bias2, stride=(1,1), padding=(0,0))
        x_win = self.range_conv(x)
        x_win = x_win.permute(0,4,2,3,1)
        x_win = x_win.permute(0,1,2,4,3)
        x_win = x_win.reshape(batch_size, channel, d, self.range_out_channels, w)
        # x_win = x_win.view(batch_size, channel, d, self.range_out_channels, w)
        x_win = self.angle_conv(x_win)
        x_win = x_win.permute(0,4,2,3,1)
        for pp in range(batch_size):
            x_out[pp, 0, 0, :, :] = torch.sqrt(torch.real(x_win[pp, 0, 0, :, :]) ** 2 + torch.imag(x_win[pp, 0, 0, :, :]) ** 2)
            x_out[pp, 0, 1, :, :] = torch.sqrt(torch.real(x_win[pp, 0, 1, :, :]) ** 2 + torch.imag(x_win[pp, 0, 1, :, :]) ** 2)
            x_out[pp, 0, 2, :, :] = torch.sqrt(torch.real(x_win[pp, 0, 2, :, :]) ** 2 + torch.imag(x_win[pp, 0, 2, :, :]) ** 2)
            x_out[pp, 0, 3, :, :] = torch.sqrt(torch.real(x_win[pp, 0, 3, :, :]) ** 2 + torch.imag(x_win[pp, 0, 3, :, :]) ** 2)
        # x_out = x_out.reshape(batch_size, channel, d, self.range_out_channels, self.angle_out_channels)
        # x_out = x_win.reshape(batch_size, 1, 128, self.angle_out_channels)
        # x_out = x_win
        return x_out


class RODEncode_RA(nn.Module):

    def __init__(self):
        super(RODEncode_RA, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=1, out_channels=64,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(2, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(3, 5, 5), stride=(2, 1, 1), padding=(1, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(2, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2))
        # self.conv4a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv4b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        # self.conv5a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv5b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        # self.bn4a = nn.BatchNorm3d(num_features=64)
        # self.bn4b = nn.BatchNorm3d(num_features=64)
        # self.bn5a = nn.BatchNorm3d(num_features=64)
        # self.bn5b = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 1, W, 128, 128) -> (B, 64, W, 128, 128) Note: W~2W in this case
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/8, 16, 16)
        # x = self.relu(self.bn4a(self.conv4a(x)))
        # x = self.relu(self.bn4b(self.conv4b(x)))
        # x = self.relu(self.bn5a(self.conv5a(x)))
        # x = self.relu(self.bn5b(self.conv5b(x)))

        return x


class RODDecode_RA(nn.Module):

    def __init__(self):
        super(RODDecode_RA, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(2, 6, 6), stride=(1, 2, 2), padding=(0, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(3, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=32,
                                         kernel_size=(2, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt4 = nn.ConvTranspose3d(in_channels=32, out_channels=1,
                                         kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.conv5 = nn.Conv3d(in_channels=1, out_channels=1,
                                kernel_size=(4, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        # self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
        #                                   radar_configs['ramap_asize']), mode='nearest')

    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, 256, W/8, 16, 16) -> (B, 128, W/4, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/4, 32, 32) -> (B, 64, W/2, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, 64, W/2, 64, 64) -> (B, 32, W/2, 128, 128)
        x = self.prelu(self.convt4(x))
        x = self.conv5(x)
        # x = self.upsample(x)
        x = self.sigmoid(x)

        return x


if __name__ == '__main__':
    # input = np.random.random((1,1,4,8,128))
    # input = torch.tensor(input,dtype=torch.complex128)
    path = r'F:\RadarNet\ADC-data\AWR1843 Automotive\2019_04_30_pcms001\radar_raw_frame\000542.mat'
    matdata = io.loadmat(path)
    rawdata = np.zeros([128, 8, 255], dtype=np.complex128)
    rawData = np.zeros([4, 128, 8, 4], dtype=np.complex128)
    radar_adc = matdata['adcData']
    ww = 0
    for pp in range(2):
        for kk in range(4):
            rawdata[:, ww, :] = radar_adc[:, :, kk, pp]
            ww = ww + 1
    rawData[0, :, :, 0] = rawdata[:, :, 60]
    rawData[0, :, :, 1] = rawdata[:, :, 120]
    rawData[0, :, :, 2] = rawdata[:, :, 180]
    rawData[0, :, :, 3] = rawdata[:, :, 240]
    rawData[1, :, :, :] = rawData[0, :, :, :]
    rawData[2, :, :, :] = rawData[0, :, :, :]
    rawData[3, :, :, :] = rawData[0, :, :, :]
    rawData = np.transpose(rawData, (0, 3, 2, 1))
    # radar_adc = np.reshape(radar_adc,(128,255,8))
    input = torch.tensor(rawData)
    input = input.reshape(4, 1, 4, 8, 128)
    encode = RODEncode_RA()
    decode = RODDecode_RA()
    cfel = CFEL(128, 128)
    input = cfel(input)
    img = input.detach().numpy()
    plt.subplot(2, 2, 1)
    plt.imshow(img[0,0,0,:,:], origin='lower')
    plt.subplot(2, 2, 2)
    plt.imshow(img[1,0,1,:,:], origin='lower')
    plt.subplot(2, 2, 3)
    plt.imshow(img[2,0,2,:,:], origin='lower')
    plt.subplot(2, 2, 4)
    plt.imshow(img[3,0,3,:,:], origin='lower')
    plt.show()
    output = encode(input)
    target = decode(output)
    pass