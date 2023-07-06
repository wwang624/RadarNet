import os
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import scipy.constants


radar_configs = {
    'ramap_rsize': 128,             # RAMap range size
    'ramap_asize': 128,             # RAMap angle size
    'ramap_vsize': 128,             # RAMap angle size
    'frame_rate': 30,
    'crop_num': 3,                  # crop some indices in range domain
    'n_chirps': 255,                # number of chirps in one frame
    'sample_freq': 4e6,
    'sweep_slope': 21.0017e12,
    'data_type': 'RISEP',           # 'RI': real + imaginary, 'AP': amplitude + phase
    'ramap_rsize_label': 122,       # TODO: to be updated
    'ramap_asize_label': 121,       # TODO: to be updated
    'ra_min_label': -60,            # min radar angle
    'ra_max_label': 60,             # max radar angle
    'rr_min': 1.0,                  # min radar range (fixed)
    'rr_max': 25.0,                 # max radar range (fixed)
    'ra_min': -90,                  # min radar angle (fixed)
    'ra_max': 90,                   # max radar angle (fixed)
    'ramap_folder': 'WIN_HEATMAP',
}


def confmap2ra(radar_configs, name, radordeg=None):
    """
    Map confidence map to range(m) and angle(deg): not uniformed angle
    :param name: 'range' for range mapping, 'angle' for angle mapping
    :return: mapping grids
    """
    # TODO: add more args for different network settings
    Fs = radar_configs['sample_freq']
    sweepSlope = radar_configs['sweep_slope']
    num_crop = radar_configs['crop_num']
    fft_Rang = radar_configs['ramap_rsize'] + 2*num_crop
    fft_Ang = radar_configs['ramap_asize']
    c = scipy.constants.speed_of_light

    if name == 'range':
        freq_res = Fs / fft_Rang
        freq_grid = np.arange(fft_Rang) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
        return rng_grid

    if name == 'angle':
        # for [-90, 90], w will be [-1, 1]
        w = np.linspace(math.sin(math.radians(radar_configs['ra_min'])),
                        math.sin(math.radians(radar_configs['ra_max'])),
                        radar_configs['ramap_asize'])
        if radordeg is None or radordeg == 'deg':
            agl_grid = np.degrees(np.arcsin(w))  # rad to deg
        elif radordeg == 'rad':
            agl_grid = np.arcsin(w)
        else:
            raise TypeError
        return agl_grid


n_angle = 128
n_vel = 128
n_range = 128
n_chirp = 255
n_rx = 8
n_sample = 128
noma_rcs = 30000
range_grid = confmap2ra(radar_configs, name='range')


def produce_RV_slice(data):
    hanning_win = np.hamming(n_vel)
    win_data1 = np.zeros([data.shape[0], data.shape[1], n_vel], dtype=np.complex128)
    win_data2 = np.zeros([data.shape[0], data.shape[1], n_vel], dtype=np.complex128)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            win_data1[i, j, :] = np.multiply(data[i, j, 0:n_vel], hanning_win)
            win_data2[i, j, :] = np.multiply(data[i, j, n_vel - 1:], hanning_win)

    fft_data_raw1 = np.fft.fft(win_data1, n_vel, axis=2)
    fft_data_raw1 = np.fft.fftshift(fft_data_raw1, axes=2)
    fft3d_data1 = np.sum(np.abs(fft_data_raw1), axis=1) / n_rx
    fft3d_data1 = np.expand_dims(fft3d_data1, axis=2)

    fft_data_raw2 = np.fft.fft(win_data2, n_vel, axis=2)
    fft_data_raw2 = np.fft.fftshift(fft_data_raw2, axes=2)
    fft3d_data2 = np.sum(np.abs(fft_data_raw2), axis=1) / n_rx
    fft3d_data2 = np.expand_dims(fft3d_data2, axis=2)

    # output format [range, velocity, 2chirps]
    fft3d_data = np.float32(np.concatenate((fft3d_data1, fft3d_data2), axis=2))
    return fft3d_data, fft_data_raw1, fft_data_raw2


def produce_VA_slice(rv_raw1, rv_raw2):
    hanning_win = np.hamming(n_rx)
    win_data1 = np.zeros([rv_raw1.shape[0], rv_raw1.shape[1], rv_raw1.shape[2]], dtype=np.complex128)
    win_data2 = np.zeros([rv_raw2.shape[0], rv_raw2.shape[1], rv_raw2.shape[2]], dtype=np.complex128)
    for i in range(rv_raw1.shape[0]):
        for j in range(rv_raw1.shape[2]):
            win_data1[i, :, j] = np.multiply(rv_raw1[i, :, j], hanning_win)
            win_data2[i, :, j] = np.multiply(rv_raw2[i, :, j], hanning_win)

    fft_data_raw1 = np.fft.fft(win_data1, n_angle, axis=1)
    fft3d_data1 = np.sum(np.abs(np.fft.fftshift(fft_data_raw1, axes=1)), axis=0) / rv_raw1.shape[0]
    fft3d_data1 = np.expand_dims(fft3d_data1, axis=2)

    fft_data_raw2 = np.fft.fft(win_data2, n_angle, axis=1)
    fft3d_data2 = np.sum(np.abs(np.fft.fftshift(fft_data_raw2, axes=1)), axis=0) / rv_raw2.shape[0]
    fft3d_data2 = np.expand_dims(fft3d_data2, axis=2)

    # output format [angle, velocity, 2chirps]
    fft3d_data = np.float32(np.concatenate((fft3d_data1, fft3d_data2), axis=2))
    return fft3d_data


def produce_RA_slice(data, filter_static=False, keep_complex=False):
    hanning_win = np.hamming(n_rx)
    win_data = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=np.complex128)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            win_data[i, :, j] = np.multiply(data[i, :, j], hanning_win)

    fft_data_raw = np.fft.fft(win_data, n_angle, axis=1)
    fft3d_data_cmplx = np.fft.fftshift(fft_data_raw, axes=1)
    if keep_complex is True:
        fft3d_data = fft3d_data_cmplx
    else:
        fft_data_real = np.expand_dims(fft3d_data_cmplx.real, axis=3)
        fft_data_imag = np.expand_dims(fft3d_data_cmplx.imag, axis=3)
        # output format [range, angle, chirps, real/imag]
        fft3d_data = np.float32(np.concatenate((fft_data_real, fft_data_imag), axis=3))
    if filter_static:
        fft3d_data = fft3d_data - np.mean(fft3d_data, axis=2, keepdims=True)

    return fft3d_data


def produce_RCSmap(data):
    hanning_win = np.hamming(n_rx)
    win_data = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=np.complex128)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            win_data[i, :, j] = np.multiply(data[i, :, j], hanning_win)

    fft_data_raw = np.fft.fft(win_data, n_angle, axis=1)
    fft3d_data_amp = np.abs(np.fft.fftshift(fft_data_raw, axes=1)) ** 2
    fft3d_data_amp = np.sum(fft3d_data_amp, axis=2) / data.shape[2]
    range_weight = np.tile(np.expand_dims(range_grid, axis=1) ** 4, (1, n_angle))
    rcs_data = np.multiply(range_weight, fft3d_data_amp) / noma_rcs

    return rcs_data


def range_fft(data):
    hanning_win = np.hamming(n_sample)
    win_data = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=np.complex128)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            win_data[:, i, j] = np.multiply(data[:, i, j], hanning_win)
    fft_data_raw = np.fft.fft(win_data, n_range, axis=0)

    return fft_data_raw


def save_ra_slice(data, save_dir_ra, new_file_name):
    for i in range(data.shape[2]):
        save_fod = os.path.join(save_dir_ra, str(i).zfill(4))
        if not os.path.exists(save_fod):
            os.makedirs(save_fod)
        save_dir = os.path.join(save_fod, new_file_name)
        np.save(save_dir, data[:, :, i, :])


def main():
    """
    This function preprocess the raw data and save the data to the local
    Input: RV data cube and RA data cube
    Output: RA slice (real and imaginary part of the first chirp after the denoise)
    RV slice (accumulate along the Angle domain)
    VA slice (accumulate along the Range domain)
    """
    root_dir = 'F:\\RadarNet\\ADC-data\\AWR1843 Automotive'
    dates = ['2019_04_09_bms1000','2019_04_09_cms1000','2019_04_09_css1000','2019_04_09_pms1000','2019_04_09_pms2000',
             '2019_04_09_pms3000','2019_04_30_cm1s000','2019_04_30_mlms000','2019_04_30_mlms001','2019_04_30_pbms002',
             '2019_04_30_pbss000','2019_04_30_pcms001','2019_05_09_bm1s007','2019_05_09_cm1s003','2019_05_09_mlms003',
             '2019_05_09_pbms004','2019_05_09_pcms002','2019_05_29_bcms000','2019_05_29_cm1s014','2019_05_29_mlms006',
             '2019_05_29_pbms007','2019_05_29_pcms005']
    raw_data = np.zeros([128,8,255],dtype=np.complex128)
    for capture_date in dates:
        # capture_date = '2019_04_09_pms2000'
        cap_folder_dir = os.path.join(root_dir, capture_date)
        seqs = sorted(os.listdir(cap_folder_dir))
        # seqs_dir = os.path.join(cap_folder_dir, seqs[1])
        # seqs = sorted(os.listdir(seqs_dir))

        seq_file = os.path.join(cap_folder_dir, seqs[1])
        files = sorted(os.listdir(seq_file))
        print('Processing ', capture_date)
        for idf, file in enumerate(files[0:len(files)]):
            print('data', file)
            file_dir = os.path.join(seq_file, file)
            file_dir = r'F:\RadarNet\ADC-data\AWR1843 Automotive\2019_04_30_pcms001\radar_raw_frame\000542.mat'
            mat = spio.loadmat(file_dir, squeeze_me=True)
            data = np.asarray(mat["adcData"])
            ww = 0
            for pp in range(2):
                for kk in range(4):
                    raw_data[:,ww,:] = data[:, :, kk, pp]
                    ww=ww+1
            # raw_data = raw_data.view(128,8,255)
            # Range FFT
            data = range_fft(raw_data)
            # generate RV slice
            # RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data) # (128, 128, 2)
            # # generate VA slice
            # VA_slice = produce_VA_slice(rv_raw1, rv_raw2)   # (128, 128, 2)
            # generate RA slice
            RA_slice = produce_RA_slice(data)   # (128, 128, 255, 2)

            # Create 1x3 sub plots
            # gs = gridspec.GridSpec(1, 3)
            # visualize 3 slices
            plt.figure(tight_layout=True)
            # ax = plt.subplot(gs[0, 0])  # row 0, col 0
            plt.imshow(np.sqrt(RA_slice[:, :, 0, 0] ** 2 + RA_slice[:, :, 0, 1] ** 2), origin='lower', aspect='auto')
            # ax.set_title("RA Slice")

            # ax2 = plt.subplot(gs[0, 1])  # row 0, col 1
            # plt.imshow(RV_slice[:, :, 0], origin='lower')
            # ax2.set_title("RV Slice")
            #
            # ax3 = plt.subplot(gs[0, 2])  # row 0, col 2
            # plt.imshow(VA_slice[:, :, 0])
            # ax3.set_title("VA Slice")
            plt.show()
            pass


if __name__ == '__main__':
    main()