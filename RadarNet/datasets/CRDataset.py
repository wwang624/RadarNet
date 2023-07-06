import os
import pickle
import numpy as np
from tqdm import tqdm
import scipy.io as io
from torch.utils import data

from .loaders import list_pkl_filenames_from_prepared


class CRDataset(data.Dataset):
    def __init__(self, data_dir, split, is_random_chirp=True, subset=None):
        # parameters settings
        self.data_dir = data_dir
        self.split = split
        self.win_size = 4
        if split == 'train' or split == 'valid':
            self.step = 1
            self.stride = 4
        else:
            self.step = 1
            self.stride = 8
        self.is_random_chirp = is_random_chirp
        self.n_chirps = 4

        self.chirp_ids = [60, 120, 180, 240]

        self.image_paths = []
        self.radar_paths = []
        self.obj_infos = []
        self.confmaps = []
        self.n_data = 0
        self.index_mapping = []

        if subset is not None:
            self.data_files = [subset + '.pkl']
        else:
            # self.data_files = list_pkl_filenames(config_dict['dataset_cfg'], split)
            self.data_files = list_pkl_filenames_from_prepared(data_dir)  # 暂时没分测试集
        self.seq_names = [name.split('.')[0] for name in self.data_files]
        self.n_seq = len(self.seq_names)

        split_folder = split
        for seq_id, data_file in enumerate(tqdm(self.data_files)):
            data_file_path = os.path.join(data_dir, data_file)
            data_details = pickle.load(open(data_file_path, 'rb'))
            if split == 'train' or split == 'valid':
                assert data_details['anno'] is not None
            n_frame = data_details['n_frame']
            self.image_paths.append(data_details['image_paths'])  # 加载所有图像数据
            self.radar_paths.append(data_details['radar_paths'])  # 加载所有雷达adc数据
            n_data_in_seq = (n_frame - (self.win_size * self.step - 1)) // self.stride + (
                1 if (n_frame - (self.win_size * self.step - 1)) % self.stride > 0 else 0)   # win_size具体指代什么？ 需要用到吗
            self.n_data += n_frame
            for data_id in range(n_frame):   # 一个.mat文件只使用一次 即只使用4个chirps
                self.index_mapping.append([seq_id, data_id])
            if data_details['anno'] is not None:
                self.obj_infos.append(data_details['anno']['meta_data'])
                self.confmaps.append(data_details['anno']['confmap'])

    def __len__(self):
        """Total number of data/label pairs"""
        return self.n_data

    def __getitem__(self, index):

        seq_id, data_id = self.index_mapping[index]
        seq_name = self.seq_names[seq_id]
        image_paths = self.image_paths[seq_id]  # 取对应的 eg:2019_04_09_cms1000中的图像和雷达数据
        radar_paths = self.radar_paths[seq_id]
        if len(self.confmaps) != 0:
            this_seq_obj_info = self.obj_infos[seq_id]  # 取对应的 eg:2019_04_09_cms1000中的标签数据即confmap
            this_seq_confmap = self.confmaps[seq_id]

        data_dict = dict(
            status=True,
            seq_names=seq_name,
            image_paths=[]
        )

        chirp_id = self.chirp_ids

        ramap_rsize = 128
        ramap_asize = 128

        # Load radar data
        rawdata = np.zeros([128, 8, 255], dtype=np.complex128)
        rawData = np.zeros([128, 8, 4], dtype=np.complex128)
        if isinstance(chirp_id, list):

            matdata = io.loadmat(radar_paths[data_id])
            radar_adc = matdata['adcData']
            ww = 0
            for pp in range(2):
                for kk in range(4):
                    rawdata[:, ww, :] = radar_adc[:, :, kk, pp]
                    ww = ww + 1
            rawData[:, :, 0] = rawdata[:, :, chirp_id[0]]
            rawData[:, :, 1] = rawdata[:, :, chirp_id[1]]
            rawData[:, :, 2] = rawdata[:, :, chirp_id[2]]
            rawData[:, :, 3] = rawdata[:, :, chirp_id[3]]
            rawData = np.transpose(rawData, (2, 1, 0))  # 4*8*128
            rawData = rawData.reshape((1, 4, 8, 128))
            assert rawData.shape == (1, self.n_chirps, 8, 128)
            data_dict['image_paths'].append(image_paths[data_id])

        data_dict['start_frame'] = data_id
        data_dict['radar_data'] = rawData

        # Load annotations
        if len(self.confmaps) != 0:
            confmap_gt = this_seq_confmap[data_id]
            confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            obj_info = this_seq_obj_info[data_id]

            data_dict['anno'] = dict(
                obj_infos=obj_info,
                confmaps=confmap_gt,
            )
        else:
            data_dict['anno'] = None

        return data_dict