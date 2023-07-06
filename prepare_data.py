import os
import pickle
from tools.read_ra_labels_csv import read_ra_labels_csv
from tools.mapping import confmap2ra
from tools.generate_confmaps import generate_confmaps


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

save_dir = 'F:\\RadarNet\\dataset1'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
root_dir = 'F:\\RadarNet\\ADC-data\\AWR1843 Automotive'
dates = ['2019_04_09_bms1000','2019_04_09_cms1000','2019_04_09_css1000','2019_04_09_pms1000','2019_04_09_pms2000',
         '2019_04_09_pms3000','2019_04_30_cm1s000','2019_04_30_mlms000','2019_04_30_mlms001','2019_04_30_pbms002',
         '2019_04_30_pbss000','2019_04_30_pcms001','2019_05_09_bm1s007','2019_05_09_cm1s003','2019_05_09_mlms003',
         '2019_05_09_pbms004','2019_05_09_pcms002','2019_05_29_bcms000','2019_05_29_cm1s014','2019_05_29_mlms006',
         '2019_05_29_pbms007']  # '2019_05_29_pcms005'标注label有问题
range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle')
for capture_date in dates:
    # if capture_date == '2019_04_30_pcms001' or '2019_05_29_cm1s014':
    #     continue  # test dataset
    # capture_date = '2019_04_09_pms2000'
    image_paths = []
    radar_paths = []
    anno_obj = {'meta_data': [], 'confmap': []}
    cap_folder_dir = os.path.join(root_dir, capture_date)
    seqs = sorted(os.listdir(cap_folder_dir))
    seq_label_file = os.path.join(cap_folder_dir, seqs[2])
    anno_files = sorted(os.listdir(seq_label_file))
    image_files = os.path.join(cap_folder_dir, seqs[0])
    radar_data_files = os.path.join(cap_folder_dir, seqs[1])
    save_path = os.path.join(save_dir, capture_date + '.pkl')
    print("Sequence %s saving to %s" % (capture_date, save_path))
    n_data = len(anno_files)
    for file in anno_files:
        num = int(file.split('.')[0])
        if not (os.path.exists(os.path.join(radar_data_files, '{:0>6d}.mat'.format(num))) and
                os.path.exists(os.path.join(image_files, '{:0>10d}.jpg'.format(num)))):
            print("label {} has not raw data or image".format(file))
            n_data -= 1
            continue
        else:
            image_paths.append(os.path.join(image_files, '{:0>10d}.jpg'.format(num)))
            radar_paths.append(os.path.join(radar_data_files, '{:0>6d}.mat'.format(num)))
            try:
                files_dir = os.path.join(seq_label_file, file)
                # anno_obj['metadata'] = load_anno_txt(seq_label_file, n_data, range_grid, angle_grid)
                obj_information = read_ra_labels_csv(files_dir, num, range_grid, angle_grid)
                anno_obj['meta_data'].append(obj_information)
            except Exception as e:
                print("Load sequence %s failed!" % file)
                print(e)
                continue
            anno_obj['confmap'].append(generate_confmaps(anno_obj['meta_data'][-1], range_grid, angle_grid, viz=False,
                                                         nclass=1))
    assert len(anno_obj['confmap']) == len(radar_paths)
    print('pack data {} correct!\n'.format(capture_date))
    data_dict = dict(
        data_root=root_dir,
        data_path=cap_folder_dir,
        seq_name=capture_date,
        n_frame=n_data,
        image_paths=image_paths,
        radar_paths=radar_paths,
        anno=anno_obj,
    )
    pickle.dump(data_dict, open(save_path, 'wb'))
    pass

