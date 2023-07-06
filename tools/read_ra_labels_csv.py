import os
import math
import pandas as pd
import json
import numpy as np
from tools.mapping import cart2pol_ramap


def find_nearest(array, value):
    """Find nearest value to 'value' in 'array'."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def read_ra_labels_csv(seq_path, idx, Range, Angle):
    frame_idx = idx
    label_csv_name = seq_path
    range_grid = Range
    angle_grid = Angle
    data = pd.read_csv(label_csv_name, names=['uid', 'class', 'px', 'py', 'wid', 'len'], header=None)
    n_row, n_col = data.shape
    obj_info_list = []
    cur_idx = -1
    obj_info = []
    cur_idx = frame_idx
    for r in range(n_row):
        ang = data['px'][r]
        ra = data['py'][r]
        Class = data['class'][r]
        pol_range, pol_angle = cart2pol_ramap(ang, ra)  # 最后一个数据集可能x y反了
        #     if region_count != 0:
        #         region_shape_attri = json.loads(data['region_shape_attributes'][r])
        #         region_attri = json.loads(data['region_attributes'][r])
        #
        #         cx = region_shape_attri['cx']
        #         cy = region_shape_attri['cy']
        #         distance = range_grid_label[cy]
        #         angle = angle_grid_label[cx]
        #     if ra > 25.0 or ra < 1.0:
        #         continue
        #     if ang > 1.5708 or ang < -1.5708:
        #         continue
        range_idx, distance = find_nearest(range_grid, pol_range)
        angle_idx, angle = find_nearest(angle_grid, pol_angle)
        #         rng_idx, _ = find_nearest(range_grid, distance)
        #         agl_idx, _ = find_nearest(angle_grid, angle)
        #         try:
        #             class_str = region_attri['class']
        #         except:
        #             print("missing class at row %d" % r)
        #             continue
        #         try:
        #             class_id = class_ids[class_str]
        #         except:
        #             if class_str == '':
        #                 print("no class label provided!")
        #                 raise ValueError
        #             else:
        #                 class_id = -1000
        #                 print("Warning class not found! %s %010d" % (seq_path, frame_idx))
        obj_info.append([range_idx, angle_idx, distance, angle, frame_idx, Class])
    #
    return obj_info
