import numpy as np
import math
import matplotlib.pyplot as plt


def generate_confmaps_class(metadata_dict, r_grid, a_grid, viz):
    confmaps = []

    n_obj = len(metadata_dict)
    obj_info = metadata_dict
    if n_obj == 0:
        confmap_gt = np.zeros((3, 128, 128), dtype=float)
        # confmap_gt[-1, :, :] = 0
    else:
        confmap_gt = generate_confmap_class(n_obj, obj_info, r_grid, a_grid)
        confmap_gt = normalize_confmap(confmap_gt)
        # confmap_gt = add_noise_channel(confmap_gt, dataset, config_dict)
    assert confmap_gt.shape == (3, 128, 128)
    if viz:
        visualize_confmap(confmap_gt, pps=metadata_dict)
    confmaps.append(confmap_gt)
    confmaps = np.array(confmaps)
    # confmaps = confmaps.reshape((1, 128, 128))
    return confmaps


def generate_confmaps(metadata_dict, r_grid, a_grid, viz):
    confmaps = []
    n_obj = len(metadata_dict)
    obj_info = metadata_dict
    if n_obj == 0:
        confmap_gt = np.zeros((1, 128, 128), dtype=float)
        confmap_gt[-1, :, :] = 0
    else:
        confmap_gt = generate_confmap(n_obj, obj_info, r_grid, a_grid)
        confmap_gt = normalize_confmap(confmap_gt)
        # confmap_gt = add_noise_channel(confmap_gt, dataset, config_dict)
    assert confmap_gt.shape == (1, 128, 128)
    if viz:
        visualize_confmap(confmap_gt, pps=metadata_dict)
    confmaps.append(confmap_gt)
    confmaps = np.array(confmaps)
    # confmaps = confmaps.reshape((1, 128, 128))
    return confmaps


def generate_confmap_class(n_obj, obj_info, R_grid, A_grid, gaussian_thres=36):  # 36
    """
    Generate confidence map a radar frame.
    :param n_obj: number of objects in this frame
    :param obj_info: obj_info includes metadata information
    :param dataset: dataset object
    :param config_dict: rodnet configurations
    :param gaussian_thres: threshold for gaussian distribution in confmaps
    :return: generated confmap
    """
    confmap_sigmas = [15, 20, 30]
    confmap_sigmas_interval = [[5, 15], [8, 20], [10, 30]]
    confmap_length = [1, 2, 3]

    range_grid = R_grid
    angle_grid = A_grid

    confmap = np.zeros((3, 128, 128), dtype=float)
    for objid in range(n_obj):
        rng_idx = obj_info[objid][0]
        agl_idx = obj_info[objid][1]
        class_name = obj_info[objid][5]
        if class_name == 0:  # person
            class_id = 0
        elif class_name == 80:  # cyclist
            class_id = 1
        else:  # car
            class_id = 2
        sigma = 2 * np.arctan(confmap_length[class_id] / (2 * range_grid[rng_idx])) * confmap_sigmas[class_id]
        sigma_interval = confmap_sigmas_interval[class_id]
        if sigma > sigma_interval[1]:
            sigma = sigma_interval[1]
        if sigma < sigma_interval[0]:
            sigma = sigma_interval[0]
        for i in range(128):
            for j in range(128):
                distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma ** 2
                if distant < gaussian_thres:  # threshold for confidence maps
                    value = np.exp(- distant / 2) / (2 * math.pi)
                    confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]

    return confmap


def generate_confmap(n_obj, obj_info, R_grid, A_grid, gaussian_thres=10):  # 36
    """
    Generate confidence map a radar frame.
    :param n_obj: number of objects in this frame
    :param obj_info: obj_info includes metadata information
    :param dataset: dataset object
    :param config_dict: rodnet configurations
    :param gaussian_thres: threshold for gaussian distribution in confmaps
    :return: generated confmap
    """
    confmap_sigmas = 15
    confmap_sigmas_interval = [5, 15]
    confmap_length = 1

    range_grid = R_grid
    angle_grid = A_grid

    confmap = np.zeros((1, 128, 128), dtype=float)
    for objid in range(n_obj):
        rng_idx = obj_info[objid][0]
        agl_idx = obj_info[objid][1]
        sigma = 2 * np.arctan(confmap_length / (2 * range_grid[rng_idx])) * confmap_sigmas
        sigma_interval = confmap_sigmas_interval
        if sigma > sigma_interval[1]:
            sigma = sigma_interval[1]
        if sigma < sigma_interval[0]:
            sigma = sigma_interval[0]
        for i in range(128):
            for j in range(128):
                distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma ** 2
                if distant < gaussian_thres:  # threshold for confidence maps
                    value = np.exp(- distant / 2) / (2 * math.pi)
                    confmap[0, i, j] = value if value > confmap[0, i, j] else confmap[0, i, j]

    return confmap


def normalize_confmap(confmap):
    conf_min = np.min(confmap)
    conf_max = np.max(confmap)
    if conf_max - conf_min != 0:
        confmap_norm = (confmap - conf_min) / (conf_max - conf_min)
    else:
        confmap_norm = confmap
    return confmap_norm


def visualize_confmap(confmap, pps):
    if len(confmap.shape) == 2:
        plt.imshow(confmap, origin='lower', aspect='auto')
        for pp in pps:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.show()
        return
    else:
        n_channel, _, _ = confmap.shape
    if n_channel == 1:
        confmap_viz = np.transpose(confmap, (1, 2, 0))
    else:
        print("Warning: wrong shape of confmap!")
        return
    plt.imshow(confmap_viz[:,:,0], origin='lower')
    # for pp in range(len(pps)):
    #     plt.scatter(pps[pp][1], pps[pp][0], s=5, c='white')
    plt.show()