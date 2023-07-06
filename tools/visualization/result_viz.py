import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .fig_configs import fig, fp, symbols


def visualize_train_img(fig_name, img_path, input_radar, output_confmap, confmap_gt):
    fig = plt.figure(figsize=(8, 8))
    img_data = mpimg.imread(img_path)

    fig.add_subplot(2, 2, 1)
    plt.imshow(img_data.astype(np.uint8))

    fig.add_subplot(2, 2, 2)
    plt.imshow(np.sqrt(input_radar[:, :, 0, 0] ** 2 + input_radar[:, :, 0, 1] ** 2), origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 3)
    if output_confmap.shape[0] == 3:
        output_confmap = np.transpose(output_confmap, (1, 2, 0))
    output_confmap[output_confmap < 0] = 0
    plt.imshow(output_confmap, vmin=0, vmax=1, origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 4)
    if confmap_gt.shape[0] == 3:
        confmap_gt = np.transpose(confmap_gt, (1, 2, 0))
    plt.imshow(confmap_gt, vmin=0, vmax=1, origin='lower', aspect='auto')

    plt.savefig(fig_name)
    plt.close(fig)


def visualize_cfel_out(cfel_dir, cfel_out, batch):
    fig = plt.figure(figsize=(8, 8))
    # img_data = mpimg.imread(cfel_dir)

    fig.add_subplot(2, 2, 1)
    plt.imshow(cfel_out[0, 0, 0, :, :], origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 2)
    plt.imshow(cfel_out[0, 0, 1, :, :], origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 3)
    plt.imshow(cfel_out[0, 0, 2, :, :], origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 4)
    plt.imshow(cfel_out[0, 0, 3, :, :], origin='lower', aspect='auto')

    plt.savefig(cfel_dir)
    plt.close(fig)