import os
import torch
from torch.utils.tensorboard import SummaryWriter
from RadarNet.datasets.CRDataset_class import CRDataset
from tools.solvedir import create_dir_for_new_model
from torch.utils.data import DataLoader
from RadarNet.datasets.collate_functions import cr_collate
from net_class import vanille_net
from tqdm import tqdm
from slice3d import range_fft, produce_RA_slice
from tools.visualization.result_viz import visualize_train_img


train_data_dir = 'F:\\RadarNet\\dataset_test'
test_model_path = 'F:\\RadarNet\\checkpoints_class_test'
checkpoint_path = 'F:\\RadarNet\\checkpoints_class\\RadarNet_V1-20230415-071105\\epoch_36_final.pkl'
model_name = 'RadarNet_V1'
cp_path = None
epoch_start = 0
iter_start = 0
loss_name = 'bce_dice_loss'
n_class = 3
n_epoch = 50
batch_size = 4
lr = 1e-5
print("Building dataloader ... (Mode: %s)" % "normal")
crdata_train = CRDataset(data_dir=train_data_dir, split='train')
# train_size = int(0.8 * len(crdata_train))
# test_size = len(crdata_train) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(crdata_train, [train_size, test_size])
# crdata_val = CRDataset(data_dir=val_data_dir, split='val')
seq_names = crdata_train.seq_names
index_mapping = crdata_train.index_mapping
train_dataloader = DataLoader(crdata_train, batch_size=1, shuffle=False, num_workers=0, collate_fn=cr_collate,
                              drop_last=False)
radarnet = vanille_net(range_channel=128, angle_channel=128).cuda()
checkpoint = torch.load(checkpoint_path)
if 'optimizer_state_dict' in checkpoint:
    radarnet.load_state_dict(checkpoint['model_state_dict'])
else:
    radarnet.load_state_dict(checkpoint)
if 'model_name' in checkpoint:
    model_name = checkpoint['model_name']
radarnet.eval()
test_res_dir = os.path.join(os.path.join(test_model_path, 'mlms006'))
if not os.path.exists(test_res_dir):
    os.makedirs(test_res_dir)

# train_viz_path = os.path.join(test_res_dir, 'train_viz')
# if not os.path.exists(train_viz_path):
#     os.makedirs(train_viz_path)
for iter, data_dict in enumerate(tqdm(train_dataloader)):
    data = data_dict['radar_data']
    image_paths = data_dict['image_paths']
    confmap_gt = data_dict['anno']['confmaps']
    confmap_preds, cefl_out = radarnet(data.cuda())
    confmap_pred = confmap_preds.cpu().detach().numpy()
    confmap_gt = confmap_gt.cpu().detach().numpy()
    cfel_out = cefl_out.cpu().detach().numpy()

    chirp_amp_curr = range_fft(data[0, 0, :, :, :].permute(2, 1, 0).numpy())
    chirp_amp_curr = produce_RA_slice(chirp_amp_curr)

    fig_name = os.path.join(test_res_dir,
                            '%d.png' % (iter + 1))

    img_path = image_paths[0][0]
    visualize_train_img(fig_name, img_path, chirp_amp_curr,
                        confmap_pred[0, :3, 0, :, :],
                        confmap_gt[0, :3, 0, :, :])


overwrite = False
framerate = "20"

# model_res_root = test_res_dir
# video_root = os.path.join(test_model_path, "demos")
# if not os.path.exists(video_root):
#     os.makedirs(video_root)
#
#     # img_root = os.path.join(model_res_root, seq, 'train_viz')
# video_path = os.path.join(video_root + "_demo.mp4")
# cmd = "ffmpeg -r " + framerate + " -i " + model_res_root + "\\%d.png -c:v libx264 -vf fps=" + framerate + \
#       " -pix_fmt yuv420p " + video_path
# print(cmd)
# os.system(cmd)

#  ffmpeg -framerate 20 -f image2 -i F:\RadarNet\checkpoints_class_test\pbms002\%d.png -c:v libx265 -vf scale=-1:1024
#  -pix_fmt yuv420p output_pbms002.mp4
