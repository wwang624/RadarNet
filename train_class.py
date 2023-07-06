import os
import time
import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from net_class import vanille_net, vanille_net_2
import matplotlib.pyplot as plt
from tools.solvedir import create_dir_for_new_model
from RadarNet.datasets.CRDataset_class import CRDataset
from RadarNet.datasets.collate_functions import cr_collate
from slice3d import range_fft, produce_RA_slice
from tools.visualization.result_viz import visualize_train_img, visualize_cfel_out
from tqdm import tqdm
from tools.metrics import compute_iou, precision_recall, f1_score, radar_score
from tools.loss import FocalLoss, dice_loss, generalized_dice_loss, bce_dice_loss, bce_focal_loss, focal_dice_loss, \
    mse_focal_loss

# model_v1:
# os.makedirs('E:\\RadarNet\\checkpoints_class')
train_data_dir = 'F:\\RadarNet\\dataset_class'
train_model_path = 'F:\\RadarNet\\checkpoints_class'
net_name = 'RadarNet_V2'
cp_path = None
epoch_start = 0
iter_start = 0
loss_name = 'mse_focal_loss'
model_dir, model_name = create_dir_for_new_model(net_name, train_model_path)
train_viz_path = os.path.join(model_dir, 'train_viz')
if not os.path.exists(train_viz_path):
    os.makedirs(train_viz_path)
writer = SummaryWriter(model_dir)
train_log_name = os.path.join(model_dir, "train.log")
val_log_name = os.path.join(model_dir, "val.log")
with open(val_log_name, 'w'):
    pass
with open(train_log_name, 'w'):
    pass
n_class = 3
n_epoch = 50
batch_size = 4
lr = 1e-5
print("Building dataloader ... (Mode: %s)" % "normal")
crdata_train = CRDataset(data_dir=train_data_dir, split='train')
train_size = int(0.8 * len(crdata_train))
test_size = len(crdata_train) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(crdata_train, [train_size, test_size])
# crdata_val = CRDataset(data_dir=val_data_dir, split='val')
seq_names = crdata_train.seq_names
index_mapping = crdata_train.index_mapping
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate,
                              drop_last=True)
val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=cr_collate,
                            drop_last=True)

if net_name == 'RadarNet_V1':
    radarnet = vanille_net(range_channel=128, angle_channel=128).cuda()
    print("Building model_V1 ... (%s)" % model_name)
elif net_name == 'RadarNet_V2':
    radarnet = vanille_net_2(range_channel=128, angle_channel=128).cuda()
    print("Building model_V2 ... (%s)" % model_name)

if loss_name == 'BCE_weighted':
    criterion = nn.BCELoss(reduction='none')
elif loss_name == 'focal_loss':
    criterion = FocalLoss(gamma=2, alpha=0.25)
elif loss_name == 'dice_loss':
    criterion = dice_loss()
elif loss_name == 'generalized_dice_loss':
    criterion = generalized_dice_loss()
elif loss_name == 'bce_dice_loss':
    criterion = bce_dice_loss()
elif loss_name == 'bce_focal_loss':
    criterion = bce_focal_loss(_lamda=1)
elif loss_name == 'focal_dice_loss':
    criterion = focal_dice_loss(_lamda=1)
elif loss_name == 'mse_focal_loss':
    criterion = mse_focal_loss(_lamda=0.5)
else:
    criterion = nn.BCELoss()
optimizer = optim.Adam(radarnet.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

iter_count = 0
loss_ave = 0

val_loss_average = 0
val_iter_count = 0
print("Model name: %s" % model_name)
print("Number of sequences to train: %d" % crdata_train.n_seq)
print("Training dataset length: %d" % len(crdata_train))
print("Batch size: %d" % batch_size)
print("Number of iterations in each epoch: %d" % int(len(crdata_train) / batch_size))

for epoch in range(epoch_start, n_epoch):
    tic_load = time.time()
    radarnet.train()
    for iter, data_dict in enumerate(tqdm(train_dataloader)):
        data = data_dict['radar_data']
        image_paths = data_dict['image_paths']
        confmap_gt = data_dict['anno']['confmaps']
        location = data_dict['anno']['obj_infos']
        location_gt = []
        for i in range(len(location)):
            for j in range(len(location[i])):
                location_gt.append(location[i][j][2])
                location_gt.append(location[i][j][3])
            if len(location[i]) <= 12:
                for k in range(12 - len(location[i])):
                    location_gt.append(0)
                    location_gt.append(0)
        location_gt = torch.tensor(location_gt).cuda()

        tic = time.time()
        optimizer.zero_grad()  # zero the parameter gradients
        confmap_preds, cefl_out, loc = radarnet(data.cuda())
        # confmap_show = confmap_preds.cpu().detach().numpy()
        # plt.imshow(confmap_show[0,0,0,:,:], origin='lower')
        # plt.show()
        # 将损失函数设计为有weight的BCELoss，weight为有目标的像素为0.8，无目标的像素为0.2
        loss_confmap = 0

        if loss_name == 'BCE_weighted':
            loss_weight = torch.zeros(confmap_gt.shape).cuda()
            loss_weight[confmap_gt == 1] = 0.7
            loss_weight[confmap_gt == 0] = 0.3  # 有目标的像素为0.8，无目标的像素为0.2
            loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda())
            loss_confmap = torch.mean(loss_confmap * loss_weight)
        else:
            # loss_confmap = 0
            loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda(), loc, location_gt.float())
        loss_confmap.backward()
        optimizer.step()

        tic_back = time.time()

        loss_ave = np.average([loss_ave, loss_confmap.item()], weights=[iter_count, 1])

        if iter % 100 == 0:
            # print statistics
            load_time = tic - tic_load
            back_time = tic_back - tic
            confmap_pred = confmap_preds.cpu().detach().numpy()
            confmap_gt = confmap_gt.cpu().detach().numpy()
            train_iou = compute_iou(confmap_pred, confmap_gt)
            train_pre, train_recall = precision_recall(confmap_pred, confmap_gt)
            train_f1 = f1_score(train_pre, train_recall)
            train_radar_score = radar_score(train_pre, train_recall)
            print('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.2f | back time: %.2f | iou: %.4f | '
                  'precision: %.4f | recall: %.4f | f1: %.4f | radar_score: %.4f' %
                  (epoch + 1, iter + 1, loss_confmap.item(), loss_ave, load_time, back_time, train_iou, train_pre,
                   train_recall, train_f1, train_radar_score))
            with open(train_log_name, 'a+') as f_log:
                f_log.write('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.2f | back time: %.2f | iou: %.4f'
                            'precision: %.4f | recall: %.4f | f1: %.4f | radar_score: %.4f\n' %
                            (epoch + 1, iter + 1, loss_confmap.item(), loss_ave, load_time, back_time, train_iou,
                             train_pre, train_recall, train_f1, train_radar_score))

            writer.add_scalar('loss/loss_all', loss_confmap.item(), iter_count)
            writer.add_scalar('loss/loss_ave', loss_ave, iter_count)
            writer.add_scalar('time/time_load', load_time, iter_count)
            writer.add_scalar('time/time_back', back_time, iter_count)
            writer.add_scalar('param/param_lr', scheduler.get_last_lr()[0], iter_count)
            writer.add_scalar('train_metrics/train_iou', train_iou, iter_count)
            writer.add_scalar('train_metrics/train_pre', train_pre, iter_count)
            writer.add_scalar('train_metrics/train_recall', train_recall, iter_count)
            writer.add_scalar('train_metrics/train_f1', train_f1, iter_count)
            writer.add_scalar('train_metrics/train_radar_score', train_radar_score, iter_count)

            cfel_out = cefl_out.cpu().detach().numpy()

            chirp_amp_curr = range_fft(data[0, 0, :, :, :].permute(2, 1, 0).numpy())
            chirp_amp_curr = produce_RA_slice(chirp_amp_curr)

            fig_name = os.path.join(train_viz_path,
                                    '%03d_%010d_%06d.png' % (epoch + 1, iter_count, iter + 1))
            cfel_dir = os.path.join(train_viz_path, 'cfel-%03d_%010d_%06d.png' % (epoch + 1, iter_count, iter + 1))
            img_path = image_paths[0][0]
            visualize_train_img(fig_name, img_path, chirp_amp_curr,
                                confmap_pred[0, :3, 0, :, :],
                                confmap_gt[0, :3, 0, :, :])
            visualize_cfel_out(cfel_dir, cfel_out, batch_size)

        # if (iter + 1) % 10000 == 0:
        #     # save current model
        #     print("saving current model ...")
        #     status_dict = {
        #         'model_name': model_name,
        #         'epoch': epoch + 1,
        #         'iter': iter + 1,
        #         'model_state_dict': radarnet.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss_confmap.item(),
        #         'loss_ave': loss_ave,
        #         'iter_count': iter_count,
        #     }
        #     save_model_path = '%s/epoch_%02d_iter_%010d.pkl' % (model_dir, epoch + 1, iter_count + 1)
        #     torch.save(status_dict, save_model_path)

        iter_count += 1
        tic_load = time.time()
    print('model validation ...')
    radarnet.eval()
    ave_iou = 0
    ave_pre = 0
    ave_rec = 0
    ave_f1 = 0
    ave_score = 0
    val_viz_path = os.path.join(train_viz_path, 'val')
    if not os.path.exists(val_viz_path):
        os.makedirs(val_viz_path)
    for iter, data_dict in enumerate(tqdm(val_dataloader)):
        data = data_dict['radar_data']
        image_paths = data_dict['image_paths']
        confmap_gt = data_dict['anno']['confmaps']
        location = data_dict['anno']['obj_infos']
        location_gt = []
        for i in range(len(location)):
            for j in range(len(location[i])):
                location_gt.append(location[i][j][2])
                location_gt.append(location[i][j][3])
            if len(location[i]) <= 12:
                for k in range(12 - len(location[i])):
                    location_gt.append(0)
                    location_gt.append(0)
        location_gt = torch.tensor(location_gt).cuda()

        confmap_preds, cefl_out, loc = radarnet(data.cuda())
        # val_loss_confmap = 0

        if loss_name == 'BCE_weighted':
            loss_weight = torch.zeros(confmap_gt.shape).cuda()
            loss_weight[confmap_gt == 1] = 0.7
            loss_weight[confmap_gt == 0] = 0.3
            val_loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda())
            val_loss_confmap = torch.mean(val_loss_confmap * loss_weight)
        else:
            val_loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda(), loc, location_gt.float())
        val_loss_average = np.average([val_loss_average, val_loss_confmap.item()], weights=[val_iter_count, 1])
        # 计算confmap_gt和confmap_preds的iou
        confmap_preds = confmap_preds.cpu().detach().numpy()
        confmap_gt = confmap_gt.cpu().detach().numpy()
        cfel_out = cefl_out.cpu().detach().numpy()
        iou = compute_iou(confmap_preds, confmap_gt)
        pre, recall = precision_recall(confmap_preds, confmap_gt)
        f1 = f1_score(pre, recall)
        score = radar_score(pre, recall)
        ave_iou = ave_iou + iou
        ave_pre = ave_pre + pre
        ave_rec = ave_rec + recall
        ave_f1 = ave_f1 + f1
        ave_score = ave_score + score
        if iter % 20 == 0:
            val_fig_name = os.path.join(val_viz_path,
                                        '%03d_%010d_%06d.png' % (epoch + 1, val_iter_count, iter + 1))
            chirp_amp_curr = range_fft(data[0, 0, :, :, :].permute(2, 1, 0).numpy())
            chirp_amp_curr = produce_RA_slice(chirp_amp_curr)
            visualize_train_img(val_fig_name, image_paths[0][0], chirp_amp_curr,
                                confmap_preds[0, :3, 0, :, :],
                                confmap_gt[0, :3, 0, :, :])
            visualize_cfel_out(
                os.path.join(val_viz_path, 'cfel-%03d_%010d_%06d.png' % (epoch + 1, val_iter_count, iter + 1)),
                cfel_out, batch_size)
    miou = ave_iou / (iter + 1)
    mpre = ave_pre / (iter + 1)
    mrec = ave_rec / (iter + 1)
    mf1 = ave_f1 / (iter + 1)
    mscore = ave_score / (iter + 1)
    print(
        'epoch %2d: val_loss: %.4f (%.4f) | miou: %.4f | mprecision: %.4f | mrecall: %.4f | mf1: %.4f | mscore: %.4f' %
        (epoch + 1, val_loss_confmap.item(), val_loss_average, miou, mpre, mrec, mf1, mscore))
    with open(val_log_name, 'a+') as f_log:
        f_log.write('epoch %2d: val_loss: %.4f (%.4f) | miou: %.4f | mprecision: %.4f | mrecall: %.4f | mf1: %.4f '
                    '| mscore: %.4f\n' %
                    (epoch + 1, val_loss_confmap.item(), val_loss_average, miou, mpre, mrec, mf1, mscore))
    writer.add_scalar('val_loss/val_loss_all', val_loss_confmap.item(), val_iter_count)
    writer.add_scalar('val_loss/val_loss_ave', val_loss_average, val_iter_count)
    writer.add_scalar('val_metrics/val_miou', miou, val_iter_count)
    writer.add_scalar('val_metrics/val_mpre', mpre, val_iter_count)
    writer.add_scalar('val_metrics/val_mrec', mrec, val_iter_count)
    writer.add_scalar('val_metrics/val_mf1', mf1, val_iter_count)
    writer.add_scalar('val_metrics/val_mscore', mscore, val_iter_count)

    val_iter_count += 1

    print("saving current epoch model ...")
    status_dict = {
        'model_name': model_name,
        'epoch': epoch,
        'iter': iter,
        'model_state_dict': radarnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_confmap.item(),
        'loss_ave': loss_ave,
        'iter_count': iter_count,
    }
    save_model_path = '%s/epoch_%02d_final.pkl' % (model_dir, epoch + 1)
    torch.save(status_dict, save_model_path)

    scheduler.step()

print('Training Finished.')
