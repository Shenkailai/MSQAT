from gettext import npgettext
import logging
import platform
import random
import models
import dataloader
import argparse
import os
import ast
import pickle
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn import metrics
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_epoch(model_name, net, test_loader):
    with torch.no_grad():
       
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for batch_x, batch_y in tqdm(test_loader):
            pred = 0
            for i in range(5):
                x_d = batch_x.cuda()
                labels = batch_y
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                # x_d = five_point_crop(i, d_img=x_d, config=config)
                pred += net(x_d)

            pred /= 5
            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = metrics.mean_squared_error(np.squeeze(
            pred_epoch), np.squeeze(labels_epoch)) ** 0.5

        logging.info('Model:{}  ===== SRCC:{:.4} ===== PLCC:{:.4} ===== RMSE:{:.4}'.format(
            model_name, rho_s, rho_p, rmse))
        return rho_s, rho_p, rmse


print("I am process %s, running on %s: starting (%s)" %
      (os.getpid(), platform.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--datapath", type=str,
                    default=r'E:\SQA\NISQA_Corpus', help="evaluation data json")
parser.add_argument("--label-csv", type=str,
                    default=r'E:\SQA\NISQA_Corpus\NISQA_corpus_file.csv', help="csv with class labels")
parser.add_argument("--dataset", type=str, default="nisqa",
                    help="the dataset used", choices=["nisqa", "tencent", "pstn"])
parser.add_argument('-b', '--batch-size', default=8,
                    type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=0, type=int,
                    metavar='NW', help='# of workers for dataloading (default: 32)')

parser.add_argument('--load_pretrained_mdl_path', help='if use SSL audio spectrogram transformer model',
                    type=str, default=r'F:\NISQA\multi-dimension-attention-network\src\pre_models\SSAST-Base-Patch-400.pth')
parser.add_argument('--model_size', help='if use ImageNet pretrained audio spectrogram transformer model',
                    default='base')
parser.add_argument('--seed', type=int, default=20)

parser.add_argument('--eval', default="NISQA_TEST_FOR")

parser.add_argument('--eval_model_path', type=str,
                    default=r'F:\NISQA\multi-dimension-attention-network\nisqa_ssast_pre\ssast_fshape-16_tshape-16_100_4_mse_base_seed20_premodel-SSAST-Base-Patch-400.pth-repetitive-new_loss_scale0.13\epoch_82.pth')

args = parser.parse_args()

setup_seed(args.seed)
# transformer based model

# dataset spectrogram mean and std, used to normalize the input
if args.eval == 'NISQA_TEST_LIVETALK':
    norm_stats_val = {'nisqa': [-9.051971, 3.7531793]}
elif args.eval == 'NISQA_TEST_FOR':
    norm_stats_val = {'nisqa': [-8.937617, 4.2769117]}
elif args.eval == 'NISQA_TEST_P501':
    norm_stats_val = {'nisqa': [-9.90131, 4.708985]}
elif args.eval == 'NISQA_VAL_LIVE':
    norm_stats_val = {'nisqa': [-9.823734, 3.6818407]}
elif args.eval == 'NISQA_VAL_SIM':
    norm_stats_val = {'nisqa': [-8.027123, 4.3762627]}
elif args.eval == 'NISQA_VAL':
    norm_stats_val = {'nisqa': [-8.185567, 4.3552947]}
elif args.eval == 'tencent_with':
    norm_stats_val = {'tencent': [-8.642287, 4.199733]}
elif args.eval == 'tencent_without':
    norm_stats_val = {'tencent': [-9.084293, 5.4488106]}
target_length = {'nisqa': 1024, 'tencent': 1024}

val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': norm_stats_val[args.dataset][0], 'std': norm_stats_val[args.dataset][1],
                  'padding_mode': 'repetitive'}

val_loader = torch.utils.data.DataLoader(
    dataloader.NisqaDataset(
        args.datapath, label_csv=args.label_csv, audio_conf=val_audio_conf, isTrain=False, Eval=args.eval),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)



net = models.MANIQA(model_size='base', fshape=16, fstride=16, tshape=16,
                    tstride=16, load_pretrained_mdl_path=args.load_pretrained_mdl_path, ast_type='ssast')

    
net = torch.load(args.eval_model_path)
net = net.cuda()

loss, rho_s, rho_p, rmse = eval_epoch(
    args.eval_model_path, net,  val_loader)
print('Eval model of {}, SRCC:{}, PLCC:{}, RMSE:{}'.format(
    args.eval_model_path, rho_s, rho_p, rmse))
