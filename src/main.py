import logging
import platform
import random
import models
import dataloader
import argparse
import os
import ast
import sys
import time
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from sklearn import metrics
from config import Config
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


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    for batch_x, batch_y in tqdm(train_loader):
        x_d = batch_x.cuda()
        labels = batch_y
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()

        pred_d = net(x_d)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rmse = metrics.mean_squared_error(np.squeeze(
        pred_epoch), np.squeeze(labels_epoch)) ** 0.5

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}/ RMSE:{:.4}'.format(
        epoch + 1, ret_loss, rho_s, rho_p, rmse))

    return ret_loss, rho_s, rho_p, rmse


def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
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
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

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

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ===== RMSE:{:.4}'.format(
            epoch + 1, np.mean(losses), rho_s, rho_p, rmse))
        return np.mean(losses), rho_s, rho_p, rmse


def clipped_mse(y_hat, label, tau=0.5):
    mse = F.mse_loss(y_hat, label, reduction='none')
    threshold = torch.abs(y_hat - label) > tau
    mse = torch.mean(threshold * mse)
    return mse

def clipped_mae(y_hat, label, tau=0.5):
    mae = F.l1_loss(y_hat, label, reduction='none')
    threshold = torch.abs(y_hat - label) > tau
    mae = torch.mean(threshold * mae)
    return mae


if __name__ == '__main__':
    print("I am process %s, running on %s: starting (%s)" %
        (os.getpid(), platform.uname()[1], time.asctime()))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str,
                        default=r'E:\SQA\NISQA_Corpus', help="evaluation data json")
    parser.add_argument("--label-csv", type=str,
                        default=r'E:\SQA\NISQA_Corpus\NISQA_corpus_file.csv', help="csv with class labels")
    parser.add_argument("--dataset", type=str, default="nisqa",
                        help="the dataset used", choices=["audioset", "esc50", "speechcommands"])
    parser.add_argument("--exp_dir", type=str, default="./nisqa_ssast_pre",
                        help="directory to dump experiments")
    parser.add_argument("--log_dir", type=str, default="./nisqa_ssast_pre_logs",
                        help="directory to dump logs")
    parser.add_argument('-b', '--batch_size', default=4,
                        type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-w', '--num-workers', default=0, type=int,
                        metavar='NW', help='# of workers for dataloading (default: 32)')
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="number of maximum training epochs")
    # the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
    parser.add_argument("--fstride", type=int, default=16,
                        help="soft split freq stride, overlap=patch_size-stride")
    parser.add_argument("--tstride", type=int, default=16,
                        help="soft split time stride, overlap=patch_size-stride")
    parser.add_argument("--fshape", type=int, default=16,
                        help="soft split freq stride, overlap=patch_size-stride")
    parser.add_argument("--tshape", type=int, default=16,
                        help="soft split time stride, overlap=patch_size-stride")
    parser.add_argument('--load_pretrained_mdl_path', help='if use SSL audio spectrogram transformer model',
                        type=str, default=r'F:\NISQA\multi-dimension-attention-network\src\pre_models\SSAST-Base-Patch'
                                        r'-400.pth')
    parser.add_argument('--model_size', help='if use ImageNet pretrained audio spectrogram transformer model',
                        default='base')
    parser.add_argument('--ast_type', help='use ast or ssast',
                        default='ssast')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--pad', type=str, default='repetitive')
    parser.add_argument('--scale', type=float, default=0.13)
    parser.add_argument('-lt', '--loss_type', default='clipped_mse', type=str,
                        help='loss type (default: norm-in-norm)')
    parser.add_argument(
        '--comment', default="new_loss")

    args = parser.parse_args()

    setup_seed(args.seed)
    # transformer based model

    if args.pad == 'zero_padding':
    # dataset spectrogram mean and std, used to normalize the input
        norm_stats = {'nisqa': [-6.9492087, 5.004692]}
        norm_stats_val = {'nisqa': [-6.659985, 5.0349174]}
    elif args.pad == 'repetitive':
        norm_stats = {'nisqa': [-8.352813, 4.29885], 'pstn':[-7.476995,4.0516253], 'tencent':[-8.980769, 5.1074333]}
        norm_stats_val = {'nisqa': [-8.185567, 4.3552947], 'pstn':[-7.472444, 4.0493846], 'tencent':[-8.983544, 5.2011952]}
    target_length = {'nisqa': 1024, 'pstn': 1024, 'tencent':1024}
    # if add noise for data augmentation, only use for speech commands
    noise = {'nisqa': False, 'pstn': False, 'tencent':False}

    audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': args.freqm,
                'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode': 'train',
                'mean': norm_stats[args.dataset][0], 'std': norm_stats[args.dataset][1],
                'noise': noise[args.dataset], 'padding_mode': args.pad}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0,
                    'dataset': args.dataset,
                    'mode': 'evaluation', 'mean': norm_stats_val[args.dataset][0], 'std': norm_stats_val[args.dataset][1],
                    'noise': False, 'padding_mode': args.pad}

    train_loader = torch.utils.data.DataLoader(
        dataloader.NisqaDataset(
            args.datapath, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.NisqaDataset(
            args.datapath, label_csv=args.label_csv, audio_conf=val_audio_conf, isTrain=False),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)



    model_tag = 'ssast_fshape-{}_tshape-{}_{}_{}_{}_{}_seed{}_premodel-{}-{}-{}_scale{}'.format(
        args.fshape, args.tshape, args.n_epochs, args.batch_size, args.loss_type, args.model_size, args.seed, os.path.basename(
            args.load_pretrained_mdl_path), args.pad, args.comment, args.scale
    )
    net = models.MANIQA(model_size=args.model_size, fshape=args.fshape, fstride=args.fstride, tshape=args.tshape,
                        tstride=args.tstride, load_pretrained_mdl_path=args.load_pretrained_mdl_path, ast_type='ssast', scale=args.scale)
    # net = torch.load(
    #     r'F:\NISQA\multi-dimension-attention-network\nisqa_ssast_pre\ssast_fshape-128_tshape-2_100_4_mae_base_seed20_premodel-SSAST-Base-Frame-400.pth-repetitive-ast_tab_swim_branch_scale0.13\epoch_53.pth')
    net = net.cuda()
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logging.basicConfig(filename=os.path.join(
        args.log_dir, model_tag) + '.log', level=logging.DEBUG, format=LOG_FORMAT)
    print("\nCreating experiment directory: %s" % args.exp_dir)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    model_save_path = os.path.join(args.exp_dir, model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    print('Now starting training for {:d} epochs'.format(args.n_epochs))

    # config file
    config = Config({
        "tensorboard_path": "./output/tensorboard/{}/".format(model_tag),
    })
    # loss function
    if args.loss_type == 'mae':
        criterion = torch.nn.L1Loss()
    elif args.loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss_type == 'clipped_mse':
        criterion = clipped_mse
    elif args.loss_type == 'clipped_mae':
        criterion = clipped_mae
        
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=1e-5,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=0)
    writer = SummaryWriter(config.tensorboard_path)
    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    besst_rmse = 1
    for epoch in range(0, args.n_epochs):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p, rmse = train_epoch(
            epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)
        writer.add_scalar("RMSE", rmse, epoch)

        if (epoch + 1) % 1 == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p, rmse = eval_epoch(
                config, epoch, net, criterion, val_loader)
            print('Eval model of epoch{}, SRCC:{}, PLCC:{}, RMSE:{}'.format(
                epoch + 1, rho_s, rho_p, rmse))
            logging.info('Eval done...')

            if rho_s > best_srocc or rho_p > best_plcc or rmse < besst_rmse:
                best_srocc = rho_s
                best_plcc = rho_p
                besst_rmse = rmse
                # save weights
                
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}, RMSE:{}'.format(
                    epoch + 1, best_srocc, best_plcc, besst_rmse))
            torch.save(net, os.path.join(
                    model_save_path, 'epoch_{}.pth'.format(epoch + 1)))

        logging.info('Epoch {} done. Time: {:.2}min'.format(
            epoch + 1, (time.time() - start_time) / 60))
