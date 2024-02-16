import torch
import math

import yaml
from util import utils
import os
import sys
import random

import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from dataset.datasetV2 import Benchmark
import numpy as np

from metrics.latitude_weighted_loss import LatitudeLoss
from tqdm import tqdm
from jacksung.utils.log import LogClass, oprint
from jacksung.utils.time import RemainTime, Stopwatch, cur_timestamp_str
from datetime import datetime
import jacksung.utils.fastnumpy as fnp
from jacksung.utils.log import StdLog
from util.data_parallelV2 import BalancedDataParallel
from util.norm_util import Normalization
from metrics.metrics import Metrics
from einops import rearrange
from util.utils import EXCLUDE_DATE

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

    device, args = utils.parse_config()
    # definitions of model

    model = utils.get_model(args)

    # load pretrain
    if args.pretrain is not None:
        print('load pretrained model: {}!'.format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load(ckpt['model_state_dict'], strict=False)
    # definition of loss and optimizer
    loss_func = eval(args.loss)
    if args.fp == 16:
        eps = 1e-3
    elif args.fp == 64:
        eps = 1e-13
    else:
        eps = 1e-8
    optimizer = eval(f'torch.optim.{args.optimizer}(model.parameters(), lr=args.lr, eps=eps)')
    scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)
    # resume training
    if args.resume is not None:
        ckpt_files = os.path.join(args.resume, 'models', "model_latest.pt")
        if len(ckpt_files) != 0:
            ckpt = torch.load(ckpt_files)
            prev_epoch = ckpt['epoch']
            start_epoch = prev_epoch + 1
            model.load(ckpt['model_state_dict'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            stat_dict = ckpt['stat_dict']
            # reset folder and param
            experiment_path = args.resume
            experiment_model_path = os.path.join(experiment_path, 'models')
            print('Select {} file, resume training from epoch {}.'.format(ckpt_files, start_epoch))
        else:
            raise Exception(f'{os.path.join(args.resume, "models", "model_latest.pt")}中无有效的ckpt_files')
    else:
        start_epoch = 1
        # auto-generate the output log name
        experiment_name = None
        timestamp = cur_timestamp_str()
        experiment_name = '{}-{}'.format(args.model if args.log_name is None else args.log_name, timestamp)
        experiment_path = os.path.join(args.log_path, experiment_name)
        stat_dict = utils.get_stat_dict(
            (
                ('val-loss', float('inf'), '<'),
                ('RMSE', float('inf'), '<'),
                ('PSNR', float('0'), '>'),
                ('SSIM', float('0'), '>')
            )
        )
        # create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
        # save training parameters
        exp_params = vars(args)
        exp_params_name = os.path.join(experiment_path, 'config_saved.yml')
        with open(exp_params_name, 'w') as exp_params_file:
            yaml.dump(exp_params, exp_params_file, default_flow_style=False)
        model.init_model()
    model = model.to(device)
    if args.balanced_gpu0 >= 0:
        # balance multi gpus
        model = BalancedDataParallel(args.balanced_gpu0, model, device_ids=list(range(len(args.gpu_ids))))
    else:
        # multi gpus
        model = nn.DataParallel(model, device_ids=list(range(len(args.gpu_ids))))
    log_name = os.path.join(experiment_path, 'log.txt')
    warning_path = os.path.join(experiment_path, 'warning.txt')
    stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
    sys.stdout = StdLog(filename=log_name, common_path=warning_path)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total Number of Parameters:' + str(round(num_params / 1024 ** 2, 2)) + 'M')
    print('Data path: ' + args.data_path)
    train_dataset = \
        Benchmark(args.data_path,
                  datetime(year=args.train_start_date[0], month=args.train_start_date[1], day=args.train_start_date[2]),
                  datetime(year=args.train_end_date[0], month=args.train_end_date[1], day=args.train_end_date[2]),
                  exclude_date=EXCLUDE_DATE, skip_day=1, train=True, repeat=args.repeat)
    valid_dataset = \
        Benchmark(args.data_path,
                  datetime(year=args.valid_start_date[0], month=args.valid_start_date[1], day=args.valid_start_date[2]),
                  datetime(year=args.valid_end_date[0], month=args.valid_end_date[1], day=args.valid_end_date[2]),
                  exclude_date=EXCLUDE_DATE, skip_day=args.skip_day, train=False)
    # create dataset for training and validating
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=args.threads, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False, drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, num_workers=args.threads, batch_size=args.batch_size,
                                  shuffle=False, pin_memory=False, drop_last=False)
    # start training
    sw = Stopwatch()
    rt = RemainTime(args.epochs)
    cloudLogName = experiment_path.split(os.sep)[-1]
    log = LogClass(args.cloudlog == 'on')
    log.send_log('Start training', cloudLogName)
    log_every = max(len(train_dataloader) // args.log_lines, 1)
    norm = Normalization(args.data_path)
    norm.cldas_mean, norm.cldas_std, norm.era5_mean, norm.era5_std = \
        utils.data_to_device([norm.cldas_mean, norm.cldas_std, norm.era5_mean, norm.era5_std], device, args.fp)
    m = Metrics()
    m.psnr, m.ssim = utils.data_to_device([m.psnr, m.ssim], device, args.fp)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        opt_lr = scheduler.get_last_lr()
        print()
        print('##===============-fp{}- Epoch: {}, lr: {} =================##'.format(args.fp, epoch, opt_lr))
        train_dataloader.check_worker_number_rationality()
        # training the model
        for iter_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            topo, lr, hr = utils.data_to_device(batch, device, args.fp)
            lr, hr = norm.norm(lr), norm.norm(hr)
            # roll = random.randint(0, now_t.shape[-1] - 1)
            roll = 0
            y_ = model(topo, lr, roll)
            # print(former_t[0, 3, 7, 360, 720], now_t[0, 3, 7, 360, 720], y_[0, 3, 7, 360, 720])
            b, c, h, w = y_.shape
            loss = loss_func(y_, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            # print log
            if (iter_idx + 1) % log_every == 0:
                cur_steps = (iter_idx + 1) * args.batch_size
                total_steps = len(train_dataloader) * args.batch_size
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(args.epochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter_idx + 1)
                stat_dict['losses'].append(avg_loss)

                oprint('Epoch:{}, {}/{}, Loss: {:.4f}, T:{}'.format(
                    cur_epoch, cur_steps, total_steps, avg_loss, sw.reset()))
        # validating the model
        if epoch % args.test_every == 0:
            torch.set_grad_enabled(False)
            model = model.eval()
            epoch_loss = 0
            psnr = 0
            ssim = 0
            rmse = 0
            progress_bar = tqdm(total=len(valid_dataset), desc='Infer')
            count = 0
            for iter_idx, batch in enumerate(valid_dataloader):
                optimizer.zero_grad()
                topo, lr, hr = utils.data_to_device(batch, device, args.fp)
                lr, hr_norm = norm.norm(lr), norm.norm(hr)
                # roll = random.randint(0, now_t.shape[-1] - 1)
                roll = 0
                y_ = model(topo, lr, roll)
                # print(former_t[0, 3, 7, 360, 720], now_t[0, 3, 7, 360, 720], y_[0, 3, 7, 360, 720])
                b, c, h, w = y_.shape
                loss = loss_func(y_, hr_norm)
                y_ = norm.denorm(y_)
                # ['WIN', 'TMP', 'PRS', 'PRE']
                m_idx = 1
                y_ = rearrange(y_[:, m_idx, :, :], '(b c) h w->b c h w', c=1)
                hr = rearrange(hr[:, m_idx, :, :], '(b c) h w->b c h w', c=1)

                psnr += float(m.calc_psnr(y_, hr))
                ssim += float(m.calc_ssim(y_, hr))
                rmse += float(m.calc_rmse(y_, hr))
                epoch_loss += float(loss)
                count += 1
                progress_bar.update(len(lr))
            progress_bar.close()
            epoch_loss = epoch_loss / count
            psnr = psnr / count
            ssim = ssim / count
            rmse = rmse / count

            log_out = utils.make_best_metric(stat_dict,
                                             (
                                                 ('val-loss', float(epoch_loss)), ('RMSE', rmse), ('PSNR', psnr),
                                                 ('SSIM', ssim)
                                             ),
                                             epoch, (experiment_model_path, model, optimizer, scheduler),
                                             (log, args.epochs, cloudLogName))
            # print log & flush out
            print(log_out)
            # save stat dict
            # save training parameters
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
            torch.set_grad_enabled(True)
            model = model.train()
        # update scheduler
        scheduler.step()
        rt.update()
    log.send_log('Training Finished!', cloudLogName)
    utils.draw_lines(stat_dict_name)
