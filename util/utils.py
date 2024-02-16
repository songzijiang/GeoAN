import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

import os
import rasterio
from rasterio.transform import from_origin
import yaml
import argparse
from jacksung.utils.multi_task import type_thread, type_process, MultiTasks
import jacksung.utils.fastnumpy as fnp
from tqdm import tqdm
from datetime import datetime, timedelta
from matplotlib.ticker import MaxNLocator
from models.GeoAN.m_networkV2 import GeoAN
# from models.unet.unet_model import UNet
# from models.swinir.network_swinir import SwinIR
from PIL import Image
import netCDF4 as nc
import warnings

# 缺失
# 2020-04-24 0
# 2021-11-02 2
# 2023-10-10 0
# 2023-10-11 0
# 2023-10-12 0
# 2023-10-13 0
# 2023-10-14 0
# 2023-10-15 0
# 2023-10-16 0
#
# 文件不完整
# 2020-12-17-WIN
# 2020-12-19-TMP
# 2023-05-22-TMP
# 2023-06-27-WIN
#
# nan值
# 2020-10-15-WIN
# 2021-01-16-WIN
EXCLUDE_DATE = ['2020-4-24', '2021-11-02', '2020-12-17', '2020-12-19', '2020-10-15', '2021-01-16',
                '2023-05-22', '2023-06-27', '2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13',
                '2023-10-14', '2023-10-15', '2023-10-16']
EXCLUDE_DATE = tuple([datetime.strptime(s, '%Y-%m-%d') for s in EXCLUDE_DATE])


def get_model(args):
    # definitions of model
    if args.model in ['lgan', 'geoan']:
        model = GeoAN(window_sizes=args.window_sizes, n_geoab=args.n_geoab, c_in=4, c_geoan=args.c_geoan,
                      r_expand=args.r_expand, down_sample=args.down_sample, num_heads=args.num_heads)
    elif args.model == 'unet':
        model = None
        # model = UNet(c_in=4, down_sample=args.down_sample)
    elif args.model == 'swinir':
        model = None
        # window_size = 10
        # height = (1040 // args.down_sample // window_size + 1) * window_size
        # width = (1600 // args.down_sample // window_size + 1) * window_size
        # model = SwinIR(upscale=args.down_sample, img_size=(height, width), in_chans=4,
        #                window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
        #                embed_dim=108, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    elif args.model == 'deeplab':
        model = None
    else:
        model = None
    if args.fp == 16:
        model.half()
    elif args.fp == 64:
        model.double()
    return model


def load_model(model, state_dict, strict=True):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        name = name[name.index('.') + 1:]
        if name in own_state.keys():
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
                # own_state[name].requires_grad = False
            except Exception as e:
                err_log = f'While copying the parameter named {name}, ' \
                          f'whose dimensions in the model are {own_state[name].size()} and ' \
                          f'whose dimensions in the checkpoint are {param.size()}.'
                if not strict:
                    print(err_log)
                else:
                    raise Exception(err_log)
        elif strict:
            raise KeyError(f'unexpected key {name} in {own_state.keys()}')
        else:
            print(f'{name} not loaded by model')


def get_stat_dict(metrics):
    stat_dict = {
        'epochs': 0, 'losses': [], 'ema_loss': 0.0, 'metrics': {}}
    for idx, metrics in enumerate(metrics):
        name, default_value, op = metrics
        stat_dict['metrics'][name] = {'value': [], 'best': {'value': default_value, 'epoch': 0, 'op': op}}
    return stat_dict


def data_to_device(datas, device, fp=32):
    outs = []
    for data in datas:
        if fp == 16:
            data = data.type(torch.HalfTensor)
        if fp == 64:
            data = data.type(torch.DoubleTensor)
        data = data.to(device)
        outs.append(data)
    return outs


def draw_lines(yaml_path):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print('[TemporaryTag]Producing the LinePicture of Log...', end='[TemporaryTag]\n')
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    # 创建图表
    m_len = len(yaml_args['metrics'])
    plt.figure(figsize=(10 * (m_len + 1), 6))  # 设置图表的大小
    x = np.array(range(1, yaml_args['epochs'] + 1))
    for idx, d in enumerate(yaml_args['metrics'].items()):
        m_name, m_value = d
        y = np.array(m_value['value'])
        # 生成数据
        plt.subplot(1, m_len + 1, idx + 1)
        plt.plot(x, y)
        plt.title(m_name)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    y = np.array(yaml_args['losses'])
    plt.subplot(1, m_len + 1, m_len + 1)
    scale = len(y) / yaml_args['epochs']
    x = np.array(range(1, len(y) + 1)) / scale
    plt.plot(x, y)
    plt.title('Loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # 添加图例
    # plt.legend()
    plt.savefig(os.path.join(os.path.dirname(yaml_path), 'Metrics.jpg'))


def make_best_metric(stat_dict, metrics, epoch, save_model_param, server_log_param):
    save_model_flag = False
    experiment_model_path, model, optimizer, scheduler = save_model_param
    log, epochs, cloudLogName = server_log_param

    for name, m_value in metrics:
        stat_dict['metrics'][name]['value'].append(m_value)
        inf = float('inf')
        if eval(str(m_value) + stat_dict['metrics'][name]['best']['op'] + str(
                stat_dict['metrics'][name]['best']['value'])):
            stat_dict['metrics'][name]['best']['value'] = m_value
            stat_dict['metrics'][name]['best']['epoch'] = epoch
            log.send_log('{}:{} epoch:{}/{}'.format(name, m_value, epoch, epochs), cloudLogName)
            save_model_flag = True

    if save_model_flag:
        # sava best model
        save_model(os.path.join(experiment_model_path, 'model_{}.pt'.format(epoch)), epoch,
                   model, optimizer, scheduler, stat_dict)
    # '[Validation] nRMSE/RMSE: {:.4f}/{:.4f} (Best: {:.4f}/{:.4f}, Epoch: {}/{})\n'
    test_log = '[Val] ' \
               + '/'.join([str(m[0]) for m in metrics]) \
               + ': ' \
               + '/'.join([str(round(m[1], 4)) for m in metrics]) \
               + ' (Best: ' \
               + '/'.join([str(round(stat_dict['metrics'][m[0]]['best']['value'], 4)) for m in metrics]) \
               + ', Epoch: ' \
               + '/'.join([str(stat_dict['metrics'][m[0]]['best']['epoch']) for m in metrics]) \
               + ')'
    save_model(os.path.join(experiment_model_path, 'model_latest.pt'), epoch, model, optimizer, scheduler, stat_dict)
    return test_log


def save_np2file(data_list, name_lists, save_path):
    progress_bar = tqdm(total=len(data_list), desc='saving npy')
    for idx, data in enumerate(data_list):
        if name_lists:
            name = name_lists[idx]
        else:
            name = idx
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, str(name).replace(' ', '-') + '.npy'), data)
        progress_bar.update(1)


def save_model(_path, _epoch, _model, _optimizer, _scheduler, _stat_dict):
    # torch.save(model.state_dict(), saved_model_path)
    torch.save({
        'epoch': _epoch,
        'model_state_dict': _model.state_dict(),
        'optimizer_state_dict': _optimizer.state_dict(),
        'scheduler_state_dict': _scheduler.state_dict(),
        'stat_dict': _stat_dict
    }, _path)


def parse_config():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config', type=str, default=None, help='pre-config file for training')
    args = parser.parse_args()

    if args.config:
        opt = vars(args)
        yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(yaml_args)

    # set visible gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.gpu_ids])

    # select active gpu devices
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        # device = torch.device('cuda:' + str(args.gpu_ids[0]))
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    return device, args


def read_nc(file_path, keys):
    dataset = nc.Dataset(file_path, 'r')  # 'r' 表示只读模式
    # print(dataset.variables.keys())  # 打印所有变量的名称
    numpy_out = None
    for key in keys:
        variable_data = dataset.variables[key]  # 读取变量数据
        # 将NetCDF变量数据转换为NumPy数组
        np_data = np.array(variable_data)
        if len(np_data.shape) == 3:
            np_data = np_data[:, np.newaxis, :, :]
        elif len(np_data.shape) == 4:
            np_data = np_data[:, np.newaxis, :, :, :]
        if numpy_out is None:
            numpy_out = np_data
        else:
            numpy_out = np.concatenate([numpy_out, np_data], axis=1)
    dataset.close()
    return numpy_out


if __name__ == '__main__':
    # weights_lat = np.repeat(fnp.load('../constant_masks/weight.npy')[:, np.newaxis], repeats=1440, axis=1)
    # print()
    # 将溢出警告设置为忽略
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    # make_train_validate(r'/mnt/data1/download', r'/mnt/data1/szj/wf_dataset_20',
    #                     datetime(year=1979, month=1, day=1, hour=0),
    #                     datetime(year=1999, month=12, day=31, hour=18), np.float32)

    # 计算指定时间的均值和方差
    # cal_mean_std(r'C:\Users\ECNU\Desktop\wf_dataset',
    #              datetime(year=2020, month=12, day=1, hour=0),
    #              datetime(year=2020, month=12, day=3, hour=18), delay_time=24)
    # 创建merge.np文件，同时求出均值和方差

    # draw_lines('log.txt')

    # draw_lines('../exclude/stat_dict.yml')
    # d1 = np.load(r'C:\Users\ECNU\Desktop\mean_pixel.npy')
    # d2 = np.load(r'C:\Users\ECNU\Desktop\mean_level.npy')

    print('Finished!')
