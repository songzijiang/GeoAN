import numpy as np
import yaml
from util import utils
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.datasetV2 import Benchmark
from datetime import datetime
from jacksung.utils.time import RemainTime, Stopwatch, cur_timestamp_str
from jacksung.utils.data_convert import np2tif
from tqdm import tqdm
from util.norm_util import Normalization
from util.data_parallelV2 import BalancedDataParallel
from util.utils import EXCLUDE_DATE

if __name__ == '__main__':
    device, args = utils.parse_config()
    test_dataset = \
        Benchmark(args.data_path,
                  datetime(year=args.valid_start_date[0], month=args.valid_start_date[1], day=args.valid_start_date[2]),
                  datetime(year=args.valid_end_date[0], month=args.valid_end_date[1], day=args.valid_end_date[2]),
                  exclude_date=EXCLUDE_DATE, skip_day=1, train=False, read_cldas=False)
    test_dataloader = DataLoader(dataset=test_dataset, num_workers=args.threads, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False, drop_last=False)
    model = utils.get_model(args)
    norm = Normalization(args.data_path)
    norm.cldas_mean, norm.cldas_std, norm.era5_mean, norm.era5_std = \
        utils.data_to_device([norm.cldas_mean, norm.cldas_std, norm.era5_mean, norm.era5_std], device, args.fp)
    # load pretrain
    if args.model_path is None:
        raise Exception('进行数据生成，请在config中指定 pretrain 参数')
    print('load pretrained model: {}!'.format(args.model_path))
    ckpt = torch.load(args.model_path)
    model.load(ckpt['model_state_dict'])
    model = model.to(device)
    if args.balanced_gpu0 >= 0:
        # balance multi gpus
        model = BalancedDataParallel(args.balanced_gpu0, model, device_ids=list(range(len(args.gpu_ids))))
    else:
        # multi gpus
        model = nn.DataParallel(model, device_ids=list(range(len(args.gpu_ids))))
    save_path = args.save_path
    timestamp = cur_timestamp_str()

    root_path = os.path.join(args.save_path, args.model + '-' + timestamp)
    os.makedirs(root_path, exist_ok=True)
    img_path = os.path.join(root_path, 'img')
    os.makedirs(img_path, exist_ok=True)
    torch.set_grad_enabled(False)
    model = model.eval()
    epoch_loss = 0
    psnr = 0
    ssim = 0
    rmse = 0
    progress_bar = tqdm(total=len(test_dataset), desc='Infer')
    count = 0
    for iter_idx, batch in enumerate(test_dataloader):
        topo, lr = utils.data_to_device(batch, device, args.fp)
        lr = norm.norm(lr)
        # roll = random.randint(0, now_t.shape[-1] - 1)
        roll = 0
        y_ = model(topo, lr, roll)
        y_ = norm.denorm(y_)
        y_ = y_.cpu().numpy()
        for idx, each_y in enumerate(y_):
            data_idx = str(iter_idx * args.batch_size + idx)
            np.save(os.path.join(root_path, data_idx + '.npy'), each_y)
            np2tif(each_y, img_path, args.model + '-' + data_idx, left=60.125, top=64.875, x_res=0.0625, y_res=0.0625,
                   dim_value=[{'value': ['WIN', 'TMP', 'PRS', 'PRE']}])
        # ['WIN', 'TMP', 'PRS', 'PRE']
        progress_bar.update(len(lr))
    progress_bar.close()
