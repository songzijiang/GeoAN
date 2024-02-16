import math
import os
import random
import time

import numpy as np
import torch
import torch.utils.data as data
from datetime import datetime, timedelta
from jacksung.utils.multi_task import MultiTasks
from jacksung.utils.time import Stopwatch
import jacksung.utils.fastnumpy as fnp
from tqdm import tqdm


class Benchmark(data.Dataset):
    def __init__(self, data_dir, start_date, end_date, exclude_date, skip_day=1, train=False, repeat=1, read_era5=True,
                 read_cldas=True):
        super(Benchmark, self).__init__()
        self.train = train
        self.repeat = repeat
        self.data_dir = data_dir
        self.read_cldas = read_cldas
        self.read_era5 = read_era5
        self.data = {}
        now_date = start_date
        count = 0
        total_bar = tqdm(total=math.ceil(((end_date - start_date).days + 1) / skip_day), desc='Loading data')

        # 全球的标准化
        self.topography = ((fnp.load('mask/clip_topography.npy') - 3723.7740) / 8349.2742)[np.newaxis, :, :]
        # 中国的标准化
        # self.topography = ((fnp.load('mask/clip_topography.npy') - 5090.5376) / 9811.776)[np.newaxis, :, :]
        while now_date <= end_date:
            if now_date not in exclude_date:
                self.data[count] = self.load_data(now_date)
                count += 1
            total_bar.update()
            now_date += timedelta(days=skip_day)
        self.nums_train_set = count

    def __len__(self):
        if self.train:
            return self.nums_train_set * self.repeat
        else:
            return self.nums_train_set

    def load_data(self, loaded_date):
        file_path = os.path.join(self.data_dir, str(loaded_date.year), str(loaded_date.month), str(loaded_date.day))
        if self.read_cldas:
            cldas_np = np.load(os.path.join(file_path, 'cldas.np.npy'))
        else:
            cldas_np = None
        if self.read_era5:
            era5_np = np.load(os.path.join(file_path, 'era5.np.npy'))
        else:
            era5_np = None
        r = [self.topography, era5_np, cldas_np]
        r = [e for e in r if e is not None]
        return r

    def __getitem__(self, idx):
        idx = idx % self.nums_train_set
        result = self.data[idx]
        return result


if __name__ == '__main__':
    print()
    # train_dataloader = DataLoader(dataset=ds, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)
    # for t_s, t_u in train_dataloader:
    #     print('*' * 40)
