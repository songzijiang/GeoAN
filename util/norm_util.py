import os.path

import torch
import numpy as np
from einops import rearrange


class Normalization:
    def __init__(self, data_path):
        self.cldas_mean = torch.from_numpy(
            np.load(os.path.join(data_path, 'cldas_global_mean_level.npy')).astype(np.float32))
        self.cldas_std = torch.from_numpy(
            np.load(os.path.join(data_path, 'cldas_global_std_level.npy')).astype(np.float32))
        self.era5_mean = torch.from_numpy(
            np.load(os.path.join(data_path, 'era5_global_mean_level.npy')).astype(np.float32))
        self.era5_std = torch.from_numpy(
            np.load(os.path.join(data_path, 'era5_global_std_level.npy')).astype(np.float32))

    def norm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = (data - self.era5_mean) / self.era5_std
        return rearrange(data, 'b h w c->b c h w')

    def denorm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = data * self.cldas_std + self.cldas_mean
        return rearrange(data, 'b h w c->b c h w')
