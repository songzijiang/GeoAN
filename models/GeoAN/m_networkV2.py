import torch.nn as nn
from models.GeoAN.m_blockV2 import FEB, Tail, Head
from einops import rearrange
from datetime import datetime
from torch.utils.data import DataLoader
import torch
import jacksung.utils.fastnumpy as fnp


class GeoAN(nn.Module):
    def __init__(self, window_sizes, n_geoab, c_in, c_geoan, r_expand=4, down_sample=4, num_heads=8):
        super(GeoAN, self).__init__()
        self.window_sizes = window_sizes
        self.n_geoab = n_geoab
        self.c_in = c_in
        self.c_geoan = c_geoan
        self.r_expand = r_expand
        self.down_sample = down_sample

        # define head module
        self.head = Head(self.c_geoan, self.c_in, self.down_sample)
        self.head_f = Head(self.c_geoan, 1, self.down_sample)
        # define body module
        number = self.n_geoab // len(self.window_sizes)
        self.body = nn.ModuleList()
        for window_size in self.window_sizes:
            for i in range(number // 1):
                self.body.append(
                    FEB(self.c_geoan, self.r_expand, window_size, num_heads=num_heads))

        self.tail = Tail(self.c_geoan, self.c_in, self.down_sample)

    def forward(self, f, x, roll=0):
        # head
        if roll > 0:
            f, x = torch.roll(f, shifts=roll, dims=-1), torch.roll(x, shifts=roll, dims=-1)
        # f, x = rearrange(x, 'b c z h w->b (c z) h w'), rearrange(f, 'b c z h w->b (c z) h w')
        x = nn.functional.interpolate(x, size=[260, 400], mode='bilinear')
        f = nn.functional.interpolate(f, size=[260, 400], mode='bilinear')
        x, f = self.head(x), self.head_f(f)
        shortcut = x
        # body
        for idx, stage in enumerate(self.body):
            if idx % 2 == 0:
                f, x = stage(f, x)
            else:
                f, x = stage(f, x, True)
        x = shortcut + x
        # tail
        x = self.tail(x)
        if roll > 0:
            x = torch.roll(x, shifts=-roll, dims=-1)
        return x

    def init_model(self):
        print('Initializing the model!')
        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

    def load(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name[name.index('.') + 1:]
            if 'LGAB' in name:
                name = name.replace('LGAB', 'ATT')
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


if __name__ == '__main__':
    pass
