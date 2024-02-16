import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from pytorch_msssim import ssim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
import importlib


def compute_rmse(da_fc, da_true):
    error = da_fc - da_true
    error = error ** 2
    number = torch.sqrt(error.mean((-2, -1)))
    return number.mean()


class Metrics:
    def __init__(self):
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()

    def calc_psnr(self, sr, hr):
        return self.psnr(sr, hr)

    def calc_ssim(self, sr, hr):
        return self.ssim(sr, hr)

    def calc_rmse(self, sr, hr):
        return compute_rmse(sr, hr)


if __name__ == '__main__':
    preds = torch.rand(2, 3, 3, 5)
    target = torch.rand(2, 3, 3, 5)
    m = Metrics()
    print(m.calc_psnr(preds, target))
    print(m.calc_ssim(preds, target))
    print(m.calc_rmse(preds, target))
