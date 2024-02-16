import torch
from torch import nn
import numpy as np
from einops import rearrange


class LatitudeLoss(nn.Module):

    def __init__(self):
        super(LatitudeLoss, self).__init__()
        self.weights_lat = np.arange(0, 65, 0.0625)
        self.weights_lat = np.cos(self.weights_lat * np.pi / 180)
        self.weights_lat = torch.from_numpy(1040 * self.weights_lat / np.sum(self.weights_lat))

    def forward(self, output, target):
        if self.weights_lat.device != output.device:
            self.weights_lat = self.weights_lat.to(output.device)
        loss = rearrange(torch.abs(output - target), 'b c h w->b c w h') * self.weights_lat
        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    loss = LatitudeLoss()
    print(loss.weights_lat)
