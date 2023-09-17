
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
device = torch.device("cuda")


def InstanceNorm2D_wrap(input_channels, momentum=0.1, affine=True,
                        track_running_stats=False, **kwargs):
    
    instance_norm_layer = nn.InstanceNorm2d(input_channels, 
        momentum=momentum, affine=affine,
        track_running_stats=track_running_stats)
    return instance_norm_layer

def ChannelNorm2D_wrap(input_channels, momentum=0.1, affine=True,
                       track_running_stats=False, **kwargs):
   
    channel_norm_layer = ChannelNorm2D(input_channels, 
        momentum=momentum, affine=affine,
        track_running_stats=track_running_stats)

    return channel_norm_layer

class ChannelNorm2D(nn.Module):
    

    def __init__(self, input_channels, momentum=0.1, eps=1e-3,
                 affine=True, **kwargs):
        super(ChannelNorm2D, self).__init__()

        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1).to(device))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1).to(device))

    def forward(self, x):
       
        mu, var = torch.mean(x, dim=1, keepdim=True), torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mu) * torch.rsqrt(var + self.eps)

        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta
        return x_normed
