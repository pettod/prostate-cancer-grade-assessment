import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):
        super().__init__()
        m = torch.hub.load(
            'facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.adaptive_concat_pool = AdaptiveConcatPool2d()
        self.linear_1 = nn.Sequential(nn.Linear(2*nc, 512), Mish())
        self.batchnorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(512, n)

    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x, 1).view(-1, shape[1], shape[2], shape[3])
        # x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        # x: bs*N x C x 4 x 4
        shape = x.shape
        # Concatenate the output for tiles into a single map
        x = x.view(-1, n, shape[1], shape[2], shape[3]).permute(
            0, 2, 1, 3, 4).contiguous().view(
                -1, shape[1], shape[2]*n, shape[3])
        # x: bs x C x N*4 x 4
        x = self.adaptive_concat_pool(x)
        x = x.flatten(start_dim=1)
        x = self.linear_1(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *(torch.tanh(F.softplus(x)))


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`." # from pytorch
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)
