import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim

from my_dataset import HDF5Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt

class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, 
        output_padding=0, bias=True, weight_norm=True, scale=False, transpose=False):
        """Intializes a Conv2d augmented with weight normalization.

        (See torch.nn.utils.weight_norm for detail.)

        Args:
            in_dim: number of input channels.
            out_dim: number of output channels.
            kernel_size: size of convolving kernel.
            stride: stride of convolution.
            padding: zero-padding added to both sides of input.
            output_padding: for inferring output shape (only for transposed convolution).
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
            transpose: True if transposed convolution, False otherwise.
        """
        super(WeightNormConv2d, self).__init__()
        if weight_norm:
            if transpose:
                self.conv = nn.utils.weight_norm(
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, 
                        padding=padding, output_padding=output_padding, bias=bias))
            else:
                self.conv = nn.utils.weight_norm(
                    nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, 
                        padding=padding, bias=bias))
            if not scale:
                self.conv.weight_g.data = torch.ones_like(self.conv.weight_g.data)
                self.conv.weight_g.requires_grad = False    # freeze scaling
        else:
            if transpose:
                self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, 
                    stride=stride, padding=padding, output_padding=output_padding, 
                    bias=bias)
            else:
                self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, 
                    padding=padding, bias=bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, bottleneck, weight_norm, transpose=False):
        """Initializes a ResidualBlock.

        Args:
            dim: number of input and output features.
            bottleneck: True if use bottleneck, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            transpose: True if transposed convolution, False otherwise.
        """
        super(ResidualBlock, self).__init__()
        self.in_block = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU())
        if bottleneck:
            self.res_block = nn.Sequential(
                WeightNormConv2d(dim, dim, (1, 1), stride=1, padding=0, bias=False, 
                    weight_norm=weight_norm, scale=False, transpose=transpose),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, bias=False, 
                    weight_norm=weight_norm, scale=False, transpose=transpose),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (1, 1), stride=1, padding=0, bias=True, 
                    weight_norm=weight_norm, scale=True, transpose=transpose))
        else:
            self.res_block = nn.Sequential(
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, bias=False, 
                    weight_norm=weight_norm, scale=False, transpose=transpose),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, bias=True, 
                    weight_norm=weight_norm, scale=True, transpose=transpose))

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return x + self.res_block(x)

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of planar flow.

        Args:
            dim: input dimensionality.
        """
        super(PlanarFlow, self).__init__()

        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        def m(x):
            return F.softplus(x) - 1.
        def h(x):
            return torch.tanh(x)
        def h_prime(x):
            return 1. - h(x)**2

        inner = (self.w * self.u).sum()
        u = self.u + (m(inner) - inner) * self.w / self.w.norm()**2
        activation = (self.w * x).sum(dim=1, keepdim=True) + self.b
        x = x + u * h(activation)
        psi = h_prime(activation) * self.w
        log_det = torch.log(torch.abs(1. + (u * psi).sum(dim=1, keepdim=True)))

        return x, log_det

class RadialFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of radial flow.

        Args:
            dim: input dimensionality.
        """
        super(RadialFlow, self).__init__()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        def m(x):
            return F.softplus(x)
        def h(r):
            return 1. / (a + r)
        def h_prime(r):
            return -h(r)**2

        a = torch.exp(self.a)
        b = -a + m(self.b)
        r = (x - self.c).norm(dim=1, keepdim=True)
        tmp = b * h(r)
        x = x + tmp * (x - self.c)
        log_det = (self.d - 1) * torch.log(1. + tmp) + torch.log(1. + tmp + b * h_prime(r) * r)

        return x, log_det

class HouseholderFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of householder flow.

        Args:
            dim: input dimensionality.
        """
        super(HouseholderFlow, self).__init__()

        self.v = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        outer = self.v.t() * self.v
        v_sqr = self.v.norm()**2
        H = torch.eye(self.d).to(self.device) - 2. * outer / v_sqr
        x = torch.mm(H, x.t()).t()
        
        return x, 0

class NiceFlow(nn.Module):
    def __init__(self, dim, mask, final=False):
        """Instantiates one step of NICE flow.

        Reference:
        NICE: Non-linear Independent Components Estimation
        Laurent Dinh, David Krueger, Yoshua Bengio
        (https://arxiv.org/abs/1410.8516)

        Args:
            dim: input dimensionality.
            mask: mask that determines active variables.
            final: True if the final step, False otherwise.
        """
        super(NiceFlow, self).__init__()

        self.final = final
        if final:
            self.scale = nn.Parameter(torch.zeros(1, dim))
        else:
            self.mask = mask
            self.coupling = nn.Sequential(
                nn.Linear(dim//2, dim*5), nn.ReLU(), 
                nn.Linear(dim*5, dim*5), nn.ReLU(), 
                nn.Linear(dim*5, dim//2))

    def forward(self, x):
        if self.final:
            x = x * torch.exp(self.scale)
            log_det = torch.sum(self.scale)
            
            return x, log_det
        else:
            [B, W] = list(x.size())
            x = x.reshape(B, W//2, 2)
            
            if self.mask:
                on, off = x[:, :, 0], x[:, :, 1]
            else:
                off, on = x[:, :, 0], x[:, :, 1]
            
            on = on + self.coupling(off)

            if self.mask:
                x = torch.stack((on, off), dim=2)
            else:
                x = torch.stack((off, on), dim=2)
            
            return x.reshape(B, W), 0

class Flow(nn.Module):
    def __init__(self, dim, type, length, device):
        """Instantiates a chain of flows.

        Args:
            dim: input dimensionality.
            type: type of flow.
            length: length of flow.
        """
        super(Flow, self).__init__()
        self.device = device
        if type == 'planar':
            self.flow = nn.ModuleList([PlanarFlow(dim) for _ in range(length)])
        elif type == 'radial':
            self.flow = nn.ModuleList([RadialFlow(dim) for _ in range(length)])
        elif type == 'householder':
            self.flow = nn.ModuleList([HouseholderFlow(dim) for _ in range(length)])
        elif type == 'nice':
            self.flow = nn.ModuleList([NiceFlow(dim, i//2, i==(length-1)) for i in range(length)])
        else:
            self.flow = nn.ModuleList([])

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        [B, _] = list(x.size())
        log_det = torch.zeros(B, 1).to(self.device)
        for i in range(len(self.flow)):
            x, inc = self.flow[i](x)
            log_det = log_det + inc

        return x, log_det

class VAE(nn.Module):
    def __init__(self, hps):
        """Instantiates a VAE.

        Args:
            hps: hyperparameters for the VAE.
        """
        super(VAE, self).__init__()
        self.device = torch.device(hps.device if torch.cuda.is_available() else "cpu")
        self.channels = hps.channels
        self.img_size = hps.img_size
        self.latent_dim = hps.latent_dim
        self.base_dim = hps.base_dim         # features in the first layer of encoder
        self.scale = hps.scale               # number of scales in encoder
        self.res_blocks = hps.res_blocks     # number of residual blocks in each scale
        self.bottleneck = hps.bottleneck     # whether to use bottleneck in residual blocks
        self.weight_norm = hps.weight_norm   # whether to use weight normalization

        self.encoder, self.feature_dim = self._build_encoder()
        self.feature_size = self.img_size // 2**self.scale
        flat_dim = self.feature_dim * self.feature_size**2

        self.mean = nn.Linear(flat_dim, hps.latent_dim)
        # self.log_var = nn.Linear(flat_dim, hps.latent_dim)
        self.log_var = nn.Sequential(nn.Linear(flat_dim, hps.latent_dim), nn.Tanh())

        self.flow = Flow(hps.latent_dim, hps.flow, hps.length, hps.device)
        self.project = nn.Linear(hps.latent_dim, flat_dim)
        self.decoder = self._build_decoder(self.feature_dim)

    def _build_encoder(self):
        """Builds an encoder.

        Returns:
            an encoder.
        """
        in_block = WeightNormConv2d(self.channels, self.base_dim, (3, 3), 
            stride=1, padding=1, bias=True, weight_norm=self.weight_norm, scale=True)
        
        core_block = nn.Sequential()
        dim = self.base_dim
        for i in range(self.scale):
            for j in range(self.res_blocks):
                core_block.add_module('scale_%d_res_%d' % (i, j), 
                    ResidualBlock(dim, self.bottleneck, self.weight_norm))
                core_block.add_module('active_%d_res_%d' % (i, j), nn.LeakyReLU())
            core_block.add_module('scale_%d_out' % i, 
                WeightNormConv2d(dim, dim*2, (3, 3), stride=2, padding=1, 
                    bias=True, weight_norm=self.weight_norm, scale=True))
            core_block.add_module('active_%d_res_%d' % (i, j), nn.ReLU())
            dim *= 2

        out_block = nn.Sequential(nn.BatchNorm2d(dim), nn.ReLU())
        
        return nn.Sequential(in_block, core_block, out_block), dim

    def _build_decoder(self, dim):
        """Builds a decoder.

        Args:
            dim: number of features in the layer closest to latent space.
        Returns:
            a decoder.
        """
        in_block = nn.Sequential(nn.BatchNorm2d(dim), nn.ReLU())

        core_block = nn.Sequential()
        for i in reversed(range(self.scale)):
            for j in range(self.res_blocks):
                core_block.add_module('scale_%d_res_%d' % (i, j),
                    ResidualBlock(dim, self.bottleneck, self.weight_norm, transpose=True))
                # core_block.add_module('tanh_%d_res_%d' % (i, j), nn.LeakyReLU(negative_slope=0.05))
            core_block.add_module('scale_%d_out' % i, 
                WeightNormConv2d(dim, dim//2, (3, 3), stride=2, padding=1, output_padding=1, 
                    bias=True, weight_norm=self.weight_norm, scale=True, transpose=True))
            core_block.add_module('active_%d_out' % i, nn.LeakyReLU(negative_slope=0.05))
            dim //= 2

        assert dim == self.base_dim
        out_block = nn.Sequential(WeightNormConv2d(
            dim, self.channels, (3, 3), stride=1, padding=1, bias=True, 
            weight_norm=self.weight_norm, scale=True, transpose=True)
            , nn.ReLU())

        return nn.Sequential(in_block, core_block, out_block)

    def encode(self, x):
        """Encodes images into latent codes.

        Args:
            x: input images.
        Returns:
            mean and log-variance of the gaussian approximate posterior.
        """
        x = self.encoder(x)
        [_, C, H, W] = list(x.size())
        assert C == self.feature_dim
        assert H == self.feature_size
        assert W == self.feature_size
        x = x.reshape(-1, C*H*W)

        return self.mean(x), self.log_var(x)

    def transform(self, mean, log_var):
        """Transforms approximate posterior.

        Args:
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
        Returns:
            transformed latent codes and the log-determinant of the Jacobian.
        """
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return self.flow(z)

    def decode(self, z):
        """Decodes latent codes in images.

        Args:
            z: latent codes.
        Returns:
            reconstructed images.
        """
        x = self.project(z)
        x = x.reshape(-1, self.feature_dim, self.feature_size, self.feature_size)

        return self.decoder(x)

    def sample(self, size):
        """Generates samples from the prior.

        Args:
            size: number of samples to generate.
        Returns:
            generated samples.
        """
        z = torch.randn(size, self.latent_dim).to(self.device)
        return self.decode(z)

    def reconstruction_loss(self, x, x_hat):
        """Computes reconstruction loss.

        Args:
            x: original input (B x D).
            x_hat: reconstructed input (B x D).
        Returns: 
            sum of reconstruction loss over the minibatch.
        """
        return nn.MSELoss(reduction='none')(x_hat, x).sum(dim=(1, 2, 3)).view(-1, 1)

    def latent_loss(self, mean, log_var, log_det):
        """Computes KL loss.

        Args:
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
            log_det: log-determinant of the Jacobian.
        Returns: sum of KL loss over the minibatch.
        """
        # KLD_element = mean.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        # KLD = torch.sum(KLD_element).mul_(-0.5)
        # return KLD - log_det
        kl = -.5 * torch.sum(1. + log_var - mean.pow(2) - log_var.exp(), dim=1, keepdim=True)
        return kl - log_det

    def loss(self, x, x_hat, mean, log_var, log_det):
        """Computes overall loss.

        Args:
            x: original input (B x D).
            x_hat: reconstructed input (B x D).
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
            log_det: log-determinant of the Jacobian.
        Returns:
            sum of reconstruction and KL loss over the minibatch.
        """
        # print('x',x.max())
        # print('x_hat',x_hat.max())
        # print('mean',mean.max())
        # print('log_var',log_var.max())
        # print('log_det',log_det.max())
        # print('rec',self.reconstruction_loss(x, x_hat).max())
        # print('lat',self.latent_loss(mean, log_var, log_det).max())
        return self.reconstruction_loss(x, x_hat) + self.latent_loss(mean, log_var, log_det) 

    def forward(self, x):
        """Forward pass.

        Args:
            x: input images.
        Returns:
            z: latent codes.
            log_qz: log-likelihood of latent codes under approximate posterior.
            x_hat: reconstructed images.
            weight_scale: scaling parameters subject to L2 regularization.
        """
        mean, log_var = self.encode(x)
        # log_var = torch.log10(log_var)
        z, log_det = self.transform(mean, log_var)
        x_hat = self.decode(z)
        loss = self.loss(x, x_hat, mean, log_var, log_det).mean()

        # collect scaling parameters
        weight_scale = None
        if self.weight_norm:
            for name, param in self.named_parameters():
                param_name = name.split('.')[-1]
                if param_name == 'weight_g' and param.requires_grad:
                    if weight_scale is None:
                        weight_scale = param.pow(2).sum()
                    else:
                        weight_scale = weight_scale + param.pow(2).sum()

        return x_hat, loss, weight_scale

def logit_transform(x, constraint=0.9, reverse=False):
    '''Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
        x *= 2.             # [0.1, 1.9]
        x -= 1.             # [-0.9, 0.9]
        x /= constraint     # [-1, 1]
        x += 1.             # [0, 2]
        x /= 2.             # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())
        
        # dequantization
        noise = distributions.Uniform(0., 1.).sample((B, C, H, W))
        x = (x * 255. + noise) / 256.
        
        # restrict data
        x *= 2.             # [0, 2]
        x -= 1.             # [-1, 1]
        x *= constraint     # [-0.9, 0.9]
        x += 1.             # [0.1, 1.9]
        x /= 2.             # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3)).mean()