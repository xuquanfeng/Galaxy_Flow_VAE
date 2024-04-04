import os
import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision.utils as utils

from torchvision import transforms
from tqdm import tqdm

from my_dataset import HDF5Dataset
from flow_conv_vae import VAE
import numpy as np

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
import sys

import warnings
warnings.filterwarnings('ignore')

def Flow_model(num_layers, input_dim, output_dim):
    base_dist = ConditionalDiagonalNormal(shape=[input_dim], 
                                        context_encoder=nn.Linear(output_dim, input_dim*2))

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=input_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, 
                                                            hidden_features=4, 
                                                            context_features=output_dim))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs('./weights/', exist_ok=True)

    f_vae = VAE(args)
    # print(f_vae)
    f_vae.to(device)
    weights_dict = torch.load("./weights/model_train_vae_best.pth", map_location=device)
    f_vae.load_state_dict(weights_dict, strict=False)
    f_vae.eval()
    
    # 实例化训练数据集
    train_dataset = HDF5Dataset(args.data_path)
    print('Using {} images to train'.format(len(train_dataset)))

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers

    dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
    
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # Optimizers
    flow = Flow_model(num_layers = 10, input_dim = args.latent_dim, output_dim = args.n_classes)
    flow = flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    ll = nn.L1Loss()
    # optimizer = torch.optim.Adam(f_vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        losss = []
        dataloader = tqdm(dataloader)
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            # Configure input
            imgs = Variable(imgs.type(FloatTensor).to(device))
            labels = Variable(labels.type(FloatTensor).to(device))

            # -----------------
            #  Generate Flow-VAE
            # -----------------
            mean, log_var = f_vae.encode(imgs)
            x_hat, _ = f_vae.transform(mean, log_var)

            # x_hat = torch.log10(x_hat-x_hat.min()+0.001)
            # labels = torch.log10(labels-labels.min()+0.001)
            # print('x_hat size', x_hat.size())
            # print('labels size', labels.size())

            x_hat = torch.tensor(x_hat, dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            loss = -flow.log_prob(inputs=x_hat, context=labels).mean()
            loss.backward()
            optimizer.step()
            # print(loss.item())

            # sys.exit()
            losss.append(loss.item())
            f_loss1 = np.mean(losss)
            with torch.no_grad():
                x_exact = labels
                z_iaf = flow.sample(num_samples=1, context = x_exact.to(device)).squeeze(1)
            mae_l = ll(z_iaf,x_hat)
            dataloader.desc = "[Epoch {}/{}] [Batch {}/{}] [Flow loss: {:.3f}] [Flow mae: {:.3f}] [Flow average_loss: {:.3f}]".format(epoch, opt.n_epochs, i, len(dataloader), loss.item(), mae_l.item(), f_loss1)

        # print(f_loss1)
        if epoch == 0:
            save_loss = f_loss1
        elif save_loss > f_loss1:
            save_loss = f_loss1
            torch.save(flow.state_dict(), "./weights/model_train_flow_vae_best.pth")
            print('Save best models in {} !'.format(epoch))
        # if epoch%10==9:
        #     torch.save(f_vae.state_dict(), "./weights/model_fvae_{}.pth".format(epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument('--data_path', type=str,default="/data/public/renhaoye/MockCSST/dataset/merged.h5")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=60, help="latent space dimensionality")
    parser.add_argument('--base_dim', type=int, default=32, help='features in residual blocks of first scale.')
    parser.add_argument("--class_latent", type=int, default=1000, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=3, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument('--scale', type=int, default=3, help='number of scales.')
    parser.add_argument('--res_blocks', type=int, default=1, help='number of residual blocks per group.')
    parser.add_argument('--bottleneck', type=int, default=0, help='whether to use bottleneck in residual blocks.')
    parser.add_argument('--weight_norm', type=int, default=0, help='whether to apply weight normalization.')
    parser.add_argument('--flow', type=str, default='none', help='type of flow to use.')
    # parser.add_argument('--flow', type=str, default='planar', help='type of flow to use.')
    parser.add_argument('--length', type=int, default=8, help='number of steps in the flow.')
    parser.add_argument('--sample_size', type=int, default=32, help='number of images to generate')
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
