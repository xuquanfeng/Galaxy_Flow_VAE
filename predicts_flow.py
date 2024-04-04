import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torch
import torch.nn as nn

from torchvision import transforms
import h5py
from flow_conv_vae import VAE

from my_dataset import MyDataSet, HDF5Dataset2
from datasets import ImageFolderDataset, arcsinh_rgb
from models import Generator, Discriminator, ViT
import numpy as np
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
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
    os.makedirs("images", exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # create model
    model = VAE(args)
    # load model weights
    # model_weight_path = "./weights/model_gen_best.pth"
    model_weight_path = "./weights/model_train_vae_best.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    model.to(device)

    data_transform = {
        "train": transforms.Compose([#transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.Resize(args.img_size),
                                     transforms.ToTensor()]),
                                    #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        "val": transforms.Compose([transforms.Resize(args.img_size),
                                   transforms.ToTensor()])}

    # load image
    train_dataset = HDF5Dataset2(args.data_path, transform=data_transform["train"])
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} images to train'.format(len(train_dataset)))

    dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Optimizers
    flow = Flow_model(num_layers = 10, input_dim = 40, output_dim = 3)
    flow = flow.to(device)
    model_weight_path = "./weights/model_train_flow_vae_best2.pth"
    flow.load_state_dict(torch.load(model_weight_path, map_location=device))
    flow.eval()
    flow.to(device)
    
    # raw_idx = 150
    raw_idx_min = 140
    raw_idx_max = 150
    for i, (imgs, labels) in enumerate(dataloader):
        if i < raw_idx_min:
            pass
        elif i > raw_idx_max:
            break
        else:
            # Configure input
            imgs = Variable(imgs.type(FloatTensor).to(device))
            labels = Variable(labels.type(FloatTensor).to(device))

            with torch.no_grad():
                x_exact = labels
                z_iaf = flow.sample(num_samples=1, context = x_exact.to(device)).squeeze(1)

            x_hat = torch.squeeze(model.decode(z_iaf)).cpu()
            # reconst = torch.tensor([arcsinh_rgb(part.detach().cpu(), ["g", "r", "z"]) for part in  x_hat])
            # imgs = torch.tensor([arcsinh_rgb(part.detach().cpu(), ["g", "r", "z"]) for part in  imgs])
            reconst = x_hat.detach().cpu().squeeze(0)
            imgs = imgs.detach().cpu().squeeze(0)
            orig = imgs.reshape(1, -1, args.channels, args.img_size, args.img_size).to(device)
            reconst = reconst.reshape(1, -1, args.channels, args.img_size, args.img_size).to(device)
            comparison = torch.cat(
                (orig, reconst), dim=0).permute(1, 0, 2, 3, 4).reshape(-1, args.channels, args.img_size, args.img_size)
            os.makedirs('./images' +'/', exist_ok=True)
            # os.makedirs('./samples' +'/', exist_ok=True)
            save_image(make_grid(comparison),
                './images' +'/' + 'recon_flow_%d.png' % i)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
    parser.add_argument('--data_path', type=str,default="/data/public/renhaoye/MockCSST/dataset/merged.h5")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=40, help="dimensionality of the latent space")
    parser.add_argument("--class_latent", type=int, default=1000, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=7, help="number of classes for dataset")
    parser.add_argument('--scale', type=int, default=3, help='number of scales.')
    parser.add_argument('--res_blocks', type=int, default=1, help='number of residual blocks per group.')
    parser.add_argument('--bottleneck', type=int, default=0, help='whether to use bottleneck in residual blocks.')
    parser.add_argument('--weight_norm', type=int, default=0, help='whether to apply weight normalization.')
    parser.add_argument('--flow', type=str, default='none', help='type of flow to use.')
    parser.add_argument('--base_dim', type=int, default=32, help='features in residual blocks of first scale.')
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument('--length', type=int, default=8, help='number of steps in the flow.')
    parser.add_argument('--sample_size', type=int, default=32, help='number of images to generate')
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
