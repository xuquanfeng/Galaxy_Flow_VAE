import os
import argparse

from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision.utils as utils
import itertools

from tqdm import tqdm

from my_dataset import HDF5Dataset, HDF5_Dataset, arcsinh_rgb
from ssim import ssim
from conv_vae import VAE
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
    project_name = '_FlowVAE'
    save_model_name = 'model_train_Fvae_vae_best'
    save_model_name1 = 'model_train_Fvae_flow_best'

    os.makedirs('./train_process/', exist_ok=True)
    os.makedirs('./reconstruction'+ project_name +'/', exist_ok=True)
    os.makedirs('./samples'+ project_name +'/', exist_ok=True)
    os.makedirs('./weights/', exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tb_writer = SummaryWriter('./train_process/'+project_name[1:])
    pre_train_vae_epoch = 10
    # -----------------
    #  Bulild Flow-VAE
    # -----------------

    #  Bulild VAE
    f_vae = VAE(args)
    f_vae.to(device)
    #  Bulild Flow
    flow = Flow_model(num_layers = 10, input_dim = args.latent_dim, output_dim = args.n_classes)
    flow = flow.to(device)
    
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
    optimizer = torch.optim.Adam(f_vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        losss = []
        ssimm = []
        floww = []
        if epoch == pre_train_vae_epoch+1:
            print("Stop the pre-train!")
        if epoch > pre_train_vae_epoch:
            optimizer = torch.optim.Adam(itertools.chain(f_vae.parameters(), flow.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
        dataloader = tqdm(dataloader)
        for i, (imgs, labels) in enumerate(dataloader):
            # Configure input
            imgs = Variable(imgs.type(FloatTensor).to(device))
            labels = Variable(labels.type(FloatTensor).to(device))

            # -----------------
            #  Train Flow-VAE
            # -----------------
            x_hat, loss1, weight_scale = f_vae(imgs)
            # add L2 regularization on scaling factors
            if args.weight_norm:
                scale_reg = 5e-5    # L2 regularization strength
                loss1 = loss1 + scale_reg * weight_scale
                # print(weight_scale)
            if epoch > pre_train_vae_epoch:
                mean, log_var = f_vae.encode(imgs)
                z_hat, _ = f_vae.transform(mean, log_var)

                z_hat = torch.tensor(z_hat, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.float32).to(device)
                # optimizer.zero_grad()
                loss2 = - flow.log_prob(inputs=z_hat, context=labels).mean()
                loss_total = loss1 + loss2
                with torch.no_grad():
                    x_exact = labels
                    z_iaf = flow.sample(num_samples=1, context = x_exact.to(device)).squeeze(1)
                x_hat = torch.squeeze(f_vae.decode(z_iaf))
            else:
                loss_total = loss1
                loss2 = loss1 - loss1

            ssim_val = ssim(imgs, x_hat, data_range=1, size_average=True)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losss.append(loss_total.item())
            ssimm.append(ssim_val.item())
            floww.append(loss2.item())
            f_loss1 = np.mean(losss)
            ssim1 = np.mean(ssimm)
            flow1 = np.mean(floww)
            if epoch > pre_train_vae_epoch:
                dataloader.desc = "[Epoch {}/{}] [Batch {}/{}] [VAE loss: {:.3f}] [SSIM: {:.3f}] [Flow loss: {:.3f}] [Total average_loss: {:.3f}] [average_ssim: {:.3f}]".format(epoch, opt.n_epochs, i, len(dataloader), loss1.item(), ssim_val, loss2.item(), f_loss1, ssim1)
            else:
                dataloader.desc = "[Epoch {}/{}] [Batch {}/{}] [VAE loss: {:.3f}] [SSIM: {:.3f}] [Total average_loss: {:.3f}] [average_ssim: {:.3f}]".format(epoch, opt.n_epochs, i, len(dataloader), loss1.item(), ssim_val, f_loss1, ssim1)

        if epoch % 10 == 0:
            samples = f_vae.sample(args.sample_size)
            samples = torch.tensor([arcsinh_rgb(part.detach().cpu(), ["g", "r", "z"]) for part in  samples]).permute(0, 3, 1, 2)
            x_hat = x_hat[0:args.sample_size,:,:,:]
            imgs = imgs[0:args.sample_size,:,:,:]
            reconst = torch.tensor([arcsinh_rgb(part.detach().cpu(), ["g", "r", "z"]) for part in  x_hat]).permute(0, 3, 1, 2)
            imgs = torch.tensor([arcsinh_rgb(part.detach().cpu(), ["g", "r", "z"]) for part in  imgs]).permute(0, 3, 1, 2)

            orig = imgs.reshape(1, -1, args.channels, args.img_size, args.img_size).to(device)
            reconst = reconst.reshape(1, -1, args.channels, args.img_size, args.img_size).to(device)
            comparison = torch.cat(
                (orig, reconst), dim=0).permute(1, 0, 2, 3, 4).reshape(-1, args.channels, args.img_size, args.img_size)
    
            os.makedirs('./reconstruction'+ project_name +'/', exist_ok=True)
            os.makedirs('./samples'+ project_name +'/', exist_ok=True)
            utils.save_image(utils.make_grid(comparison),
                './reconstruction'+ project_name +'/' + 'recon_%d.png' % epoch, normalize=False)
            utils.save_image(utils.make_grid(samples),
                './samples'+ project_name +'/' + 'sample_%d.png' % epoch, normalize=False)
        
        if epoch == 0:
            save_loss = f_loss1
        elif save_loss > f_loss1:
            save_loss = f_loss1
            torch.save(f_vae.state_dict(), "./weights/" + save_model_name + ".pth")
            torch.save(flow.state_dict(), "./weights/" + save_model_name1 + ".pth")
            print('Save best models in {} !'.format(epoch))
        
        tags = ["learning_rate", "vae_loss", "flow_loss", "ssim", "learning_rate"]
        tb_writer.add_scalar(tags[0], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[1], f_loss1-flow1, epoch)
        tb_writer.add_scalar(tags[2], flow1, epoch)
        tb_writer.add_scalar(tags[3], ssim1, epoch)
        # tb_writer.add_scalar(tags[4], , epoch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
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
    parser.add_argument('--length', type=int, default=8, help='number of steps in the flow.')
    parser.add_argument('--sample_size', type=int, default=32, help='number of images to generate')
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
