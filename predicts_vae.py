import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torch

from torchvision import transforms
import h5py

from my_dataset import HDF5Dataset, arcsinh_rgb
from conv_vae import VAE
from astropy.io import fits
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main(args):
    os.makedirs("images", exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # create model
    model = VAE(args)
    # load model weights
    model_weight_path = "./weights/model_train_vae_best.pth"
    # model.load_state_dict(torch.load(model_weight_path))
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    model.to(device)

    # load image
    # train_dataset = HDF5Dataset(args.data_path, transform=data_transform["train"])
    # print('Using {} images to train'.format(len(train_dataset)))

    # file = h5py.File(args.data_path, 'r')
    # raw_idx = 10000
    # with h5py.File(args.data_path, 'r') as f:
    #     num_images = f['images'].shape[0]  # 获取图像数量
    #     image = f['images'][raw_idx]
    #     psf = f["psfs"][raw_idx]
    #     labels = f["labels"][raw_idx]
    # print(num_images)

    # flux_g = file["FLUX_G"][index]
    # flux_r = file["FLUX_R"][index]
    # flux_z = file["FLUX_Z"][index]
    # shape_r = file["SHAPE_R"][index]
    # shape_e1 = file["SHAPE_E1"][index]
    # shape_e2 = file["SHAPE_E2"][index]
    # z = file["Z"][index]
    # image = file["images"][index].transpose((1, 2, 0))
    # label = np.array([z, flux_g, flux_r, flux_z, shape_r, shape_e1, shape_e2])
    # min_per_channel = image.min(axis=(0, 1), keepdims=True)
    # max_per_channel = image.max(axis=(0, 1), keepdims=True)
    # normalized_image = (image - min_per_channel) / (max_per_channel - min_per_channel)
    # # print(normalized_image.shape)
    # normalized_image = Image.fromarray((normalized_image* 255).astype(np.uint8))
    # normalized_image = data_transform["train"](normalized_image)
    # # img = torch.unsqueeze(normalized_image, dim=0)
    # img = torch.tensor(normalized_image).cpu()

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # z = Variable(FloatTensor(np.random.normal(0, 1, (1, args.latent_dim)))).to(device)
    # # z = Variable(FloatTensor(np.random.randint(0, 255, (1, args.channels, args.img_size, args.img_size)))).to(device)
    # label = torch.tensor(label)
    # labels = Variable(label.type(FloatTensor)).view(1,-1).to(device)

    train_dataset = HDF5Dataset(args.data_path)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers

    dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
    raw_idx = 11
    for i, (imgs, labels) in enumerate(dataloader):
        if i == raw_idx:
            imgs = Variable(imgs.type(FloatTensor).to(device))
            labels = Variable(labels.type(FloatTensor).to(device))
            print(imgs.size())
            print(labels.size())
            break
    x_hat, _, _ = model(imgs)

    # with torch.no_grad():
    # # 这边可以选择用不用豪子的图像可视化方法
    # output = x_hat.detach().cpu().squeeze(0)
    # images = imgs.detach().cpu().squeeze(0)

    # output = torch.tensor(arcsinh_rgb(x_hat.detach().cpu().squeeze(0), ["g", "r", "z"]))
    # images = torch.tensor(arcsinh_rgb(imgs.detach().cpu().squeeze(0), ["g", "r", "z"]))
    # save_image([output, images], "images/%d.png" % raw_idx, nrow=2, normalize=True)
    # outputs = output.numpy()
    # imagess = images.numpy()
    # fits.writeto("./images/%d_original.fits" % raw_idx, imagess, overwrite=True)
    # fits.writeto("./images/%d_generate.fits" % raw_idx, outputs, overwrite=True)
    x_hat = x_hat[0:args.sample_size,:,:,:]
    imgs = imgs[0:args.sample_size,:,:,:]
    reconst = torch.tensor([arcsinh_rgb(part.detach().cpu(), ["g", "r", "z"]) for part in  x_hat]).permute(0, 3, 1, 2)
    imgs = torch.tensor([arcsinh_rgb(part.detach().cpu(), ["g", "r", "z"]) for part in  imgs]).permute(0, 3, 1, 2)
    orig = imgs.reshape(1, -1, args.channels, args.img_size, args.img_size).to(device)
    reconst = reconst.reshape(1, -1, args.channels, args.img_size, args.img_size).to(device)
    comparison = torch.cat(
        (orig, reconst), dim=0).permute(1, 0, 2, 3, 4).reshape(-1, args.channels, args.img_size, args.img_size)
    os.makedirs('./images' +'/', exist_ok=True)
    # os.makedirs('./samples' +'/', exist_ok=True)
    save_image(make_grid(comparison),
        './images' +'/' + 'recon_vae_%d.png' % raw_idx)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument('--data_path', type=str,default="/data/public/renhaoye/MockCSST/dataset/merged.h5")
    # parser.add_argument('--data_path', type=str,default="/data/public/renhaoye/stylegan3/DESI_GAN.h5")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
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
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
