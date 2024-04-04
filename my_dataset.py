from PIL import Image
from skimage import io
import torch
from torch.utils.data import Dataset
import os
import numpy as np

import h5py

def arcsinh_rgb(imgs, bands, scales=None,m = 0.03):
    rgbscales =dict(g=(2,6.0), r=(1,3.4), z=(0,2.2))
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    Q = 50
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    # fI = MTF(I)
    I += (I == 0.) * 1e-9
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I
    rgb = np.clip(rgb, 0, 1)
    return rgb
# arcsinh_rgb(img, ["g", "r", "z"])

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.file = h5py.File(hdf5_file, 'r')
        self.image_key = "images"
        self.transform = transform
        self.first_image_shape = [3,128,128]

    def __getitem__(self, index):
        # flux_g = self.file["FLUX_G"][index]
        # flux_r = self.file["FLUX_R"][index]
        # flux_z = self.file["FLUX_Z"][index]
        # shape_r = self.file["SHAPE_R"][index]
        shape_e1 = self.file["SHAPE_E1"][index]
        shape_e2 = self.file["SHAPE_E2"][index]
        z = self.file["Z"][index]
        label = np.array([z, shape_e1, shape_e2])
        # label = np.array([z, flux_g, flux_r, flux_z, shape_r, shape_e1, shape_e2])
        # label = np.log10(label)*10

        image = self.file[self.image_key][index]
        raw_shape = image.shape
        if raw_shape[0] != self.first_image_shape[0]:
            raise IOError('Image files do not match the channel Dimensions')
        if raw_shape[1] >= self.first_image_shape[1] and raw_shape[2] >= self.first_image_shape[2]:
            add_size1 = int((raw_shape[1] - self.first_image_shape[1]) / 2)
            add_size2 = int((raw_shape[2] - self.first_image_shape[2]) / 2)
            image = image[:, add_size1 : add_size1 + self.first_image_shape[1], add_size2 : add_size2 + self.first_image_shape[2]]
        else:
            raise IOError('Image files do not match the image Dimensions')

        # image = self.file[self.image_key][index].transpose((1, 2, 0))
        # min_per_channel = image.min(axis=(0, 1), keepdims=True)
        # max_per_channel = image.max(axis=(0, 1), keepdims=True)
        # normalized_image = (image - min_per_channel) / (max_per_channel - min_per_channel)
        # # # print(normalized_image.shape)
        # normalized_image = Image.fromarray((normalized_image* 255).astype(np.uint8))
        # normalized_image = Image.fromarray(image)
        # # # print(np.max(np.array(normalized_image)))

        # if self.transform is not None:
        #     # normalized_image = self.transform(image)
        #     normalized_image = self.transform(normalized_image)
        # # # print(np.max(np.array(normalized_image)))
        # return normalized_image, label
        return image, label

    def __len__(self):
        return len(self.file[self.image_key])
        # return 10000

class HDF5_Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.file = h5py.File(hdf5_file, 'r')
        self.image_key = "images"
        self.transform = transform
        self.first_image_shape = [3,128,128]

    def __getitem__(self, index):
        shape_e1 = self.file["SHAPE_E1"][index]
        shape_e2 = self.file["SHAPE_E2"][index]
        z = self.file["Z"][index]
        label = np.array([z, shape_e1, shape_e2])

        image = self.file[self.image_key][index]
        raw_shape = image.shape
        if raw_shape[0] != self.first_image_shape[0]:
            raise IOError('Image files do not match the channel Dimensions')
        if raw_shape[1] >= self.first_image_shape[1] and raw_shape[2] >= self.first_image_shape[2]:
            add_size1 = int((raw_shape[1] - self.first_image_shape[1]) / 2)
            add_size2 = int((raw_shape[2] - self.first_image_shape[2]) / 2)
            image = image[:, add_size1 : add_size1 + self.first_image_shape[1], add_size2 : add_size2 + self.first_image_shape[2]]
        else:
            raise IOError('Image files do not match the image Dimensions')

        image = image.transpose((1, 2, 0))
        min_per_channel = image.min(axis=(0, 1), keepdims=True)
        max_per_channel = image.max(axis=(0, 1), keepdims=True)
        normalized_image = (image - min_per_channel) / (max_per_channel - min_per_channel)
        # # print(normalized_image.shape)
        normalized_image = Image.fromarray((normalized_image* 255).astype(np.uint8))
        # normalized_image = Image.fromarray(image)
        # # print(np.max(np.array(normalized_image)))

        if self.transform is not None:
            # normalized_image = self.transform(image)
            normalized_image = self.transform(normalized_image)
        # # print(np.max(np.array(normalized_image)))
        return normalized_image, label

    def __len__(self):
        return len(self.file[self.image_key])
        # return 100000

def scale_data(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._type = "h5"
        with h5py.File(self._path, 'r') as f:
            num_images = f['images'].shape[0]  # 获取图像数量
            self._image_fnames = list(range(20))  # 创建一个图像索引列表
            # self._image_fnames = list(range(num_images))  # 创建一个图像索引列表
            # 获取第一张图像的形状以确定整个数据集的形状
            # first_image_shape = [3,128,128]
            first_image_shape = [3,256,256]

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [num_images, first_image_shape[0], first_image_shape[1], first_image_shape[2]]
        # raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
