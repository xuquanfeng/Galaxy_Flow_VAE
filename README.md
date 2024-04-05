Galaxy_Flow_VAE
========================
English | [简体中文](README_zh.md)

The goal of this work is to combine ``Flow`` and ``VAE`` to build a pipeline capable of generating clean galaxy images from physical parameters and to add to this pipeline the noise of various survey programs and ``PSF`` to provide a closer simulation of galaxy images for the data of various characterization surveys.

The task is divided into two parts: a data-driven unsupervised extraction of morphological features from a large database using ``VAE``; and the construction of a reasonable bijective model between the physical parameters and the extraction of hidden variables by ``Flow``, where the galaxy images are generated using the physical parameters.

This project explores the above methods in a variety of ways, as detailed below.

## Requirements
### Main
-  torch>=1.8.1
-  torchvision>=0.8.0
-  cudatoolkit>=0.7.1
-  itertools
-  nflows

### Minor
-  warnings
-  tensorboard
-  tqdm
-  os
-  h5py
-  astropy
-  matplotlib>=1.0.2
-  argparse
-  datetime
-  random
-  numpy
-  pandas

Details of ``Galaxy_Flow_VAE``
========================

# Part I ``VAE``
## Datasets
This work uses the `DR9` data released by the `DECaLS` project, and image data of galaxies in the `g, r, z` bands were downloaded from publicly [available websites](https://www.legacysurvey.org/dr9/description/).

The downloaded galaxy image data (image size basically at `1×152×152`, with three bands stacked into a three-channel image), while the physical parameters were obtained by using `sextractor`.

Finally, save the galaxy image and physical parameters to an `h5` file using `create_dataset.ipynb`.

Read the corresponding `h5` file in `my_dataset.py` and crop the `3×182×128` size of it for training.

## Methods
This work follows the ``VAE`` model of this previous work.
```
@article{10.1093/mnras/stad3181,
    author = {Xu, Quanfeng and Shen, Shiyin and de Souza, Rafael S and Chen, Mi and Ye, Renhao and She, Yumei and Chen, Zhu and Ishida, Emille E O and Krone-Martins, Alberto and Durgesh, Rupesh},
    title = "{From images to features: unbiased morphology classification via variational auto-encoders and domain adaptation}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {526},
    number = {4},
    pages = {6391-6400},
    year = {2023},
    month = {10},
    issn = {0035-8711},
    doi = {10.1093/mnras/stad3181},
    url = {https://doi.org/10.1093/mnras/stad3181},
}
```
And based on this, a fine-tuning of the model was carried out, in which some ``ReLU`` were modified into ``ELU`` activation functions to make the model predict galaxy images of a more reasonable order of magnitude, instead of the order of magnitude limitations of ``0-255`` or ``0-1`` of the natural images.
### Training
Train the galaxy images in the `h5` file directly with the following commands. (`ssim.py` can calculate the discrepancy between the generated image and the real one.)
```
python train_vae.py
```
The ``--data_path`` parameter needs to be defined.

You can access it by clicking on 
[Github-models](https://github.com/xuquanfeng/Galaxy_Flow_VAE/blob/master/conv_vae.py)

# Part II ``Flow``

A bijection designed to connect physical parameters to vectors of latent variables connects `VAE` and `Flow`.
## Separate Training
- `train_vae.py`: Training code for the first part；
- `predicts_vae.py`: For the above `VAE` model, to the prediction of galaxy images.
- `train_flow.py`: Fastened the `VAE` model that was able to extract the galaxy morphology after training, and then trained `Flow` for mapping.
- `predicts_flow.py`: Loading `VAE` and `Flow` models to the prediction of galaxy images.
## Parallel Training
- `train_fvae.py`: The models of `VAE` and `Flow` are trained in parallel to obtain the results of dynamic optimization of the two models for each other.
- `predicts_flow.py`: Loading `VAE` and `Flow` models to predict galaxy images.
# Citation
If you'd like to cite this work, please use the following.
```
Waiting accepted.
```
