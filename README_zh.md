Galaxy_Flow_VAE
========================

[English](README.md) | 简体中文

此工作的目的是联合``Flow``和``VAE``建立一个能够从物理参数到生成干净星系图像的基本工具，且在此基本工具上可以添加各种巡天计划的噪声和``PSF``为各种特性巡天计划的数据提供更接近真实的星系图像模拟数据。

此任务主要分为两个部分：从大量的数据基础上，通过数据驱动用``VAE``无监督的提取星系图像形态特征；通过``Flow``对物理参数和提取隐变量之间构建一个合理的双射模型，于此方法用物理参数对星系图像进行生成。

本项目对以上方法进行多种探究，细节如下。

## 环境配置要求
### 主要引用的库
-  torch>=1.8.1
-  torchvision>=0.8.0
-  cudatoolkit>=0.7.1
-  itertools
-  nflows

### 次要引用的库
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

``Galaxy_Flow_VAE``的细节
========================

# 第一部分 ``VAE``
## 数据
本工作使用了`DECaLS`项目发布的`DR9`数据，从[公开网站](https://www.legacysurvey.org/dr9/description/)中下载了`g,r,z`三个波段的星系图像数据。

下载的星系图像数据（图像大小基本在`1×152×152`，三波段叠加成三通道图像），而后采用`sextractor`提取得到物理参数。

最后用`create_dataset.ipynb`保存星系图像和物理参数到`h5`文件。

在`my_dataset.py`中读取对应的`h5`文件，并且裁剪其中`3×182×128`大小来训练。

## 模型结构
本项工作沿用了之前此工作的``VAE``模型。
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
并且在此基础上，进行了微调模型，其中修改了一些``ReLU``变成``ELU``激活函数，使得模型预测星系图像数量级更合理，而非自然图像的``0-255``或者``0-1``的数量级限制。
### 训练
用以下命令直接训练`h5`文件中的星系图像。（其中`ssim.py`文件，能够计算生成图像和真实图像之间的差距。）
```
python train_vae.py
```
其中``--data_path``参数需要被自行定义。

您可以通过单击访问网络结构：
[Github-models](https://github.com/xuquanfeng/Galaxy_Flow_VAE/blob/master/conv_vae.py)
## 结果
采用以下命令获得`h5`文件中的星系图像，读取训练好的`VAE`模型进行预测。
```
python predicts_vae.py
```
得到结果可视化如下：

<img src="https://github.com/xuquanfeng/Galaxy_Flow_VAE/blob/master/reconstruction/recon_190.jpg" width="600">

# 第二部分 ``Flow``

旨在连接物理参数于隐变量向量的双射，将`VAE`和`Flow`进行连接。
## 分开训练
- `train_vae.py`: 第一部分的训练代码；
- `predicts_vae.py`: 对以上`VAE`模型，进行星系图像的预测。
- `train_flow.py`: 固定好训练完后能够提取星系形态的`VAE`模型，再训练`Flow`进行映射。
- `predicts_flow.py`: 读取`VAE`和`Flow`的模型，进行星系图像的预测。
## 合并训练
- `train_flow_vae.py`: 同时训练`VAE`和`Flow`的模型，用编码器得到隐变量而不用`Flow`生成的隐变量去解码得到星系图像于原图做重构损失，得到两个模型互相动态优化的结果。
- `train_FlowVAE.py`: 同时训练`VAE`和`Flow`的模型，用`Flow`生成的隐变量去解码得到星系图像于原图做重构损失，得到两个模型互相动态优化的结果。
- `predicts_flow.py`: 读取`VAE`和`Flow`的模型，进行星系图像的预测。
## 结果
采用以下命令获得`h5`文件中的星系图像，读取训练好的`VAE`和`Flow`模型进行预测。
```
python predicts_flow.py
```
得到结果可视化如下：

<img src="https://github.com/xuquanfeng/Galaxy_Flow_VAE/blob/master/images/recon_flow_170.jpg" width="600">

# 引用
如果引用我们的工作，请使用以下信息：
```
Waiting accepted.
```
