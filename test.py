#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :test.py
@说明        :执行单张样本测试
@时间        :2020/02/23 12:14:50
@作者        :钱彬
@版本        :1.0
'''

# from utils import *
# from torch import nn
# from models import SRResNet, Generator
# import time
# from PIL import Image
#
# # 测试图像
# imgPath = r"E:\baidudownload2\tilesClip\L15-1650E-1215N0.TIF"
#
# # 模型参数
# large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
# small_kernel_size = 3  # 中间层卷积的核大小
# n_channels = 64  # 中间层通道数
# n_blocks = 16  # 残差模块数量
# scaling_factor = 4  # 放大比例
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# if __name__ == '__main__':
#     # 预训练模型
#     srgan_checkpoint = "./results/checkpoint_srgan.pth"
#     # srresnet_checkpoint = "./results/checkpoint_srresnet.pth"
#
#     # 加载模型SRResNet 或 SRGAN
#     checkpoint = torch.load(srgan_checkpoint)
#     generator = Generator(large_kernel_size=large_kernel_size,
#                           small_kernel_size=small_kernel_size,
#                           n_channels=n_channels,
#                           n_blocks=n_blocks,
#                           scaling_factor=scaling_factor)
#     generator = generator.to(device)
#     generator.load_state_dict(checkpoint['generator'])
#
#     generator.eval()
#     model = generator
#
#     # 加载图像
#     img = Image.open(imgPath, mode='r')
#     img = img.convert('RGB')
#
#     # 双线性上采样
#     Bicubic_img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)), Image.BICUBIC)
#     Bicubic_img.save('./data/oringTestMultiTif/sentinel_bicubic/L15-1650E-1215N0.TIF')
#
#     # 图像预处理
#     lr_img = convert_image(img, source='pil', target='imagenet-norm')
#     lr_img.unsqueeze_(0)
#
#     # 记录时间
#     start = time.time()
#
#     # 转移数据至设备
#     lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
#
#     # 模型推理
#     with torch.no_grad():
#         sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
#         sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
#         sr_img.save(r'.\data\oringTestMultiTif\sentinrl_hl\L15-1650E-1215N02.TIF')
#
#     print('用时  {:.3f} 秒'.format(time.time() - start))
from utils import *
from torch import nn
from models import SRResNet, Generator
import time
import os
from PIL import Image
import torch

# 文件夹路径
folder_path = r'E:\learn\DeepLearning\Afterclip\only8bittif'

# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 预训练模型路径
srgan_checkpoint = "./results/checkpointForRM1.pth"

# 加载模型
checkpoint = torch.load(srgan_checkpoint, map_location=device)
generator = Generator(large_kernel_size=large_kernel_size,
                      small_kernel_size=small_kernel_size,
                      n_channels=n_channels,
                      n_blocks=n_blocks,
                      scaling_factor=scaling_factor)
generator = generator.to(device)
generator.load_state_dict(checkpoint['generator'])
generator.eval()

def process_image(img_path, output_folder):
    # 加载图像
    img = Image.open(img_path)
    img = img.convert('RGB')

    # 双线性上采样保存
    bicubic_img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)), Image.BICUBIC)
    bicubic_path = os.path.join(output_folder, 'bicubic', os.path.basename(img_path))
    bicubic_img.save(bicubic_path)

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)
    lr_img = lr_img.to(device)  # 转移数据至设备

    # 模型推理
    with torch.no_grad():
        sr_img = generator(lr_img).squeeze(0).cpu().detach()  # in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_path = os.path.join(output_folder, 'sr', os.path.basename(img_path))
        sr_img.save(sr_path)

    print(f"Processed {img_path}")

if __name__ == '__main__':
    tif_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.TIF')]
    output_folder = r'./data/oringTestMultiTif/sentinel_processed_test'
    os.makedirs(os.path.join(output_folder, 'bicubic'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'sr'), exist_ok=True)

    start_time = time.time()
    for tif_file in tif_files:
        process_image(tif_file, output_folder)
    print('Total time: {:.3f} seconds'.format(time.time() - start_time))



