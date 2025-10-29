"""
本代码保存常用的transforms工具

"""
import cv2
import random
import numpy as np
from typing import List, Union

import torch
from sympy.abc import theta
from torchvision.transforms import functional
from torchvision.transforms import transforms

import kornia
import kornia.augmentation
from kornia.geometry.transform import get_tps_transform
from kornia.geometry.transform import warp_points_tps, warp_image_tps

import tqdm
import time

random.seed(0)
torch.manual_seed(0)

# 组合多种变换
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target

# 变为ToTensor的张量
class ToTensor(object):
    def __call__(self, image, target):
        image = functional.to_tensor(image)
        target = functional.to_tensor(target)
        return image, target

# 随机水平翻转图像
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            target = functional.hflip(target)
        return image, target

# 对图像进行标准化（归一化）
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image, target

# 调整图像尺寸
# 初始缩放时的目标尺寸
class Resize(object):
    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        self.size = size  # [h, w] or an int
        self.resize_mask = resize_mask

    def _normalized_size(self):
        # functional.resize in some type-checking contexts expects a list-like size,
        # so convert an int to a two-element list [h, w].
        if isinstance(self.size, int):
            return [self.size, self.size]
        return list(self.size)

    def __call__(self, image, target=None):
        size = self._normalized_size()
        image = functional.resize(image, size)
        if self.resize_mask and target is not None:
            target = functional.resize(target, size)

        return image, target

# 随机裁剪图像
class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.shape[-2:])
        if min_size < self.size:
            ow, oh = img.size
            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0
            img = functional.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, image, target):
        image = self.pad_if_smaller(image)
        target = self.pad_if_smaller(target)
        crop_params = transforms.RandomCrop.get_params(image, (self.size, self.size))
        image = functional.crop(image, *crop_params)
        target = functional.crop(target, *crop_params)
        return image, target

# 非刚性图像变换
def generateRandomTPS(shape, grid = (8, 6), GLOBAL_MULTIPLIER = 0.3, prob = 0.5):
    # 网格点生成
    h, w = shape
    sh, sw = h / grid[0], w / grid[1]
    src = torch.dstack(torch.meshgrid(torch.arange(0, h + sh, sh), torch.arange(0, w + sw, sw), indexing = "ij"))

    # 随机偏移生成
    offset = torch.rand(grid[0] + 1, grid[1] + 1, 2) - 0.5 # 生成[-0.5, 0.5)范围内的随机偏移
    offset *= torch.tensor(sh / 2, sw / 2).view(1, 1, 2) * min(0.97, 2.0 * GLOBAL_MULTIPLIER) # 幅度控制
    dst = src + offset if np.random.uniform() < prob else src

    # 归一化
    src, dst = src.view(1, -1, 2), dst.view(1, -1, 2)
    src = (src / torch.tensor([h, w]).view(1, 1, 2)) * 2 - 1.0 # 归一化：将坐标从[0, h]×[0, w]映射到[-1, 1]×[-1, 1]
    dst = (dst / torch.tensor([h, w]).view(1, 1, 2)) * 2 - 1.0 # 归一化：将坐标从[0, h]×[0, w]映射到[-1, 1]×[-1, 1]
    weights, A = get_tps_transform(dst, src)

    return src, weights, A

# 刚性图像变换
def generateRandomHomography(shape, GLOBAL_MULTIPLIER = 0.3):
    # 创建随机旋转变换
    theta = np.radians(np.random.uniform(-45, 45))
    c, s = np.cos(theta), np.sin(theta)

    # 创建随机缩放变换
    scale_x, scale_y = np.random.uniform(0.35, 1.2, 2)

    # 创建随机平移变换
    # 设定坐标系原点
    # 设定坐标系原点(即将图像的坐标系原点设定在图像中心)
    tx, ty = shape[1] / 2.0, shape[0] / 2.0
    # 随机平移坐标系原点
    txn, tyn = np.random.normal(0, 120.0 * GLOBAL_MULTIPLIER, 2)

    # 仿射变换参数
    sx, sy = np.random.normal(0, 0.6 * GLOBAL_MULTIPLIER, 2)

    # 射影变换参数
    p1, p2 = np.random.normal(0, 0.006 * GLOBAL_MULTIPLIER, 2)

    # 计算单应性变换矩阵
    H_t = np.array(((1, 0, -tx), (0, 1, -ty), (0, 0, 1)))  # t                      # 水平变换
    H_r = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))  # rotation                  # 旋转变换
    H_a = np.array(((1, sy, 0), (sx, 1, 0), (0, 0, 1)))  # affine                   # 仿射变换
    H_p = np.array(((1, 0, 0), (0, 1, 0), (p1, p2, 1)))  # projective               # 射影变换
    H_s = np.array(((scale_x, 0, 0), (0, scale_y, 0), (0, 0, 1)))  # scale          # 缩放变换
    H_b = np.array(((1.0, 0, tx + txn), (0, 1, ty + tyn), (0, 0, 1)))  # t_back,    # 坐标系原点变换

    # H = H_t * H_r * H_a * H_p * H_s * H_b
    H = H_t @ H_r @ H_a @ H_p @ H_s @ H_b

    return H