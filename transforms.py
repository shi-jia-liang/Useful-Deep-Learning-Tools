# 本代码学习如何使用 transforms（变换） 对数据进行一些处理，使其适合训练

"""
torchvision.datasets 提供了许多常见的数据集，并简化了数据加载的过程
torchvision.models 提供了许多常见的模型
torchvision.transforms 用于使用常见的图像预处理和增强操作，提高了模型的泛化能力
数据预处理分为 旋转、裁剪、归一化
数据增强分为 随机翻转、随机旋转、随机裁剪，重点在随机

所有 TorchVision 数据集都有两个参数
-transform          用于修改特征
-target_transform   用于修改标签
它们接受包含变换逻辑的可调用对象。

"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
定义transforms.Compose<UNK>模块，将多个变换操作组合在一起
常见的有：
· transforms.Resize()： 调整图像大小
· transforms.ToTensor()： 将图像转换为 PyTorch 张量，值会被归一化到 [0, 1] 范围
· transforms.Normalize()：标准化图像数据，通常使用预训练模型时需要进行标准化处理
· transforms.RandomHorizontalFlip()： 随机水平翻转
· transforms.RandomRotation(30)： 随机旋转 30 度
· transforms.RandomResizedCrop(128)： 随机裁剪并调整为 128x128
"""

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])

"""
通常情况下，train = True      为加载训练集
		  train = False     为加载验证集
"""
# 下载并加载 MNIST 数据集
train_dataset = datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./datasets', train=False, download=True, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 迭代训练数据
for inputs, labels in train_loader:
    print(inputs.shape)  # 每个批次的输入数据形状
    print(labels.shape)  # 每个批次的标签形状
    print("\n")