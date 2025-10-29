# 本代码学习如何使用数据集和数据集加载器

"""
理想情况下，我们希望将数据集代码和模型训练代码解耦，以提高可读性和模块化
Dataset 用于存储数据集（样本及其对应的标签）
dataLoader 用于在Dateset周围封装了一个迭代器，以便于访问样本
"""


import torch
# 用于创建数据集，或者加载数据集
from torch.utils.data import Dataset

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, feature_data, label_data):
        """
        初始化数据集，feature_data 和 label_data 是两个列表或数组
        feature_data: 输入特征
        label_data: 目标标签
        """
        self.feature_data = feature_data
        self.label_data = label_data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.feature_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.tensor(self.feature_data[idx], dtype=torch.float32)  # 转换为 Tensor
        y = torch.tensor(self.label_data[idx], dtype=torch.float32)
        return x, y

# 示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 输入特征
Y_data = [1, 0, 1, 0]  # 目标标签

# 创建数据集实例
dataset = MyDataset(X_data, Y_data)

# 用于从Dataset中按批次（batch）加载数据，支持多线程加载并进行数据打乱。
from torch.utils.data import DataLoader

# 创建 DataLoader 实例，batch_size 设置每次加载的样本数量
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# ·batch_size:每次加载的样本数量
# ·shuffle:是否对数据进行洗牌，通常训练时需要将数据打乱
# ·drop_last: 如果数据集中的样本数不能被 batch_size 整除，设置为 True 时，丢弃最后一个不完整的 batch。

# 打印加载的数据
for epoch in range(1):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f'Batch {batch_idx + 1}:')
        print(f'Inputs: {inputs}')
        print(f'Labels: {labels}')

