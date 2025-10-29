# 本代码学习如何建立神经网络（NN）的演示

"""
神经网络由对数据执行操作的层/模块组成。
torch.nn 命名空间提供了构建自己的神经网络所需的所有构建块。
PyTorch 中的每个模块都继承自 nn.Module。
神经网络本身就是一个由其他模块（层）组成的模块。
这种嵌套结构使得构建和管理复杂的架构变得容易。

nn.Sequential() 是模块的有序容器。数据按定义的相同顺序通过所有模块。

nn.Flatten() 初始化 nn.Flatten 层，将每张 2D 28x28 图像转换为包含 784 个像素值的连续数组（小批量维度（dim=0）被保留）

常见的神经网络层
nn.Linear() 线性层是一个使用其存储的权重和偏置对输入应用线性变换的模块
nn.Conv2d() 2D卷积层，用于图像处理
nn.MaxPool2d() 2D最大池化层，用于降维

常见的激活函数
	非线性激活层在模型的输入和输出之间创建复杂的映射。它们在线性变换之后应用，以引入非线性，帮助神经网络学习各种现象
nn.ReLU() 目前最流行的激活函数之一，定义为 f(x) = max(0, x)，有助于解决梯度消失问题
nn.Sigmoid() 用于二分类问题，输出值在0~1之间
nn.Tanh() 输出值在 -1 和 1 之间，常用于输出层之前
nn.Softmax() 常用于多分类问题的输出层，将输出转换为概率分布

常见的损失函数
	损失函数用于衡量模型的预测值与真实值之间的差异
nn.MSELoss() 回归问题常用，计算输出与目标值的平方差
nn.CrossEntropyLoss() 分类问题常用，计算输出和真实标签之间的交叉熵
nn.BCEWithLogitsLoss() 二分类问题，结合了 Sigmoid 激活和二元交叉熵损失
使用 loss.backward() 会计算损失函数相对于参数的导数
使用 torch.no_grad() 禁用梯度跟踪，冻结权重，一般使用在验证过程中；现许多大模型因为权重参数过多，也会使用此函数来冻结部分权重


常见的优化器
	优化是调整模型参数以在每个训练步骤中减少模型误差的过程
optim.SGD() 随机梯度下降
optim.Adam() 自适应矩估计
optim.RMS() 均方根传播
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 选择合适的训练设备
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# 定义神经网络
class NeuralNetwork(nn.Module):
	def __init__(self):
		"""
		定义网络层
		初始化神经网络
		"""
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()                                 # 展平张量
		self.fc1 = nn.Sequential(                                   # 神经网络层（线性层——非线性激活层——线性层——非线性激活层——线性层）
			nn.Linear(784, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 10)
		)
	def forward(self, x):
		"""
		定义数据的前向推理过程
		param x: 神经网络
		return: 返回预测结果
		"""
		x = self.flatten(x)
		return self.fc1(x)

# 模型层
# 创建神经网络实例，并将其移动到训练设备上
model = NeuralNetwork().to(device)
print("----- 模型数据 -----")
print(model)
# 假设输入一张28x28的彩色图像，将数据移动到训练设备上
# 需要注意的是，一定要将数据集和网络模型都放在同意训练设备才能运行
input_image = torch.randn(3, 28, 28).to(device)

# 直接使用定义好的神经网络模型训练
# result = model(input_image) # 根据网络中的forward进行训练

print("----- 解析模型层 -----")
print(input_image.size())
# 展平数据
flatten = nn.Flatten().to(device)
flat_img = flatten(input_image)
print(f"flat_img= \n {flat_img.size()} \n")
# 使用线性层前向推理
linear_layer = nn.Linear(in_features=28 * 28, out_features=10).to(device)
linear_output = linear_layer(flat_img)
print(f"linear_output = \n {linear_output.size()} \n")
# 非线性激活层
print(f"Before ReLU= \n {linear_output} \n")
ReLU_output = nn.ReLU()(linear_output).to(device)
print(f"After ReLU= \n {ReLU_output} \n")

# 将神经网络最后一层传递给nn.Softmax模块，结果被缩放到[0, 1]的值，表示模型对每个类别的预测概率
softmax = nn.Softmax(dim=1)
pred_result = softmax(ReLU_output)
print(f"pred_result= \n {pred_result} \n")
