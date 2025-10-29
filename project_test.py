# 本代码将完整展示神经网络从数据集——数据加载——神经网络（前向推理）——优化器（梯度下降）——损失函数（反向学习）——保存模型的整个过程
"""
超参数是可调节的参数，允许你控制模型的优化过程。不同的超参数值会影响模型的训练速度和收敛率
常见的超参数：
· 训练轮数(epochs): 遍历数据集的次数
· 热身训练轮数(warn_epochs): 使用较低的学习率，热身训练遍历数据集的次数
· 批量大小(batch_size): 在更新参数之前通过网络传播的数据样本数量（与设备内存有关）
· 学习率(learning_rate): 在每个批量/训练轮中更新模型参数的幅度。较小的值会产生较慢的学习速度，而较大的值可能导致训练期间出现不可预测的行为
"""

import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

# 数据集
train_data = datasets.MNIST(root='./datasets', train=True, download=True, transform=transforms.ToTensor())
val_data = datasets.MNIST(root='./datasets', train=False, download=True, transform=transforms.ToTensor())

# 数据加载
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# 创建神经网络模型
class Net(nn.Module):
	def __init__(self):
		"""
		定义神经网络模型
		"""
		super(Net, self).__init__()
		self.flatten = nn.Flatten()
		self.linear = nn.Sequential(
			nn.Linear(28 * 28, 512),
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
		return self.linear(x)
model = Net()

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)    # 随机梯度下降

# 损失函数
loss_fn = nn.CrossEntropyLoss() # 适用于多分类任务的损失函数，使用了“负对数似然损失”

# 训练过程
def train(dataloader, model, loss_fn, optimizer,epoch):
	size = len(dataloader.dataset)
	# Set the model to training mode - important for batch normalization and dropout layers
	# 设置模型为训练模式，这对batch normalization层和dropout层非常重要，因为torch中某些层，在训练和推理时的行为是不同的
	# 训练模式中，BatchNorm 会计算当前批次的均值和方差，并更新运行均值（running mean）和方差（running variance）
	# 训练模式中，Dropout 会随机将部分神经元置零，以防止过拟合
	model.train()
	
	for batch_idx, (data, target) in enumerate(dataloader):
		# Compute prediction and loss
		# 计算预测和损失
		pred = model(data)
		loss = loss_fn(pred, target)
		
		# Backpropagation(反向学习)
		optimizer.zero_grad()   # 梯度默认会累积，为了防止重复计算，清除上一轮训练时的梯度值
		loss.backward()         # 反向学习，计算损失函数相对于模型参数的梯度
		optimizer.step()        # 根据计算出的梯度更新模型参数
		
		if (batch_idx+1) % 100 == 0:    # 每100个数据，打印数据
			loss, current = loss.item(), batch_idx * 64 +  len(data)
			print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
	torch.save({
		'epoch': epoch + 1,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
	}, f"./weights/model_{epoch+1}.pth")

# 验证过程
def val(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	# Set the model to evaluation mode - important for batch normalization and dropout layers
	# 设置模型为评价模式，这对batch normalization层和dropout层非常重要，因为torch中某些层，在训练和推理时的行为是不同的
	# 评价模式中，BatchNorm 使用之前累积的运行均值和方差，不再更新
	# 评价模式中，Dropout 被禁用，所有神经元都会参与计算
	model.eval()
	num_batches = len(dataloader)

	val_loss , correct = 0, 0
	with torch.no_grad():   # 冻结权重
		for data, target in dataloader:
			pred = model(data)
			val_loss += loss_fn(pred, target).item()
			correct += (pred.argmax(1) == target).type(torch.float).sum().item()
	val_loss /= num_batches
	correct /= size
	print(f"Val Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

# 开始训练
epochs = 10
os.makedirs("./datasets", exist_ok=True)
os.makedirs("./weight", exist_ok=True)
for i in range(epochs):
	print(f"Epoch {i+1}/{epochs}\n---------------------------------")
	train(train_loader, model, loss_fn, optimizer,i)
	val(val_loader, model, loss_fn)
print("Done!")