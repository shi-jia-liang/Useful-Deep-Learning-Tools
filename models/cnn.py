# 该代码用于展示卷积神经网络（CNN）架构

"""
PyTorch 卷积神经网络 (Convolutional Neural Networks, CNN) 是一类专门用于处理具有网格状拓扑结构数据（如图像）的深度学习模型。
CNN 是计算机视觉任务（如图像分类、目标检测和分割）的核心技术。
"""

# 导入依赖库
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# 检查训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备： {device}")

# 数据集
train_data = datasets.MNIST(root='../datasets', train=True, download=True, transform=transforms.ToTensor())
val_data = datasets.MNIST(root='../datasets', train=False, download=True, transform=transforms.ToTensor())

# 数据集加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

# 数据集预处理和数据增强
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])

# 定义模型
class SimpleCNN(nn.Module):
	def __init__(self):
		"""
		定义网络层
		"""
		super(SimpleCNN, self).__init__()
		# 定义卷积层：输入1通道，输出32通道，卷积核大小3x3
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		# 定义卷积层：输入32通道，输出64通道
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		# 定义全连接层
		self.fc1 = nn.Linear(in_features=7*7*64, out_features=128)
		self.fc2 = nn.Linear(in_features=128, out_features=10)
	def forward(self, x):
		"""
		前向推理
		param x: 神经网络
		return: 返回预测结果
		"""
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # 展平
		x = self.fc1(x)
		x = self.fc2(x)
		return x

# 实例化模型
model = SimpleCNN().to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss() # 多分裂交叉熵损失

# 优化器
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9) # 学习率和动量（学习率决定每一步的步长， 动量决定每一步的惯性）

# 训练模型
def train(dataloader, model, loss_fn, optimizer, epoch, epochs):
	total_loss = 0
	model.train()

	# 使用tqdm包装数据加载器
	batch_loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)

	for images, labels in batch_loop:
		images, labels = images.to(device), labels.to(device)
		output = model(images)
		loss = loss_fn(output, labels)

		optimizer.zero_grad()   # 清空梯度
		loss.backward()         # 反向学习
		optimizer.step()        # 更新梯度

		total_loss += loss.item()
	print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
	torch.save(model.state_dict(), f"../weights/model{epoch+1}.pth")

# 验证模型
def val(dataloader, model, loss_fn):
	val_loss , correct = 0, 0
	model.eval()

	with torch.no_grad():
		for images, labels in dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)  # 预测类别
			val_loss += labels.size(0)
			correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
	accuracy = 100 * correct / val_loss
	print(f"Test Accuracy: {accuracy:.2f}")

if __name__ == '__main__':
	epochs = 10
	for epoch in range(epochs):
		train(train_loader, model, loss_fn, optimizer, epoch, epochs)
		val(val_loader, model, loss_fn)

	# 6. 可视化测试结果
	dataiter = iter(val_loader)
	images, labels = next(dataiter)
	outputs = model(images.to(device))
	_, predictions = torch.max(outputs, 1)

	# 将预测结果移回CPU用于可视化
	predictions = predictions.cpu()

	fig, axes = plt.subplots(1, 6, figsize=(12, 4))
	for i in range(6):
		axes[i].imshow(images[i][0], cmap='gray')
		axes[i].set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
		axes[i].axis('off')
	plt.show()
