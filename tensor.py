# 本代码学习关于torch中的Tensor（张量）数据结构

"""
张量（Tensor）：PyTorch 的核心数据结构，支持多维数组，并可以在 CPU 或 GPU 上进行加速计算。

张量（Tensor）是 PyTorch 中的核心数据结构，用于存储和操作多维数组。
张量可以视为一个多维数组，支持加速计算的操作。
在 PyTorch 中，张量的概念类似于 NumPy 中的数组，但是 PyTorch 的张量可以运行在不同的设备上，比如 CPU 和 GPU，这使得它们非常适合于进行大规模并行计算，特别是在深度学习领域。
维度（Dimensionality）：张量的维度指的是数据的多维数组结构。例如，一个标量（0维张量）是一个单独的数字，一个向量（1维张量）是一个一维数组，一个矩阵（2维张量）是一个二维数组，以此类推。
形状（Shape）：张量的形状是指每个维度上的大小。例如，一个形状为(3, 4)的张量意味着它有3行4列。
数据类型（Dtype）：张量中的数据类型定义了存储每个元素所需的内存大小和解释方式。PyTorch支持多种数据类型，包括整数型（如torch.int8、torch.int32）、浮点型（如torch.float32、torch.float64）和布尔型（torch.bool）。
"""


import torch
import numpy as np

# 创建一个 3x3x3 的全 0 张量
a = torch.zeros(3, 3, 3)
print("a= ", a)

# 创建一个 3x3x3 的全 1 张量
b = torch.ones(3, 3, 3)
print("b= ", b)

# 创建一个 3x3x3 的随机整数张量
c = torch.randint(0, 10, (3, 3, 3))
print("c= ", c)

# 直接从数据创建
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print("x_data= ", x_data)

# 从 NumPy 数组创建张量
numpy_array = np.array([[1, 2], [3, 4]])
x_numpy = torch.from_numpy(numpy_array)
print("x_numpy= ", x_numpy)

# 在指定设备（CPU/GPU）上创建张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device=device)
print("d= ", d)

# 张量的操作
print("----- 张量的操作 -----")
# 张量的加减
x1_data = [[1, 1, 1], [0, 2, 0], [0, 0, 3]]
x2_data = [[4, 0, 0], [1, 5, 0], [1, 0, 6]]
x1 = torch.tensor(x1_data)
x2 = torch.tensor(x2_data)
print("x1 = ", x1)
print("x2 = ", x2)
print("x1 + x2 = ", x1 + x2)
print("x1 - x2 = ", x1 - x2)

# 张量的乘法（不等于矩阵乘法）
print(f"x1.mul(x2)= \n {x1.mul(x2)} \n")
print(f"x1 * x2= \n {x1 * x2} \n")

# 张量的转置
print(f"x1.t= \n {x1.t()} \n")

# 张量的属性
print("----- 张量的属性 -----")
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
if torch.cuda.is_available():
	# tensor = tensor.cuda()
	tensor = tensor.to('cuda')
print(f"Now Device tensor is stored on: {tensor.device}")

# CPU 上的张量和 NumPy 数组可以共享其底层内存位置，改变其中一个会改变另一个
print("----- tensor 与 numpy -----")
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# tensor.view()函数, 改变矩阵维度
print("----- tensor.view() -----")
print(f"c: {c}")
print("c.size = ", c.size())
print("len(c.size()) = ", len(c.size()))

view = c.view(9, -1)    # -1, 根据其他维度,自动调整该维度数量
print(f"view: {view}")
print("view.size() = ", view.size())
print("len(view.size()) = ", len(view.size()))


permute = x_data.permute(1, 0)  # permute()交换维度顺序
print(f"permute: {permute}")
print("permute.size() = ", permute.size())
print("len(permute.size()) = ", len(permute.size()))