import torch

# 当前安装的 PyTorch 库的版本
print(torch.__version__)
# 检查 CUDA 是否可用，即你的系统有 NVIDIA 的 GPU
print(torch.cuda.is_available())

# torch.rand() 生成的是均匀分布的随机数
# torch.randn() 生成的是正态分布的随机数
x = torch.rand(2, 3, 2, 2)      # 存储2 * 3 * 2 * 2 = 24个数据
y = torch.randn(2, 1, 2, 2)     # 存储2 * 1 * 2 * 2 = 8个数据

print("x=",x)
print("y=",y)