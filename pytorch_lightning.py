# 本代码将学习如何使用 PyTorch Lightning 来简化 PyTorch 代码结构
# 参考资料：https://pytorch-lightning.readthedocs.io/en/stable/
# 官方GitHub：https://github.com/PyTorchLightning/pytorch-lightning

# 导入模块
import os
from pathlib import Path
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


# 定义编码器解码器
class Encoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Sequential(
			nn.Linear(28 * 28, 64),
			nn.ReLU(),
			nn.Linear(64, 3),
		)

	def forward(self, x):
		return self.l1(x)


class Decoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Sequential(
			nn.Linear(3, 64),
			nn.ReLU(),
			nn.Linear(64, 28 * 28),
		)

	def forward(self, x):
		return self.l1(x)
# -------------------------------- 数据准备 --------------------------------
# 下面的代码被注释掉了，因为我们使用了 LightningDataModule 来处理
# transform = transforms.ToTensor()
# train_dataset = MNIST("./datasets", download=True, train=True, transform=transform)
# test_dataset = MNIST("./datasets", download=True, train=False, transform=transform)
# # 划分训练集和验证集
# train_size = int(0.8 * len(train_dataset))
# val_size = len(train_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])  # 随机划分训练集和验证集
# # 创建数据加载器
# # 计算推荐的 num_workers：通常使用 (cpu_count - 1) 避免占满所有 CPU
# # 注意：在 Windows 上如果你不想使用子进程（spawn），可以将其设为 0
# cpu_count = os.cpu_count() or 1
# default_workers = max(0, cpu_count - 1)
# # 可选上限，防止在某些环境（虚拟机/容器/笔记本）中创建过多 worker
# max_workers_cap = 4  # 如果你有更多 CPU 并且想使用更多 worker，可以提高这个值或移除该上限
# num_workers = min(default_workers, max_workers_cap)
# # persistent_workers 只有在 num_workers>0 时有效
# train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=16, shuffle=True, persistent_workers=(num_workers>0))
# val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=16, shuffle=False, persistent_workers=(num_workers>0))
# test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=16, shuffle=False, persistent_workers=(num_workers>0))

# 定义pytorch lightning data module
class MNISTDataModule(pl.LightningDataModule):
	# 初始化
	def __init__(self, data_path: str = "./datasets", num_workers : int = 4, batch_size: int = 4) -> None:
		super().__init__()
		self.data_path = data_path
		# 创建数据加载器
		# 计算推荐的 num_workers：通常使用 (cpu_count - 1) 避免占满所有 CPU
		# 注意：在 Windows 上如果你不想使用子进程（spawn），可以将其设为 0
		cpu_count = os.cpu_count() or 1
		default_workers = max(0, cpu_count - 1)
		self.num_workers = min(default_workers, num_workers)

		self.batch_size = batch_size
		self.transform = transforms.ToTensor()
	
	def prepare_data(self) -> None:
		# 下载数据集
		MNIST(self.data_path, download=True, train=True)
		MNIST(self.data_path, download=True, train=False)
	
	def setup(self, stage: str | None = None) -> None:
		if stage == "fit" or stage is None:
			# 训练集和验证集
			full_train_dataset = MNIST(self.data_path, download=False, train=True, transform=self.transform)
			train_size = int(0.8 * len(full_train_dataset))
			val_size = len(full_train_dataset) - train_size
			self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
		if stage == "test" or stage is None:
			# 测试集
			self.test_dataset = MNIST(self.data_path, download=False, train=False, transform=self.transform)
	
	def train_dataloader(self):
		return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True, persistent_workers=(self.num_workers>0))
	
	def val_dataloader(self):
		return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False, persistent_workers=(self.num_workers>0))

	def test_dataloader(self):
		return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False, persistent_workers=(self.num_workers>0))
# -------------------------------- 数据准备 --------------------------------


# 定义pytorch lightning model module
class LitAutoEncoder(pl.LightningModule):
	# 初始化
	def __init__(self, encoder, decoder):
		super().__init__()
		# self.save_hyperparameters() # 保存超参数，方便后续加载模型时使用
		self.encoder = encoder
		self.decoder = decoder

	# 训练步骤
	def training_step(self, batch, batch_idx):
		# training_step defines the train loop.
		x, _ = batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		return loss

	# 配置优化器
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer
	
	# 验证循环
	def validation_step(self, batch, batch_idx):
		x, _ = batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		val_loss = F.mse_loss(x_hat, x)
		self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

	# 测试循环
	def test_step(self, batch, batch_idx):
		x, _ = batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log("test_loss", loss)

if __name__ == "__main__":
	pl.seed_everything(42)  # 设置随机种子，确保结果可复现
	torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度（可选）
	# 准备数据
	data_module = MNISTDataModule(data_path="./datasets", num_workers=4, batch_size=32)
	data_module.prepare_data()  # 下载数据集
	data_module.setup()  # 划分数据集
	train_loader = data_module.train_dataloader()
	val_loader = data_module.val_dataloader()
	test_loader = data_module.test_dataloader()
	# 创建模型
	autoencoder = LitAutoEncoder(Encoder(), Decoder())
	# 创建日志
	logger = TensorBoardLogger(save_dir="logs/", name="lightning_logger", version = "mnist_autoencoder")

	# 创建回调函数
	# 保存模型
	checkpoint_callback = ModelCheckpoint(
		monitor="val_loss",  # 监控验证集的损失
		dirpath="weights/lightning",  # 保存路径
		filename="{epoch}--val_loss={val_loss:.2f}",  # 文件名（修正格式）
		save_on_train_epoch_end=True,  # 在每个训练epoch结束时保存
		save_top_k=5,  # 保存最好的模型数量
		mode="min",  # 监控指标越小越好
		verbose=True,  # 开启信息打印
	)
	# 早停机制
	early_stopping_callback = EarlyStopping(
		monitor="val_loss",  # 监控验证集的损失
		min_delta=1e-3,  # 最小变化
		patience=3,  # 容忍多少个epoch没有提升
		mode="min",  # 监控指标越小越好
		verbose=True,  # 开启信息打印
	)

	# --------- 训练模型 ---------
	# 训练模型
	trainer = pl.Trainer(
		# 基本参数配置
		max_epochs=20,  # 最大训练轮数
		accelerator="gpu" if torch.cuda.is_available() else "cpu",  # 使用GPU加速（如果可用）
		devices=1,  # 使用 1 个设备（整数形式在 CPU/GPU 上更通用）
		# 训练参数配置
		# val_check_interval=1,  # 每个step验证
		# check_val_every_n_epoch=5,  # 每个epoch验证

		# 日志和回调函数
		log_every_n_steps= 10,  # 每10个step记录一次日志
		logger=logger,  # 使用TensorBoard记录日志
		callbacks=[checkpoint_callback, early_stopping_callback],  # 回调
	)
	# 下面的代码被注释掉了，因为我们使用了 trainer.fit() 来处理
	# # 创建模型和优化器
	# autoencoder = LitAutoEncoder(Encoder(), Decoder())
	# optimizer = autoencoder.configure_optimizers()

	# for batch_idx, batch in enumerate(train_loader):
	#     loss = autoencoder.training_step(batch, batch_idx)

	#     loss.backward()
	#     optimizer.step()
	#     optimizer.zero_grad()

	print("\n----- train and val -----\n")
	# trainer.fit(model=autoencoder, train_dataloaders=train_loader)  # 如果没有验证集，可以只传入训练集
	trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)

	print("\n----- test -----\n")
	trainer.test(model=autoencoder, dataloaders=test_loader)
	# --------- 训练模型 ---------
	
	# # 加载权重文件使用模型预测
	# model = LitAutoEncoder.load_from_checkpoint(Path("weights/pl.ckpt"), encoder=Encoder(), decoder=Decoder())
	# model.eval()
	# y = model(x=torch.randn(1, 28 * 28))