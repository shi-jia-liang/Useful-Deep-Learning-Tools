# 生成特征提取网络
import numpy
import torch
import torch.nn as nn
import torchvision.models as models

def conv1x1(in_channels, out_channels, stride=1):
	"""1x1 convolution with padding"""
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(True)
	)

def conv3x3(in_channels, out_channels, stride=1, groups=1, padding=1, dilation=1) -> nn.Sequential:
	"""3x3 convolution with padding"""
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=False,dilation=dilation),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(True)
	)

# 使用VGG16作为骨干网络
def VGGBlock1(in_channels, out_channels):
	return nn.Sequential(
		conv3x3(in_channels, out_channels),
		conv3x3(out_channels, out_channels),
		nn.MaxPool2d(kernel_size=2, stride=2)
	)
	
def VGGBlock2(in_channels, out_channels):
	return nn.Sequential(
		conv3x3(in_channels, out_channels),
		conv3x3(out_channels, out_channels),
		conv3x3(out_channels, out_channels),
		nn.MaxPool2d(kernel_size=2, stride=2)
	)

class VGG16Backbone(nn.Module):
	def __init__(self, in_channels=1, dims=[64, 128, 256, 512, 512]):
		super(VGG16Backbone, self).__init__()
		self.c1 = dims[0]
		self.c2 = dims[1]
		self.c3 = dims[2]
		self.c4 = dims[3]
		self.c5 = dims[4]
		self.stage1 = VGGBlock1(in_channels, self.c1)
		self.stage2 = VGGBlock1(self.c1, self.c2)
		self.stage3 = VGGBlock2(self.c2, self.c3)
		self.stage4 = VGGBlock2(self.c3, self.c4)
		self.stage5 = VGGBlock2(self.c4, self.c5)
		self.reduce5 = conv1x1(self.c5, self.c5 // 2)
		
	def forward(self, x):
		feat1 = self.stage1(x)
		feat2 = self.stage2(feat1)
		feat3 = self.stage3(feat2)
		feat4 = self.stage4(feat3)
		feat5 = self.stage5(feat4)
		feat5 = self.reduce5(feat5)
		return feat3, feat4, feat5


# 使用ResNet50作为骨干网络（仍存在问题）
class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class ResNet50Backbone(nn.Module):
	def __init__(self, in_channels=1, dims=[64, 128, 256, 512, 512]):
		super(ResNet50Backbone, self).__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
	
		self.stage1 = self._make_layer(Bottleneck, dims[0], 3)
		self.stage2 = self._make_layer(Bottleneck, dims[1], 4, stride=2)
		self.stage3 = self._make_layer(Bottleneck, dims[2], 6, stride=2)
		self.stage4 = self._make_layer(Bottleneck, dims[3], 3, stride=2)
		
		# ResNet50 stage4 output channels = dims[3] * expansion (512 * 4 = 2048)
		out_channels = dims[3] * Bottleneck.expansion
		self.reduce5 = conv1x1(out_channels, out_channels // 2)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.stage1(x)
		feat3 = self.stage2(x)
		feat4 = self.stage3(feat3)
		feat5 = self.stage4(feat4)
		
		feat5 = self.reduce5(feat5)
		
		return feat3, feat4, feat5

# 使用RepVGG作为骨干网络（未仔细调整）
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
	result = nn.Sequential()
	result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
												  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
	result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
	return result

class RepVGGBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size,
				 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
		super(RepVGGBlock, self).__init__()
		self.deploy = deploy
		self.groups = groups
		self.in_channels = in_channels

		assert kernel_size == 3
		assert padding == 1

		padding_11 = padding - kernel_size // 2

		self.non_linearity = nn.ReLU()

		if deploy:
			self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
									  padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

		else:
			self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
			self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
			self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

	def forward(self, inputs):
		if hasattr(self, 'rbr_reparam'):
			return self.non_linearity(self.rbr_reparam(inputs))

		if self.rbr_identity is None:
			id_out = 0
		else:
			id_out = self.rbr_identity(inputs)

		return self.non_linearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

class RepVGGBackbone(nn.Module):
	def __init__(self, num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False):
		super(RepVGGBackbone, self).__init__()
		
		self.deploy = deploy
		self.override_groups_map = override_groups_map or dict()
		
		assert len(width_multiplier) == 4
		
		self.in_planes = min(64, int(64 * width_multiplier[0]))
		
		# Stage 0
		self.stage0 = RepVGGBlock(in_channels=1, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
		self.cur_layer_idx = 1
		
		self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
		self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
		self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
		self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
		self.reduce5 = conv1x1(int(512 * width_multiplier[3]), int(512 * width_multiplier[3]) // 2)

	def _make_stage(self, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		blocks = []
		for stride in strides:
			cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
			blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
									  stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
			self.in_planes = planes
			self.cur_layer_idx += 1
		return nn.Sequential(*blocks)

	def forward(self, x):
		x = self.stage0(x)
		x = self.stage1(x) # Stride 4
		feat3 = self.stage2(x) # Stride 8 -> feat3
		feat4 = self.stage3(feat3) # Stride 16 -> feat4
		feat5 = self.stage4(feat4) # Stride 32 -> feat5
		feat5 = self.reduce5(feat5)
		return feat3, feat4, feat5

def build_backbone(model_config):
	model_backbone = model_config['MODEL']['BACKBONE']

	if model_backbone == "VGG16":
		return VGG16Backbone()
	elif model_backbone == "ResNet50":
		return ResNet50Backbone()
	elif model_backbone == "RepVGG":
		return RepVGGBackbone(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5])
	else:
		raise ValueError(f"Unknown model type: {model_backbone}")
	
if __name__ == "__main__":
	default_config = {
		"MODEL": {
			"BACKBONE": "ResNet50" 
		}
	}
	
	model_config = default_config
	backbone = build_backbone(model_config)
	# printing the feature shape
	x = torch.randn(1, 1, 224, 224)  # Example input tensor
	c3, c4, c5 = backbone(x)
	print(f"Feature shapes: \nc3={c3.shape}, \nc4={c4.shape}, \nc5={c5.shape}")