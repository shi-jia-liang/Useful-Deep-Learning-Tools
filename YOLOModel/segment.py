# 实例分割
import cv2
import numpy
from ultralytics import YOLO

# 加载预训练好的Yolo模型
model = YOLO("./weight/yolo11n-seg.pt")

# 加载数据
results = model("./images")

# 遍历每张图片
for result in results:
	# 显示检测后的图片
	result.show()
	# 预测后的数据
	masks = result.masks
	print(masks)

# masks.data:所有检测目标的二值化掩膜数据,形状为[检测到的目标数量, 缩放后尺寸]
# masks.orig_shape:原始图像的大小
# masks.shape:data数据形状,第一位表示检测到的实例个数,第二、三位表示掩码图片的尺寸
# masks.xy:每个目标的掩膜轮廓的多边形坐标(绝对像素值)
# masks.xyn:每个目标的掩膜轮廓的归一化多边形坐标(绝对像素值)

# # 显示二值化掩膜图片
# for idx, result in enumerate(results):
	# masks = result.masks
	# if masks is None:
	# 	continue
	#
	# for i in range(len(masks)):
	# 	mask = masks.data[i].cpu().numpy()
	# 	binary_mask = mask * 255
	# 	cv2.imshow("binary_mask", binary_mask)
	# 	cv2.waitKey(0)