# 图像分类
from ultralytics import YOLO

# 加载预训练好的Yolo模型
model = YOLO("./weight/yolo11n-cls.pt")

# 分类图片(加载数据)
results = model("./images")

# 遍历每张图片
for result in results:
	# 显示带预测结果的图片(显示前5个置信度最大的类别)
	result.show()
	# 获取概率分布
	probs = result.probs
	print(probs)

# probs.data:所有类别的概率分布(一维张量),每个元素对应一个类别的置信度
# probs.orig_shape:原始图像的大小
# probs.shape:data数据形状,张量的形状(说明有1000个类别)
# probs.top1:最高置信度的类别索引
# probs.top1conf:最高置信度的概率值
# probs.top5:置信度前五的类别索引列表
# probs.top5conf:前五置信度的概率值