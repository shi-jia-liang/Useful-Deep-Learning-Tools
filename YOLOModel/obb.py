# 旋转目标检测
from ultralytics import YOLO

# 加载预训练好的Yolo模型
model = YOLO("./weight/yolo11n-obb.pt")

# 分类图片(加载数据)
results = model("./images/parking.png")

# 遍历每张图片
for result in results:
	# 显示带预测结果的图片(显示前5个置信度最大的类别)
	result.show()
	# 获取概率分布
	obb = result.obb
	print(obb)

# obb.cls:类别索引
# obb.conf:每个检测结果的置信度
# obb.data:每个检测结果的原始检测数据xywhr, conf, cls
# obb.id:用于目标追踪时分配的唯一ID
# obb.is_track:布尔值,表示是否启用了追踪
# obb.orig_shape:原始图像的大小
# obb.shape:data数据形状,第一位表示检测到的目标数量;第二位是data
# obb.xywhr:旋转矩形框的中心坐标+宽高
# obb.xyxy:旋转矩形框的最小外接轴对齐矩形框的坐标
# obb.xyxyxyxy:旋转矩形框的四个角点的绝对坐标
# obb.xyxyxyxyn:旋转矩形框的四个角点的归一化绝对坐标