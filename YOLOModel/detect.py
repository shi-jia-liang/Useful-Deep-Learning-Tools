# 目标检测
from ultralytics import YOLO

# 加载预训练好的Yolo模型
model = YOLO("./weight/yolo11n.pt")

# 加载数据
results = model("./images")

# 遍历每张图片
for result in results:
	# 显示检测后的图片
	result.show()
	# 检测后的数据
	boxes = result.boxes
	print(boxes)

# boxes.cls:类别索引
# boxes.onf:每个检测结果的置信度
# boxes.data:每个检测结果的原始检测数据xyxy, conf, cls
# boxes.id:用于目标追踪时分配的唯一ID
# boxes.is_track:布尔值,表示是否启用了追踪
# boxes.orig_shape:原始图像的大小
# boxes.shape:data数据形状,第一位表示检测到的目标数量;第二位是data
# boxes.xywh:中心坐标+宽高
# boxes.xywhn:归一化后中心坐标+宽高
# boxes.xyxy:绝对像素坐标的边界框(左上、右下)
# boxes.xyxyn:归一化后绝对像素坐标的边界框(左上、右下)
