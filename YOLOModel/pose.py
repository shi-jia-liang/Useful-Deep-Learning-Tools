# 姿态估计
from ultralytics import YOLO

# 加载预训练好的Yolo模型
model = YOLO("./weight/yolo11n-pose.pt")

# 加载数据
results = model("./images")

# 遍历每张图片
for result in results:
	# 显示检测后的图片
	result.show()
	# 预测后的数据
	keypoints = result.keypoints
	print(keypoints)

# keypoints.conf:表示人17个节点(人身上有17个关键点)的置信度
# keypoints.data:每个检测结果的xy坐标 和 置信度
# keypoints.has_visible:是否包含关键点的可见性信息(显示在图片中)
# keypoints.shape:data数据形状,第一位表示检测到的人的个数,第二位表示人的17个关键点,第三位是data
# keypoints.xy:每个检测目标的 17 个关键点的绝对坐标
# keypoints.xyn:每个检测目标的 17 个关键点的归一化绝对坐标