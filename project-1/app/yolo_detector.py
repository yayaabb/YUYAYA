from ultralytics import YOLO
from PIL import Image
import numpy as np

# 加载 YOLOv8 模型（只加载一次，避免重复下载和初始化）
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO('yolov8n.pt')  # 你可以换成 yolov8s.pt 等更大模型
    return _model

def detect_image_objects(image: Image.Image):
    """
    输入PIL图片，返回检测到的物体类别词汇列表（去重）。
    """
    model = get_model()
    # YOLO 支持 numpy array 或文件路径
    results = model(np.array(image))
    detected_classes = set()
    for r in results:
        for c in r.boxes.cls:
            detected_classes.add(model.model.names[int(c)])
    return list(detected_classes)