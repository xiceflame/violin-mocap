"""
自动学习模块 - 用于收集和处理训练数据，并进行模型微调
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from datetime import datetime

class AutoLearning:
    def __init__(self, base_path="data/training"):
        """初始化自动学习系统"""
        self.base_path = base_path
        self.images_path = os.path.join(base_path, "images")
        self.labels_path = os.path.join(base_path, "labels")
        self.model_path = os.path.join(base_path, "models")
        
        # 创建必要的目录
        for path in [self.images_path, self.labels_path, self.model_path]:
            os.makedirs(path, exist_ok=True)
            
        # 初始化YOLO模型
        self.model = YOLO('yolov8n.pt')
        
        # 设置类别
        self.classes = ['violin', 'bow']
        
        # 记录收集的样本数
        self.sample_count = 0
        
    def collect_sample(self, frame, detections, confidence_threshold=0.8):
        """收集高置信度的样本"""
        if detections is None or len(detections) == 0:
            return False
            
        # 检查是否有高置信度的检测结果
        high_confidence = False
        for det in detections:
            if det.conf > confidence_threshold:
                high_confidence = True
                break
                
        if not high_confidence:
            return False
            
        # 保存图像和标注
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(self.images_path, f"sample_{timestamp}.jpg")
        label_path = os.path.join(self.labels_path, f"sample_{timestamp}.txt")
        
        # 保存图像
        cv2.imwrite(image_path, frame)
        
        # 保存标注（YOLO格式）
        with open(label_path, 'w') as f:
            for det in detections:
                # 转换为YOLO格式：<class> <x_center> <y_center> <width> <height>
                class_id = int(det.cls)
                x_center = (det.xmin + det.xmax) / 2 / frame.shape[1]
                y_center = (det.ymin + det.ymax) / 2 / frame.shape[0]
                width = (det.xmax - det.xmin) / frame.shape[1]
                height = (det.ymax - det.ymin) / frame.shape[0]
                
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        self.sample_count += 1
        return True
        
    def should_train(self):
        """判断是否应该开始训练"""
        # 每收集100个样本进行一次训练
        return self.sample_count > 0 and self.sample_count % 100 == 0
        
    def train_model(self):
        """训练模型"""
        if not os.path.exists(self.images_path) or len(os.listdir(self.images_path)) == 0:
            return False
            
        # 准备训练配置
        data = {
            'path': self.base_path,
            'train': 'images',  # 训练集图片相对路径
            'val': 'images',    # 验证集图片相对路径
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        # 开始训练
        try:
            self.model.train(
                data=data,
                epochs=10,
                imgsz=640,
                batch=8,
                save=True,
                device='mps' if torch.backends.mps.is_available() else 'cpu'
            )
            return True
        except Exception as e:
            print(f"训练失败: {str(e)}")
            return False
            
    def get_latest_model(self):
        """获取最新训练的模型"""
        if not os.path.exists(self.model_path):
            return None
            
        models = [f for f in os.listdir(self.model_path) if f.endswith('.pt')]
        if not models:
            return None
            
        latest_model = max(models, key=lambda x: os.path.getctime(os.path.join(self.model_path, x)))
        return os.path.join(self.model_path, latest_model)
