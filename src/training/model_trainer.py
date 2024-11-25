import os
import torch
from ultralytics import YOLO
from PyQt6.QtCore import QThread, pyqtSignal
from ..utils.path_manager import PathManager

class ModelTrainer(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.path_manager = PathManager()
        self.running = True
        
        # 确保使用GPU（如果可用）
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def prepare_dataset(self):
        """准备数据集"""
        dataset_path = self.path_manager.get_path('data', 'dataset')
        if dataset_path is None:
            raise ValueError("数据集路径未设置")
            
        self.log_signal.emit(f"正在准备数据集: {dataset_path}")
        
        # 检查数据集目录结构
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        
        if not images_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"标注目录不存在: {labels_dir}")
            
        # 创建数据集配置文件
        data_yaml = dataset_path / 'data.yaml'
        yaml_content = f"""
# Dataset Path
path: {str(dataset_path)}

# Classes
names:
  0: violin
  1: bow

# Number of classes
nc: 2
"""
        with open(data_yaml, 'w') as f:
            f.write(yaml_content.strip())
            
        return data_yaml
        
    def train(self):
        """训练模型"""
        try:
            # 准备数据集
            data_yaml = self.prepare_dataset()
            
            # 获取预训练模型路径
            pretrained_model = self.path_manager.get_path('models', 'pretrained', 'detection', 'yolo', 'yolov8n')
            if pretrained_model is None or not pretrained_model.exists():
                raise FileNotFoundError("预训练模型不存在")
                
            # 设置训练输出目录
            output_dir = self.path_manager.get_path('models', 'trained', 'detection')
            if output_dir is None:
                raise ValueError("训练输出目录未设置")
                
            # 开始训练
            model = YOLO(str(pretrained_model))
            model.train(
                data=str(data_yaml),
                epochs=self.config.get('epochs', 100),
                batch=self.config.get('batch_size', 16),
                device=self.config.get('device', 'auto'),
                project=str(output_dir)
            )
            
        except Exception as e:
            self.log_signal.emit(f"训练失败: {e}")
            raise
            
    def run(self):
        try:
            # 准备数据集
            data_yaml = self.prepare_dataset()
            self.log_signal.emit("数据集准备完成")
            
            # 开始训练
            self.train()
            
            # 训练完成
            self.log_signal.emit("训练完成")
            self.progress_signal.emit(100)
            
        except Exception as e:
            self.log_signal.emit(f"训练出错: {str(e)}")
            raise e
            
    def stop(self):
        """停止训练"""
        self.running = False
        self.log_signal.emit("正在停止训练...")
