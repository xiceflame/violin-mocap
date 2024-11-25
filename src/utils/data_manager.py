import os
import json
import csv
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from .path_manager import PathManager

logger = logging.getLogger(__name__)

class DataManager:
    """数据管理器类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据管理器
        
        Args:
            config: 数据管理配置
        """
        self.config = config or {}
        self.path_manager = PathManager()
        
        # 获取数据路径
        self.data_path = self.path_manager.get_path('data')
        self.dataset_path = self.path_manager.get_path('data', 'dataset')
        self.raw_path = self.path_manager.get_path('data', 'raw')
        self.processed_path = self.path_manager.get_path('data', 'processed')
        self.annotations_path = self.path_manager.get_path('data', 'annotations')
        
        # 创建必要的目录
        for path in [self.data_path, self.dataset_path, self.raw_path, 
                    self.processed_path, self.annotations_path]:
            if path:
                path.mkdir(parents=True, exist_ok=True)
                
        # 当前会话数据
        self.session_data = []
        self.session_start_time = datetime.now()
    
    def _create_directories(self) -> None:
        """创建必要的目录结构"""
        try:
            for directory in [self.data_path, self.dataset_path, 
                            self.raw_path, self.processed_path, self.annotations_path]:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
                
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise
    
    def save_frame_data(self, data: Dict[str, Any]) -> None:
        """
        保存单帧数据
        
        Args:
            data: 帧数据
        """
        try:
            # 添加时间戳
            data['timestamp'] = datetime.now().isoformat()
            self.session_data.append(data)
            
        except Exception as e:
            logger.error(f"Error saving frame data: {str(e)}")
    
    def save_session(self) -> None:
        """保存当前会话数据"""
        try:
            if not self.session_data:
                return
                
            # 生成文件名
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            filename = self.data_path / f"session_{timestamp}.json"
            
            # 保存数据
            with open(filename, 'w') as f:
                json.dump({
                    'session_start': self.session_start_time.isoformat(),
                    'session_end': datetime.now().isoformat(),
                    'frame_count': len(self.session_data),
                    'frames': self.session_data
                }, f, indent=2)
            
            logger.info(f"Session data saved to {filename}")
            
            # 清空会话数据
            self.session_data = []
            self.session_start_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error saving session data: {str(e)}")
    
    def capture_training_data(self, frame: np.ndarray, 
                            annotations: Dict[str, Any]) -> None:
        """
        捕获训练数据
        
        Args:
            frame: 图像帧
            annotations: 标注数据
        """
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_file = self.raw_path / f"image_{timestamp}.jpg"
            label_file = self.annotations_path / f"label_{timestamp}.txt"
            
            # 保存图像
            cv2.imwrite(str(image_file), frame)
            
            # 保存标注
            self._save_yolo_annotations(label_file, annotations)
            
            logger.info(f"Training data captured: {image_file}")
            
        except Exception as e:
            logger.error(f"Error capturing training data: {str(e)}")
    
    def _save_yolo_annotations(self, filepath: Path, 
                             annotations: Dict[str, Any]) -> None:
        """
        保存YOLO格式的标注
        
        Args:
            filepath: 标注文件路径
            annotations: 标注数据
        """
        try:
            with open(filepath, 'w') as f:
                for obj_name, detection in annotations.items():
                    if detection is None:
                        continue
                        
                    # 转换为YOLO格式
                    x1, y1, x2, y2 = detection.bbox
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 写入标注
                    f.write(f"{detection.class_id} {x_center} {y_center} {width} {height}\n")
                    
        except Exception as e:
            logger.error(f"Error saving YOLO annotations: {str(e)}")
    
    def save_dataset_item(self, image: np.ndarray, annotations: Dict[str, Any],
                         image_id: str) -> Tuple[Path, Path]:
        """保存数据集项目"""
        if not self.dataset_path:
            raise ValueError("数据集路径未设置")
            
        # 创建图像和标注目录
        images_dir = self.dataset_path / 'images'
        labels_dir = self.dataset_path / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图像
        image_path = images_dir / f"{image_id}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # 保存标注
        annotation_path = labels_dir / f"{image_id}.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=4, ensure_ascii=False)
            
        return image_path, annotation_path
    
    def export_to_csv(self, data: List[Dict[str, Any]], 
                     filepath: str) -> None:
        """
        导出数据到CSV文件
        
        Args:
            data: 要导出的数据
            filepath: 输出文件路径
        """
        try:
            if not data:
                return
                
            # 获取所有可能的字段
            fields = set()
            for item in data:
                fields.update(item.keys())
            
            # 写入CSV
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(fields))
                writer.writeheader()
                writer.writerows(data)
                
            logger.info(f"Data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
    
    def load_session_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        加载会话数据
        
        Args:
            filepath: 会话文件路径
            
        Returns:
            会话数据列表
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('frames', [])
                
        except Exception as e:
            logger.error(f"Error loading session data: {str(e)}")
            return []

    def export_session(self, session_path: str, format: str = 'json') -> None:
        """
        导出会话数据
        
        Args:
            session_path: 会话文件路径
            format: 导出格式（'json', 'csv'等）
        """
        if not os.path.exists(session_path):
            raise FileNotFoundError(f"Session file not found: {session_path}")
            
        # 读取会话数据
        with open(session_path, 'r') as f:
            session_data = json.load(f)
            
        # 生成导出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(
            self.data_path, f"export_{timestamp}.{format}")
            
        if format == 'json':
            with open(export_path, 'w') as f:
                json.dump(session_data, f, indent=2)
        elif format == 'csv':
            # TODO: 实现CSV导出
            pass
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def _process_data_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数据以便JSON序列化
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        processed = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                processed[key] = value.tolist()
            elif isinstance(value, dict):
                processed[key] = self._process_data_for_json(value)
            elif isinstance(value, list):
                processed[key] = [
                    self._process_data_for_json(item) if isinstance(item, dict)
                    else item.tolist() if isinstance(item, np.ndarray)
                    else item
                    for item in value
                ]
            else:
                processed[key] = value
        return processed
