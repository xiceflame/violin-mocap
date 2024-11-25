import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import Dict, List, Optional, Tuple, Any
from ..utils.path_manager import PathManager

logger = logging.getLogger(__name__)

class DetectionResult:
    """检测结果类"""
    def __init__(self, class_id: int, confidence: float, bbox: Tuple[float, float, float, float]):
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        
    @property
    def center(self) -> Tuple[float, float]:
        """计算边界框中心点"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        """计算边界框宽度"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """计算边界框高度"""
        return self.bbox[3] - self.bbox[1]
    
    def get_landmarks(self, num_points: int = 8) -> List[Tuple[float, float]]:
        """
        生成小提琴形状的关键点
        
        Args:
            num_points: 关键点数量
            
        Returns:
            关键点列表 [(x, y), ...]
        """
        x1, y1, x2, y2 = self.bbox
        w = self.width
        h = self.height
        cx, cy = self.center
        
        # 定义小提琴形状的关键点
        landmarks = [
            (cx, y1),                    # 顶部中心
            (x2, y1 + h * 0.25),         # 右上角
            (x2, y1 + h * 0.75),         # 右下角
            (cx, y2),                    # 底部中心
            (x1, y1 + h * 0.75),         # 左下角
            (x1, y1 + h * 0.25),         # 左上角
            (cx - w * 0.2, cy),          # 左中心
            (cx + w * 0.2, cy)           # 右中心
        ]
        
        return landmarks[:num_points]

class ObjectDetector:
    """YOLOv8目标检测器类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化目标检测器
        
        Args:
            config: 检测器配置
        """
        self.config = config or {}
        self.path_manager = PathManager()
        
        # 加载YOLO模型
        try:
            model_path = self.path_manager.get_path('models', 'pretrained', 'detection', 'yolo', 'yolov8n')
            if model_path is None or not model_path.exists():
                raise FileNotFoundError("YOLO模型文件不存在")
            self.model = YOLO(str(model_path))
            self.confidence_threshold = self.config.get('confidence', 0.25)
            self.class_mapping = self.config.get('classes', {'violin': 0, 'bow': 1})
            self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
            logger.info("Object detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize object detector: {str(e)}")
            self.model = None
            
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        在图像中检测目标
        
        Args:
            frame: 输入图像
            
        Returns:
            检测结果列表
        """
        try:
            results = []
            yolo_results = self.model(frame, conf=self.confidence_threshold)[0]
            
            for box in yolo_results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                # 只处理小提琴和琴弓
                if class_id in self.reverse_class_mapping:
                    results.append(DetectionResult(
                        class_id=class_id,
                        confidence=confidence,
                        bbox=tuple(xyxy)
                    ))
                    
            return results
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []
            
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult],
                       draw_landmarks: bool = True):
        """
        在图像上绘制检测结果
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            draw_landmarks: 是否绘制关键点
        """
        for det in detections:
            # 绘制边界框
            color = (0, 255, 0) if det.class_id == self.class_mapping['violin'] else (0, 0, 255)
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签
            label = f"{self.reverse_class_mapping[det.class_id]}: {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制关键点
            if draw_landmarks and det.class_id == self.class_mapping['violin']:
                landmarks = det.get_landmarks()
                for i, (x, y) in enumerate(landmarks):
                    cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
                    if i > 0:  # 连接相邻点
                        prev_x, prev_y = landmarks[i-1]
                        cv2.line(frame, (int(prev_x), int(prev_y)),
                                (int(x), int(y)), (255, 0, 0), 1)
                # 连接最后一个点和第一个点
                cv2.line(frame, (int(landmarks[-1][0]), int(landmarks[-1][1])),
                        (int(landmarks[0][0]), int(landmarks[0][1])), (255, 0, 0), 1)
