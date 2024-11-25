"""姿态检测模块"""
import os
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
from ..utils.path_manager import PathManager

@dataclass
class ViolinKeypoints:
    """小提琴关键点"""
    scroll: Optional[Tuple[float, float]] = None          # 琴头
    upper_neck: Optional[Tuple[float, float]] = None      # 琴颈上端
    lower_neck: Optional[Tuple[float, float]] = None      # 琴颈下端
    upper_left: Optional[Tuple[float, float]] = None      # 琴身左上角
    upper_right: Optional[Tuple[float, float]] = None     # 琴身右上角
    lower_left: Optional[Tuple[float, float]] = None      # 琴身左下角
    lower_right: Optional[Tuple[float, float]] = None     # 琴身右下角
    tailpiece: Optional[Tuple[float, float]] = None       # 尾部
    confidence: float = 0.0

@dataclass
class BowKeypoints:
    """琴弓关键点"""
    tip: Optional[Tuple[float, float]] = None            # 弓尖
    upper_stick: Optional[Tuple[float, float]] = None    # 弓杆上部
    lower_stick: Optional[Tuple[float, float]] = None    # 弓杆下部
    frog: Optional[Tuple[float, float]] = None          # 弓根
    confidence: float = 0.0

class PoseDetector:
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.path_manager = PathManager()
        
        # YOLO参数
        self.yolo_conf = self.config.get('yolo_conf', 0.5)  # 默认置信度阈值
        self.yolo_iou = self.config.get('yolo_iou', 0.45)   # 默认IOU阈值
        
        # 初始化MediaPipe
        try:
            self.mp_holistic = mp.solutions.holistic
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # 加载YOLO模型用于辅助检测
            model_path = self.path_manager.get_path('models', 'pretrained', 'detection', 'yolo', 'yolov8n')
            if model_path is None or not model_path.exists():
                raise FileNotFoundError("YOLO模型文件不存在")
            self.yolo_model = YOLO(str(model_path))
            self.yolo_model.conf = self.yolo_conf
            self.yolo_model.iou = self.yolo_iou
            
            # 初始化姿态检测器
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,          # 视频模式
                model_complexity=2,               # 使用最高精度模型
                smooth_landmarks=True,            # 平滑关键点
                enable_segmentation=False,        # 不需要分割
                smooth_segmentation=False,        # 不需要分割
                refine_face_landmarks=True,       # 细化面部关键点
                min_detection_confidence=0.5,     # 检测置信度阈值
                min_tracking_confidence=0.5       # 追踪置信度阈值
            )
        except Exception as e:
            self.logger.error(f"初始化姿态检测器失败: {e}")
            self.holistic = None
            self.yolo_model = None

    def detect_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """检测姿态"""
        if frame is None:
            return {}
            
        # MediaPipe需要RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 设置帧为不可写，提高性能
        frame_rgb.flags.writeable = False
        
        # MediaPipe姿态检测
        results = self.holistic.process(frame_rgb)
        
        # 恢复帧为可写
        frame_rgb.flags.writeable = True
        
        # 返回结果
        return {
            'pose_landmarks': results.pose_landmarks,
            'face_landmarks': results.face_landmarks,
            'left_hand_landmarks': results.left_hand_landmarks,
            'right_hand_landmarks': results.right_hand_landmarks,
            'yolo_results': self._detect_violin(frame) if self.yolo_model else None
        }
        
    def _detect_violin(self, frame: np.ndarray) -> Any:
        """检测小提琴"""
        if self.yolo_model:
            return self.yolo_model(frame)
        else:
            return None

    def draw_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """在帧上绘制检测结果"""
        if frame is None or not results:
            return frame
            
        # 绘制MediaPipe结果
        if results.get('pose_landmarks'):
            self.mp_drawing.draw_landmarks(
                frame,
                results['pose_landmarks'],
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if results.get('face_landmarks'):
            self.mp_drawing.draw_landmarks(
                frame,
                results['face_landmarks'],
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
        if results.get('left_hand_landmarks'):
            self.mp_drawing.draw_landmarks(
                frame,
                results['left_hand_landmarks'],
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
        if results.get('right_hand_landmarks'):
            self.mp_drawing.draw_landmarks(
                frame,
                results['right_hand_landmarks'],
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
        # 绘制YOLO结果
        if results.get('yolo_results'):
            self._draw_yolo_results(frame, results['yolo_results'])
            
        return frame

    def _draw_yolo_results(self, frame: np.ndarray, yolo_results: Any) -> None:
        """绘制YOLO结果"""
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 92:  # COCO数据集中小提琴的类别ID
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    def set_yolo_params(self, conf: float = None, iou: float = None):
        """设置YOLO检测参数
        
        Args:
            conf (float, optional): 置信度阈值. Defaults to None.
            iou (float, optional): IOU阈值. Defaults to None.
        """
        if hasattr(self, 'yolo_model'):
            if conf is not None:
                self.yolo_model.conf = conf
                self.logger.info(f"YOLO置信度阈值已更新: {conf}")
            if iou is not None:
                self.yolo_model.iou = iou
                self.logger.info(f"YOLO IOU阈值已更新: {iou}")

    def release(self):
        """释放资源"""
        self.holistic.close()
