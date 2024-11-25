import cv2
import numpy as np
import time
from datetime import datetime
import logging
from typing import Optional, Dict, Any

from ..utils.camera import CameraManager
from ..utils.performance import PerformanceOptimizer
from ..utils.visualization import UIManager
from ..utils.data_manager import DataManager
from .object_detection import ObjectDetector
from .pose_estimation import PoseEstimator
from ..networking.ue_connector import UEConnector

# 设置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotionCaptureError(Exception):
    """动作捕捉相关的自定义异常"""
    pass

class MotionCapture:
    """小提琴动作捕捉系统的主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动作捕捉系统
        
        Args:
            config: 配置字典，包含所有必要的设置
        
        Raises:
            MotionCaptureError: 初始化失败时抛出
        """
        try:
            self.config = config
            self.camera_manager = CameraManager(config['camera'])
            self.detector = ObjectDetector(config['detection'])
            self.pose_estimator = PoseEstimator(config['detection']['mediapipe'])
            self.performance = PerformanceOptimizer(config['performance'])
            self.ui_manager = UIManager(config.get('visualization', {}))
            self.data_manager = DataManager(config['paths'])
            self.ue_connector = UEConnector(config['networking'])
            
            # 状态标志
            self.is_running = False
            self.is_recording = False
            self.is_streaming = False
            self.data_collection_mode = False
            
            logger.info("Motion capture system initialized successfully")
            
        except Exception as e:
            raise MotionCaptureError(f"Failed to initialize motion capture system: {str(e)}")
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            处理后的图像帧
        """
        if frame is None:
            return None
            
        try:
            start_time = time.time()
            
            # 目标检测
            detection_results = self.detector.detect(frame)
            
            # 姿态估计
            pose_results = self.pose_estimator.estimate(frame)
            
            # 合并结果
            combined_results = {
                'detection': detection_results,
                'pose': pose_results,
                'timestamp': time.time()
            }
            
            # 如果正在录制，保存数据
            if self.is_recording:
                self.data_manager.save_frame_data(combined_results)
            
            # 如果正在流式传输，发送数据
            if self.is_streaming:
                self.ue_connector.send_data(combined_results)
            
            # 更新性能指标
            self.performance.update(time.time() - start_time)
            
            # 绘制UI
            frame = self.ui_manager.draw(frame, combined_results, self.performance.get_stats())
            
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame
    
    def run(self, camera_id: int = 0) -> None:
        """
        运行动作捕捉系统
        
        Args:
            camera_id: 要使用的相机ID
        """
        try:
            cap = self.camera_manager.initialize_camera(camera_id)
            self.is_running = True
            
            while self.is_running:
                success, frame = cap.read()
                if not success:
                    logger.warning("Failed to read camera frame")
                    continue
                
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    cv2.imshow('Violin Motion Capture', processed_frame)
                
                # 处理键盘输入
                self._handle_keyboard_input()
                
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()
    
    def _handle_keyboard_input(self) -> None:
        """处理键盘输入"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            self.is_running = False
        elif key == ord('r'):
            self.toggle_recording()
        elif key == ord('s'):
            self.toggle_streaming()
        elif key == ord('d'):
            self.toggle_data_collection()
        elif key == ord(' '):
            if self.data_collection_mode:
                self.capture_training_data()
    
    def toggle_recording(self) -> None:
        """切换录制状态"""
        self.is_recording = not self.is_recording
        status = "started" if self.is_recording else "stopped"
        logger.info(f"Recording {status}")
    
    def toggle_streaming(self) -> None:
        """切换流式传输状态"""
        self.is_streaming = not self.is_streaming
        status = "started" if self.is_streaming else "stopped"
        logger.info(f"Streaming {status}")
    
    def toggle_data_collection(self) -> None:
        """切换数据收集模式"""
        self.data_collection_mode = not self.data_collection_mode
        status = "enabled" if self.data_collection_mode else "disabled"
        logger.info(f"Data collection mode {status}")
    
    def capture_training_data(self) -> None:
        """捕获训练数据"""
        if not self.data_collection_mode:
            return
        self.data_manager.capture_training_data()
    
    def cleanup(self) -> None:
        """清理资源"""
        logger.info("Cleaning up resources...")
        cv2.destroyAllWindows()
        self.camera_manager.release_all()
        self.ue_connector.close()
        if self.is_recording:
            self.data_manager.save_session()
