"""
摄像头工具模块 - 提供摄像头检测和管理功能
"""

import cv2
import time
import logging
from typing import List, Dict, Optional, Tuple

class CameraManager:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
    def list_cameras(self) -> List[Dict]:
        """
        列出所有可用的摄像头设备
        
        Returns:
            List[Dict]: 包含摄像头信息的字典列表
        """
        available_cameras = []
        max_cameras = 10  # 最多检测10个摄像头
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # 获取摄像头信息
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # 尝试读取一帧以确认摄像头确实可用
                ret, _ = cap.read()
                if ret:
                    camera_info = {
                        'id': i,
                        'name': f'Camera {i}',
                        'resolution': f'{width}x{height}',
                        'fps': fps,
                        'width': width,
                        'height': height
                    }
                    available_cameras.append(camera_info)
                
                cap.release()
            
        return available_cameras
    
    def init_camera(self, camera_id: int = 0, 
                   width: int = 1280, 
                   height: int = 720, 
                   fps: int = 30) -> Optional[cv2.VideoCapture]:
        """
        初始化摄像头，包含重试机制
        
        Args:
            camera_id: 摄像头ID
            width: 期望的宽度
            height: 期望的高度
            fps: 期望的帧率
            
        Returns:
            Optional[cv2.VideoCapture]: 成功则返回VideoCapture对象，失败返回None
        """
        for attempt in range(self.max_retries):
            try:
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open camera {camera_id}")
                
                # 设置分辨率和帧率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                
                # 验证设置是否生效
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                self.logger.info(f"Camera initialized - Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
                
                # 测试读取一帧
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to read frame from camera")
                
                return cap
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    self.logger.error(f"Failed to initialize camera {camera_id} after {self.max_retries} attempts")
                    return None
    
    def get_camera_info(self, cap: cv2.VideoCapture) -> Dict:
        """
        获取摄像头的详细信息
        
        Args:
            cap: VideoCapture对象
            
        Returns:
            Dict: 包含摄像头信息的字典
        """
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'backend': cap.getBackendName(),
            'format': int(cap.get(cv2.CAP_PROP_FORMAT)),
            'mode': int(cap.get(cv2.CAP_PROP_MODE)),
            'auto_exposure': int(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            'brightness': int(cap.get(cv2.CAP_PROP_BRIGHTNESS)),
            'contrast': int(cap.get(cv2.CAP_PROP_CONTRAST)),
            'saturation': int(cap.get(cv2.CAP_PROP_SATURATION))
        }
        return info
    
    def safe_release(self, cap: Optional[cv2.VideoCapture]) -> None:
        """
        安全释放摄像头资源
        
        Args:
            cap: VideoCapture对象
        """
        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                self.logger.error(f"Error releasing camera: {str(e)}")
