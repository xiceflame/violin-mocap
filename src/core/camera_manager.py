"""
摄像头管理模块
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple
import queue
import threading
import time
import subprocess
import json
import platform

class CameraManager:
    """摄像头管理类"""
    def __init__(self, camera_index: int = 0, buffer_size: int = 3):
        self.logger = logging.getLogger(__name__)
        self.camera_index = camera_index
        self.buffer_size = buffer_size
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
    def start(self) -> bool:
        """启动摄像头"""
        try:
            # 确保之前的摄像头已经完全释放
            self.stop()
            time.sleep(0.1)  # 减少等待时间
            
            # 尝试打开摄像头
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.logger.error(f"无法打开摄像头 {self.camera_index}")
                return False
                
            # 优化相机参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 降低分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲
            
            # 清空帧缓冲区
            while not self.frame_buffer.empty():
                self.frame_buffer.get_nowait()
                
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            # 打印相机实际参数
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"摄像头 {self.camera_index} 已启动")
            self.logger.info(f"分辨率: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"启动摄像头失败: {e}")
            return False
            
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """获取一帧图像"""
        if not self.is_running:
            return False, None
            
        try:
            frame = self.frame_buffer.get(timeout=0.1)  # 减少等待时间
            return True, frame
        except queue.Empty:
            return False, None
            
    def get_available_cameras(self):
        """获取所有可用的摄像头列表"""
        cameras = []
        
        try:
            # 在 macOS 上使用 system_profiler 获取详细信息
            if platform.system() == 'Darwin':
                cmd = ['system_profiler', 'SPCameraDataType', '-json']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if 'SPCameraDataType' in data and data['SPCameraDataType']:
                        for camera in data['SPCameraDataType']:
                            if '_name' in camera:
                                cameras.append({
                                    'id': len(cameras),
                                    'name': camera['_name'],
                                    'model': camera.get('model_id', ''),
                                    'location': camera.get('location_id', '')
                                })
            
            # 如果没有找到摄像头，使用 OpenCV 尝试
            if not cameras:
                max_to_test = 5  # 最多测试5个摄像头索引
                for i in range(max_to_test):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            cameras.append({
                                'id': i,
                                'name': f'Camera {i}',
                                'model': 'Unknown',
                                'location': ''
                            })
                        cap.release()
                
            return cameras
            
        except Exception as e:
            self.logger.error(f"获取摄像头列表失败: {e}")
            # 返回默认摄像头
            return [{'id': 0, 'name': 'Default Camera', 'model': 'Unknown', 'location': ''}]
            
    def _capture_frames(self):
        """帧捕获线程"""
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
                
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # 更新FPS计数
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # 如果缓冲区满，移除最旧的帧
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_buffer.put(frame)
                
            except Exception as e:
                self.logger.error(f"捕获帧时出错: {e}")
                time.sleep(0.1)
                
    def get_fps(self) -> float:
        """获取当前FPS"""
        return self.fps
        
    def stop(self):
        """停止摄像头"""
        self.is_running = False
        
        # 等待捕获线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            
        # 释放摄像头
        if self.cap is not None:
            # 多次尝试释放摄像头
            for _ in range(3):
                try:
                    self.cap.release()
                    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
                    time.sleep(0.1)  # 短暂等待
                except:
                    pass
            self.cap = None
            
        # 清空帧缓冲区
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except:
                pass
                
        self.logger.info("摄像头已停止")
        
    def __del__(self):
        """析构函数"""
        self.stop()
