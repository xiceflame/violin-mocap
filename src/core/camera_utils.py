"""
相机工具类
"""

import cv2
import logging
import numpy as np
from typing import Optional, Tuple, List
import sys
import subprocess
import json
import platform

class CameraManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_camera = None
        self.camera_info_cache = {}
        
    def get_available_cameras(self):
        """获取所有可用的摄像头列表"""
        cameras = []
        
        try:
            # 使用system_profiler获取摄像头信息（仅在macOS上）
            if platform.system() == 'Darwin':
                cmd = ['system_profiler', 'SPCameraDataType', '-json']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if 'SPCameraDataType' in data:
                        for camera in data['SPCameraDataType']:
                            if '_name' in camera:
                                cameras.append({
                                    'id': len(cameras),
                                    'name': camera['_name'],
                                    'model': camera.get('model_id', ''),
                                    'location': camera.get('location_id', '')
                                })
            
            # 如果没有找到摄像头，使用OpenCV尝试
            if not cameras:
                index = 0
                while True:
                    cap = cv2.VideoCapture(index)
                    if not cap.isOpened():
                        break
                    
                    # 获取摄像头信息
                    name = f"Camera {index}"
                    model = ""
                    try:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        model = f"{width}x{height}"
                    except:
                        pass
                    
                    cameras.append({
                        'id': index,
                        'name': name,
                        'model': model,
                        'location': ''
                    })
                    
                    cap.release()
                    index += 1
                    
                    # 最多检测10个摄像头
                    if index >= 10:
                        break
            
            self.logger.info(f"找到 {len(cameras)} 个摄像头")
            return cameras
            
        except Exception as e:
            self.logger.error(f"获取摄像头列表失败: {e}")
            return []
            
    def initialize_camera(self, camera_id):
        """初始化指定的摄像头"""
        try:
            # 先释放当前相机
            if self.current_camera is not None:
                self.safe_release(self.current_camera)
                self.current_camera = None
                
            # 尝试打开相机
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # 设置分辨率和帧率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # 验证相机是否真正打开
                ret, frame = cap.read()
                if not ret:
                    self.logger.error(f"相机 {camera_id} 无法读取帧")
                    cap.release()
                    return None
                    
                # 缓存摄像头信息
                self.camera_info_cache[camera_id] = {
                    'id': camera_id,
                    'name': f"Camera {camera_id}",
                    'model': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                    'fps': int(cap.get(cv2.CAP_PROP_FPS))
                }
                
                # 保存当前相机
                self.current_camera = cap
                self.logger.info(f"成功初始化相机 {camera_id}")
                return cap
                
            self.logger.error(f"无法打开相机 {camera_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"初始化相机失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def get_camera_info(self, cap):
        """获取摄像头信息"""
        info = {}
        try:
            # 获取基本信息
            info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
            info['backend'] = int(cap.get(cv2.CAP_PROP_BACKEND))
            info['format'] = int(cap.get(cv2.CAP_PROP_FORMAT))
            
            # 获取更多详细信息（仅在macOS上）
            if platform.system() == 'Darwin':
                cmd = ['system_profiler', 'SPCameraDataType', '-json']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if 'SPCameraDataType' in data:
                        for camera in data['SPCameraDataType']:
                            if '_name' in camera:
                                info['name'] = camera['_name']
                                info['model'] = camera.get('model_id', '')
                                info['location'] = camera.get('location_id', '')
                                break
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取摄像头信息失败: {e}")
            return info
            
    def safe_release(self, cap):
        """安全释放摄像头"""
        try:
            if cap is not None:
                cap.release()
                self.logger.info("相机已释放")
        except Exception as e:
            self.logger.error(f"释放相机失败: {e}")
            
    def read_frame(self, cap):
        """读取一帧图像"""
        try:
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    return frame
        except Exception as e:
            self.logger.error(f"读取帧失败: {e}")
        return None
