"""性能监控模块"""
import cv2
import numpy as np
import time
import logging
import psutil
import GPUtil
from collections import deque
from typing import Dict, Any, List, Optional

class PerformanceMonitor:
    def __init__(self, window_size: int = 30):
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        
        # 性能指标队列
        self.frame_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.fps_history = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.gpu_usage = deque(maxlen=window_size)
        
        # 性能警告阈值
        self.fps_threshold = 15.0
        self.latency_threshold = 100.0  # ms
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 80.0  # %
        self.gpu_threshold = 80.0  # %
        
        # 统计信息
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_update_time = time.time()
        self.start_time = time.time()
        
        # 初始化系统监控
        self.process = psutil.Process()
        
    def update_stats(self, frame_time: float = 0.0, detection_time: float = 0.0, 
                    dropped: bool = False):
        """更新性能统计"""
        current_time = time.time()
        
        # 更新基本指标
        if not dropped:
            self.frame_times.append(frame_time * 1000)  # 转换为毫秒
            self.detection_times.append(detection_time * 1000)
            self.total_frames += 1
        else:
            self.dropped_frames += 1
            
        # 计算FPS
        time_diff = current_time - self.last_update_time
        if time_diff >= 1.0:  # 每秒更新一次
            fps = len(self.frame_times) / time_diff
            self.fps_history.append(fps)
            self.last_update_time = current_time
            
            # 更新系统资源使用情况
            self.cpu_usage.append(self.process.cpu_percent())
            self.memory_usage.append(self.process.memory_percent())
            
            # 获取GPU使用情况（如果可用）
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_usage.append(gpus[0].load * 100)
            except:
                pass
                
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        stats = {
            'fps': np.mean(self.fps_history) if self.fps_history else 0.0,
            'frame_latency': np.mean(self.frame_times) if self.frame_times else 0.0,
            'detection_latency': np.mean(self.detection_times) if self.detection_times else 0.0,
            'cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0.0,
            'memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0.0,
            'gpu_usage': np.mean(self.gpu_usage) if self.gpu_usage else 0.0,
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'drop_rate': (self.dropped_frames / max(1, self.total_frames)) * 100
        }
        return stats
        
    def get_performance_warnings(self) -> List[str]:
        """获取性能警告"""
        warnings = []
        stats = self.get_performance_stats()
        
        if stats['fps'] < self.fps_threshold:
            warnings.append(f"低帧率警告: {stats['fps']:.1f} FPS")
            
        if stats['frame_latency'] > self.latency_threshold:
            warnings.append(f"高延迟警告: {stats['frame_latency']:.1f} ms")
            
        if stats['cpu_usage'] > self.cpu_threshold:
            warnings.append(f"CPU使用率过高: {stats['cpu_usage']:.1f}%")
            
        if stats['memory_usage'] > self.memory_threshold:
            warnings.append(f"内存使用率过高: {stats['memory_usage']:.1f}%")
            
        if stats['gpu_usage'] > self.gpu_threshold:
            warnings.append(f"GPU使用率过高: {stats['gpu_usage']:.1f}%")
            
        if stats['drop_rate'] > 5.0:  # 超过5%的丢帧率
            warnings.append(f"丢帧率过高: {stats['drop_rate']:.1f}%")
            
        return warnings
        
    def draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制性能统计信息"""
        stats = self.get_performance_stats()
        warnings = self.get_performance_warnings()
        
        # 绘制性能信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)  # 白色
        warning_color = (0, 0, 255)  # 红色
        
        # 基本性能信息
        y_offset = 30
        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += 25
        cv2.putText(frame, f"延迟: {stats['frame_latency']:.1f}ms", (10, y_offset),
                   font, font_scale, color, thickness)
        y_offset += 25
        cv2.putText(frame, f"CPU: {stats['cpu_usage']:.1f}%", (10, y_offset),
                   font, font_scale, color, thickness)
        y_offset += 25
        cv2.putText(frame, f"内存: {stats['memory_usage']:.1f}%", (10, y_offset),
                   font, font_scale, color, thickness)
        
        if stats['gpu_usage'] > 0:
            y_offset += 25
            cv2.putText(frame, f"GPU: {stats['gpu_usage']:.1f}%", (10, y_offset),
                      font, font_scale, color, thickness)
                      
        # 绘制警告信息
        for warning in warnings:
            y_offset += 25
            cv2.putText(frame, warning, (10, y_offset),
                      font, font_scale, warning_color, thickness)
                      
        return frame
        
    def reset_stats(self):
        """重置统计信息"""
        self.frame_times.clear()
        self.detection_times.clear()
        self.fps_history.clear()
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.gpu_usage.clear()
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_update_time = time.time()
        self.start_time = time.time()
