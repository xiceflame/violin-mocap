import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque

class LandmarkSmoother:
    """
    关键点平滑处理器，使用移动平均和速度权重来平滑关键点轨迹
    """
    
    def __init__(self, window_size: int = 5, velocity_weight: float = 0.5):
        """
        初始化平滑器
        
        Args:
            window_size: 平滑窗口大小
            velocity_weight: 速度权重，用于预测运动趋势
        """
        self.window_size = window_size
        self.velocity_weight = velocity_weight
        self.history = deque(maxlen=window_size)
        
    def smooth(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        平滑处理一帧的关键点数据
        
        Args:
            results: 包含关键点数据的字典
            
        Returns:
            平滑处理后的关键点数据
        """
        # 添加到历史记录
        self.history.append(results)
        
        # 如果历史记录不足，返回原始数据
        if len(self.history) < 2:
            return results
            
        # 创建输出字典
        smoothed_results = results.copy()
        
        # 平滑处理各类关键点
        for key in ['pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
            if key in results and results[key]:
                smoothed_results[key] = self._smooth_landmarks(
                    [frame[key] for frame in self.history if key in frame and frame[key]]
                )
                
        return smoothed_results
        
    def _smooth_landmarks(self, landmarks_history: List) -> Any:
        """
        平滑处理单个类型的关键点序列
        
        Args:
            landmarks_history: 关键点历史数据列表
            
        Returns:
            平滑处理后的关键点数据
        """
        if not landmarks_history:
            return None
            
        # 获取最新帧
        current = landmarks_history[-1]
        
        # 如果历史记录不足，返回当前帧
        if len(landmarks_history) < 2:
            return current
            
        # 创建平滑后的关键点对象
        smoothed = type(current)()
        
        # 获取关键点列表
        current_landmarks = current.landmark
        smoothed.landmark = []
        
        # 对每个关键点进行平滑处理
        for i in range(len(current_landmarks)):
            # 收集历史数据
            point_history = [frame.landmark[i] for frame in landmarks_history]
            
            # 计算加权平均
            weights = np.linspace(0, 1, len(point_history))
            weights = weights / weights.sum()
            
            # 创建平滑后的关键点
            smoothed_point = type(current_landmarks[i])()
            
            # 平滑x, y, z坐标
            for attr in ['x', 'y', 'z']:
                values = np.array([getattr(p, attr) for p in point_history])
                smoothed_value = np.sum(weights * values)
                
                # 应用速度权重
                if len(values) >= 2:
                    velocity = values[-1] - values[-2]
                    smoothed_value += velocity * self.velocity_weight
                    
                setattr(smoothed_point, attr, smoothed_value)
                
            # 保持可见性不变
            if hasattr(current_landmarks[i], 'visibility'):
                smoothed_point.visibility = current_landmarks[i].visibility
                
            smoothed.landmark.append(smoothed_point)
            
        return smoothed
        
    def reset(self):
        """重置平滑器状态"""
        self.history.clear()
