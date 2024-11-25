import time
from collections import deque

class FPSCounter:
    """
    FPS（每秒帧数）计数器，用于性能监控
    """
    
    def __init__(self, window_size: int = 30):
        """
        初始化FPS计数器
        
        Args:
            window_size: 用于计算平均FPS的时间窗口大小
        """
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        
    def update(self) -> None:
        """
        更新FPS计数器
        每一帧都应该调用此方法
        """
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
    def get_fps(self) -> float:
        """
        获取当前FPS
        
        Returns:
            当前的每秒帧数
        """
        if not self.frame_times:
            return 0.0
            
        # 计算平均帧时间
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        # 避免除以零
        if avg_frame_time == 0:
            return float('inf')
            
        return 1.0 / avg_frame_time
        
    def reset(self) -> None:
        """重置FPS计数器"""
        self.frame_times.clear()
        self.last_time = time.time()
