import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple

class UIManager:
    """UI管理器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化UI管理器
        
        Args:
            config: UI配置
        """
        self.config = config or {}
        
        # 显示选项
        self.show_fps = self.config.get('show_fps', True)
        self.show_detections = self.config.get('show_detections', True)
        self.show_pose = self.config.get('show_pose', True)
        self.show_landmarks = self.config.get('show_landmarks', True)
        self.show_violin_box = self.config.get('show_violin_box', True)
        self.show_bow_box = self.config.get('show_bow_box', True)
        self.show_debug_info = self.config.get('show_debug_info', True)
        
        # 窗口和样式设置
        self.window_name = self.config.get('window_name', 'Violin Motion Capture')
        self.font_scale = self.config.get('font_scale', 1.0)
        self.line_thickness = self.config.get('line_thickness', 2)
        
        # 颜色方案
        color_scheme = self.config.get('color_scheme', {})
        self.colors = {
            'landmarks': tuple(color_scheme.get('landmarks', (0, 255, 0))),
            'skeleton': tuple(color_scheme.get('skeleton', (255, 255, 255))),
            'violin_box': tuple(color_scheme.get('violin_box', (0, 255, 0))),
            'bow_box': tuple(color_scheme.get('bow_box', (255, 0, 0))),
            'text': tuple(color_scheme.get('text', (255, 255, 255)))
        }
        
        # FPS计算
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        self.fps_update_interval = self.config.get('fps_update_interval', 0.5)  # 秒
        self.last_fps_update = time.time()
        
    def update_fps(self):
        """更新FPS计算"""
        self.curr_frame_time = time.time()
        if self.curr_frame_time - self.last_fps_update >= self.fps_update_interval:
            self.fps = 1 / (self.curr_frame_time - self.prev_frame_time)
            self.last_fps_update = self.curr_frame_time
        self.prev_frame_time = self.curr_frame_time
        
    def draw_text(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                 color: Optional[Tuple[int, int, int]] = None):
        """
        在图像上绘制文本
        
        Args:
            frame: 输入图像
            text: 要绘制的文本
            position: 文本位置 (x, y)
            color: 文本颜色 (B, G, R)
        """
        if color is None:
            color = self.colors['text']
            
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   self.font_scale, color, self.line_thickness)
                   
    def draw_fps(self, frame: np.ndarray):
        """
        在图像上绘制FPS
        
        Args:
            frame: 输入图像
        """
        if self.show_fps:
            self.update_fps()
            self.draw_text(frame, f"FPS: {int(self.fps)}", (10, 30))
            
    def draw_detection_box(self, frame: np.ndarray, bbox: Tuple[float, float, float, float],
                         label: str, color: Optional[Tuple[int, int, int]] = None):
        """
        绘制检测框
        
        Args:
            frame: 输入图像
            bbox: 边界框坐标 (x1, y1, x2, y2)
            label: 标签文本
            color: 框的颜色
        """
        if not self.show_detections:
            return
            
        x1, y1, x2, y2 = map(int, bbox)
        if color is None:
            color = self.colors['violin_box'] if 'violin' in label.lower() else self.colors['bow_box']
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
        self.draw_text(frame, label, (x1, y1 - 10), color)
        
    def draw_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float]],
                      color: Optional[Tuple[int, int, int]] = None):
        """
        绘制关键点
        
        Args:
            frame: 输入图像
            landmarks: 关键点列表 [(x, y), ...]
            color: 关键点颜色
        """
        if not self.show_landmarks or not landmarks:
            return
            
        if color is None:
            color = self.colors['landmarks']
            
        for x, y in landmarks:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
            
    def draw_debug_info(self, frame: np.ndarray, info: Dict[str, Any]):
        """
        绘制调试信息
        
        Args:
            frame: 输入图像
            info: 调试信息字典
        """
        if not self.show_debug_info:
            return
            
        y_offset = 60
        for key, value in info.items():
            self.draw_text(frame, f"{key}: {value}", (10, y_offset))
            y_offset += 30
            
    def show_frame(self, frame: np.ndarray):
        """
        显示图像
        
        Args:
            frame: 要显示的图像
        """
        cv2.imshow(self.window_name, frame)
        
    def wait_key(self, delay: int = 1) -> int:
        """
        等待键盘事件
        
        Args:
            delay: 等待时间（毫秒）
            
        Returns:
            键码
        """
        return cv2.waitKey(delay) & 0xFF
