"""
图像查看器模块
"""
from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np

class ImageViewer(QLabel):
    """图像查看器类"""
    clicked = pyqtSignal(object)  # 点击信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self._pixmap = None
        self.font_scale = 0.5  # 文字大小缩放因子
        self.text_thickness = 1  # 文字粗细
        self.text_color = (0, 255, 0)  # 文字颜色 (BGR)

    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event)
        super().mousePressEvent(event)

    def load_image(self, image_path):
        """从文件加载图像
        
        Args:
            image_path: 图像文件路径
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 显示图像
            self.display_image(image)
            
        except Exception as e:
            raise Exception(f"加载图像失败: {str(e)}")

    def display_image(self, image):
        """显示图像
        
        Args:
            image: numpy数组或QImage对象
        """
        if isinstance(image, np.ndarray):
            # 处理文字渲染
            if len(image.shape) == 3:
                image = image.copy()  # 创建副本以避免修改原始图像
                
                # 获取图像尺寸
                height, width = image.shape[:2]
                
                # 计算文字大小
                base_font_size = min(width, height) / 1000
                font_size = max(0.3, base_font_size * self.font_scale)
                
                # 添加性能信息（示例）
                fps_text = "FPS: 30"  # 这里可以传入实际的FPS值
                cv2.putText(image, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                           self.text_color, self.text_thickness, 
                           cv2.LINE_AA)
                
                # 转换颜色空间
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                format = QImage.Format.Format_RGB888
                bytes_per_line = width * 3
            else:
                format = QImage.Format.Format_Grayscale8
                bytes_per_line = width
            
            # 创建QImage
            qimage = QImage(image.data, width, height, bytes_per_line, format)
        else:
            qimage = image

        # 缩放图像以适应标签大小
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self._pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)

    def set_text_properties(self, scale=None, thickness=None, color=None):
        """设置文字属性
        
        Args:
            scale: 文字大小缩放因子
            thickness: 文字粗细
            color: 文字颜色 (BGR)
        """
        if scale is not None:
            self.font_scale = scale
        if thickness is not None:
            self.text_thickness = thickness
        if color is not None:
            self.text_color = color

    def clear_image(self):
        """清除图像"""
        self._pixmap = None
        self.clear()
