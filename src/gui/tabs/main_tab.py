"""
主界面标签页模块
"""
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSlider, QComboBox,
    QTextEdit, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer

from ..widgets import ImageViewer, LogHandler

class MainTab(QWidget):
    """主界面标签页类"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        
        # 创建定时器用于更新视频帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms刷新一次，约33FPS
        
        # 刷新摄像头列表
        self.refresh_cameras()

    def _setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout()
        
        # 创建摄像头控制区域
        camera_group = QGroupBox("摄像头控制")
        camera_layout = QHBoxLayout()
        
        # 摄像头选择下拉框
        camera_layout.addWidget(QLabel("摄像头:"))
        self.camera_combo = QComboBox()
        camera_layout.addWidget(self.camera_combo)
        
        # 刷新摄像头按钮
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(refresh_button)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # 创建摄像头预览区域
        self.camera_view = ImageViewer()
        layout.addWidget(self.camera_view)
        
        # 创建控制按钮区域
        control_layout = QHBoxLayout()
        
        # 开始/停止按钮
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.parent.start_capture)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.parent.stop_capture)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        # 切换摄像头按钮
        self.switch_camera_button = QPushButton("切换摄像头")
        self.switch_camera_button.clicked.connect(self.switch_camera)
        control_layout.addWidget(self.switch_camera_button)
        
        # 添加控制按钮区域到主布局
        layout.addLayout(control_layout)
        
        # 创建设置区域
        settings_group = QGroupBox("设置")
        settings_layout = QGridLayout()
        
        # YOLO置信度阈值
        settings_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        self.yolo_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.yolo_conf_slider.setRange(1, 100)
        self.yolo_conf_slider.setValue(50)
        self.yolo_conf_slider.valueChanged.connect(self.parent.update_yolo_conf)
        settings_layout.addWidget(self.yolo_conf_slider, 0, 1)
        self.yolo_conf_label = QLabel("50%")
        settings_layout.addWidget(self.yolo_conf_label, 0, 2)
        
        # YOLO IOU阈值
        settings_layout.addWidget(QLabel("IOU阈值:"), 1, 0)
        self.yolo_iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.yolo_iou_slider.setRange(1, 100)
        self.yolo_iou_slider.setValue(50)
        self.yolo_iou_slider.valueChanged.connect(self.parent.update_yolo_iou)
        settings_layout.addWidget(self.yolo_iou_slider, 1, 1)
        self.yolo_iou_label = QLabel("50%")
        settings_layout.addWidget(self.yolo_iou_label, 1, 2)
        
        # 模型选择
        settings_layout.addWidget(QLabel("模型:"), 2, 0)
        self.model_type = QComboBox()
        self.model_type.addItems(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"])
        settings_layout.addWidget(self.model_type, 2, 1, 1, 2)
        
        # 应用设置按钮
        self.apply_settings_button = QPushButton("应用设置")
        self.apply_settings_button.clicked.connect(self.parent.apply_model_settings)
        settings_layout.addWidget(self.apply_settings_button, 3, 0, 1, 3)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # 添加日志显示区域
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # 添加日志控制按钮
        log_control_layout = QHBoxLayout()
        
        clear_log_button = QPushButton("清除日志")
        clear_log_button.clicked.connect(self.parent.clear_log)
        log_control_layout.addWidget(clear_log_button)
        
        save_log_button = QPushButton("保存日志")
        save_log_button.clicked.connect(self.parent.save_log)
        log_control_layout.addWidget(save_log_button)
        
        log_layout.addLayout(log_control_layout)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
        
        # 设置日志处理器
        log_handler = LogHandler(self.log_text)
        self.logger.addHandler(log_handler)

    def update_frame(self):
        """更新视频帧"""
        try:
            if self.parent and hasattr(self.parent, 'capture'):
                ret, frame = self.parent.capture.get_frame()
                if ret and frame is not None:
                    self.camera_view.display_image(frame)
        except Exception as e:
            self.logger.error(f"更新视频帧失败: {e}")

    def refresh_cameras(self):
        """刷新摄像头列表"""
        try:
            # 清空当前列表
            self.camera_combo.clear()
            
            # 获取可用摄像头
            cameras = self.parent.capture.get_available_cameras()
            
            # 添加到下拉框
            for camera in cameras:
                # 使用更详细的摄像头描述
                desc = f"{camera['name']}"
                if camera['model']:
                    desc += f" ({camera['model']})"
                self.camera_combo.addItem(desc, camera)  # 将整个camera字典存储为用户数据
                
            # 如果有摄像头，启用相关按钮
            has_cameras = len(cameras) > 0
            self.switch_camera_button.setEnabled(has_cameras)
            self.start_button.setEnabled(has_cameras)
            
            if has_cameras:
                self.logger.info(f"找到 {len(cameras)} 个摄像头")
            else:
                self.logger.warning("未找到可用摄像头")
                
        except Exception as e:
            self.logger.error(f"刷新摄像头列表失败: {e}")
            
    def switch_camera(self):
        """切换到选中的摄像头"""
        try:
            current_index = self.camera_combo.currentIndex()
            if current_index >= 0:
                # 获取存储的camera字典
                camera_data = self.camera_combo.currentData()
                if camera_data:
                    # 停止当前捕捉
                    self.parent.stop_capture()
                    # 切换摄像头
                    if self.parent.capture.change_camera(camera_data):
                        self.logger.info(f"已切换到摄像头: {camera_data['name']}")
                        # 如果之前在捕捉，重新开始
                        if self.stop_button.isEnabled():
                            self.parent.start_capture()
                    else:
                        self.logger.error("切换摄像头失败")
                        
        except Exception as e:
            self.logger.error(f"切换摄像头时出错: {e}")
