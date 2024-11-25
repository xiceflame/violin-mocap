"""
标注工具标签页模块
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QGridLayout
)
from PyQt6.QtCore import Qt

from ..widgets import ImageViewer

class AnnotationTab(QWidget):
    """标注工具标签页类"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._setup_ui()

    def _setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout()
        
        # 创建图片显示区域
        self.image_view = ImageViewer()
        layout.addWidget(self.image_view)
        
        # 创建控制区域
        control_layout = QHBoxLayout()
        
        # 选择图片目录按钮
        self.select_dir_button = QPushButton("选择图片目录")
        self.select_dir_button.clicked.connect(self.parent.select_image_directory)
        control_layout.addWidget(self.select_dir_button)
        
        # 上一张/下一张按钮
        self.prev_button = QPushButton("上一张")
        self.prev_button.clicked.connect(self.parent.show_prev_image)
        control_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("下一张")
        self.next_button.clicked.connect(self.parent.show_next_image)
        control_layout.addWidget(self.next_button)
        
        layout.addLayout(control_layout)
        
        # 创建标注设置区域
        annotation_group = QGroupBox("标注设置")
        annotation_layout = QGridLayout()
        
        # 标注类别选择
        annotation_layout.addWidget(QLabel("标注类别:"), 0, 0)
        self.class_combo = QComboBox()
        self.class_combo.addItems(["小提琴", "琴弓", "琴弓毛", "琴码", "琴弦"])
        annotation_layout.addWidget(self.class_combo, 0, 1)
        
        # 标注模式选择
        annotation_layout.addWidget(QLabel("标注模式:"), 1, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["关键点", "边界框"])
        annotation_layout.addWidget(self.mode_combo, 1, 1)
        
        # 标注操作按钮
        self.start_annotation_button = QPushButton("开始标注")
        self.start_annotation_button.clicked.connect(self.parent.start_annotation)
        annotation_layout.addWidget(self.start_annotation_button, 2, 0)
        
        self.save_annotation_button = QPushButton("保存标注")
        self.save_annotation_button.clicked.connect(self.parent.save_annotation)
        annotation_layout.addWidget(self.save_annotation_button, 2, 1)
        
        self.clear_annotation_button = QPushButton("清除标注")
        self.clear_annotation_button.clicked.connect(self.parent.clear_annotation)
        annotation_layout.addWidget(self.clear_annotation_button, 2, 2)
        
        annotation_group.setLayout(annotation_layout)
        layout.addWidget(annotation_group)
        
        self.setLayout(layout)
