import sys
import os
import json
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                          QLabel, QPushButton, QFileDialog, QListWidget,
                          QMessageBox, QComboBox, QSpinBox)
from PyQt6.QtCore import Qt, QPoint, QRect, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

class KeypointType:
    VIOLIN = "violin"
    BOW = "bow"
    HAND = "hand"

class ViolinKeypointAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小提琴关键点标注工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.current_image = None
        self.current_image_path = None
        self.keypoints = {
            KeypointType.VIOLIN: [],  # [(x, y, label), ...]
            KeypointType.BOW: [],
            KeypointType.HAND: []
        }
        self.current_keypoint_type = KeypointType.VIOLIN
        self.scale_factor = 1.0
        self.image_offset = QPoint(0, 0)
        
        # 标注点的标签
        self.violin_labels = ["bridge", "scroll", "tailpiece", "chin_rest"]
        self.bow_labels = ["tip", "frog", "stick_center"]
        self.hand_labels = ["thumb", "index", "middle", "ring", "pinky"]
        
        self.init_ui()
        
    def init_ui(self):
        # 创建主窗口布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)
        
        # 文件操作按钮
        self.load_btn = QPushButton("加载图像")
        self.load_btn.clicked.connect(self.load_image)
        self.save_btn = QPushButton("保存标注")
        self.save_btn.clicked.connect(self.save_annotations)
        
        # 关键点类型选择
        self.type_combo = QComboBox()
        self.type_combo.addItems([KeypointType.VIOLIN, KeypointType.BOW, KeypointType.HAND])
        self.type_combo.currentTextChanged.connect(self.change_keypoint_type)
        
        # 标签选择
        self.label_combo = QComboBox()
        self.update_label_combo()
        
        # 图像列表
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_selected_image)
        
        # 添加控件到控制面板
        control_layout.addWidget(QLabel("标注工具"))
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(QLabel("关键点类型:"))
        control_layout.addWidget(self.type_combo)
        control_layout.addWidget(QLabel("标签:"))
        control_layout.addWidget(self.label_combo)
        control_layout.addWidget(QLabel("图像列表:"))
        control_layout.addWidget(self.image_list)
        control_layout.addStretch()
        
        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.mousePressEvent = self.add_keypoint
        
        # 添加到主布局
        layout.addWidget(control_panel)
        layout.addWidget(self.image_label, stretch=1)
        
    def update_label_combo(self):
        self.label_combo.clear()
        if self.current_keypoint_type == KeypointType.VIOLIN:
            self.label_combo.addItems(self.violin_labels)
        elif self.current_keypoint_type == KeypointType.BOW:
            self.label_combo.addItems(self.bow_labels)
        else:
            self.label_combo.addItems(self.hand_labels)
            
    def change_keypoint_type(self, keypoint_type):
        self.current_keypoint_type = keypoint_type
        self.update_label_combo()
        self.update_display()
        
    def load_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg)"
        )
        
        if image_path:
            self.load_image_from_path(image_path)
            
    def load_image_from_path(self, image_path):
        self.current_image = cv2.imread(image_path)
        if self.current_image is not None:
            self.current_image_path = image_path
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.update_display()
            
            # 检查是否存在对应的标注文件
            annotation_path = self.get_annotation_path(image_path)
            if os.path.exists(annotation_path):
                self.load_annotations(annotation_path)
                
    def get_annotation_path(self, image_path):
        directory = os.path.dirname(image_path)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(directory, f"{filename}_annotations.json")
        
    def update_display(self):
        if self.current_image is None:
            return
            
        # 创建显示图像的副本
        display_image = self.current_image.copy()
        
        # 绘制所有关键点
        for kp_type, points in self.keypoints.items():
            color = self.get_color_for_type(kp_type)
            for x, y, label in points:
                # 绘制点
                cv2.circle(display_image, (int(x), int(y)), 5, color, -1)
                # 添加标签
                cv2.putText(display_image, label, (int(x)+10, int(y)+10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # 转换为QImage并显示
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 缩放图像以适应显示区域
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self.image_label.setPixmap(scaled_pixmap)
        
    def get_color_for_type(self, keypoint_type):
        colors = {
            KeypointType.VIOLIN: (0, 255, 0),    # 绿色
            KeypointType.BOW: (255, 0, 0),       # 红色
            KeypointType.HAND: (0, 0, 255)       # 蓝色
        }
        return colors.get(keypoint_type, (255, 255, 255))
        
    def add_keypoint(self, event):
        if self.current_image is None:
            return
            
        # 获取点击位置
        pos = event.pos()
        label_size = self.image_label.size()
        pixmap_size = self.image_label.pixmap().size()
        
        # 计算图像在标签中的实际位置
        x_scale = self.current_image.shape[1] / pixmap_size.width()
        y_scale = self.current_image.shape[0] / pixmap_size.height()
        
        # 调整点击坐标到实际图像坐标
        x = int(pos.x() * x_scale)
        y = int(pos.y() * y_scale)
        
        # 添加关键点
        label = self.label_combo.currentText()
        self.keypoints[self.current_keypoint_type].append((x, y, label))
        
        # 更新显示
        self.update_display()
        
    def save_annotations(self):
        if self.current_image_path is None:
            return
            
        # 准备保存的数据
        data = {
            "image_path": self.current_image_path,
            "keypoints": self.keypoints
        }
        
        # 保存到JSON文件
        annotation_path = self.get_annotation_path(self.current_image_path)
        with open(annotation_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        QMessageBox.information(self, "成功", "标注已保存")
        
    def load_annotations(self, annotation_path):
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
                self.keypoints = data["keypoints"]
            self.update_display()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载标注失败: {str(e)}")
            
    def clear_annotations(self):
        self.keypoints = {
            KeypointType.VIOLIN: [],
            KeypointType.BOW: [],
            KeypointType.HAND: []
        }
        self.update_display()

    def load_selected_image(self, item):
        """从列表中加载选中的图像"""
        image_path = item.text()
        if os.path.exists(image_path):
            self.load_image_from_path(image_path)
        else:
            QMessageBox.warning(self, "错误", f"找不到图像文件: {image_path}")

class AnnotationTool:
    def __init__(self, image_dir: str, class_name: str):
        self.image_dir = image_dir
        self.class_name = class_name
        self.current_index = 0
        self.annotations = {}
        
        # 获取图片列表
        self.image_files = self._get_image_files()
        
        # 加载已有标注
        self.labels_dir = os.path.join(os.path.dirname(image_dir), 'labels')
        os.makedirs(self.labels_dir, exist_ok=True)
        self._load_annotations()
        
    def _get_image_files(self) -> list[str]:
        """获取图片文件列表"""
        image_files = []
        for file in os.listdir(self.image_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(self.image_dir, file))
        return sorted(image_files)
        
    def _load_annotations(self):
        """加载已有标注"""
        for image_file in self.image_files:
            label_file = self._get_label_file(image_file)
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    self.annotations[image_file] = json.load(f)
                    
    def _get_label_file(self, image_file: str) -> str:
        """获取标注文件路径"""
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        return os.path.join(self.labels_dir, f"{base_name}.json")
        
    def get_current_image(self) -> str | None:
        """获取当前图片路径"""
        if 0 <= self.current_index < len(self.image_files):
            return self.image_files[self.current_index]
        return None
        
    def next_image(self) -> bool:
        """切换到下一张图片"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return True
        return False
        
    def prev_image(self) -> bool:
        """切换到上一张图片"""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False
        
    def get_progress(self) -> str:
        """获取标注进度"""
        total = len(self.image_files)
        annotated = len(self.annotations)
        return f"{annotated}/{total} ({annotated/total*100:.1f}%)"
        
    def save_annotation(self, image_path: str, bbox: list[float] | None = None):
        """保存标注"""
        if bbox is None:
            # 如果没有提供边界框，使用整个图片
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            bbox = [0, 0, width, height]
            
        # 将坐标转换为YOLO格式 (x_center, y_center, width, height)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        x_center = (bbox[0] + bbox[2]) / 2 / width
        y_center = (bbox[1] + bbox[3]) / 2 / height
        box_width = (bbox[2] - bbox[0]) / width
        box_height = (bbox[3] - bbox[1]) / height
        
        # 创建标注数据
        annotation = {
            "class": self.class_name,
            "bbox": [x_center, y_center, box_width, box_height]
        }
        
        # 保存标注文件
        label_file = self._get_label_file(image_path)
        with open(label_file, 'w') as f:
            json.dump(annotation, f, indent=2)
            
        # 更新内存中的标注
        self.annotations[image_path] = annotation
