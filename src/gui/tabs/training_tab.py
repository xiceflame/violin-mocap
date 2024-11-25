"""
训练标签页模块
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QGridLayout, QLineEdit
)
from PyQt6.QtCore import Qt
from ...utils.path_manager import PathManager

class TrainingTab(QWidget):
    """训练标签页类"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.path_manager = PathManager()
        self._setup_ui()

    def _setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout()
        
        # 创建数据集设置区域
        dataset_group = QGroupBox("数据集设置")
        dataset_layout = QGridLayout()
        
        # 选择数据集目录
        dataset_layout.addWidget(QLabel("数据集目录:"), 0, 0)
        self.dataset_path = QLineEdit()
        default_dataset = self.path_manager.get_path('data', 'dataset')
        if default_dataset:
            self.dataset_path.setText(str(default_dataset))
        dataset_layout.addWidget(self.dataset_path, 0, 1)
        
        self.select_dataset_button = QPushButton("浏览")
        self.select_dataset_button.clicked.connect(self.parent.select_dataset_directory)
        dataset_layout.addWidget(self.select_dataset_button, 0, 2)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # 创建模型设置区域
        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout()
        
        # 训练轮数
        model_layout.addWidget(QLabel("训练轮数:"), 0, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        model_layout.addWidget(self.epochs, 0, 1)
        
        # 批次大小
        model_layout.addWidget(QLabel("批次大小:"), 1, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(16)
        model_layout.addWidget(self.batch_size, 1, 1)
        
        # 学习率
        model_layout.addWidget(QLabel("学习率:"), 2, 0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.1)
        self.learning_rate.setValue(0.01)
        self.learning_rate.setSingleStep(0.001)
        model_layout.addWidget(self.learning_rate, 2, 1)
        
        # 使用预训练模型
        self.pretrained = QCheckBox("使用预训练模型")
        self.pretrained.setChecked(True)
        model_layout.addWidget(self.pretrained, 3, 0, 1, 2)
        
        # 使用多GPU训练
        self.multi_gpu = QCheckBox("使用多GPU训练")
        model_layout.addWidget(self.multi_gpu, 4, 0, 1, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 创建训练控制区域
        control_group = QGroupBox("训练控制")
        control_layout = QVBoxLayout()
        
        # 进度条
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)
        
        # 训练控制按钮
        button_layout = QHBoxLayout()
        
        self.start_training_button = QPushButton("开始训练")
        self.start_training_button.clicked.connect(self.parent.start_training)
        button_layout.addWidget(self.start_training_button)
        
        self.stop_training_button = QPushButton("停止训练")
        self.stop_training_button.clicked.connect(self.parent.stop_training)
        self.stop_training_button.setEnabled(False)
        button_layout.addWidget(self.stop_training_button)
        
        control_layout.addLayout(button_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        self.setLayout(layout)
