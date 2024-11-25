"""
小提琴动作捕捉系统 - 主窗口
"""
import os
import sys
import json
import logging
import cv2

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                          QTabWidget, QMenuBar, QStatusBar, QTextEdit,
                          QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QImage, QPixmap

from src.core.violin_capture import ViolinCapture
from src.utils.path_manager import PathManager
from configs.manager import ConfigManager
from .tabs.main_tab import MainTab
from .tabs.training_tab import TrainingTab
from .tabs.annotation_tab import AnnotationTab
from src.annotation import ViolinKeypointAnnotator, KeypointType

class QTextEditLogger(logging.Handler):
    """QTextEdit日志处理器"""
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)

class ModelTrainer(QThread):
    """模型训练线程"""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = False

    def run(self):
        try:
            self.is_running = True
            # 模型训练逻辑
            self.finished.emit()
        except Exception as e:
            self.log.emit(f"训练出错: {e}")
        finally:
            self.is_running = False

class ViolinMocapGUI(QMainWindow):
    """小提琴动作捕捉系统GUI类"""
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.path_manager = PathManager()
        self.config_manager = ConfigManager()
        
        # 初始化小提琴捕捉系统
        try:
            self.capture = ViolinCapture()
            # 加载摄像头列表但不启动
            self.capture.list_cameras()
            self.logger.info("摄像头系统初始化成功")
        except Exception as e:
            self.logger.error(f"摄像头系统初始化失败: {e}")
            QMessageBox.critical(self, "错误", f"摄像头系统初始化失败: {str(e)}")
            sys.exit(1)
            
        # 初始化UI
        self._init_ui()
        
        # 初始化标注相关变量
        self.current_image_dir = None
        self.image_files = []
        self.current_image_index = -1
        self.annotator = None
        self.is_annotating = False
        
        # 功能状态
        self.network_stream_enabled = False
        self.auto_learning_enabled = False
        
        # 应用配置
        self._apply_config()
        
    def _init_ui(self):
        """初始化UI"""
        # 设置窗口标题和大小
        self.setWindowTitle('小提琴动作捕捉系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.main_tab = MainTab(self)
        self.annotation_tab = AnnotationTab(self)
        self.training_tab = TrainingTab(self)
        
        self.tab_widget.addTab(self.main_tab, "主界面")
        self.tab_widget.addTab(self.annotation_tab, "标注")
        self.tab_widget.addTab(self.training_tab, "训练")
        
        layout.addWidget(self.tab_widget)
        
        # 创建菜单栏
        self._create_menu_bar()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 创建日志区域
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)
        
        # 设置日志处理器
        log_handler = QTextEditLogger(self.log_text)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.INFO)
        
    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        open_action = QAction('打开', self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction('保存', self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        # 功能菜单
        feature_menu = menubar.addMenu('功能')
        
        # 网络流开关
        self.network_stream_action = QAction('启用网络流', self, checkable=True)
        self.network_stream_action.triggered.connect(self.toggle_network_stream)
        feature_menu.addAction(self.network_stream_action)
        
        # 自动学习开关
        self.auto_learning_action = QAction('启用自动学习', self, checkable=True)
        self.auto_learning_action.triggered.connect(self.toggle_auto_learning)
        feature_menu.addAction(self.auto_learning_action)
        
    def _apply_config(self):
        """应用配置"""
        try:
            # 应用相机设置
            camera_settings = self.config_manager.camera_settings
            if camera_settings:
                self.change_camera(camera_settings.get('camera_index', 0))
                
            # 应用模型设置
            model_settings = self.config_manager.model_settings
            if model_settings:
                conf = model_settings.get('conf_threshold', 0.5)
                iou = model_settings.get('iou_threshold', 0.45)
                self.main_tab.yolo_conf_slider.setValue(int(conf * 100))
                self.main_tab.yolo_iou_slider.setValue(int(iou * 100))
                
            # 应用训练设置
            training_settings = self.config_manager.training_settings
            if training_settings:
                self.training_tab.epochs.setValue(training_settings.get('epochs', 100))
                self.training_tab.batch_size.setValue(training_settings.get('batch_size', 16))
                self.training_tab.learning_rate.setValue(training_settings.get('learning_rate', 0.01))
                self.training_tab.pretrained.setChecked(training_settings.get('use_pretrained', True))
                self.training_tab.multi_gpu.setChecked(training_settings.get('use_multi_gpu', False))
                
        except Exception as e:
            self.logger.error(f"应用配置失败: {e}")
            QMessageBox.warning(self, "警告", f"应用配置失败: {str(e)}")
            
    def toggle_network_stream(self, checked):
        """切换网络流状态"""
        if checked:
            self.capture.enable_network_stream()
            self.network_stream_enabled = True
            self.status_bar.showMessage("网络流已启用")
        else:
            self.capture.disable_network_stream()
            self.network_stream_enabled = False
            self.status_bar.showMessage("网络流已禁用")
            
    def toggle_auto_learning(self, checked):
        """切换自动学习状态"""
        if checked:
            self.capture.enable_auto_learning()
            self.auto_learning_enabled = True
            self.status_bar.showMessage("自动学习已启用")
        else:
            self.capture.disable_auto_learning()
            self.auto_learning_enabled = False
            self.status_bar.showMessage("自动学习已禁用")
            
    def change_camera(self, camera_id):
        """切换摄像头"""
        try:
            self.capture.change_camera(camera_id)
            self.config_manager.set_gui_config(camera_id, 'camera_settings', 'camera_index')
            self.logger.info(f"已切换到摄像头 {camera_id}")
        except Exception as e:
            self.logger.error(f"切换摄像头失败: {e}")
            QMessageBox.critical(self, "错误", f"切换摄像头失败: {str(e)}")
            
    def update_yolo_conf(self, value):
        """更新YOLO置信度阈值"""
        conf = value / 100.0
        self.config_manager.set_gui_config(conf, 'model_settings', 'conf_threshold')
        self.capture.set_yolo_conf(conf)
        
    def update_yolo_iou(self, value):
        """更新YOLO IOU阈值"""
        iou = value / 100.0
        self.config_manager.set_gui_config(iou, 'model_settings', 'iou_threshold')
        self.capture.set_yolo_iou(iou)
            
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, '关于',
            '小提琴动作捕捉系统\n'
            '版本 1.0.0\n\n'
            '使用计算机视觉技术实时捕捉和分析小提琴演奏动作。')
            
    def closeEvent(self, event):
        """关闭事件处理"""
        if hasattr(self, 'capture'):
            self.capture.stop()
        event.accept()
        
    def open_file(self):
        """打开文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开文件",
            "",
            "图像文件 (*.png *.jpg *.jpeg);;所有文件 (*.*)"
        )
        if file_path:
            try:
                self.load_image(file_path)
            except Exception as e:
                self.logger.error(f"打开文件失败: {e}")
                QMessageBox.critical(self, "错误", f"打开文件失败: {str(e)}")

    def save_file(self):
        """保存文件"""
        if not hasattr(self, 'current_image'):
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存文件",
            "",
            "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;所有文件 (*.*)"
        )
        if file_path:
            try:
                # 保存当前图像
                if hasattr(self, 'current_image'):
                    cv2.imwrite(file_path, self.current_image)
                    self.logger.info(f"文件已保存: {file_path}")
            except Exception as e:
                self.logger.error(f"保存文件失败: {e}")
                QMessageBox.critical(self, "错误", f"保存文件失败: {str(e)}")
                
    def select_image_directory(self):
        """选择图片目录"""
        try:
            dir_path = QFileDialog.getExistingDirectory(
                self, "选择图片目录", "",
                QFileDialog.Option.ShowDirsOnly |
                QFileDialog.Option.DontResolveSymlinks)
                
            if dir_path:
                self.current_image_dir = dir_path
                self.image_files = [
                    os.path.join(dir_path, f) for f in os.listdir(dir_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                self.current_image_index = 0 if self.image_files else -1
                
                if not self.image_files:
                    QMessageBox.warning(
                        self, "警告",
                        "所选目录中没有找到支持的图片文件 (.png, .jpg, .jpeg)")
                else:
                    self.load_current_image()
                    
        except Exception as e:
            self.logger.error(f"选择图片目录失败: {e}")
            QMessageBox.critical(self, "错误", f"选择图片目录失败: {str(e)}")
            
    def load_current_image(self):
        """加载当前图片"""
        try:
            if 0 <= self.current_image_index < len(self.image_files):
                image_path = self.image_files[self.current_image_index]
                self.annotation_tab.image_view.load_image(image_path)
                self.logger.info(f"已加载图片: {image_path}")
        except Exception as e:
            self.logger.error(f"加载图片失败: {e}")
            QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")
            
    def start_capture(self):
        """开始捕捉"""
        try:
            if not hasattr(self, 'capture'):
                self.capture = ViolinCapture()
                self.main_tab.refresh_cameras()  # 刷新摄像头列表
            self.capture.start()
            self.main_tab.start_button.setEnabled(False)
            self.main_tab.stop_button.setEnabled(True)
            self.logger.info("开始捕捉")
        except Exception as e:
            self.logger.error(f"启动失败: {e}")
            QMessageBox.critical(self, "错误", f"启动失败: {e}")
            
    def stop_capture(self):
        """停止捕捉"""
        try:
            if hasattr(self, 'capture'):
                self.capture.stop()
                self.main_tab.start_button.setEnabled(True)
                self.main_tab.stop_button.setEnabled(False)
                self.logger.info("停止捕捉")
        except Exception as e:
            self.logger.error(f"停止失败: {e}")
            QMessageBox.critical(self, "错误", f"停止失败: {e}")
            
    def apply_model_settings(self):
        """应用模型设置"""
        try:
            if hasattr(self, 'capture'):
                # 获取模型设置
                model_type = self.main_tab.model_type.currentText()
                conf_threshold = self.main_tab.yolo_conf_slider.value() / 100.0
                iou_threshold = self.main_tab.yolo_iou_slider.value() / 100.0
                
                # 更新设置
                self.capture.model_type = model_type
                self.capture.yolo_conf = conf_threshold
                self.capture.yolo_iou = iou_threshold
                
                self.logger.info(f"已应用模型设置: {model_type}, conf={conf_threshold:.2f}, iou={iou_threshold:.2f}")
                
        except Exception as e:
            self.logger.error(f"应用模型设置失败: {e}")
            QMessageBox.critical(self, "错误", f"应用模型设置失败: {str(e)}")
            
    def clear_log(self):
        """清除日志"""
        try:
            self.log_text.clear()
            self.logger.info("日志已清除")
        except Exception as e:
            self.logger.error(f"清除日志失败: {e}")
            QMessageBox.critical(self, "错误", f"清除日志失败: {str(e)}")
            
    def save_log(self):
        """保存日志"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存日志", "", "日志文件 (*.log);;所有文件 (*.*)"
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.logger.info(f"日志已保存到: {file_path}")
        except Exception as e:
            self.logger.error(f"保存日志失败: {e}")
            QMessageBox.critical(self, "错误", f"保存日志失败: {str(e)}")
            
    def select_dataset_directory(self):
        """选择数据集目录"""
        try:
            directory = QFileDialog.getExistingDirectory(
                self, "选择数据集目录",
                str(self.path_manager.get_path('data', 'dataset') or self.path_manager.root)
            )
            if directory:
                self.training_tab.dataset_path.setText(directory)
                self.logger.info(f"已选择数据集目录: {directory}")
        except Exception as e:
            self.logger.error(f"选择数据集目录失败: {e}")
            QMessageBox.critical(self, "错误", f"选择数据集目录失败: {str(e)}")
            
    def start_training(self):
        """开始训练"""
        try:
            if not hasattr(self, 'trainer'):
                self.trainer = ModelTrainer(self.config_manager.get_config())
                self.trainer.progress.connect(self.training_tab.progress_bar.setValue)
                self.trainer.log.connect(self.logger.info)
                self.trainer.finished.connect(self._on_training_finished)
            
            if not self.trainer.is_running:
                self.trainer.start()
                self.training_tab.start_training_button.setEnabled(False)
                self.training_tab.stop_training_button.setEnabled(True)
                self.logger.info("开始训练")
                
        except Exception as e:
            self.logger.error(f"开始训练失败: {e}")
            QMessageBox.critical(self, "错误", f"开始训练失败: {str(e)}")
            
    def stop_training(self):
        """停止训练"""
        try:
            if hasattr(self, 'trainer') and self.trainer.is_running:
                self.trainer.is_running = False
                self.logger.info("正在停止训练...")
        except Exception as e:
            self.logger.error(f"停止训练失败: {e}")
            QMessageBox.critical(self, "错误", f"停止训练失败: {str(e)}")
            
    def _on_training_finished(self):
        """训练完成回调"""
        self.training_tab.start_training_button.setEnabled(True)
        self.training_tab.stop_training_button.setEnabled(False)
        self.logger.info("训练已完成")

    def start_annotation(self):
        """开始标注"""
        try:
            if not self.image_files:
                QMessageBox.warning(self, "警告", "请先选择图片目录")
                return
            
            # 切换标注状态
            self.is_annotating = not self.is_annotating
            
            if self.is_annotating:
                # 创建标注器
                self.annotator = ViolinKeypointAnnotator()
                # 更新按钮文本
                self.annotation_tab.start_annotation_button.setText("停止标注")
                self.logger.info("开始标注")
            else:
                # 清理标注器
                self.annotator = None
                # 更新按钮文本
                self.annotation_tab.start_annotation_button.setText("开始标注")
                self.logger.info("停止标注")
                
        except Exception as e:
            self.logger.error(f"标注操作失败: {e}")
            QMessageBox.critical(self, "错误", f"标注操作失败: {str(e)}")
            
    def save_annotation(self):
        """保存标注"""
        try:
            if not self.annotator:
                QMessageBox.warning(self, "警告", "请先开始标注")
                return
                
            if not self.image_files:
                QMessageBox.warning(self, "警告", "没有可标注的图片")
                return
                
            # 获取当前图片路径
            image_path = self.image_files[self.current_image_index]
            
            # 保存标注
            annotation_dir = os.path.join(os.path.dirname(self.current_image_dir), "annotations")
            os.makedirs(annotation_dir, exist_ok=True)
            
            base_name = os.path.splitext(self.image_files[self.current_image_index])[0]
            annotation_path = os.path.join(annotation_dir, f"{base_name}.json")
            
            self.annotator.save_annotations(annotation_path)
            self.logger.info(f"标注已保存到: {annotation_path}")
            
        except Exception as e:
            self.logger.error(f"保存标注失败: {e}")
            QMessageBox.critical(self, "错误", f"保存标注失败: {str(e)}")
            
    def clear_annotation(self):
        """清除标注"""
        try:
            if not self.annotator:
                QMessageBox.warning(self, "警告", "请先开始标注")
                return
                
            # 清除当前图片的标注
            self.annotator.clear_keypoints()
            # 刷新显示
            self.load_current_image()
            self.logger.info("已清除当前标注")
            
        except Exception as e:
            self.logger.error(f"清除标注失败: {e}")
            QMessageBox.critical(self, "错误", f"清除标注失败: {str(e)}")

    def show_next_image(self):
        """显示下一张图片"""
        try:
            if not self.image_files:
                return
            
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            self.load_current_image()
            
        except Exception as e:
            self.logger.error(f"显示下一张图片失败: {e}")
            QMessageBox.critical(self, "错误", f"显示下一张图片失败: {str(e)}")

    def show_prev_image(self):
        """显示上一张图片"""
        try:
            if not self.image_files:
                return
            
            self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            self.load_current_image()
            
        except Exception as e:
            self.logger.error(f"显示上一张图片失败: {e}")
            QMessageBox.critical(self, "错误", f"显示上一张图片失败: {str(e)}")
