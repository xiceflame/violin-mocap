"""性能优化模块，特别针对Apple Silicon芯片优化"""
import torch
import logging
from typing import Any, Dict, List, Optional

class MSeries:
    """Apple Silicon M系列芯片优化"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = self._get_optimal_device()
        self.performance_tips = []
        self._analyze_system()

    def _get_optimal_device(self) -> torch.device:
        """获取最优设备"""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _analyze_system(self):
        """分析系统性能和特性"""
        device = self.get_device()
        if device.type == 'mps':
            self.performance_tips.extend([
                "使用Metal性能着色器(MPS)进行加速",
                "启用神经引擎优化",
                "使用多核CPU优化"
            ])
        elif device.type == 'cuda':
            self.performance_tips.extend([
                "使用CUDA加速",
                "启用cuDNN优化"
            ])
        else:
            self.performance_tips.extend([
                "使用OpenCV优化",
                "启用多线程处理"
            ])

    def optimize_model(self, model: Any) -> Any:
        """优化模型"""
        try:
            if model is None:
                return None

            device = self.get_device()
            if device.type == 'mps':
                # MPS优化
                model.to(device)
                self.logger.info(f"模型已优化为MPS设备: {device}")
            elif device.type == 'cuda':
                # CUDA优化
                model.to(device)
                self.logger.info(f"模型已优化为CUDA设备: {device}")
            else:
                # CPU优化
                model.to(device)
                self.logger.info(f"模型已优化为CPU设备: {device}")

            return model
        except Exception as e:
            self.logger.error(f"模型优化失败: {e}")
            return model

    def get_device(self) -> torch.device:
        """获取当前设备"""
        return self.device

    def get_performance_tips(self) -> List[str]:
        """获取性能优化建议"""
        return self.performance_tips

class AdaptivePerformance:
    """自适应性能管理"""
    def __init__(self, m_series: MSeries):
        self.logger = logging.getLogger(__name__)
        self.m_series = m_series
        self.performance_mode = 'balanced'  # 'speed', 'balanced', 'quality'
        self.settings = self._init_settings()

    def _init_settings(self) -> Dict[str, Dict[str, Any]]:
        """初始化性能设置"""
        return {
            'speed': {
                'frame_skip': 2,
                'resolution_scale': 0.75,
                'detection_confidence': 0.3,
                'tracking_confidence': 0.3
            },
            'balanced': {
                'frame_skip': 0,
                'resolution_scale': 1.0,
                'detection_confidence': 0.5,
                'tracking_confidence': 0.5
            },
            'quality': {
                'frame_skip': 0,
                'resolution_scale': 1.0,
                'detection_confidence': 0.7,
                'tracking_confidence': 0.7
            }
        }

    def set_performance_mode(self, mode: str):
        """设置性能模式"""
        if mode in self.settings:
            self.performance_mode = mode
            self.logger.info(f"性能模式已设置为: {mode}")
        else:
            self.logger.warning(f"未知的性能模式: {mode}")

    def get_current_settings(self) -> Dict[str, Any]:
        """获取当前性能设置"""
        return self.settings[self.performance_mode]

    def optimize_frame_processing(self, frame_time: float):
        """根据帧处理时间自动调整性能设置"""
        if frame_time > 0.1:  # 帧处理时间超过100ms
            self.set_performance_mode('speed')
        elif frame_time < 0.033:  # 帧处理时间小于33ms
            self.set_performance_mode('quality')
        else:
            self.set_performance_mode('balanced')
