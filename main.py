#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import threading
import time
from PyQt6.QtWidgets import QApplication

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gui import ViolinMocapGUI
from src.core.performance_optimizer import MSeries, AdaptivePerformance

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('violin_mocap.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 初始化性能优化
        m_series = MSeries()
        performance = AdaptivePerformance(m_series)
        logger.info(f"性能优化初始化完成，使用设备: {m_series.device}")
        
        # 创建应用
        app = QApplication(sys.argv)
        
        # 创建主窗口
        window = ViolinMocapGUI()
        window.show()
        
        # 运行应用
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
