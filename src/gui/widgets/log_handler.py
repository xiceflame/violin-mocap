"""
日志处理器模块
"""
import logging
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtCore import Qt

class LogHandler(logging.Handler):
    """日志处理器类"""
    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self.text_edit = text_edit
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        """输出日志记录"""
        msg = self.format(record)
        self.text_edit.append(msg)
        self.text_edit.verticalScrollBar().setValue(
            self.text_edit.verticalScrollBar().maximum()
        )
