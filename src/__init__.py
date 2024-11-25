"""
小提琴动作捕捉系统
"""
from .core import ViolinCapture
from .gui import ViolinMocapGUI
from .training import ModelTrainer
from .annotation import ViolinKeypointAnnotator, KeypointType, AnnotationTool

__version__ = '1.0.0'

__all__ = [
    'ViolinCapture',
    'ViolinMocapGUI',
    'ModelTrainer',
    'ViolinKeypointAnnotator',
    'KeypointType',
    'AnnotationTool'
]
