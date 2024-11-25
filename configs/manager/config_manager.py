"""
配置管理器模块 (ConfigManager)

此模块提供了一个统一的配置管理接口，用于处理小提琴动作捕捉系统的所有配置需求。
主要功能包括：
1. 系统配置管理 (/configs/system/config.json)
2. GUI配置管理 (/configs/system/gui_config.json)
3. 路径配置管理 (/configs/system/paths.json)

配置结构：
- system/
  - config.json: 系统基础配置
  - gui_config.json: 界面和交互配置
  - paths.json: 路径配置

使用示例：
    config_manager = ConfigManager()
    
    # 获取配置
    camera_settings = config_manager.camera_settings
    model_settings = config_manager.model_settings
    
    # 设置配置
    config_manager.set_gui_config(value, *keys)
    config_manager.set_system_config(value, *keys)

作者: Cascade AI
更新: 2024-01-24
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        
        # 获取项目根目录
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # 配置文件路径
        self.config_dir = self.project_root / 'configs'
        self.system_dir = self.config_dir / 'system'
        
        # 检查配置目录
        if not self.system_dir.exists():
            raise RuntimeError("系统配置目录不存在")
            
        # 加载配置
        self.system_config = self._load_system_config()
        self.gui_config = self._load_gui_config()
        self.paths_config = self._load_paths_config()
        
    def _load_system_config(self) -> Dict[str, Any]:
        """加载系统配置"""
        system_config_path = self.system_dir / 'config.json'
        if not system_config_path.exists():
            raise FileNotFoundError("系统配置文件不存在")
            
        try:
            with open(system_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载系统配置失败: {e}")
            raise
            
    def _load_gui_config(self) -> Dict[str, Any]:
        """加载GUI配置"""
        gui_config_path = self.system_dir / 'gui_config.json'
        if not gui_config_path.exists():
            raise FileNotFoundError("GUI配置文件不存在")
            
        try:
            with open(gui_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载GUI配置失败: {e}")
            raise
            
    def _load_paths_config(self) -> Dict[str, Any]:
        """加载路径配置"""
        paths_config_path = self.system_dir / 'paths.json'
        if not paths_config_path.exists():
            raise FileNotFoundError("路径配置文件不存在")
            
        try:
            with open(paths_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载路径配置失败: {e}")
            raise
            
    def save_system_config(self):
        """保存系统配置"""
        try:
            with open(self.system_dir / 'config.json', 'w') as f:
                json.dump(self.system_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存系统配置失败: {e}")
            raise
            
    def save_gui_config(self):
        """保存GUI配置"""
        try:
            with open(self.system_dir / 'gui_config.json', 'w') as f:
                json.dump(self.gui_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存GUI配置失败: {e}")
            raise
            
    def get_system_config(self, *keys: str) -> Any:
        """获取系统配置值"""
        return self._get_value(self.system_config, keys)
        
    def get_gui_config(self, *keys: str) -> Any:
        """获取GUI配置值"""
        return self._get_value(self.gui_config, keys)
        
    def get_path(self, *keys: str) -> Optional[Path]:
        """获取路径配置值"""
        path = self._get_value(self.paths_config, keys)
        if path is None:
            return None
        return self.project_root / path
        
    def _get_value(self, config: Dict[str, Any], keys: tuple) -> Optional[Any]:
        """从配置中获取值"""
        current = config
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return None
            
    def set_system_config(self, value: Any, *keys: str):
        """设置系统配置值"""
        self._set_value(self.system_config, value, keys)
        self.save_system_config()
        
    def set_gui_config(self, value: Any, *keys: str):
        """设置GUI配置值"""
        self._set_value(self.gui_config, value, keys)
        self.save_gui_config()
        
    def _set_value(self, config: Dict[str, Any], value: Any, keys: tuple):
        """设置配置值"""
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        
    @property
    def camera_settings(self) -> Dict[str, Any]:
        """获取相机设置"""
        return self.gui_config.get('camera_settings', {})
        
    @property
    def model_settings(self) -> Dict[str, Any]:
        """获取模型设置"""
        return self.gui_config.get('model_settings', {})
        
    @property
    def training_settings(self) -> Dict[str, Any]:
        """获取训练设置"""
        return self.gui_config.get('training_settings', {})
