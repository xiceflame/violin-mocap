import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class PathManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PathManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.paths_config = self._load_paths_config()
        
    def _load_paths_config(self) -> Dict[str, Any]:
        """Load paths configuration from the paths.json file."""
        config_path = self.project_root / 'configs' / 'system' / 'paths.json'
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading paths config: {e}")
            return {}
            
    def get_path(self, *keys: str) -> Optional[Path]:
        """Get a path from the configuration using dot notation.
        
        Args:
            *keys: The keys to navigate the paths configuration.
            
        Returns:
            Path object if the path exists, None otherwise.
            
        Example:
            >>> path_manager = PathManager()
            >>> model_path = path_manager.get_path('models', 'pretrained', 'detection', 'yolo', 'yolov8n')
        """
        current = self.paths_config
        try:
            for key in keys:
                current = current[key]
            if isinstance(current, str):
                return self.project_root / current
            return None
        except (KeyError, TypeError):
            return None
            
    def get_absolute_path(self, path: str) -> Path:
        """Convert a relative path to absolute path based on project root."""
        return self.project_root / path
        
    @property
    def root(self) -> Path:
        """Get the project root directory."""
        return self.project_root
