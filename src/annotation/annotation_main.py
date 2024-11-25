import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import QApplication
from src.annotation.annotation_tool import ViolinKeypointAnnotator

def main():
    app = QApplication(sys.argv)
    window = ViolinKeypointAnnotator()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
