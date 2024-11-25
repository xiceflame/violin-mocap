#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip3 install -r requirements.txt

# 运行主程序
python3 src/main.py
