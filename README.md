# Violin Motion Capture System

A comprehensive motion tracking solution for analyzing violin performance movements, combining MediaPipe Holistic tracking with YOLOv8 object detection.

[English](#english) | [中文](#中文)

---

## English

### 🎯 Overview

This system provides real-time motion capture capabilities specifically designed for violin performance analysis. It combines multiple computer vision technologies to track body posture, hand movements, and instrument position simultaneously.

### ✨ Key Features

#### Motion Tracking
- Full body pose estimation using MediaPipe Holistic
- Precise hand movement tracking with landmark smoothing
- Real-time angle calculations and performance metrics
- Violin and bow position tracking using YOLOv8
- Auto-learning capability for improved tracking

#### Data Management
- Comprehensive data recording system
- Automatic dataset generation with annotations
- Performance monitoring and analysis
- Flexible data export formats

#### User Interface
- Modern PyQt6-based GUI with tab organization
- Multi-camera support with device switching
- Real-time visualization with performance metrics
- Interactive annotation system
- Training progress monitoring

### 🛠 Technical Requirements

#### Software Requirements
- Python 3.11+
- PyQt6
- OpenCV
- MediaPipe
- YOLOv8 (Ultralytics)
- PyTorch with MPS support

#### Hardware Requirements
- Webcam (1080p recommended)
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 8GB minimum, 16GB recommended
- Apple Silicon (M1/M2/M3) or Intel Mac
- Storage: 2GB for software and models

### 📦 Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd violin_mocap_test
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Model Downloads

The YOLO models are not included in this repository due to their size. Please download them manually:

1. Download YOLOv8x model:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -P models/pretrained/detection/yolo/
   ```

2. Download YOLOv8n model (optional, for testing):
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/pretrained/detection/yolo/
   ```

### 🚀 Usage Guide

#### Basic Operation
1. Start the program:
```bash
python main.py
```

2. Main Interface Features:
- Camera Selection: Choose between available cameras
- Start/Stop Capture: Begin or end motion tracking
- Performance Metrics: View real-time tracking statistics
- Settings: Adjust detection and tracking parameters

#### Annotation Mode
1. Select the Annotation tab
2. Load image directory
3. Use the annotation tools to mark keypoints
4. Save annotations for training

#### Training Mode
1. Navigate to the Training tab
2. Select dataset directory
3. Configure training parameters
4. Monitor training progress

### 🎓 Development Guide

#### Project Structure
```
violin_mocap_test/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   │   ├── violin_capture.py    # Main capture system
│   │   ├── pose_estimation.py   # Pose tracking
│   │   ├── object_detection.py  # YOLO detection
│   │   ├── camera_manager.py    # Camera handling
│   │   └── data_recorder.py     # Data recording
│   ├── gui/               # User interface
│   │   ├── main_window.py      # Main GUI
│   │   └── tabs/              # Interface tabs
│   ├── annotation/        # Annotation tools
│   ├── training/         # Model training
│   ├── networking/       # Network features
│   └── utils/           # Utility functions
├── models/              # Model storage
├── data/               # Data directory
└── configs/            # Configurations
```

#### Key Components
1. Core System
   - Multi-device camera management
   - Real-time pose estimation pipeline
   - Object detection integration
   - Performance optimization

2. User Interface
   - Main capture interface
   - Annotation system
   - Training management
   - Performance monitoring

3. Data Processing
   - Real-time data recording
   - Keypoint smoothing
   - Performance metrics calculation
   - Dataset management

### 🔧 Developer Guide

#### 1. 代码规范和架构

##### 代码组织
- 模块化设计：
  * 核心功能模块解耦
  * 接口定义清晰
  * 依赖注入模式
- 设计模式：
  * 观察者模式：用于UI更新和数据同步
  * 工厂模式：创建检测器和估计器
  * 策略模式：切换不同的处理算法
  * 单例模式：管理全局资源

##### 代码质量
- 类型注解：
  * 使用Python类型提示
  * mypy静态类型检查
  * 文档字符串规范
- 错误处理：
  * 异常层次结构
  * 日志记录系统
  * 错误恢复机制

##### 测试策略
- 单元测试：
  * pytest测试框架
  * 模拟对象和依赖注入
  * 参数化测试
- 集成测试：
  * 端到端测试
  * 性能基准测试
  * 并发测试

#### 2. 配置管理

##### 配置系统
- 分层配置：
  * 默认配置
  * 环境配置
  * 用户配置
  * 运行时配置
- 配置验证：
  * JSON Schema验证
  * 类型检查
  * 默认值处理

##### 环境管理
- 虚拟环境：
  * 环境隔离
  * 依赖版本控制
  * 开发/生产环境分离
- 依赖管理：
  * requirements.txt
  * setup.py配置
  * 版本约束

#### 3. 性能优化

##### 代码优化
- 性能分析：
  * cProfile性能分析
  * 内存使用监控
  * 并发瓶颈分析
- 优化策略：
  * 算法优化
  * 缓存机制
  * 并发处理

##### 资源管理
- 内存管理：
  * 内存池
  * 垃圾回收优化
  * 大对象处理
- 并发控制：
  * 线程池
  * 异步处理
  * 资源锁定

#### 4. 持续集成/持续部署

##### CI/CD流程
- 代码检查：
  * pylint代码分析
  * 代码覆盖率检查
  * 风格检查
- 自动化测试：
  * 单元测试自动化
  * 集成测试自动化
  * 性能测试自动化

##### 部署流程
- 版本控制：
  * 语义化版本
  * 更新日志
  * 发布说明
- 部署自动化：
  * 环境配置
  * 依赖安装
  * 系统检查

#### 5. 监控和维护

##### 监控系统
- 性能监控：
  * CPU使用率
  * 内存占用
  * 帧率统计
- 错误监控：
  * 异常捕获
  * 错误报告
  * 状态检查

##### 日志系统
- 日志配置：
  * 分级日志
  * 日志轮转
  * 格式化输出
- 日志分析：
  * 错误追踪
  * 性能分析
  * 用户行为分析

#### 6. 文档管理

##### 代码文档
- API文档：
  * 函数文档
  * 类文档
  * 模块文档
- 架构文档：
  * 系统设计
  * 数据流程
  * 接口定义

##### 用户文档
- 安装指南
- 使用手册
- 故障排除
- 最佳实践

#### 7. 安全性

##### 数据安全
- 数据加密：
  * 配置文件加密
  * 敏感数据保护
  * 安全传输
- 访问控制：
  * 权限管理
  * 身份验证
  * 会话控制

##### 代码安全
- 依赖检查：
  * 安全更新
  * 漏洞扫描
  * 依赖审计
- 代码审查：
  * 安全最佳实践
  * 代码注入防护
  * 错误处理审查

### 📈 Performance Features

1. Hardware Acceleration
   - MPS (Metal Performance Shaders) support
   - Neural Engine optimization
   - Multi-core CPU utilization
   - Adaptive performance management

2. Optimization Features
   - Real-time performance monitoring
   - Automatic quality adjustment
   - Memory usage optimization
   - Frame rate management

### 🍎 Mac M1/M2/M3 Optimization

1. Hardware Acceleration
   - Native ARM64 support
   - PyTorch MPS acceleration
   - Neural Engine utilization
   - Multi-core optimization

2. Performance Modes
   - Balanced mode for battery efficiency
   - Performance mode for maximum speed
   - Quality mode for best tracking

---

## 中文

### 🎯 系统概述

小提琴动作捕捉系统是一个专门用于分析小提琴演奏动作的实时动作捕捉解决方案。系统结合了多种计算机视觉技术，可以同时追踪身体姿势、手部运动和乐器位置。

### ✨ 主要功能

#### 动作追踪
- 使用 MediaPipe Holistic 进行全身姿势估计
- 精确的手部运动追踪与平滑处理
- 实时角度计算和性能指标分析
- 使用 YOLOv8 追踪小提琴和琴弓位置
- 自动学习功能提升追踪效果

#### 数据管理
- 完整的数据记录系统
- 自动数据集生成与标注
- 性能监控和分析
- 灵活的数据导出格式

#### 用户界面
- 基于 PyQt6 的现代化界面，标签页组织
- 多摄像头支持，设备切换
- 实时可视化与性能指标
- 交互式标注系统
- 训练进度监控

### 🛠 技术要求

#### 软件要求
- Python 3.11+
- PyQt6
- OpenCV
- MediaPipe
- YOLOv8 (Ultralytics)
- 支持 MPS 的 PyTorch

#### 硬件要求
- 摄像头（推荐 1080p）
- CPU：Intel i5/AMD Ryzen 5 或更好
- 内存：最少 8GB，推荐 16GB
- Apple Silicon (M1/M2/M3) 或 Intel Mac
- 存储：软件和模型需要 2GB

### 📦 安装说明

1. 克隆仓库：
```bash
git clone [repository-url]
cd violin_mocap_test
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # MacOS
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

### 模型下载

YOLO 模型由于体积较大，不包含在仓库中。请手动下载：

1. 下载 YOLOv8x 模型：
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -P models/pretrained/detection/yolo/
   ```

2. 下载 YOLOv8n 模型（可选，用于测试）：
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/pretrained/detection/yolo/
   ```

### 🚀 使用指南

#### 基本操作
1. 启动程序：
```bash
python main.py
```

2. 主界面功能：
- 摄像头选择：选择可用摄像头
- 开始/停止捕捉：控制动作追踪
- 性能指标：查看实时追踪统计
- 设置：调整检测和追踪参数

#### 标注模式
1. 选择标注标签页
2. 加载图像目录
3. 使用标注工具标记关键点
4. 保存标注用于训练

#### 训练模式
1. 切换到训练标签页
2. 选择数据集目录
3. 配置训练参数
4. 监控训练进度

### 🎓 开发指南

#### 项目结构
```
violin_mocap_test/
├── src/                    # 源代码
│   ├── core/              # 核心功能
│   │   ├── violin_capture.py    # 主捕捉系统
│   │   ├── pose_estimation.py   # 姿势追踪
│   │   ├── object_detection.py  # YOLO检测
│   │   ├── camera_manager.py    # 摄像头管理
│   │   └── data_recorder.py     # 数据记录
│   ├── gui/               # 用户界面
│   │   ├── main_window.py      # 主界面
│   │   └── tabs/              # 界面标签页
│   ├── annotation/        # 标注工具
│   ├── training/         # 模型训练
│   ├── networking/       # 网络功能
│   └── utils/           # 工具函数
├── models/              # 模型存储
├── data/               # 数据目录
└── configs/            # 配置文件
```

#### 核心组件
1. 核心系统
   - 多设备摄像头管理
   - 实时姿势估计流程
   - 目标检测集成
   - 性能优化

2. 用户界面
   - 主捕捉界面
   - 标注系统
   - 训练管理
   - 性能监控

3. 数据处理
   - 实时数据记录
   - 关键点平滑
   - 性能指标计算
   - 数据集管理

### 🔧 开发者指南

#### 1. 代码规范和架构

##### 代码组织
- 模块化设计：
  * 核心功能模块解耦
  * 接口定义清晰
  * 依赖注入模式
- 设计模式：
  * 观察者模式：用于UI更新和数据同步
  * 工厂模式：创建检测器和估计器
  * 策略模式：切换不同的处理算法
  * 单例模式：管理全局资源

##### 代码质量
- 类型注解：
  * 使用Python类型提示
  * mypy静态类型检查
  * 文档字符串规范
- 错误处理：
  * 异常层次结构
  * 日志记录系统
  * 错误恢复机制

##### 测试策略
- 单元测试：
  * pytest测试框架
  * 模拟对象和依赖注入
  * 参数化测试
- 集成测试：
  * 端到端测试
  * 性能基准测试
  * 并发测试

#### 2. 配置管理

##### 配置系统
- 分层配置：
  * 默认配置
  * 环境配置
  * 用户配置
  * 运行时配置
- 配置验证：
  * JSON Schema验证
  * 类型检查
  * 默认值处理

##### 环境管理
- 虚拟环境：
  * 环境隔离
  * 依赖版本控制
  * 开发/生产环境分离
- 依赖管理：
  * requirements.txt
  * setup.py配置
  * 版本约束

#### 3. 性能优化

##### 代码优化
- 性能分析：
  * cProfile性能分析
  * 内存使用监控
  * 并发瓶颈分析
- 优化策略：
  * 算法优化
  * 缓存机制
  * 并发处理

##### 资源管理
- 内存管理：
  * 内存池
  * 垃圾回收优化
  * 大对象处理
- 并发控制：
  * 线程池
  * 异步处理
  * 资源锁定

#### 4. 持续集成/持续部署

##### CI/CD流程
- 代码检查：
  * pylint代码分析
  * 代码覆盖率检查
  * 风格检查
- 自动化测试：
  * 单元测试自动化
  * 集成测试自动化
  * 性能测试自动化

##### 部署流程
- 版本控制：
  * 语义化版本
  * 更新日志
  * 发布说明
- 部署自动化：
  * 环境配置
  * 依赖安装
  * 系统检查

#### 5. 监控和维护

##### 监控系统
- 性能监控：
  * CPU使用率
  * 内存占用
  * 帧率统计
- 错误监控：
  * 异常捕获
  * 错误报告
  * 状态检查

##### 日志系统
- 日志配置：
  * 分级日志
  * 日志轮转
  * 格式化输出
- 日志分析：
  * 错误追踪
  * 性能分析
  * 用户行为分析

#### 6. 文档管理

##### 代码文档
- API文档：
  * 函数文档
  * 类文档
  * 模块文档
- 架构文档：
  * 系统设计
  * 数据流程
  * 接口定义

##### 用户文档
- 安装指南
- 使用手册
- 故障排除
- 最佳实践

#### 7. 安全性

##### 数据安全
- 数据加密：
  * 配置文件加密
  * 敏感数据保护
  * 安全传输
- 访问控制：
  * 权限管理
  * 身份验证
  * 会话控制

##### 代码安全
- 依赖检查：
  * 安全更新
  * 漏洞扫描
  * 依赖审计
- 代码审查：
  * 安全最佳实践
  * 代码注入防护
  * 错误处理审查

### 📈 性能特性

1. 硬件加速
   - MPS (Metal Performance Shaders) 支持
   - 神经引擎优化
   - 多核 CPU 利用
   - 自适应性能管理

2. 优化功能
   - 实时性能监控
   - 自动质量调节
   - 内存使用优化
   - 帧率管理

### 🍎 Mac M1/M2/M3 优化

1. 硬件加速
   - 原生 ARM64 支持
   - PyTorch MPS 加速
   - 神经引擎利用
   - 多核心优化

2. 性能模式
   - 平衡模式：节省电池
   - 性能模式：最大速度
   - 质量模式：最佳追踪
