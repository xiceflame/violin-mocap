# Violin Motion Capture System - 设计文档

## 1. 系统架构

### 1.1 目录结构
```
violin_mocap_test/
├── configs/            # 配置文件
├── data/              # 数据存储
│   ├── training/      # 训练数据
│   ├── sessions/      # 捕捉会话
│   └── annotations/   # 标注数据
├── logs/              # 日志文件
├── models/            # 模型文件
├── src/               # 源代码
│   ├── annotation/    # 标注工具
│   ├── core/          # 核心功能
│   ├── gui/           # 图形界面
│   ├── networking/    # 网络功能
│   ├── training/      # 训练相关
│   └── utils/         # 工具函数
├── tests/             # 测试代码
└── venv/              # 虚拟环境
```

### 1.2 核心组件
- ViolinCapture: 系统核心控制器
- PoseDetector: 姿态检测系统
- CameraManager: 摄像头管理
- DataRecorder: 数据记录系统
- NetworkStreamer: 网络传输
- AutoLearning: 自动学习系统
- PerformanceMonitor: 性能监控

## 2. 功能模块详解

### 2.1 核心功能 (src/core/)

#### 2.1.1 ViolinCapture
- 功能：系统核心控制器
- 职责：
  * 协调各组件工作
  * 处理帧捕获流程
  * 管理系统状态
  * 性能优化控制

#### 2.1.2 PoseDetector
- 功能：姿态检测系统
- 技术栈：
  * MediaPipe Holistic
  * YOLOv8
- 检测内容：
  * 人体姿态
  * 手部动作
  * 乐器位置

#### 2.1.3 CameraManager
- 功能：摄像头管理
- 特性：
  * 多摄像头支持
  * 自动设备发现
  * 参数配置
  * 异常处理

### 2.2 自动学习系统 (src/training/)

#### 2.2.1 AutoLearning
- 功能：自动数据收集和模型优化
- 工作流程：
  1. 高置信度样本收集
  2. 自动标注生成
  3. 增量训练
  4. 模型评估

#### 2.2.2 ModelTrainer
- 功能：模型训练管理
- 特性：
  * GPU加速支持
  * 训练过程监控
  * 自动评估
  * 模型版本管理

### 2.3 数据管理 (src/core/data_recorder.py)

#### 2.3.1 DataRecorder
- 功能：数据记录和管理
- 数据类型：
  * 原始视频帧
  * 检测结果
  * 标注数据
  * 训练数据
- 存储结构：
  * 按会话组织
  * 自动备份
  * 数据验证

### 2.4 标注系统 (src/annotation/)

#### 2.4.1 ViolinKeypointAnnotator
- 功能：关键点标注工具
- 标注类型：
  * 小提琴关键点
  * 琴弓关键点
  * 手部关键点
- 特性：
  * 实时预览
  * 批量处理
  * 标注验证

### 2.5 性能优化 (src/core/performance_monitor.py)

#### 2.5.1 PerformanceMonitor
- 功能：性能监控和优化
- 监控指标：
  * FPS
  * 处理延迟
  * CPU使用率
  * GPU使用率
  * 内存使用
- 优化策略：
  * 自适应帧率
  * 资源分配
  * 并行处理

### 2.6 网络传输 (src/networking/)

#### 2.6.1 NetworkStreamer
- 功能：实时数据传输
- 特性：
  * UDP高效传输
  * 数据压缩
  * 丢包处理
  * 延迟控制

### 2.7 用户界面 (src/gui/)

#### 2.7.1 ViolinMocapGUI
- 功能：主界面系统
- 组件：
  * 实时预览
  * 参数配置
  * 数据管理
  * 训练控制
- 特性：
  * 多标签页设计
  * 实时更新
  * 状态监控

## 3. 技术栈

### 3.1 核心依赖
- Python 3.11+
- PyQt6: GUI框架
- OpenCV: 图像处理
- MediaPipe: 姿态估计
- YOLOv8: 物体检测
- PyTorch: 深度学习
- NumPy: 数值计算

### 3.2 性能优化
- MPS (Metal Performance Shaders)
- 多线程处理
- GPU加速
- 内存优化

## 4. 部署说明

### 4.1 环境要求
- macOS (支持Metal)
- Python 3.11+
- 足够的GPU内存
- 稳定的网络连接

### 4.2 安装步骤
1. 创建虚拟环境
2. 安装依赖
3. 下载预训练模型
4. 配置系统参数

### 4.3 配置说明
- 摄像头设置
- 检测参数
- 网络设置
- 性能参数

## 5. 开发规范

### 5.1 代码规范
- PEP 8
- 类型注解
- 文档字符串
- 单元测试

### 5.2 Git工作流
- 主分支：main
- 开发分支：dev
- 功能分支：feature/*
- 修复分支：bugfix/*

### 5.3 版本控制
- 语义化版本
- 更新日志
- 发布说明

## 6. 未来规划

### 6.1 功能增强
- 3D姿态重建
- 多人跟踪
- 动作分析
- 云端集成

### 6.2 性能优化
- 深度优化
- 算法改进
- 资源利用
- 网络优化

### 6.3 用户体验
- 界面优化
- 操作流程
- 反馈机制
- 帮助系统
