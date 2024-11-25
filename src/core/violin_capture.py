"""
小提琴动作捕捉系统 - 主程序
"""
import cv2
import numpy as np
import logging
import time
import os
import sys
from ..utils.path_manager import PathManager

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .camera_manager import CameraManager
from .pose_detector import PoseDetector, ViolinKeypoints, BowKeypoints
from .data_recorder import DataRecorder
from .performance_monitor import PerformanceMonitor
from .network_streamer import NetworkStreamer
from .performance_optimizer import MSeries, AdaptivePerformance
from .auto_learning import AutoLearning

class ViolinCapture:
    def __init__(self, config=None):
        """初始化小提琴动作捕捉系统"""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 初始化路径管理器
        self.path_manager = PathManager()
        
        # 性能优化参数
        self.frame_interval = 1  # 自适应帧间隔
        self.last_process_time = 0
        self.target_fps = 30
        self.performance_stats = {
            'fps': 0,
            'process_time': 0,
            'detection_time': 0,
            'frame_count': 0
        }
        
        # 初始化核心模块
        self.camera_manager = CameraManager()
        self.pose_detector = PoseDetector(config)
        self.data_recorder = DataRecorder()
        self.performance_monitor = PerformanceMonitor()
        
        # 初始化可选模块（默认不启用）
        self.network_stream = None
        self.auto_learning = None
        
        # 结果缓存
        self.last_results = None
        self.results_timestamp = 0
        self.cache_timeout = 0.1  # 100ms缓存超时
        
        # 系统状态
        self.is_running = False
        self.current_frame = None
        
    def start(self):
        """启动系统"""
        if self.is_running:
            return
            
        # 初始化摄像头
        if not self.camera_manager.start():
            self.logger.error("初始化摄像头失败")
            return
            
        self.is_running = True
        self.logger.info("系统启动成功")
        
    def process_frame(self):
        """处理当前帧"""
        current_time = time.time()
        
        # 帧率控制
        if current_time - self.last_process_time < (1.0 / self.target_fps):
            return None
            
        # 获取帧
        ret, frame = self.camera_manager.get_frame()
        if not ret or frame is None:
            self.performance_monitor.update_stats(dropped=True)
            return None
            
        # 性能自适应
        process_start = time.time()
        frame_processed = False
        
        try:
            # 检查缓存是否可用
            if (self.last_results is not None and 
                current_time - self.results_timestamp < self.cache_timeout):
                results = self.last_results
            else:
                # 姿态检测
                results = self.pose_detector.detect_pose(frame)
                self.last_results = results
                self.results_timestamp = current_time
                frame_processed = True
            
            # 绘制结果
            processed_frame = frame.copy()
            processed_frame = self.pose_detector.draw_results(processed_frame, results)
            
            # 处理可选功能
            if results:
                # 网络流传输
                if self.network_stream and self.network_stream.is_streaming:
                    self.network_stream.send_data(results)
                    
                # 自动学习
                if self.auto_learning:
                    if self.auto_learning.collect_sample(frame, results):
                        if self.auto_learning.should_train():
                            self.auto_learning.train_model()
            
            # 更新性能统计
            process_time = time.time() - process_start
            self.performance_stats['process_time'] = process_time
            self.performance_stats['frame_count'] += 1
            
            # 自适应帧率调整
            if process_time > 1.0 / self.target_fps:
                self.frame_interval = min(self.frame_interval + 1, 3)
            else:
                self.frame_interval = max(self.frame_interval - 1, 1)
            
            self.last_process_time = current_time
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"处理帧时出错: {e}")
            return frame  # 发生错误时返回原始帧
            
        finally:
            # 更新FPS
            if frame_processed:
                elapsed = time.time() - process_start
                self.performance_stats['fps'] = 1.0 / elapsed if elapsed > 0 else 0
                
    def stop(self):
        """停止系统"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.camera_manager.release()
        self.pose_detector.release()
        self.network_streamer.close()
        self.logger.info("系统已停止")
        
    def toggle_auto_learning(self, enabled: bool):
        """切换自动学习状态"""
        self.auto_learning_enabled = enabled
        self.logger.info(f"自动学习已{'启用' if enabled else '禁用'}")
        
    def __del__(self):
        """析构函数"""
        self.stop()

    def capture_frame(self, frame, results):
        """采集单帧数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存图像
        image_path = os.path.join(self.output_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        
        # 保存姿态数据
        data = self.extract_pose_data(results)
        data['timestamp'] = timestamp
        
        json_path = os.path.join(self.output_dir, f"pose_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"已捕获帧 {timestamp}")

    def record_frame(self, results):
        """记录帧数据用于录制"""
        if results.pose_landmarks:
            frame_data = {
                'timestamp': time.time() - self.recording_start_time,
                'pose_data': self.extract_pose_data(results)
            }
            self.motion_data.append(frame_data)

    def save_recording(self):
        """保存录制的数据"""
        if not self.motion_data:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"recording_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.motion_data, f, indent=4)
            
        print(f"录制数据已保存到 {filename}")

    def extract_pose_data(self, results):
        """提取姿态数据"""
        data = {
            'pose_landmarks': None,
            'left_hand_landmarks': None,
            'right_hand_landmarks': None,
            'face_landmarks': None
        }
        
        if results.pose_landmarks:
            data['pose_landmarks'] = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            
        if results.left_hand_landmarks:
            data['left_hand_landmarks'] = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            
        if results.right_hand_landmarks:
            data['right_hand_landmarks'] = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            
        if results.face_landmarks:
            data['face_landmarks'] = [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
            
        return data

    def get_available_cameras(self):
        """获取可用的摄像头列表"""
        if hasattr(self, 'camera_manager'):
            return self.camera_manager.get_available_cameras()
        return [{'id': 0, 'name': 'Default Camera', 'model': 'Unknown', 'location': ''}]

    def list_cameras(self):
        """兼容性方法，调用get_available_cameras"""
        return self.get_available_cameras()

    def start_capture(self):
        """开始捕捉"""
        if not self.is_running:
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            self.logger.info("动作捕捉系统已启动")
            
    def _capture_loop(self):
        """捕捉循环"""
        while self.is_running:
            try:
                ret, frame = self.camera_manager.get_frame()
                if not ret:
                    self.logger.warning("无法读取摄像头画面")
                    continue
                    
                # 处理帧
                processed_frame = self.process_frame()
                
                # 存储当前帧
                self.current_frame = processed_frame
                    
            except Exception as e:
                self.logger.error(f"捕捉循环出错: {e}")
                break
                
        self.logger.info("捕捉循环已停止")
        
    def get_frame(self):
        """获取当前帧"""
        if not self.is_running or not hasattr(self, 'camera_manager'):
            return False, None
            
        try:
            # 获取并处理帧
            frame = self.process_frame()
            if frame is None:
                return False, None
                
            return True, frame
            
        except Exception as e:
            self.logger.error(f"获取帧失败: {e}")
            return False, None

    def change_camera(self, camera_id):
        """切换摄像头
        
        Args:
            camera_id: 摄像头ID或摄像头信息字典
        """
        try:
            # 如果传入的是字典，提取ID
            if isinstance(camera_id, dict):
                camera_id = camera_id.get('id', 0)
            # 确保camera_id是整数
            camera_id = int(camera_id)
            
            self.logger.info(f"切换到摄像头 {camera_id}")
            
            # 停止当前摄像头
            if hasattr(self, 'camera_manager'):
                self.camera_manager.stop()
                time.sleep(0.5)  # 等待资源释放
                
            # 创建新的摄像头管理器
            self.camera_manager = CameraManager(camera_id)
            
            # 尝试启动新摄像头
            if not self.camera_manager.start():
                raise RuntimeError(f"无法启动摄像头 {camera_id}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"无法初始化新摄像头: {str(e)}")
            # 如果切换失败，尝试回退到默认摄像头
            try:
                self.camera_manager = CameraManager(0)
                if self.camera_manager.start():
                    self.logger.info("已回退到默认摄像头")
                    return True
            except:
                pass
            return False

    def __del__(self):
        """清理资源"""
        self.stop()
            
    def release(self):
        """释放资源"""
        if self.is_running:
            self.camera_manager.stop()
            self.is_running = False

    def toggle_streaming(self):
        """切换实时流数据状态"""
        self.is_streaming = not self.is_streaming
        status = "开启" if self.is_streaming else "关闭"
        print(f"实时数据流已{status}")

    def send_to_ue(self, landmarks_data):
        """发送数据到UE"""
        try:
            # 将数据转换为紧凑的格式
            data = {
                'timestamp': time.time(),
                'landmarks': landmarks_data
            }
            json_data = json.dumps(data).encode('utf-8')
            self.udp_socket.sendto(json_data, self.ue_address)
        except Exception as e:
            print(f"发送数据失败: {str(e)}")

    def init_camera(self):
        """初始化摄像头"""
        try:
            # 获取可用摄像头列表
            available_cameras = self.camera_manager.get_available_cameras()
            if not available_cameras:
                self.logger.error("未找到可用的摄像头")
                return False
                
            # 使用第一个可用的摄像头
            camera_id = available_cameras[0]['id']
            self.camera_manager = CameraManager(camera_id)
            
            if not self.camera_manager.start():
                self.logger.error(f"无法打开摄像头 {camera_id}")
                return False
                
            self.current_camera_id = camera_id
            self.logger.info(f"已初始化摄像头 {camera_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化摄像头失败: {e}")
            return False
            
    def toggle_data_collection_mode(self):
        """切换数据收集模式"""
        self.data_collection_mode = not self.data_collection_mode
        if self.data_collection_mode:
            self.collection_config['current_pose'] = 0
            self.collection_config['pose_start_time'] = time.time()
            self.collection_config['current_session'] += 1
            print("\nData Collection Mode Started!")
            print(f"Session {self.collection_config['current_session']}")
            print(f"Current pose: {self.collection_config['poses'][0]}")
            print(f"Duration: {self.collection_config['pose_duration']} seconds")
        else:
            print("\nData Collection Mode Stopped!")
            self.save_dataset_info()

    def toggle_auto_capture(self):
        """切换自动拍摄模式"""
        self.collection_config['auto_capture'] = not self.collection_config['auto_capture']
        status = "enabled" if self.collection_config['auto_capture'] else "disabled"
        print(f"\nAuto capture {status}")
        print(f"Interval: {self.collection_config['capture_interval']} seconds")

    def save_dataset_info(self):
        """保存数据集信息"""
        dataset_path = self.path_manager.get_path('data', 'dataset')
        if dataset_path is None:
            self.logger.error("无法获取数据集路径")
            return
            
        info = {
            'total_images': self.total_captures,
            'date_created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'classes': ['violin', 'bow'],
            'poses': self.pose_list
        }
        
        info_path = dataset_path / 'dataset_info.json'
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
            
        # 创建YOLO格式的dataset.yaml
        yaml_content = f"""
# Dataset Path
path: {str(dataset_path)}

# Classes
names:
  0: violin
  1: bow

# Number of classes
nc: 2
"""
        yaml_path = dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())

    def capture_training_image(self, frame, force=False):
        """捕获训练图像"""
        current_time = time.time()
        
        # 检查是否应该捕获图像
        should_capture = force or (
            self.collection_config['auto_capture'] and 
            current_time - self.collection_config['last_capture_time'] >= self.collection_config['capture_interval']
        )
        
        if should_capture:
            # 保存图像
            timestamp = int(time.time() * 1000)
            session = self.collection_config['current_session']
            pose = self.collection_config['current_pose']
            
            image_name = f"session_{session:02d}_pose_{pose:02d}_{timestamp}.jpg"
            label_name = f"session_{session:02d}_pose_{pose:02d}_{timestamp}.txt"
            
            image_path = os.path.join(self.images_dir, image_name)
            label_path = os.path.join(self.labels_dir, label_name)
            
            # 保存图像
            cv2.imwrite(image_path, frame)
            
            # 保存标签（如果检测到物体）
            if self.violin_keypoints.scroll is not None or self.bow_keypoints.tip is not None:
                with open(label_path, 'w') as f:
                    if self.violin_keypoints.scroll is not None:
                        # 转换为YOLO格式
                        x = (self.violin_keypoints.scroll[0] + self.violin_keypoints.lower_left[0]) / 2 / frame.shape[1]
                        y = (self.violin_keypoints.scroll[1] + self.violin_keypoints.lower_left[1]) / 2 / frame.shape[0]
                        w = (self.violin_keypoints.lower_left[0] - self.violin_keypoints.scroll[0]) / frame.shape[1]
                        h = (self.violin_keypoints.lower_left[1] - self.violin_keypoints.scroll[1]) / frame.shape[0]
                        f.write(f'0 {x} {y} {w} {h}\n')
                    
                    if self.bow_keypoints.tip is not None:
                        x = (self.bow_keypoints.tip[0] + self.bow_keypoints.frog[0]) / 2 / frame.shape[1]
                        y = (self.bow_keypoints.tip[1] + self.bow_keypoints.frog[1]) / 2 / frame.shape[0]
                        w = (self.bow_keypoints.frog[0] - self.bow_keypoints.tip[0]) / frame.shape[1]
                        h = (self.bow_keypoints.frog[1] - self.bow_keypoints.tip[1]) / frame.shape[0]
                        f.write(f'1 {x} {y} {w} {h}\n')
            
            self.collection_config['total_captures'] += 1
            self.collection_config['last_capture_time'] = current_time
            
            return True
        return False

    def update_data_collection_state(self):
        """更新数据收集状态"""
        if not self.data_collection_mode:
            return
        
        current_time = time.time()
        pose_elapsed = current_time - self.collection_config['pose_start_time']
        
        # 检查是否需要切换到下一个姿势
        if pose_elapsed >= self.collection_config['pose_duration']:
            self.collection_config['current_pose'] += 1
            if self.collection_config['current_pose'] >= len(self.collection_config['poses']):
                self.collection_config['current_pose'] = 0
                print("\nCompleted all poses! Starting next round...")
            
            self.collection_config['pose_start_time'] = current_time
            print(f"\nNext pose: {self.collection_config['poses'][self.collection_config['current_pose']]}")
            print(f"Duration: {self.collection_config['pose_duration']} seconds")

    def get_available_cameras(self):
        """获取可用的摄像头列表"""
        if hasattr(self, 'camera_manager'):
            return self.camera_manager.get_available_cameras()
        return [{'id': 0, 'name': 'Default Camera', 'model': 'Unknown', 'location': ''}]

    def _initialize_yolo_model(self):
        """初始化YOLO模型"""
        try:
            # 获取M系列优化配置
            m_series_params = configure_m_series()
            inference_settings = optimize_inference_settings()
            
            # 首先尝试加载自定义的小提琴检测模型
            model_path = os.path.join(os.path.dirname(__file__), '../../models/violin_detector.pt')
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                self.logger.info("已加载自定义小提琴检测模型")
            else:
                # 如果没有自定义模型，使用最大的预训练模型以获得最高精度
                self.yolo_model = YOLO('yolov8x.pt')  # 使用最大模型
                # 设置小提琴类别ID（在COCO数据集中为77）
                self.violin_class_id = 77
                self.logger.info("已加载预训练YOLO-X模型")
            
            # 应用M系列优化配置
            if m_series_params:
                self.yolo_conf = m_series_params['conf']
                self.yolo_iou = m_series_params['iou']
                self.inference_size = (m_series_params['imgsz'], m_series_params['imgsz'])
                
                # 配置模型以获得最佳性能
                if inference_settings:
                    self.yolo_model.fuse()  # 模型融合优化
                    if torch.backends.mps.is_available():
                        self.yolo_model.to('mps')  # 使用Apple M2 GPU
                        self.logger.info("已启用MPS加速")
            else:
                # 使用默认配置
                self.yolo_conf = 0.3
                self.yolo_iou = 0.7
                self.inference_size = (640, 640)
            
        except Exception as e:
            self.logger.error(f"YOLO模型初始化失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.yolo_model = None

    def _initialize_keypoint_model(self):
        """初始化关键点检测模型"""
        try:
            model_path = 'models/violin_keypoints.pt'
            if os.path.exists(model_path):
                self.keypoint_model = YOLO(model_path)
                self.logger.info("成功加载关键点检测模型")
            else:
                self.logger.info("未找到专门训练的关键点检测模型，将使用MediaPipe进行姿态检测")
                self.keypoint_model = None
        except Exception as e:
            self.logger.info(f"加载关键点检测模型失败: {e}，将使用MediaPipe进行姿态检测")
            self.keypoint_model = None

    def reload_models(self, config):
        """重新加载模型"""
        try:
            # 释放现有模型
            self.release_models()
            
            # 更新配置
            self.config = config
            
            # 重新加载模型
            self._load_models()
            
            return True
            
        except Exception as e:
            self.logger.error(f"重新加载模型失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def release_models(self):
        """释放模型资源"""
        try:
            if hasattr(self, 'yolo_model') and self.yolo_model is not None:
                del self.yolo_model
                self.yolo_model = None
                
            if hasattr(self, 'keypoint_model') and self.keypoint_model is not None:
                del self.keypoint_model
                self.keypoint_model = None
                
            self.models_info = {}
            
        except Exception as e:
            self.logger.error(f"释放模型资源失败: {e}")

    def release(self):
        """释放所有资源"""
        try:
            # 释放模型
            self.release_models()
            
            # 释放摄像头
            if hasattr(self, 'camera_manager') and self.camera_manager is not None:
                self.camera_manager.stop()
                
            # 释放MediaPipe资源
            if hasattr(self, 'holistic'):
                self.holistic.close()
                
        except Exception as e:
            self.logger.error(f"释放资源失败: {e}")

    def _init_cache(self):
        """初始化缓存系统"""
        self.frame_cache = {}
        self.detection_cache = {}
        self.keypoint_cache = {}
        self.cache_size = self.config.get('cache_size', 30)  # 默认缓存30帧
        
    def _update_performance_stats(self, frame_time=None, detection_time=None, keypoint_time=None):
        """更新性能统计信息"""
        current_time = time.time()
        
        # 更新帧处理时间
        if frame_time is not None:
            self.perf_stats['frame_times'].append(frame_time)
            if len(self.perf_stats['frame_times']) > 100:
                self.perf_stats['frame_times'].pop(0)
        
        # 更新检测时间
        if detection_time is not None:
            self.perf_stats['detection_times'].append(detection_time)
            if len(self.perf_stats['detection_times']) > 100:
                self.perf_stats['detection_times'].pop(0)
        
        # 更新关键点检测时间
        if keypoint_time is not None:
            self.perf_stats['keypoint_times'].append(keypoint_time)
            if len(self.perf_stats['keypoint_times']) > 100:
                self.perf_stats['keypoint_times'].pop(0)
        
        # 更新FPS
        if current_time - self.perf_stats['last_fps_update'] >= 1.0:
            self.perf_stats['current_fps'] = len(self.perf_stats['frame_times'])
            self.perf_stats['last_fps_update'] = current_time
            
    def get_performance_stats(self):
        """获取性能统计信息"""
        stats = {
            'fps': self.perf_stats['current_fps'],
            'avg_frame_time': sum(self.perf_stats['frame_times']) / len(self.perf_stats['frame_times']) if self.perf_stats['frame_times'] else 0,
            'avg_detection_time': sum(self.perf_stats['detection_times']) / len(self.perf_stats['detection_times']) if self.perf_stats['detection_times'] else 0,
            'avg_keypoint_time': sum(self.perf_stats['keypoint_times']) / len(self.perf_stats['keypoint_times']) if self.perf_stats['keypoint_times'] else 0,
            'total_frames': self.perf_stats['total_frames'],
            'dropped_frames': self.perf_stats['dropped_frames']
        }
        return stats
        
    def _draw_performance_info(self, frame):
        """在帧上绘制性能信息"""
        try:
            stats = self.get_performance_stats()
            info_text = [
                f"FPS: {stats['fps']:.1f}",
                f"Frame Time: {stats['avg_frame_time']*1000:.1f}ms",
                f"Detection: {stats['avg_detection_time']*1000:.1f}ms",
                f"Keypoint: {stats['avg_keypoint_time']*1000:.1f}ms"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                           
        except Exception as e:
            self.logger.error(f"绘制性能信息失败: {e}")

    def start(self):
        """启动动作捕捉系统"""
        try:
            # 初始化相机
            if not self.camera_manager.start():
                raise RuntimeError("无法初始化摄像头")
                
            self.is_running = True
            self.logger.info("动作捕捉系统已启动")
            
        except Exception as e:
            self.logger.error(f"启动失败: {e}")
            raise

    def stop(self):
        """停止动作捕捉系统"""
        if self.is_running:
            self.is_running = False
            if hasattr(self, 'camera_manager') and self.camera_manager is not None:
                self.camera_manager.stop()
            self.logger.info("动作捕捉系统已停止")

    def init_camera(self):
        """初始化摄像头"""
        try:
            # 获取可用摄像头列表
            available_cameras = self.camera_manager.get_available_cameras()
            if not available_cameras:
                self.logger.error("未找到可用的摄像头")
                return False
                
            # 使用第一个可用的摄像头
            camera_id = available_cameras[0]['id']
            self.camera_manager = CameraManager(camera_id)
            
            if not self.camera_manager.start():
                self.logger.error(f"无法打开摄像头 {camera_id}")
                return False
                
            self.current_camera_id = camera_id
            self.logger.info(f"已初始化摄像头 {camera_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化摄像头失败: {e}")
            return False

    def enable_network_stream(self, host="localhost", port=8765):
        """启用网络流功能"""
        if self.network_stream is None:
            from .network_stream import NetworkStream
            self.network_stream = NetworkStream(host, port)
            self.network_stream.start()
            
    def disable_network_stream(self):
        """禁用网络流功能"""
        if self.network_stream is not None:
            self.network_stream.stop()
            self.network_stream = None
            
    def enable_auto_learning(self, **kwargs):
        """启用自动学习功能"""
        if self.auto_learning is None:
            from .auto_learning import AutoLearning
            self.auto_learning = AutoLearning(**kwargs)
            
    def disable_auto_learning(self):
        """禁用自动学习功能"""
        if self.auto_learning is not None:
            self.auto_learning = None

    def set_yolo_conf(self, conf: float):
        """设置YOLO置信度阈值"""
        if hasattr(self.pose_detector, 'set_yolo_params'):
            self.pose_detector.set_yolo_params(conf=conf)
            
    def set_yolo_iou(self, iou: float):
        """设置YOLO IOU阈值"""
        if hasattr(self.pose_detector, 'set_yolo_params'):
            self.pose_detector.set_yolo_params(iou=iou)

def select_cameras():
    """让用户选择要使用的摄像头"""
    capture = ViolinCapture()
    available_cameras = capture.list_cameras()
    
    if not available_cameras:
        print("Error: No cameras found")
        return []
    
    print("\nAvailable cameras:")
    for cam in available_cameras:
        print(f"[{cam['id']}] {cam['name']} - Resolution: {cam['resolution']}, FPS: {cam['fps']}")
    
    while True:
        try:
            selection = input("\nSelect camera ID (or press Enter for default camera 0): ")
            if selection.strip() == "":
                return [0]
            
            selected_id = int(selection)
            valid_ids = [cam['id'] for cam in available_cameras]
            
            if selected_id in valid_ids:
                return [selected_id]
            else:
                print("Error: Invalid camera ID, please try again")
        except ValueError:
            print("Error: Please enter a valid number")
        except EOFError:
            return [0]

def main():
    try:
        camera_ids = select_cameras()
        for camera_id in camera_ids:
            capture = ViolinCapture()
            capture.start_capture(camera_id)
    except Exception as e:
        print(f"程序发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
