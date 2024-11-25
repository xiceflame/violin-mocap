"""
数据记录器模块
"""
import os
import cv2
import json
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

class DataRecorder:
    """数据记录器类"""
    def __init__(self, base_dir: str = "data"):
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir
        self.current_session = None
        self.frame_count = 0
        
        # 创建基础目录
        self._create_directories()
        
    def _create_directories(self):
        """创建必要的目录结构"""
        try:
            # 创建主目录
            os.makedirs(self.base_dir, exist_ok=True)
            
            # 创建子目录
            subdirs = ['images', 'annotations', 'models', 'logs']
            for subdir in subdirs:
                os.makedirs(os.path.join(self.base_dir, subdir), exist_ok=True)
                
            self.logger.info("数据目录结构已创建")
            
        except Exception as e:
            self.logger.error(f"创建目录结构失败: {e}")
            
    def start_session(self) -> bool:
        """开始新的记录会话"""
        try:
            # 生成会话ID
            self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 创建会话目录
            session_dirs = [
                os.path.join(self.base_dir, 'images', self.current_session),
                os.path.join(self.base_dir, 'annotations', self.current_session)
            ]
            
            for directory in session_dirs:
                os.makedirs(directory, exist_ok=True)
                
            self.frame_count = 0
            self.logger.info(f"开始新会话: {self.current_session}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建会话失败: {e}")
            return False
            
    def save_frame(self, frame: np.ndarray, keypoints: Dict[str, Any],
                  index: int) -> Tuple[str, str]:
        """保存帧和关键点数据"""
        if self.current_session is None:
            self.logger.error("未开始会话")
            return "", ""
            
        try:
            # 生成文件名
            image_filename = f"frame_{index:06d}.jpg"
            json_filename = f"frame_{index:06d}.json"
            
            # 构建完整路径
            image_path = os.path.join(self.base_dir, 'images',
                                    self.current_session, image_filename)
            json_path = os.path.join(self.base_dir, 'annotations',
                                   self.current_session, json_filename)
                                   
            # 保存图像
            cv2.imwrite(image_path, frame)
            
            # 保存关键点数据
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(keypoints, f, indent=4, ensure_ascii=False)
                
            self.frame_count += 1
            return image_path, json_path
            
        except Exception as e:
            self.logger.error(f"保存帧数据失败: {e}")
            return "", ""
            
    def record_frame(self, results):
        """记录单帧的姿态数据"""
        if self.current_session is None:
            return
            
        try:
            # 提取姿态数据
            frame_data = {
                'timestamp': datetime.now().isoformat(),
                'frame_index': self.frame_count,
                'pose_landmarks': None,
                'left_hand_landmarks': None,
                'right_hand_landmarks': None
            }
            
            # 处理姿态关键点
            if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                frame_data['pose_landmarks'] = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                    for lm in results.pose_landmarks.landmark
                ]
                
            # 处理左手关键点
            if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
                frame_data['left_hand_landmarks'] = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z}
                    for lm in results.left_hand_landmarks.landmark
                ]
                
            # 处理右手关键点
            if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
                frame_data['right_hand_landmarks'] = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z}
                    for lm in results.right_hand_landmarks.landmark
                ]
                
            # 保存数据
            json_filename = f"frame_{self.frame_count:06d}.json"
            json_path = os.path.join(self.base_dir, 'annotations',
                                   self.current_session, json_filename)
                                   
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(frame_data, f, indent=4, ensure_ascii=False)
                
            self.frame_count += 1
            
        except Exception as e:
            self.logger.error(f"记录帧数据失败: {e}")
            
    def end_session(self) -> bool:
        """结束当前会话"""
        if self.current_session is None:
            self.logger.warning("没有活动的会话")
            return False
            
        try:
            # 生成会话摘要
            summary = {
                'session_id': self.current_session,
                'frame_count': self.frame_count,
                'start_time': self.current_session,
                'end_time': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            # 保存会话摘要
            summary_path = os.path.join(self.base_dir, 'logs',
                                      f"session_{self.current_session}.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"会话 {self.current_session} 已结束，"
                           f"共记录 {self.frame_count} 帧")
            
            self.current_session = None
            self.frame_count = 0
            return True
            
        except Exception as e:
            self.logger.error(f"结束会话失败: {e}")
            return False
            
    def get_session_data(self, session_id: str) -> Tuple[List[str], List[str]]:
        """获取指定会话的所有数据文件路径"""
        try:
            image_dir = os.path.join(self.base_dir, 'images', session_id)
            anno_dir = os.path.join(self.base_dir, 'annotations', session_id)
            
            if not (os.path.exists(image_dir) and os.path.exists(anno_dir)):
                self.logger.error(f"会话 {session_id} 的数据目录不存在")
                return [], []
                
            # 获取所有图像和标注文件
            image_files = sorted([os.path.join(image_dir, f)
                                for f in os.listdir(image_dir)
                                if f.endswith('.jpg')])
            anno_files = sorted([os.path.join(anno_dir, f)
                               for f in os.listdir(anno_dir)
                               if f.endswith('.json')])
                               
            return image_files, anno_files
            
        except Exception as e:
            self.logger.error(f"获取会话数据失败: {e}")
            return [], []
            
    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID列表"""
        try:
            sessions_dir = os.path.join(self.base_dir, 'images')
            if not os.path.exists(sessions_dir):
                return []
                
            # 获取所有会话目录
            sessions = [d for d in os.listdir(sessions_dir)
                       if os.path.isdir(os.path.join(sessions_dir, d))]
            return sorted(sessions)
            
        except Exception as e:
            self.logger.error(f"获取会话列表失败: {e}")
            return []
