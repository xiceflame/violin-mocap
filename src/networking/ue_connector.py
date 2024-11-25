import socket
import json
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from threading import Thread, Lock

class UEConnector:
    """
    与Unreal Engine通信的连接器类
    用于发送动作捕捉数据到UE进行可视化
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 12345,
                 packet_size: int = 8192, retry_interval: float = 1.0):
        """
        初始化UE连接器
        
        Args:
            host: UE服务器地址
            port: UE服务器端口
            packet_size: UDP数据包大小
            retry_interval: 重连间隔（秒）
        """
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.packet_size = packet_size
        self.retry_interval = retry_interval
        
        # 创建UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 连接状态
        self.connected = False
        self.lock = Lock()
        
        # 启动心跳线程
        self.heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
    def send_data(self, data: Dict[str, Any]) -> bool:
        """
        发送数据到UE
        
        Args:
            data: 要发送的数据字典
            
        Returns:
            是否发送成功
        """
        try:
            # 转换数据格式
            processed_data = self._process_data(data)
            
            # 转换为JSON
            json_data = json.dumps(processed_data)
            
            # 发送数据
            with self.lock:
                self.socket.sendto(json_data.encode(), (self.host, self.port))
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send data: {str(e)}")
            return False
            
    def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数据以适应UE的格式要求
        
        Args:
            data: 原始数据字典
            
        Returns:
            处理后的数据字典
        """
        processed = {
            'timestamp': time.time(),
            'pose': self._process_pose_data(data.get('pose_landmarks')),
            'hands': {
                'left': self._process_hand_data(data.get('left_hand_landmarks')),
                'right': self._process_hand_data(data.get('right_hand_landmarks'))
            },
            'violin': self._process_violin_data(data.get('violin_detection')),
            'joint_angles': data.get('joint_angles', {})
        }
        
        return processed
        
    def _process_pose_data(self, pose_landmarks) -> Optional[Dict[str, Any]]:
        """处理姿态数据"""
        if not pose_landmarks:
            return None
            
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x * 100,  # 转换为厘米
                'y': landmark.y * 100,
                'z': landmark.z * 100,
                'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            })
            
        return {'landmarks': landmarks}
        
    def _process_hand_data(self, hand_landmarks) -> Optional[Dict[str, Any]]:
        """处理手部数据"""
        if not hand_landmarks:
            return None
            
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append({
                'x': landmark.x * 100,
                'y': landmark.y * 100,
                'z': landmark.z * 100
            })
            
        return {'landmarks': landmarks}
        
    def _process_violin_data(self, violin_detection) -> Optional[Dict[str, Any]]:
        """处理小提琴检测数据"""
        if not violin_detection:
            return None
            
        return {
            'landmarks': violin_detection['landmarks'],
            'confidence': float(violin_detection['confidence'])
        }
        
    def _heartbeat_loop(self):
        """心跳循环，定期检查连接状态"""
        while True:
            try:
                # 发送心跳包
                with self.lock:
                    self.socket.sendto(b'heartbeat', (self.host, self.port))
                self.connected = True
                
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {str(e)}")
                self.connected = False
                
            time.sleep(self.retry_interval)
            
    def is_connected(self) -> bool:
        """
        检查是否已连接
        
        Returns:
            连接状态
        """
        return self.connected
        
    def close(self):
        """关闭连接"""
        try:
            self.socket.close()
        except Exception as e:
            self.logger.error(f"Error closing socket: {str(e)}")

class UEDataProcessor:
    """处理和转换发送到UE的数据"""
    
    def __init__(self):
        """初始化数据处理器"""
        # 定义骨骼映射
        self.bone_mapping = {
            'Hips': 'pose_24',           # 右髋
            'Spine': 'pose_23',          # 脊椎
            'Spine1': 'pose_11',         # 上脊椎
            'Neck': 'pose_10',           # 颈部
            'Head': 'pose_0',            # 头部
            'LeftShoulder': 'pose_11',   # 左肩
            'LeftArm': 'pose_13',        # 左上臂
            'LeftForeArm': 'pose_15',    # 左前臂
            'LeftHand': 'pose_17',       # 左手
            'RightShoulder': 'pose_12',  # 右肩
            'RightArm': 'pose_14',       # 右上臂
            'RightForeArm': 'pose_16',   # 右前臂
            'RightHand': 'pose_18',      # 右手
            'LeftUpLeg': 'pose_23',      # 左大腿
            'LeftLeg': 'pose_25',        # 左小腿
            'LeftFoot': 'pose_27',       # 左脚
            'RightUpLeg': 'pose_24',     # 右大腿
            'RightLeg': 'pose_26',       # 右小腿
            'RightFoot': 'pose_28'       # 右脚
        }
        
        # 手指关节映射
        self.finger_mapping = {
            'Thumb': [1, 2, 3, 4],
            'Index': [5, 6, 7, 8],
            'Middle': [9, 10, 11, 12],
            'Ring': [13, 14, 15, 16],
            'Pinky': [17, 18, 19, 20]
        }
    
    def process_landmarks(self, landmarks_data: Dict) -> Dict:
        """
        处理和转换关键点数据为UE可用格式
        
        Args:
            landmarks_data: 原始关键点数据
            
        Returns:
            转换后的数据字典
        """
        ue_data = {
            'skeleton': self._process_skeleton(landmarks_data),
            'hands': self._process_hands(landmarks_data),
            'face': self._process_face(landmarks_data),
            'violin': self._process_violin(landmarks_data)
        }
        
        return ue_data
    
    def _process_skeleton(self, landmarks_data: Dict) -> Dict:
        """处理骨骼数据"""
        skeleton_data = {}
        
        if 'pose' in landmarks_data and landmarks_data['pose']:
            pose_landmarks = landmarks_data['pose']
            
            for ue_bone, mp_landmark in self.bone_mapping.items():
                landmark_idx = int(mp_landmark.split('_')[1])
                if landmark_idx < len(pose_landmarks):
                    landmark = pose_landmarks[landmark_idx]
                    
                    # 转换为UE坐标系统
                    skeleton_data[ue_bone] = {
                        'location': self._convert_to_ue_coordinates(
                            landmark['x'], landmark['y'], landmark['z']
                        ),
                        'rotation': self._calculate_bone_rotation(
                            landmarks_data, ue_bone
                        ),
                        'visibility': landmark['visibility']
                    }
        
        return skeleton_data
    
    def _process_hands(self, landmarks_data: Dict) -> Dict:
        """处理手部数据"""
        hands_data = {
            'left': {},
            'right': {}
        }
        
        # 处理左手
        if 'left_hand' in landmarks_data and landmarks_data['left_hand']:
            hands_data['left'] = self._process_single_hand(
                landmarks_data['left_hand'], 'left'
            )
        
        # 处理右手
        if 'right_hand' in landmarks_data and landmarks_data['right_hand']:
            hands_data['right'] = self._process_single_hand(
                landmarks_data['right_hand'], 'right'
            )
        
        return hands_data
    
    def _process_single_hand(self, hand_landmarks: list, side: str) -> Dict:
        """处理单个手的数据"""
        hand_data = {}
        
        for finger, indices in self.finger_mapping.items():
            finger_data = []
            for idx in indices:
                if idx < len(hand_landmarks):
                    landmark = hand_landmarks[idx]
                    finger_data.append({
                        'location': self._convert_to_ue_coordinates(
                            landmark['x'], landmark['y'], landmark['z']
                        ),
                        'rotation': self._calculate_finger_rotation(
                            hand_landmarks, idx
                        )
                    })
            hand_data[finger] = finger_data
        
        return hand_data
    
    def _process_face(self, landmarks_data: Dict) -> Dict:
        """处理面部数据"""
        face_data = {}
        
        if 'face' in landmarks_data and landmarks_data['face']:
            face_landmarks = landmarks_data['face']
            
            # 提取关键面部特征点
            face_data = {
                'jaw': self._process_face_region(face_landmarks, range(0, 17)),
                'right_eyebrow': self._process_face_region(face_landmarks, range(17, 22)),
                'left_eyebrow': self._process_face_region(face_landmarks, range(22, 27)),
                'nose': self._process_face_region(face_landmarks, range(27, 36)),
                'right_eye': self._process_face_region(face_landmarks, range(36, 42)),
                'left_eye': self._process_face_region(face_landmarks, range(42, 48)),
                'outer_mouth': self._process_face_region(face_landmarks, range(48, 60)),
                'inner_mouth': self._process_face_region(face_landmarks, range(60, 68))
            }
        
        return face_data
    
    def _process_violin(self, landmarks_data: Dict) -> Dict:
        """处理小提琴相关数据"""
        violin_data = {}
        
        if 'violin_position' in landmarks_data:
            position = landmarks_data['violin_position']
            if position:
                violin_data = {
                    'center': self._convert_to_ue_coordinates(
                        position['center']['x'],
                        position['center']['y'],
                        position['center']['z']
                    ),
                    'angle': position['angle']
                }
        
        return violin_data
    
    def _convert_to_ue_coordinates(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        转换MediaPipe坐标系到UE坐标系
        MediaPipe: +X右, +Y下, +Z前
        UE: +X前, +Y右, +Z上
        """
        return (z * 100, x * 100, -y * 100)  # 缩放到厘米
    
    def _calculate_bone_rotation(self, landmarks_data: Dict, bone_name: str) -> Tuple[float, float, float]:
        """计算骨骼旋转"""
        # 这里需要根据具体骨骼实现旋转计算
        # 返回欧拉角(Pitch, Yaw, Roll)
        return (0.0, 0.0, 0.0)
    
    def _calculate_finger_rotation(self, hand_landmarks: list, joint_idx: int) -> Tuple[float, float, float]:
        """计算手指关节旋转"""
        # 这里需要实现手指关节的旋转计算
        return (0.0, 0.0, 0.0)
    
    def _process_face_region(self, face_landmarks: list, indices: range) -> List[Dict]:
        """处理面部区域关键点"""
        region_data = []
        for idx in indices:
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                region_data.append({
                    'location': self._convert_to_ue_coordinates(
                        landmark['x'], landmark['y'], landmark['z']
                    )
                })
        return region_data
    
    def compress_data(self, json_data: str) -> str:
        """压缩JSON数据（可选）"""
        # 这里可以实现数据压缩算法
        return json_data
