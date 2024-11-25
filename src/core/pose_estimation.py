import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from ultralytics import YOLO
from ..utils.path_manager import PathManager

class PoseEstimator:
    """姿态估计器类，用于检测和跟踪人体姿态、手部和小提琴"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化姿态估计器
        
        Args:
            config: MediaPipe配置字典
        """
        self.logger = logging.getLogger(__name__)
        
        # 从配置中获取参数
        min_detection_confidence = config.get('min_detection_confidence', 0.5)
        min_tracking_confidence = config.get('min_tracking_confidence', 0.5)
        model_complexity = config.get('model_complexity', 2)
        refine_face_landmarks = config.get('refine_face_landmarks', True)
        
        # 初始化MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            refine_face_landmarks=refine_face_landmarks
        )
        
        # 初始化PathManager
        self.path_manager = PathManager()
        
        # 初始化YOLO模型用于小提琴检测
        try:
            # 加载YOLO模型
            model_path = self.path_manager.get_path('models', 'pretrained', 'detection', 'yolo', 'yolov8n')
            if model_path is None or not model_path.exists():
                raise FileNotFoundError("YOLO模型文件不存在")
            self.yolo_model = YOLO(str(model_path))
            self.logger.info("Successfully loaded YOLO model")
        except Exception as e:
            self.logger.error("Failed to load YOLO model: %s", str(e))
            self.yolo_model = None
            
        # 缓存上一帧的结果用于平滑处理
        self.prev_results = None
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            包含检测结果的字典
        """
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe处理
        results = self.holistic.process(frame_rgb)
        
        # YOLO检测小提琴
        violin_detection = None
        if self.yolo_model:
            yolo_results = self.yolo_model(frame)
            violin_detection = self._process_yolo_results(yolo_results)
            
        # 组合所有结果
        combined_results = {
            'pose_landmarks': results.pose_landmarks,
            'face_landmarks': results.face_landmarks,
            'left_hand_landmarks': results.left_hand_landmarks,
            'right_hand_landmarks': results.right_hand_landmarks,
            'violin_detection': violin_detection
        }
        
        # 计算关节角度
        if results.pose_landmarks:
            combined_results['joint_angles'] = self._calculate_joint_angles(results.pose_landmarks)
            
        # 估计小提琴位置
        if results.pose_landmarks and violin_detection:
            combined_results['violin_position'] = self._estimate_violin_position(
                results.pose_landmarks,
                violin_detection
            )
            
        self.prev_results = combined_results
        return combined_results
        
    def _process_yolo_results(self, results) -> Optional[Dict[str, Any]]:
        """处理YOLO检测结果"""
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # 假设0是小提琴类别
                    return {
                        'box': box.xyxy[0].cpu().numpy(),
                        'confidence': box.conf[0].cpu().numpy()
                    }
        return None
        
    def _calculate_joint_angles(self, pose_landmarks) -> Dict[str, float]:
        """计算关节角度"""
        angles = {}
        
        # 计算右臂角度
        right_shoulder = np.array([
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].y
        ])
        right_elbow = np.array([
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_ELBOW].x,
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_ELBOW].y
        ])
        right_wrist = np.array([
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_WRIST].x,
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_WRIST].y
        ])
        
        angles['right_elbow'] = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # 计算左臂角度
        left_shoulder = np.array([
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].y
        ])
        left_elbow = np.array([
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_ELBOW].x,
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_ELBOW].y
        ])
        left_wrist = np.array([
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_WRIST].x,
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_WRIST].y
        ])
        
        angles['left_elbow'] = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        return angles
        
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """计算三个点形成的角度"""
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
        
    def _estimate_violin_position(self, pose_landmarks, violin_detection) -> Dict[str, Any]:
        """估计小提琴相对于身体的位置"""
        # 获取肩膀中点作为参考
        left_shoulder = np.array([
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].y
        ])
        right_shoulder = np.array([
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
            pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].y
        ])
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # 计算小提琴框的中心
        box = violin_detection['box']
        violin_center = np.array([
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2
        ])
        
        # 计算相对位置和角度
        relative_position = violin_center - shoulder_center
        angle = np.degrees(np.arctan2(relative_position[1], relative_position[0]))
        
        return {
            'relative_position': relative_position.tolist(),
            'angle': angle,
            'distance': np.linalg.norm(relative_position)
        }
        
    def draw_pose(self, frame: np.ndarray, results: Dict[str, Any]):
        """绘制姿态骨架"""
        if results['pose_landmarks']:
            self.mp_drawing.draw_landmarks(
                frame,
                results['pose_landmarks'],
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
    def draw_hands(self, frame: np.ndarray, results: Dict[str, Any]):
        """绘制手部关键点"""
        if results['left_hand_landmarks']:
            self.mp_drawing.draw_landmarks(
                frame,
                results['left_hand_landmarks'],
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
            
        if results['right_hand_landmarks']:
            self.mp_drawing.draw_landmarks(
                frame,
                results['right_hand_landmarks'],
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
            
    def draw_violin_detection(self, frame: np.ndarray, results: Dict[str, Any]):
        """绘制小提琴检测框"""
        if results.get('violin_detection'):
            box = results['violin_detection']['box']
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                2
            )
            
    def draw_joint_angles(self, frame: np.ndarray, results: Dict[str, Any]):
        """绘制关节角度"""
        if 'joint_angles' in results:
            angles = results['joint_angles']
            y_offset = 60
            for joint, angle in angles.items():
                cv2.putText(
                    frame,
                    f"{joint}: {angle:.1f}°",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_offset += 20
