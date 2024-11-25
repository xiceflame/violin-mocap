import socket
import struct
import pickle
import cv2
import torch
import logging
from ultralytics import YOLO
from datetime import datetime

class ViolinCaptureServer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.logger = self._setup_logger()
        self.model = self._initialize_model()
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler(f'server_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def _initialize_model(self):
        """初始化YOLO模型"""
        try:
            model = YOLO('yolov8n.pt')
            if torch.cuda.is_available():
                model.to('cuda')
                self.logger.info("使用CUDA进行模型加速")
            return model
        except Exception as e:
            self.logger.error(f"初始化模型失败: {e}")
            return None
            
    def start(self):
        """启动服务器"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            self.logger.info(f"服务器启动在 {self.host}:{self.port}")
            
            while True:
                client_socket, address = server_socket.accept()
                self.logger.info(f"接受连接来自: {address}")
                self._handle_client(client_socket)
                
        except Exception as e:
            self.logger.error(f"服务器启动失败: {e}")
            
    def _handle_client(self, client_socket):
        """处理客户端连接"""
        try:
            while True:
                # 接收数据大小
                data_size = struct.unpack(">L", client_socket.recv(4))[0]
                
                # 接收数据
                received_data = b""
                while len(received_data) < data_size:
                    received_data += client_socket.recv(4096)
                    
                # 解码图像
                img_data = pickle.loads(received_data)
                frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                
                # 处理图像
                if self.model is not None:
                    results = self.model(frame)
                    processed_frame = results[0].plot()
                else:
                    processed_frame = frame
                    
                # 编码处理后的图像
                _, img_encoded = cv2.imencode('.jpg', processed_frame)
                data = pickle.dumps(img_encoded)
                
                # 发送数据大小
                client_socket.sendall(struct.pack(">L", len(data)))
                # 发送数据
                client_socket.sendall(data)
                
        except ConnectionResetError:
            self.logger.info("客户端断开连接")
        except Exception as e:
            self.logger.error(f"处理客户端数据时出错: {e}")
        finally:
            client_socket.close()

if __name__ == '__main__':
    server = ViolinCaptureServer()
    server.start()
