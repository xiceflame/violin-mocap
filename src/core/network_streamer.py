"""网络流传输模块"""
import socket
import pickle
import struct
import logging
from typing import Dict, Any, Optional

class NetworkStreamer:
    def __init__(self, host: str = '127.0.0.1', port: int = 12345):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.is_streaming = False
        self.remote_address = (host, port)

    def start_streaming(self):
        """开始流传输"""
        self.is_streaming = True
        self.logger.info(f"开始流传输到 {self.host}:{self.port}")

    def stop_streaming(self):
        """停止流传输"""
        self.is_streaming = False
        self.logger.info("停止流传输")

    def send_data(self, data: Dict[str, Any]) -> bool:
        """发送数据"""
        if not self.is_streaming:
            return False
            
        try:
            # 序列化数据
            serialized_data = pickle.dumps(data)
            
            # 如果数据太大，需要分片发送
            max_chunk_size = 65507  # UDP最大数据包大小
            
            if len(serialized_data) > max_chunk_size:
                # 分片发送
                chunks = [serialized_data[i:i + max_chunk_size] 
                         for i in range(0, len(serialized_data), max_chunk_size)]
                
                # 发送分片数量
                header = struct.pack('!I', len(chunks))
                self.udp_socket.sendto(header, self.remote_address)
                
                # 发送每个分片
                for i, chunk in enumerate(chunks):
                    # 添加分片索引
                    chunk_header = struct.pack('!I', i)
                    self.udp_socket.sendto(chunk_header + chunk, self.remote_address)
            else:
                # 直接发送
                self.udp_socket.sendto(serialized_data, self.remote_address)
                
            return True
        except Exception as e:
            self.logger.error(f"发送数据失败: {e}")
            return False

    def set_remote_address(self, host: Optional[str] = None, port: Optional[int] = None):
        """设置远程地址"""
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port
        self.remote_address = (self.host, self.port)

    def close(self):
        """关闭连接"""
        self.stop_streaming()
        self.udp_socket.close()
