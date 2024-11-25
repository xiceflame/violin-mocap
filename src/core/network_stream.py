"""
网络流模块 - 处理实时数据流传输
"""
import logging
import json
import time
import threading
import queue
import websockets
import asyncio
from typing import Dict, Any, Optional

class NetworkStream:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.is_streaming = False
        self.server = None
        self.clients = set()
        self.data_queue = queue.Queue(maxsize=100)
        self._stream_thread = None
        
    async def _handle_client(self, websocket, path):
        """处理客户端连接"""
        self.clients.add(websocket)
        try:
            while True:
                # 保持连接活跃
                await websocket.ping()
                await asyncio.sleep(1)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            
    async def _broadcast_data(self):
        """广播数据到所有连接的客户端"""
        while True:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    if self.clients:
                        # 将数据转换为JSON字符串
                        json_data = json.dumps(data)
                        # 广播到所有客户端
                        await asyncio.gather(
                            *[client.send(json_data) for client in self.clients],
                            return_exceptions=True
                        )
            except Exception as e:
                self.logger.error(f"广播数据时出错: {e}")
            await asyncio.sleep(0.01)  # 避免CPU过度使用
            
    def _run_server(self):
        """在独立线程中运行websocket服务器"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        start_server = websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )
        
        loop.run_until_complete(start_server)
        loop.create_task(self._broadcast_data())
        loop.run_forever()
        
    def start(self):
        """启动网络流服务"""
        if self.is_streaming:
            return
            
        try:
            self._stream_thread = threading.Thread(target=self._run_server)
            self._stream_thread.daemon = True
            self._stream_thread.start()
            self.is_streaming = True
            self.logger.info(f"网络流服务已启动 - ws://{self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"启动网络流服务失败: {e}")
            
    def stop(self):
        """停止网络流服务"""
        self.is_streaming = False
        # 清空数据队列
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
                
    def send_data(self, data: Dict[str, Any]):
        """发送数据到流"""
        if not self.is_streaming:
            return
            
        try:
            if not self.data_queue.full():
                self.data_queue.put_nowait(data)
        except queue.Full:
            # 如果队列满了，移除最旧的数据
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(data)
            except (queue.Empty, queue.Full):
                pass
                
    @property
    def client_count(self) -> int:
        """获取当前连接的客户端数量"""
        return len(self.clients)
        
    def get_status(self) -> Dict[str, Any]:
        """获取流状态信息"""
        return {
            "is_streaming": self.is_streaming,
            "client_count": self.client_count,
            "queue_size": self.data_queue.qsize(),
            "address": f"ws://{self.host}:{self.port}"
        }
