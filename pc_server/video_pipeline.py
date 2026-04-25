"""
video_pipeline.py
危险区域人员闯入检测与报警系统 —— 纯后端视频处理管道

架构：生产者-消费者模型
  - StreamProducer  (生产者线程)：拉流 → 丢弃旧帧队列
  - InferenceConsumer (消费者线程)：队列 → YOLO+ByteTrack → 多边形入侵判定
"""

from __future__ import annotations

import abc
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================================
#  1. 丢弃旧帧的线程安全队列
# ============================================================================

class DropOldQueue:
    """
    固定容量的线程安全队列，队列满时自动丢弃最旧的一帧，
    保证消费者始终拿到最新画面，避免安防场景中的视频延迟。
    """

    def __init__(self, maxsize: int = 2) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        """放入一帧；若队列已满，先丢弃队头最旧帧再放入。"""
        with self._lock:
            if self._queue.full():
                try:
                    self._queue.get_nowait()          # 丢弃旧帧
                except queue.Empty:
                    pass
            self._queue.put_nowait(frame)

    def get(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """取出一帧；超时返回 None。"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ============================================================================
#  2. 检测器抽象基类 & YoloTracker 实现
# ============================================================================

@dataclass
class Detection:
    """单个检测目标的数据结构。"""
    track_id: int                        # ByteTrack 分配的跟踪 ID（无追踪时为 -1）
    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2)
    confidence: float
    class_id: int

    @property
    def center(self) -> Tuple[int, int]:
        """检测框中心点坐标。"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class BaseDetector(abc.ABC):
    """
    推理逻辑基类——主流程只依赖此接口，
    方便未来替换为 TensorRT / ONNX / 边缘端推理引擎。
    """

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """对单帧执行检测+追踪，返回检测结果列表。"""
        ...


class YoloTracker(BaseDetector):
    """
    集成 YOLOv8 + ByteTrack 的检测追踪器。
    仅检测 person 类别 (class_id=0)。
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.5,
        device: str = "",               # 空串让 ultralytics 自动选 CPU/GPU
    ) -> None:
        self._model = YOLO(model_path)
        self._conf = conf
        self._device = device

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        执行 YOLO 推理 + ByteTrack 多目标追踪。
        由于跳帧可能导致 ByteTrack 丢失目标，此处做了异常容错：
        若追踪失败则回退到无追踪的纯检测结果。
        """
        try:
            results = self._model.track(
                source=frame,
                persist=True,              # 保持 ByteTrack 内部状态
                tracker="bytetrack.yaml",  # 使用 ByteTrack 追踪器
                classes=[0],               # 仅检测 person
                conf=self._conf,
                device=self._device,
                verbose=False,
            )
        except Exception as e:
            # ByteTrack 因跳帧等原因异常时的容错：回退到纯检测
            print(f"[YoloTracker] 追踪异常，回退到纯检测: {e}")
            try:
                results = self._model.predict(
                    source=frame,
                    classes=[0],
                    conf=self._conf,
                    device=self._device,
                    verbose=False,
                )
            except Exception as e2:
                print(f"[YoloTracker] 推理失败: {e2}")
                return []

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                # track id 可能为 None（首帧/追踪丢失时）
                tid = int(boxes.id[i]) if boxes.id is not None else -1
                detections.append(Detection(
                    track_id=tid,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                ))
        return detections


# ============================================================================
#  3. 生产者线程 —— 拉流
# ============================================================================

class StreamProducer(threading.Thread):
    """
    拉流线程（生产者）：从本地摄像头 / 视频文件 / RTSP 流读取帧，
    放入 DropOldQueue。读取失败时自动重试，线程可通过 stop() 安全退出。
    """

    def __init__(
        self,
        source: int | str,              # 0=本地摄像头, 或视频文件路径/RTSP URL
        frame_queue: DropOldQueue,
    ) -> None:
        super().__init__(daemon=True, name="StreamProducer")    # 设置成守护线程，主线程退出则退出
        self._source = source
        self._queue = frame_queue
        self._running = threading.Event()       # 线程安全标志位默认是False
        self._running.set()                     # 将标志位设置为Ture

    # ------------------------------------------------------------------
    def run(self) -> None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            print(f"[StreamProducer] 无法打开视频源: {self._source}")
            return

        # 获取视频原始帧率，用于视频文件按原速播放
        # 摄像头 / RTSP 流由硬件/网络自然限速，不需要额外延时
        fps = cap.get(cv2.CAP_PROP_FPS)
        is_file = isinstance(self._source, str) and not self._source.lower().startswith("rtsp")
        frame_interval = 1.0 / fps if (is_file and fps > 0) else 0

        print(f"[StreamProducer] 已连接视频源: {self._source}"
              f"{f'  (原始帧率: {fps:.1f} FPS)' if frame_interval > 0 else ''}")
        try:
            while self._running.is_set():
                t_start = time.perf_counter()   # 记录当前时间戳

                ret, frame = cap.read()     # 元组解包,ret：bool,代表这次抓取成功与否
                if not ret:
                    # 视频文件播放完毕 / 摄像头断开
                    print("[StreamProducer] 视频源读取结束或断开，尝试重连...")
                    cap.release()       # 释放
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(self._source)        # 重连
                    if not cap.isOpened():
                        print("[StreamProducer] 重连失败，退出拉流线程。")
                        break
                    continue

                # 放入丢弃旧帧队列，保证实时性
                self._queue.put(frame)

                # 视频文件按原始帧率节流，避免全速读取导致播放加速
                if frame_interval > 0:
                    elapsed = time.perf_counter() - t_start     # 计算读图 + 塞进队列总共花了多长时间
                    sleep_time = frame_interval - elapsed       # 这一帧的标准间隔时间，减去刚才消耗的时间，剩下的就是要“睡”的时间
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        finally:
            cap.release()
            print("[StreamProducer] 拉流线程已退出，资源已释放。")

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """安全停止拉流线程。"""
        self._running.clear()       # 主程序点“停止”按钮时调用。它把（is_set）改成False。


# ============================================================================
#  3.5 Socket 拉流线程 —— 从开发板 TCP 连接接收 JPEG 帧
# ============================================================================

class SocketStreamProducer(threading.Thread):
    """
    Socket 拉流线程（生产者）：作为 TCP **服务端** 监听指定端口，
    等待嵌入式开发板（i.MX6ULL 等）主动连接后，持续接收 JPEG 视频帧。

    协议格式（与原 main.py 一致）：
      [4 字节小端 uint32: 帧大小] + [N 字节 JPEG 数据]

    粘包/分包处理：
      TCP 是字节流协议，单次 recv() 可能返回不完整数据（分包）
      或多帧数据粘在一起（粘包）。本类通过以下方式解决：
      1) _recv_exactly(n)：循环 recv 直到恰好收满 n 字节
      2) 先读 4 字节包头获取帧长度，再精确读取该长度的 payload
      这样无论底层如何分片/合并，都能正确切割每一帧的边界。
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8888,
        frame_queue: DropOldQueue = None,
    ) -> None:
        super().__init__(daemon=True, name="SocketStreamProducer")
        self._host = host
        self._port = port
        self._queue = frame_queue
        self._running = threading.Event()
        self._running.set()
        # 保存 socket 引用，以便 stop() 时从外部关闭来打断阻塞的 accept/recv
        # 执行recv()accept()时程序死等,外部光改变标志位不能停止程序,这里存为成员变量以便主线程从外部操控
        self._server_socket: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        # 开发板连接后记录其真实 IP（供 AlarmSender 使用）
        self._peer_ip: Optional[str] = None

    @property       # 把一个方法（Method）伪装成一个属性（Attribute）。只读。ip = producer.peer_ip 无需括号即可调用
    def peer_ip(self) -> Optional[str]:
        """开发板连接后的真实 IP 地址，未连接时为 None。"""
        return self._peer_ip

    # ------------------------------------------------------------------
    #  精确接收 n 字节 —— 解决 TCP 分包问题的核心函数
    # ------------------------------------------------------------------

    def _recv_exactly(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """
        从 socket 中精确接收 n 字节数据。
        TCP 是字节流，单次 recv() 返回的数据量可能 < n（分包），
        因此必须循环拼接，直到收满 n 字节为止。
        返回 None 表示连接已断开。
        """
        buf = b""       # 准备一个空的二进制容器
        while len(buf) < n:     # 直到接收到的数据长度够n才退出
            if not self._running.is_set():      # 这里使用了非阻塞写法防止程序睡死
                return None
            try:
                chunk = sock.recv(n - len(buf))     # 确保不多拿少拿一字节
            except (OSError, ConnectionError):
                return None
            if not chunk:
                # recv 返回空字节 → 对端已关闭连接
                return None
            buf += chunk
        return buf

    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        主循环：创建 TCP 服务端 → 等待开发板连接 → 循环接收帧。
        开发板断开后自动回到等待连接状态（支持重连）。
        """
        # ---- 创建 TCP 服务端 ----
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # SO_REUSEADDR 防止程序重启时 "端口被占用"
            self._server_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            self._server_socket.settimeout(1.0)   # accept 超时，在后面的 _wait_for_connection 里被捕获并返回 None，这样程序就能进入下一轮循环，便于检查 _running
            self._server_socket.bind((self._host, self._port))
            self._server_socket.listen(1)
            print(f"[SocketStreamProducer] 服务器已启动，监听 {self._host}:{self._port}，等待开发板连接...")
        except OSError as e:
            print(f"[SocketStreamProducer] 绑定端口失败: {e}")
            return

        try:
            while self._running.is_set():
                # ---- 等待开发板连接（带超时，可被 stop() 中断） ----
                conn = self._wait_for_connection()
                if conn is None:
                    continue               # 超时或已停止，重新检查 _running

                self._conn = conn
                self._peer_ip = conn.getpeername()[0]
                print(f"[SocketStreamProducer] 开发板已连接: {conn.getpeername()}")

                # ---- 持续接收帧 ----
                self._receive_frames(conn)      # 只要网络没断，程序就会一直停在这一行里面不出来

                # 开发板断开，清理连接，回到等待状态
                conn.close()
                self._conn = None
                print("[SocketStreamProducer] 开发板断开，等待重新连接...")
        finally:
            if self._conn is not None:
                self._conn.close()
            if self._server_socket is not None:
                self._server_socket.close()
            print("[SocketStreamProducer] Socket 拉流线程已退出，资源已释放。")

    # ------------------------------------------------------------------
    def _wait_for_connection(self) -> Optional[socket.socket]:
        """等待一个客户端连接，超时返回 None。"""
        try:
            conn, addr = self._server_socket.accept()
            conn.settimeout(10.0)          # recv 超时，需大于边缘端心跳间隔(2s)，留足余量
            return conn
        except socket.timeout:
            return None
        except OSError:     # 执行stop，关闭套接字捕获异常
            return None

    # ------------------------------------------------------------------
    def _receive_frames(self, conn: socket.socket) -> None:
        """
        持续从已连接的 socket 中接收帧。
        协议：[4 字节小端 uint32 帧大小] + [帧大小字节的 JPEG 数据]
        利用 _recv_exactly 解决粘包/分包问题。
        """
        while self._running.is_set():
            # ---- 步骤 1：读取 4 字节包头（帧大小） ----
            # 这 4 个字节可能被 TCP 拆成多次 recv，_recv_exactly 会拼完整
            header = self._recv_exactly(conn, 4)
            if header is None:
                break                      # 连接断开

            frame_size = struct.unpack("<I", header)[0]     # < 表示小端序，I 表示这是一个 Unsigned Int

            # 基本合法性校验：防止错误数据导致分配超大内存
            if frame_size == 0 or frame_size > 10 * 1024 * 1024:  # >10MB 视为异常
                print(f"[SocketStreamProducer] 异常帧大小: {frame_size}，跳过")
                continue

            # ---- 步骤 2：根据帧大小，精确接收完整的 JPEG 数据 ----
            # 即使 JPEG 数据被 TCP 拆成几十个小包，_recv_exactly 也会拼接完整
            data = self._recv_exactly(conn, frame_size)
            if data is None:
                break                      # 连接断开

            # ---- 步骤 3：JPEG 解码 ----
            try:
                img_array = np.frombuffer(data, dtype=np.uint8)         # 二进制数据解码成1 维数组（类型是uint8）
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)       # 解压展开得到三维矩阵（高度 x 宽度 x 3个颜色通道 BGR），得到图片；失败返回None
            except Exception as e:
                print(f"[SocketStreamProducer] imdecode 异常: {e}")
                continue

            if frame is None:
                # JPEG 数据损坏，解码失败
                print("[SocketStreamProducer] 警告：收到损坏的图片，解码失败，跳过")
                continue

            # 放入丢弃旧帧队列，保证实时性
            self._queue.put(frame)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """
        安全停止 Socket 拉流线程。
        通过关闭 socket 来打断可能阻塞在 accept/recv 上的操作。
        """
        self._running.clear()
        # 关闭 socket 使阻塞的 recv/accept 立即抛出异常并退出
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass


# ============================================================================
#  4. 消费者线程 —— 推理 + 入侵检测
# ============================================================================

class InferenceConsumer(threading.Thread):
    """
    推理线程（消费者）：从队列取帧 → 检测追踪 → 多边形入侵判定。
    处理后的带标注帧和统计数据通过回调函数输出（为将来接入 Qt Signal 预留）。
    """

    def __init__(
        self,
        frame_queue: DropOldQueue,
        detector: BaseDetector,
        polygon: List[Tuple[int, int]],    # 多边形区域顶点坐标列表
        alarm_threshold: int = 1,          # 区域内人数达到此值触发警报
        on_frame_ready: Optional[Callable[[np.ndarray, int], None]] = None,
    ) -> None:
        super().__init__(daemon=True, name="InferenceConsumer")
        self._queue = frame_queue
        self._detector = detector
        self._alarm_threshold = alarm_threshold
        self._running = threading.Event()
        self._running.set()

        # 多边形坐标——线程安全拷贝  与UI线程共享资源
        # 未来可通过 update_polygon() 由 UI 线程动态更新
        self._polygon_lock = threading.Lock()
        self._polygon = np.array(polygon, dtype=np.int32)       # 直接把传入的普通 Python 列表转换成 Numpy 矩阵，并强制类型为 32 位整数

        # 回调：(标注帧, 区域内人数) → 主线程/UI
        self._on_frame_ready = on_frame_ready

    # ------------------------------------------------------------------
    def update_polygon(self, polygon: List[Tuple[int, int]]) -> None:
        """线程安全地更新多边形区域坐标（供 UI 线程调用）。"""
        with self._polygon_lock:
            self._polygon = np.array(polygon, dtype=np.int32)

    # ------------------------------------------------------------------
    def _get_polygon(self) -> np.ndarray:
        """获取多边形坐标的线程安全拷贝。"""
        with self._polygon_lock:
            return self._polygon.copy()     # 复制了一份，返回副本的指针

    # ------------------------------------------------------------------
    def run(self) -> None:
        print("[InferenceConsumer] 推理线程已启动。")
        while self._running.is_set():
            frame = self._queue.get(timeout=0.5)
            if frame is None:
                continue                  # 队列超时，继续等待

            try:
                annotated, intrusion_count = self._process_frame(frame)
            except Exception as e:
                print(f"[InferenceConsumer] 帧处理异常: {e}")
                continue

            # 通过回调输出结果（将来替换为 Qt Signal）
            if self._on_frame_ready is not None:
                self._on_frame_ready(annotated, intrusion_count)

        print("[InferenceConsumer] 推理线程已退出。")

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        单帧处理流水线：
        1) YOLO + ByteTrack 检测追踪
        2) 多边形碰撞检测（中心点是否在区域内）
        3) 绘制标注 & 报警判定
        返回 (标注帧, 区域内人数)。
        """
        polygon = self._get_polygon()
        detections = self._detector.detect(frame)
        annotated = frame.copy()        # 复制一份原始图片用来标注

        # ---------- 绘制多边形警戒区域 ----------
        if len(polygon) >= 3:
            overlay = annotated.copy()      # 在 overlay（覆盖层）上画一个实心的红色多边形
            cv2.fillPoly(overlay, [polygon], color=(0, 0, 80))       # 半透明红色填充
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
            cv2.polylines(annotated, [polygon], isClosed=True,
                          color=(0, 0, 255), thickness=2)

        # ---------- 逐目标判定入侵 ----------
        intrusion_count = 0
        for det in detections:
            cx, cy = det.center
            x1, y1, x2, y2 = det.bbox

            # cv2.pointPolygonTest: (射线法检测)返回 >=0 表示点在多边形内部或边上
            inside = False
            if len(polygon) >= 3:
                dist = cv2.pointPolygonTest(
                    polygon.reshape(-1, 1, 2).astype(np.float32),
                    (float(cx), float(cy)),
                    measureDist=False,
                )
                inside = dist >= 0

            if inside:
                intrusion_count += 1
                # 入侵目标：红色框
                color = (0, 0, 255)
                label = f"INTRUDER #{det.track_id} {det.confidence:.2f}"
            else:
                # 区域外目标：绿色框
                color = (0, 255, 0)
                label = f"ID:{det.track_id} {det.confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (cx, cy), 4, color, -1)
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ---------- 报警判定 ----------
        if intrusion_count >= self._alarm_threshold:
            alarm_text = (
                f"!! ALARM !! {intrusion_count} person(s) in restricted zone!"
            )
            print(f"[ALARM] {alarm_text}")
            cv2.putText(annotated, alarm_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # 左上角统计信息
        cv2.putText(annotated,
                    f"Zone: {intrusion_count}/{self._alarm_threshold}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return annotated, intrusion_count

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """安全停止推理线程。"""
        self._running.clear()


# ============================================================================
#  5. 报警指令发送器 —— 向 i.MX6ULL 发送 0x01(开) / 0x00(关) 报警指令
# ============================================================================

class AlarmSender:
    """
    报警指令发送器：通过独立的 TCP 连接向 i.MX6ULL 开发板发送报警指令。

    协议格式（极简单字节指令）：
      0x01 —— 开启报警（蜂鸣器/LED）
      0x00 —— 关闭报警

    设计要点：
      - 仅在报警状态**变化**时才发送指令，避免每帧重复发送
      - 连接失败时不阻塞推理线程，后台自动重试
      - 线程安全：可从推理线程调用 set_alarm()
    """

    # 报警指令端口，与视频流端口 (8888) 区分开
    DEFAULT_PORT = 8889

    def __init__(self, host: str, port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._last_state: Optional[bool] = None   # 上次发送的报警状态
        self._running = True

    def connect(self) -> bool:
        """尝试连接开发板报警端口。连接失败返回 False，不会抛异常。"""
        # 注意：此方法内部会获取 _lock，不可在已持有 _lock 的上下文中调用
        with self._lock:
            if self._sock is not None:
                return True
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3.0)               # TCP三次握手超时
                s.connect((self._host, self._port))
                s.settimeout(2.0)               # 发送或接收超时
                self._sock = s
                print(f"[AlarmSender] 已连接报警通道: {self._host}:{self._port}")
                return True
            except (OSError, ConnectionError) as e:
                print(f"[AlarmSender] 连接报警通道失败: {e}")
                return False

    def _connect_unlocked(self) -> bool:
        """在已持有 _lock 的情况下尝试连接（不再获取锁）。
        连接超时设为 1 秒，避免长时间阻塞推理线程。"""
        if self._sock is not None:
            return True
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)      # 短超时，不阻塞推理线程
            s.connect((self._host, self._port))
            s.settimeout(2.0)
            self._sock = s
            print(f"[AlarmSender] 已连接报警通道: {self._host}:{self._port}")
            return True
        except (OSError, ConnectionError) as e:
            print(f"[AlarmSender] 连接报警通道失败: {e}")
            return False

    def set_alarm(self, alarm_on: bool) -> None:
        """
        设置报警状态。仅在状态发生变化时才实际发送指令。
        由推理线程调用，内部线程安全。
        """
        if not self._running:
            return
        if alarm_on == self._last_state:
            return                            # 状态未变化，跳过

        cmd = b'\x01' if alarm_on else b'\x00'
        self._last_state = alarm_on

        with self._lock:
            # 若未连接，尝试连接（使用不获取锁的版本，避免死锁）
            if self._sock is None:
                if not self._connect_unlocked():
                    return

            try:
                self._sock.sendall(cmd)
                state_str = "开启" if alarm_on else "关闭"
                print(f"[AlarmSender] 已发送报警指令: {state_str} (0x{cmd[0]:02X})")
            except (OSError, ConnectionError) as e:
                print(f"[AlarmSender] 发送报警指令失败: {e}，关闭连接待重连")
                self._close_sock()
                self._last_state = None        # 重置状态，下次重新发送

    def _close_sock(self) -> None:
        """关闭 socket（需在 _lock 内调用）。"""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def close(self) -> None:
        """关闭连接并标记停止，用于程序退出时清理。"""
        self._running = False
        with self._lock:
            # 发送关闭指令（尽力而为）
            if self._sock is not None:
                try:
                    self._sock.sendall(b'\x00')
                except OSError:
                    pass
                self._close_sock()
        print("[AlarmSender] 报警通道已关闭。")


# ============================================================================
#  6. 测试入口
# ============================================================================

def main() -> None:
    """
    测试入口：
      - 使用本地摄像头 (source=0) 或指定视频文件路径
      - cv2.imshow 显示标注画面
      - 按 'q' 退出并安全清理所有线程
    """
    # -------- 配置 --------
    VIDEO_SOURCE = 0                     # 本地摄像头；可改为视频文件路径
    MODEL_PATH = "yolov8n.pt"
    ALARM_THRESHOLD = 1                  # 区域内 >=3 人触发警报

    # 硬编码的多边形警戒区域（四边形，单位：像素坐标）
    # 实际使用时由 UI 鼠标绘制，通过 update_polygon() 传入
    POLYGON = [
        (100, 100),
        (500, 100),
        (500, 400),
        (100, 400),
    ]

    # -------- 初始化组件 --------
    frame_queue = DropOldQueue(maxsize=2)
    detector = YoloTracker(model_path=MODEL_PATH, conf=0.5)

    # 回调：将标注帧存入主线程可读的容器
    latest_frame_lock = threading.Lock()        # 一个在写、一个在读，必须加一把 latest_frame_lock 锁
    latest_frame: dict = {"img": None}

    def on_frame_ready(img: np.ndarray, count: int) -> None:
        with latest_frame_lock:
            latest_frame["img"] = img

    producer = StreamProducer(source=VIDEO_SOURCE, frame_queue=frame_queue)
    consumer = InferenceConsumer(
        frame_queue=frame_queue,
        detector=detector,
        polygon=POLYGON,
        alarm_threshold=ALARM_THRESHOLD,
        on_frame_ready=on_frame_ready,
    )

    # -------- 启动线程 --------
    producer.start()
    consumer.start()
    print("[Main] 管道已启动，按 'q' 退出。")

    # -------- 主循环：显示画面 --------
    try:
        while True:
            with latest_frame_lock:
                img = latest_frame["img"]

            if img is not None:
                cv2.imshow("Human Detection Pipeline", img)

            # waitKey 必须在主线程调用（OpenCV 限制）
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # -------- 安全清理 --------
        print("[Main] 正在停止所有线程...")
        producer.stop()
        consumer.stop()
        producer.join(timeout=3.0)
        consumer.join(timeout=3.0)
        cv2.destroyAllWindows()
        print("[Main] 已安全退出。")


if __name__ == "__main__":
    main()
