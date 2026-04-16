"""
main.py
危险区域人员闯入检测与报警系统 —— 主程序入口

架构（严格遵循 CLAUDE.md 规范）：
  - UI 主线程：PySide6 界面渲染、用户交互、接收信号更新画面
  - 拉流线程 (StreamProducer)：生产者，读取视频帧 → DropOldQueue
  - 推理线程 (InferenceWorker)：消费者，QThread，队列 → YOLO+ByteTrack → Qt Signal → 主线程

绝对禁止在子线程中直接操作 UI 控件。
"""

from __future__ import annotations

import sys
import time
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QWidget, QComboBox, QSpinBox, QFileDialog,
)

from video_pipeline import (
    DropOldQueue, BaseDetector, YoloTracker, StreamProducer,
    SocketStreamProducer, Detection,
)
from ui_components import VideoDisplayLabel

import cv2
import threading


# ============================================================================
#  1. 推理线程 —— 继承 QThread，通过 Signal 向主线程输出结果
# ============================================================================

class InferenceWorker(QThread):
    """
    推理消费者线程（QThread 版本）。
    从 DropOldQueue 取帧 → YOLO+ByteTrack 检测追踪 → 多边形入侵判定
    → 通过 Qt Signal 将标注帧和统计数据发送给 UI 主线程。

    信号：
      frame_ready(QImage, int, bool)
        - QImage: 带标注的画面（BGR→RGB 已转换）
        - int:    当前区域内入侵人数
        - bool:   是否触发报警
    """

    # ---------- 信号定义 ----------
    frame_ready = Signal(QImage, int, bool)

    # 额外信号：发送未标注的原始帧（用于停止后清除区域时恢复干净画面）
    raw_frame_ready = Signal(QImage)

    def __init__(
        self,
        frame_queue: DropOldQueue,
        detector: BaseDetector,
        alarm_threshold: int = 3,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._queue = frame_queue
        self._detector = detector
        self._alarm_threshold = alarm_threshold
        self._running = True

        # 多边形坐标——线程安全读写
        self._polygon_lock = threading.Lock()
        self._polygon = np.array([], dtype=np.int32)

        # 帧率统计（在推理线程内计算，保证精确）
        self._fps_timestamps: deque = deque(maxlen=60)
        self._current_fps: float = 0.0

    # ------------------------------------------------------------------
    #  多边形坐标的线程安全更新接口（由 UI 主线程调用）
    # ------------------------------------------------------------------

    def update_polygon(self, polygon: List[Tuple[int, int]]) -> None:
        """线程安全地更新多边形区域坐标。"""
        with self._polygon_lock:
            if polygon:
                self._polygon = np.array(polygon, dtype=np.int32)
            else:
                self._polygon = np.array([], dtype=np.int32)

    def _get_polygon(self) -> np.ndarray:
        """获取多边形坐标的线程安全拷贝。"""
        with self._polygon_lock:
            return self._polygon.copy()

    # ------------------------------------------------------------------
    #  QThread 主循环
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("[InferenceWorker] 推理线程已启动。")
        while self._running:
            frame = self._queue.get(timeout=0.5)
            if frame is None:
                continue                  # 队列超时，继续等待

            # 帧率统计
            now = time.perf_counter()
            self._fps_timestamps.append(now)
            if len(self._fps_timestamps) >= 2:
                elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
                if elapsed > 0:
                    self._current_fps = (len(self._fps_timestamps) - 1) / elapsed

            try:
                annotated, count, alarm = self._process_frame(frame)
            except Exception as e:
                print(f"[InferenceWorker] 帧处理异常: {e}")
                continue

            # 将 OpenCV BGR 帧转为 QImage（RGB），通过信号发送给主线程
            h, w, ch = annotated.shape
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

            # 同时发送原始未标注帧（用于停止后清除区域恢复干净画面）
            rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg_raw = QImage(rgb_raw.data, w, h, ch * w, QImage.Format.Format_RGB888)

            # QImage 引用的是 numpy buffer，跨线程必须深拷贝
            self.frame_ready.emit(qimg.copy(), count, alarm)
            self.raw_frame_ready.emit(qimg_raw.copy())

        print("[InferenceWorker] 推理线程已退出。")

    # ------------------------------------------------------------------
    #  单帧处理流水线
    # ------------------------------------------------------------------

    def _process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, int, bool]:
        """
        单帧处理：
        1) YOLO + ByteTrack 检测追踪
        2) 多边形碰撞检测（中心点判定）
        3) 绘制标注 & 报警判定
        返回 (标注帧, 区域内人数, 是否报警)。
        """
        polygon = self._get_polygon()
        has_polygon = len(polygon) >= 3
        detections = self._detector.detect(frame)
        annotated = frame.copy()

        # ---------- 绘制多边形警戒区域（半透明红色） ----------
        if has_polygon:
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], color=(0, 0, 80))
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
            cv2.polylines(annotated, [polygon], isClosed=True,
                          color=(0, 0, 255), thickness=2)

        # ---------- 逐目标判定入侵 ----------
        intrusion_count = 0
        for det in detections:
            cx, cy = det.center
            x1, y1, x2, y2 = det.bbox

            inside = False
            if has_polygon:
                dist = cv2.pointPolygonTest(
                    polygon.reshape(-1, 1, 2).astype(np.float32),
                    (float(cx), float(cy)),
                    measureDist=False,
                )
                inside = dist >= 0

            if inside:
                intrusion_count += 1
                color = (0, 0, 255)        # 入侵：红色
                label = f"INTRUDER #{det.track_id} {det.confidence:.2f}"
            else:
                color = (0, 255, 0)        # 区域外：绿色
                label = f"ID:{det.track_id} {det.confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (cx, cy), 4, color, -1)
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ---------- 报警判定 ----------
        alarm = intrusion_count >= self._alarm_threshold
        if alarm:
            text = f"!! ALARM !! {intrusion_count} person(s) in restricted zone!"
            cv2.putText(annotated, text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # 左上角统计
        cv2.putText(annotated,
                    f"Zone: {intrusion_count}/{self._alarm_threshold}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ---------- 右上角 OSD：帧率 + 时间 ----------
        h, w = annotated.shape[:2]
        fps_text = f"FPS: {self._current_fps:.1f}"
        time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # FPS 颜色：绿(>=20) / 黄(>=10) / 红(<10)
        if self._current_fps >= 20:
            fps_color = (0, 230, 118)      # 绿 (BGR)
        elif self._current_fps >= 10:
            fps_color = (0, 170, 255)      # 黄
        else:
            fps_color = (51, 51, 255)      # 红

        # 先画半透明背景条，让文字在各种画面上都清晰可读
        overlay_osd = annotated.copy()
        cv2.rectangle(overlay_osd, (w - 280, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay_osd, 0.5, annotated, 0.5, 0, annotated)

        cv2.putText(annotated, fps_text, (w - 270, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        cv2.putText(annotated, time_text, (w - 270, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return annotated, intrusion_count, alarm

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """安全停止推理线程。"""
        self._running = False


# ============================================================================
#  2. 主窗口
# ============================================================================

class MainWindow(QMainWindow):
    """
    系统主窗口，整合视频显示、多边形绘制、状态面板和后台线程管理。
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("危险区域人员闯入检测与报警系统")
        self.resize(1100, 750)

        # ---------- 后台线程引用（延迟初始化） ----------
        self._producer: StreamProducer | SocketStreamProducer | None = None
        self._worker: InferenceWorker | None = None
        self._frame_queue = DropOldQueue(maxsize=2)
        self._is_running = False

        # 最后一帧原始画面（无标注），用于停止后清除区域时恢复干净画面
        self._last_raw_pixmap: Optional[QPixmap] = None

        # ---------- UI 控件 ----------
        self._init_ui()
        self._connect_signals()

    # ==================================================================
    #  UI 初始化
    # ==================================================================

    def _init_ui(self) -> None:
        # ---- 视频画面 ----
        self._video_label = VideoDisplayLabel()

        # ---- 控制栏 ----
        self._combo_source = QComboBox()
        self._combo_source.addItem("本地摄像头 (0)", 0)
        self._combo_source.addItem("本地摄像头 (1)", 1)
        self._combo_source.addItem("开发板 Socket (192.168.1.100:8888)", "socket://192.168.1.100:8888")
        self._combo_source.setEditable(True)
        self._combo_source.setToolTip(
            "输入摄像头编号、视频文件路径、RTSP URL\n"
            "或 socket://IP:PORT 格式连接开发板"
        )

        self._btn_browse = QPushButton("浏览文件")
        self._spin_threshold = QSpinBox()
        self._spin_threshold.setRange(1, 50)
        self._spin_threshold.setValue(3)
        self._spin_threshold.setPrefix("报警阈值: ")
        self._spin_threshold.setSuffix(" 人")

        self._btn_start = QPushButton("▶ 启动检测")
        self._btn_stop = QPushButton("■ 停止")
        self._btn_stop.setEnabled(False)
        self._btn_clear_poly = QPushButton("清除区域")

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("视频源:"))
        ctrl_layout.addWidget(self._combo_source, stretch=1)
        ctrl_layout.addWidget(self._btn_browse)
        ctrl_layout.addWidget(self._spin_threshold)
        ctrl_layout.addWidget(self._btn_start)
        ctrl_layout.addWidget(self._btn_stop)
        ctrl_layout.addWidget(self._btn_clear_poly)

        # ---- 状态栏 ----
        self._label_status = QLabel("就绪  |  左键绘制警戒区域  |  右键闭合多边形")
        self._label_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_status.setStyleSheet(
            "font-size: 14px; padding: 6px; background: #2b2b2b; color: #ccc;"
        )

        self._label_alarm = QLabel("")
        self._label_alarm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_alarm.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self._label_alarm.setFixedHeight(36)

        # ---- 布局 ----
        main_layout = QVBoxLayout()
        main_layout.addWidget(self._video_label, stretch=1)
        main_layout.addLayout(ctrl_layout)
        main_layout.addWidget(self._label_alarm)
        main_layout.addWidget(self._label_status)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ==================================================================
    #  信号 / 槽连接
    # ==================================================================

    def _connect_signals(self) -> None:
        self._btn_browse.clicked.connect(self._on_browse)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_clear_poly.clicked.connect(self._on_clear_polygon)

        # 多边形绘制完成 → 更新给推理线程
        self._video_label.polygon_finished.connect(self._on_polygon_finished)

    # ==================================================================
    #  槽函数
    # ==================================================================

    @Slot()
    def _on_clear_polygon(self) -> None:
        """清除 UI 上的多边形，同时清除推理线程中的多边形坐标。
        若已停止，用缓存的原始帧替换冻结画面，去掉红色区域但保留视频画面。"""
        self._video_label.clear_polygon()
        if self._worker is not None:
            self._worker.update_polygon([])
        if not self._is_running and self._last_raw_pixmap is not None:
            # 用无标注的原始帧替换带红色区域的冻结画面
            self._video_label.set_pixmap_image(self._last_raw_pixmap)

    @Slot()
    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv);;所有文件 (*)",
        )
        if path:
            # 将文件路径作为新条目加入 combo 并选中，
            # 确保 currentData() 返回 None 而非之前预设项的数据
            idx = self._combo_source.findText(path)
            if idx < 0:
                self._combo_source.addItem(path)       # data 默认为 None
                idx = self._combo_source.findText(path)
            self._combo_source.setCurrentIndex(idx)

    @Slot()
    def _on_start(self) -> None:
        """启动拉流线程和推理线程。"""
        if self._is_running:
            return

        # ---------- 解析视频源 ----------
        # 优先使用 combo 的 itemData（预设选项的整数/字符串），
        # 若用户手动输入则 currentData() 为 None，回退到解析文本
        source_data = self._combo_source.currentData()
        source_text = self._combo_source.currentText().strip()

        # 判断是否为 Socket 模式：
        #   - 预设选项的 data 以 "socket://" 开头
        #   - 用户也可以手动输入 "socket://IP:PORT"
        use_socket = False
        socket_port = 8888

        if isinstance(source_data, str) and source_data.startswith("socket://"):
            use_socket = True
            addr_str = source_data[len("socket://"):]
        elif source_text.lower().startswith("socket://"):
            use_socket = True
            addr_str = source_text[len("socket://"):]
        else:
            addr_str = ""

        if use_socket and addr_str:
            # 解析端口号（服务端始终 bind 0.0.0.0，这里的 IP 仅作显示）
            try:
                _, port_part = addr_str.rsplit(":", 1)
                socket_port = int(port_part)
            except (ValueError, IndexError):
                self._label_status.setText(
                    "Socket 地址格式错误，请使用 socket://IP:PORT"
                )
                return

        # 非 Socket 模式：使用 itemData（整数）或解析文本
        if not use_socket:
            if isinstance(source_data, int):
                source = source_data       # 预设选项，直接用整数摄像头编号
            elif source_text.isdigit():
                source = int(source_text)  # 用户手动输入的数字
            else:
                source = source_text       # 文件路径或 RTSP URL

        alarm_threshold = self._spin_threshold.value()

        # ---- 初始化检测器 ----
        self._label_status.setText("正在加载 YOLOv8 模型...")
        QApplication.processEvents()

        try:
            detector = YoloTracker(model_path="yolov8n.pt", conf=0.5, device="cuda")
        except Exception as e:
            self._label_status.setText(f"模型加载失败: {e}")
            return

        # ---- 创建并启动线程 ----
        self._frame_queue = DropOldQueue(maxsize=2)

        if use_socket:
            # Socket 模式：服务端始终 bind 0.0.0.0，等待开发板主动连接
            self._producer = SocketStreamProducer(
                host="0.0.0.0",
                port=socket_port,
                frame_queue=self._frame_queue,
            )
            source_display = f"Socket 监听端口 {socket_port}"
        else:
            # 本地摄像头 / 视频文件 / RTSP 模式
            self._producer = StreamProducer(
                source=source,
                frame_queue=self._frame_queue,
            )
            source_display = str(source)

        self._worker = InferenceWorker(
            frame_queue=self._frame_queue,
            detector=detector,
            alarm_threshold=alarm_threshold,
            parent=None,
        )

        # 推理线程信号 → 主线程槽
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.raw_frame_ready.connect(self._on_raw_frame)

        # 如果用户已经画好多边形，立即传给推理线程
        poly = self._video_label.polygon
        if poly:
            self._worker.update_polygon(poly)

        self._producer.start()
        self._worker.start()

        self._is_running = True
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._label_status.setText(
            f"检测运行中  |  视频源: {source_display}  |  报警阈值: {alarm_threshold} 人"
        )

    @Slot()
    def _on_stop(self) -> None:
        """停止所有后台线程。"""
        self._shutdown_threads()
        self._label_alarm.setText("")
        self._label_alarm.setStyleSheet("")
        self._label_status.setText("已停止  |  左键绘制警戒区域  |  右键闭合多边形")

    @Slot(list)
    def _on_polygon_finished(self, coords: list) -> None:
        """用户在 VideoDisplayLabel 上闭合多边形后，更新给推理线程。"""
        if self._worker is not None:
            self._worker.update_polygon(coords)
        n = len(coords)
        self._label_status.setText(
            self._label_status.text().split("|")[0].strip()
            + f"  |  警戒区域已设置 ({n} 个顶点)"
        )

    @Slot(QImage)
    def _on_raw_frame(self, qimg: QImage) -> None:
        """缓存最后一帧原始画面（无标注），供停止后清除区域使用。"""
        self._last_raw_pixmap = QPixmap.fromImage(qimg)

    @Slot(QImage, int, bool)
    def _on_frame_ready(self, qimg: QImage, count: int, alarm: bool) -> None:
        """
        主线程槽函数：接收推理线程发来的标注帧并更新 UI。
        绝对禁止在子线程中调用此方法——Qt Signal 保证在主线程执行。
        """
        # ---------- 更新视频画面 ----------
        self._video_label.set_pixmap_image(QPixmap.fromImage(qimg))

        # 更新报警状态
        if alarm:
            self._label_alarm.setText(
                f"⚠ 警报！区域内检测到 {count} 人闯入！"
            )
            self._label_alarm.setStyleSheet(
                "color: #ff3333; background: #4a0000; border-radius: 4px;"
            )
        elif count > 0:
            self._label_alarm.setText(f"区域内检测到 {count} 人")
            self._label_alarm.setStyleSheet(
                "color: #ffaa00; background: transparent;"
            )
        else:
            self._label_alarm.setText("")
            self._label_alarm.setStyleSheet("")

    # ==================================================================
    #  线程生命周期管理
    # ==================================================================

    def _shutdown_threads(self) -> None:
        """优雅地停止所有后台线程并等待退出。"""
        if not self._is_running:
            return

        print("[MainWindow] 正在停止后台线程...")

        # 先停止生产者（不再往队列放帧）
        if self._producer is not None:
            self._producer.stop()

        # 再停止消费者
        if self._worker is not None:
            self._worker.stop()
            self._worker.frame_ready.disconnect(self._on_frame_ready)
            self._worker.raw_frame_ready.disconnect(self._on_raw_frame)

        # 等待线程退出（给予足够超时）
        if self._producer is not None:
            self._producer.join(timeout=3.0)
            self._producer = None

        if self._worker is not None:
            self._worker.wait(3000)        # QThread.wait() 参数为毫秒
            self._worker = None

        self._is_running = False
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        print("[MainWindow] 所有后台线程已停止。")

    # ==================================================================
    #  窗口关闭事件 —— 确保线程安全退出
    # ==================================================================

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭窗口时优雅清理线程，防止崩溃。"""
        self._shutdown_threads()
        event.accept()


# ============================================================================
#  入口
# ============================================================================

def main() -> None:
    app = QApplication(sys.argv)

    # 全局深色风格
    app.setStyleSheet("""
        QMainWindow { background: #1e1e1e; }
        QLabel { color: #ddd; }
        QPushButton {
            background: #3c3c3c; color: #ddd; border: 1px solid #555;
            padding: 5px 12px; border-radius: 3px;
        }
        QPushButton:hover { background: #505050; }
        QPushButton:disabled { color: #666; }
        QComboBox, QSpinBox {
            background: #2b2b2b; color: #ddd; border: 1px solid #555;
            padding: 4px; border-radius: 3px;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
