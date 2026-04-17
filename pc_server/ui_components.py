"""
ui_components.py
危险区域人员闯入检测与报警系统 —— PySide6 可交互视频显示控件

功能：
  - VideoDisplayLabel：继承 QLabel，支持鼠标点击绘制多边形警戒区域
  - 左键：添加顶点
  - 右键：闭合多边形
  - 提供 polygon 属性向外输出坐标列表，供后端推理线程使用
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor,
    QMouseEvent, QPaintEvent, QPolygon,
)
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QWidget, QFileDialog, QSizePolicy,
)


class VideoDisplayLabel(QLabel):
    """
    可交互的视频显示控件。

    使用方式：
      1. set_frame(np.ndarray) 或 set_pixmap(QPixmap) 更新底图
      2. 用户左键点击添加多边形顶点，右键闭合
      3. 读取 polygon 属性获取坐标列表 [(x, y), ...]
      4. polygon_finished 信号在闭合时发射，携带坐标列表

    坐标系说明：
      所有坐标均为**原始图像像素坐标**，而非控件显示坐标，
      因此缩放显示不会影响坐标精度。
    """

    # 多边形闭合时发射，参数为顶点坐标列表
    polygon_finished = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #1e1e1e;")

        # ---------- 内部状态 ----------
        self._source_pixmap: Optional[QPixmap] = None   # 原始底图
        self._points: List[QPoint] = []                  # 当前绘制中的顶点（图像坐标）
        self._closed: bool = False                       # 多边形是否已闭合
        self._display_rect = None                        # 底图在控件中的实际绘制区域

    # ==================================================================
    #  公开接口
    # ==================================================================

    @property
    def polygon(self) -> List[Tuple[int, int]]:
        """返回已闭合多边形的顶点坐标列表（原始图像像素坐标）。
        未闭合时返回空列表。"""
        if not self._closed:
            return []
        return [(p.x(), p.y()) for p in self._points]

    @property
    def polygon_normalized(self) -> List[Tuple[float, float]]:
        """返回归一化坐标 (0.0~1.0)，用于跨分辨率保存/恢复。
        坐标 = 像素坐标 / 图像宽高。未闭合或无底图时返回空列表。"""
        if not self._closed or self._source_pixmap is None:
            return []
        img_w = self._source_pixmap.width()
        img_h = self._source_pixmap.height()
        if img_w == 0 or img_h == 0:
            return []
        return [(p.x() / img_w, p.y() / img_h) for p in self._points]

    @property
    def is_drawing(self) -> bool:
        """是否正在绘制中（有顶点但尚未闭合）。"""
        return len(self._points) > 0 and not self._closed

    def clear_polygon(self) -> None:
        """清除当前多边形，重置为未绘制状态。"""
        self._points.clear()
        self._closed = False
        self.update()

    def clear_display(self) -> None:
        """清除底图，恢复到无画面的初始状态。"""
        self._source_pixmap = None
        self._display_rect = None
        self.update()

    def set_polygon(self, coords: List[Tuple[int, int]]) -> None:
        """
        以编程方式设置多边形（绝对像素坐标）。
        设置后状态为已闭合，会触发 polygon_finished 信号。
        """
        self._points = [QPoint(x, y) for x, y in coords]
        self._closed = True
        self.polygon_finished.emit(self.polygon)
        self.update()

    def set_polygon_normalized(self, norm_coords: List[Tuple[float, float]],
                                img_w: int, img_h: int) -> None:
        """
        用归一化坐标 (0.0~1.0) 恢复多边形，按指定的图像分辨率转换为像素坐标。
        用于从配置文件恢复警戒区域（跨分辨率通用）。
        """
        pixel_coords = [(int(nx * img_w), int(ny * img_h))
                        for nx, ny in norm_coords]
        self.set_polygon(pixel_coords)

    def set_frame(self, frame: np.ndarray) -> None:
        """
        用 OpenCV BGR 帧更新底图。
        在主线程中调用（由 Qt Signal 触发）。
        """
        if frame is None:
            return
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # OpenCV BGR → Qt RGB
        rgb = frame[..., ::-1].copy()
        qimg = QImage(rgb.data, w, h, bytes_per_line,
                      QImage.Format.Format_RGB888)
        self._source_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_pixmap_image(self, pixmap: QPixmap) -> None:
        """直接设置 QPixmap 底图。"""
        self._source_pixmap = pixmap
        self.update()

    # ==================================================================
    #  坐标转换：控件坐标 ↔ 图像坐标
    # ==================================================================

    def _calc_display_rect(self) -> None:
        """
        计算底图在控件中按比例缩放后的实际绘制区域，
        用于鼠标坐标与图像坐标之间的换算。
        """
        if self._source_pixmap is None:
            self._display_rect = None
            return

        pm = self._source_pixmap
        widget_w, widget_h = self.width(), self.height()
        img_w, img_h = pm.width(), pm.height()

        # 保持宽高比缩放
        scale = min(widget_w / img_w, widget_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        offset_x = (widget_w - disp_w) // 2
        offset_y = (widget_h - disp_h) // 2

        self._display_rect = (offset_x, offset_y, disp_w, disp_h, scale)

    def _widget_to_image(self, pos: QPoint) -> Optional[QPoint]:
        """将控件坐标转换为原始图像像素坐标。落在图像外则返回 None。"""
        self._calc_display_rect()
        if self._display_rect is None:
            return None

        ox, oy, dw, dh, scale = self._display_rect
        x = pos.x() - ox
        y = pos.y() - oy

        if x < 0 or y < 0 or x >= dw or y >= dh:
            return None                   # 点击在图像区域之外

        img_x = int(x / scale)
        img_y = int(y / scale)
        return QPoint(img_x, img_y)

    def _image_to_widget(self, pt: QPoint) -> QPoint:
        """将原始图像坐标转换为控件坐标（用于绘制叠加层）。"""
        if self._display_rect is None:
            return pt
        ox, oy, _, _, scale = self._display_rect
        return QPoint(int(pt.x() * scale) + ox,
                      int(pt.y() * scale) + oy)

    # ==================================================================
    #  鼠标事件
    # ==================================================================

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._source_pixmap is None:
            return

        img_pt = self._widget_to_image(event.position().toPoint())
        if img_pt is None:
            return                        # 点击在图像之外，忽略

        # ---------- 左键：添加顶点 ----------
        if event.button() == Qt.MouseButton.LeftButton:
            if self._closed:
                # 已闭合状态下左键点击 → 自动清除旧多边形，开始新绘制
                self.clear_polygon()
            self._points.append(img_pt)
            self.update()

        # ---------- 右键：闭合多边形 ----------
        elif event.button() == Qt.MouseButton.RightButton:
            if len(self._points) >= 3 and not self._closed:
                self._closed = True
                self.polygon_finished.emit(self.polygon)
                self.update()

    # ==================================================================
    #  绘制
    # ==================================================================

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ---------- 1. 绘制底图（按比例缩放居中） ----------
        if self._source_pixmap is not None:
            self._calc_display_rect()
            ox, oy, dw, dh, _ = self._display_rect
            scaled = self._source_pixmap.scaled(
                dw, dh,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(ox, oy, scaled)
        else:
            # 无底图时显示提示文字
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "暂无画面\n请加载图片或启动视频流")
            painter.end()
            return

        # ---------- 2. 绘制多边形叠加层 ----------
        if len(self._points) == 0:
            painter.end()
            return

        # 转为控件坐标系
        widget_pts = [self._image_to_widget(p) for p in self._points]

        if self._closed and len(widget_pts) >= 3:
            # 已闭合：半透明填充 + 实线边框
            q_polygon = QPolygon(widget_pts)
            painter.setBrush(QBrush(QColor(255, 0, 0, 60)))
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine))
            painter.drawPolygon(q_polygon)
        else:
            # 绘制中：虚线连线 + 顶点圆点
            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            for i in range(len(widget_pts) - 1):
                painter.drawLine(widget_pts[i], widget_pts[i + 1])

        # 绘制每个顶点
        painter.setPen(Qt.PenStyle.NoPen)
        for i, wp in enumerate(widget_pts):
            if self._closed:
                painter.setBrush(QBrush(QColor(255, 0, 0)))
            else:
                painter.setBrush(QBrush(QColor(0, 255, 0)))
            painter.drawEllipse(wp, 5, 5)

            # 顶点序号标注
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(wp.x() + 8, wp.y() - 8, str(i + 1))
            painter.setPen(Qt.PenStyle.NoPen)

        painter.end()


# ============================================================================
#  测试主窗口
# ============================================================================

class _TestWindow(QMainWindow):
    """用于测试 VideoDisplayLabel 多边形绘制功能的主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VideoDisplayLabel 绘制测试")
        self.resize(900, 650)

        # ---------- 控件 ----------
        self._video_label = VideoDisplayLabel()
        self._btn_load = QPushButton("加载图片")
        self._btn_clear = QPushButton("清除区域")
        self._btn_print = QPushButton("打印坐标")
        self._status = QLabel("左键添加顶点 | 右键闭合多边形 | 闭合后左键重新绘制")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: #888; font-size: 13px;")

        # ---------- 布局 ----------
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self._btn_load)
        btn_layout.addWidget(self._btn_clear)
        btn_layout.addWidget(self._btn_print)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._video_label, stretch=1)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self._status)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # ---------- 信号连接 ----------
        self._btn_load.clicked.connect(self._on_load)
        self._btn_clear.clicked.connect(self._video_label.clear_polygon)
        self._btn_print.clicked.connect(self._on_print)
        self._video_label.polygon_finished.connect(self._on_polygon_finished)

    # ------------------------------------------------------------------
    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)",
        )
        if path:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self._video_label.set_pixmap_image(pixmap)
                self._status.setText(f"已加载: {path}")

    def _on_print(self) -> None:
        coords = self._video_label.polygon
        if coords:
            print(f"[多边形坐标] {coords}")
            self._status.setText(f"坐标: {coords}")
        else:
            self._status.setText("多边形尚未闭合，无坐标可输出。")

    def _on_polygon_finished(self, coords: list) -> None:
        self._status.setText(f"多边形已闭合，共 {len(coords)} 个顶点")
        print(f"[polygon_finished] {coords}")


# ============================================================================
#  入口
# ============================================================================

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = _TestWindow()
    win.show()
    sys.exit(app.exec())
