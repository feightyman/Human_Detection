import cv2
from ultralytics import YOLO
import socket
import struct
import numpy as np

# 配置服务器的 IP 和端口
HOST = '0.0.0.0'  # 0.0.0.0 表示监听本机所有的 IP 地址
PORT = 8888  # 必须跟你 C 语言里定义的 S_PORT 一模一样


def main():
    # 1. 加载模型
    model = YOLO("yolov8n.pt")

    # 2. 初始化 Socket 服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 这行代码是为了防止程序重启时报 "端口被占用" 的错误
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"服务器已启动，正在监听 {PORT} 端口，等待开发板连接...")

    # 3. 阻塞等待开发板连接
    conn, addr = server_socket.accept()
    print(f"太棒了！开发板已连接，IP地址是: {addr}")

    try:
        while True:
            # ---------------- 解决粘包：先收 4 字节的包头 ----------------
            header = conn.recv(4)
            if not header:
                print("开发板断开了连接")
                break

            # 解析图像大小
            frame_size = struct.unpack('<I', header)[0]

            # ---------------- 循环接收完整的一张照片 ----------------
            data = b''
            while len(data) < frame_size:
                packet = conn.recv(frame_size - len(data))
                if not packet:
                    break
                data += packet

            if not data:
                break

            # ---------------- 解码并显示图像 ----------------
            img_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # ---------------- 安全的推理逻辑 ----------------
            # 确保解码成功了，才进行 YOLO 推理
            if frame is not None:
                # 可选：如果你想同时看原始画面，可以取消注释下面这行
                # cv2.imshow('i.MX6ULL Camera Stream (Raw)', frame)

                # 进行检测
                results = model.predict(source=frame, stream=True, classes=[0], conf=0.5, show_labels=True,
                                        show_conf=True)

                for result in results:
                    # 获取带有标注的图像
                    annotated_frame = result.plot()

                    # 显示最终结果画面
                    cv2.imshow("YOLOv8 人形检测 (Processed)", annotated_frame)

                # 全局唯一的按键检测，放在帧处理的最后
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("手动退出视频流")
                    break
            else:
                print("警告：收到了一张损坏的图片，解码失败跳过")

    except Exception as e:
        print(f"发生运行异常: {e}")

    finally:
        # 5. 清理战场 (修复了这里的逻辑)
        print("正在清理资源并关闭服务...")
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()  # 销毁所有 OpenCV 窗口即可
        print("服务已完全关闭。")


if __name__ == '__main__':
    main()