#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <linux/fb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <signal.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <time.h>
#include <math.h>

#define FRAMEBUFFER_COUNT   3               //帧缓冲数量
#define FRAME_WIDTH         640             //帧宽度
#define FRAME_HEIGHT        480             //帧高度

#define S_PORT              8888            //服务器端口号（视频流）
#define ALARM_PORT          8889            //报警指令监听端口

/* ---- 边缘端帧差法运动检测参数 ---- */
#define MOTION_WIDTH        160             //运动检测降采样宽度
#define MOTION_HEIGHT       120             //运动检测降采样高度
#define MOTION_THRESHOLD    25              //像素差阈值（0~255）
#define MOTION_RATIO        0.02            //变化像素占比阈值（2%即判定为运动）
#define HEARTBEAT_INTERVAL  2               //心跳帧间隔（秒），静止时仍定期发帧

/* 正点原子 Alpha 开发板 LED / 蜂鸣器 sysfs 路径
 * 若使用自定义字符设备驱动，可替换为 /dev/beep 等设备节点 */
#define BEEP_SYSFS_PATH     "/sys/class/leds/beep/brightness"
#define LED_SYSFS_PATH      "/sys/class/leds/sys-led/brightness"

/*** 摄像头像素格式及其描述信息 ***/
typedef struct camera_format {
    unsigned char description[32];  //字符串描述信息
    unsigned int pixelformat;       //像素格式
} cam_fmt;

/*** 描述一个帧缓冲的信息 ***/
typedef struct cam_buf_info {
    unsigned short *start;      //帧缓冲起始地址
    unsigned long length;       //帧缓冲长度
} cam_buf_info;


static int v4l2_fd = -1;                //摄像头设备文件描述符
static cam_buf_info buf_infos[FRAMEBUFFER_COUNT];
static cam_fmt cam_fmts[10];
static int frm_width, frm_height;   //视频帧宽度和高度
static int sockfd = -1;

/**
 * 连接 PC 端视频流服务器（单次尝试）
 *
 * 与旧版不同：连接失败不再 exit()，而是返回 -1，
 * 由调用方决定是否重试，支持断线重连机制。
 *
 * @param s_ip  服务器 IP 地址字符串
 * @return 0 成功, -1 失败（sockfd 已关闭）
 */
static int connect_server(const char *s_ip)
{
    struct sockaddr_in server_addr = {0};

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("[Connect] socket error");
        return -1;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(S_PORT);
    inet_pton(AF_INET, s_ip, &server_addr.sin_addr.s_addr);

    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        fprintf(stderr, "[Connect] 连接 %s:%d 失败: %s\n",
                s_ip, S_PORT, strerror(errno));
        close(sockfd);
        sockfd = -1;
        return -1;
    }

    printf("[Connect] 服务器 %s:%d 连接成功\n", s_ip, S_PORT);
    return 0;
}

/**
 * 带自动重试的服务器连接
 *
 * 首次连接立即尝试，失败后以退避间隔重试（1s → 2s → 4s → 最大 10s）。
 * 重连过程中摄像头仍在采集（内核驱动缓冲），连上后立即拿到最新帧。
 *
 * @param s_ip  服务器 IP 地址字符串
 */
static void connect_server_with_retry(const char *s_ip)
{
    int retry_count = 0;
    int delay = 1;                          /* 初始重试间隔（秒） */
    const int max_delay = 10;               /* 最大重试间隔（秒） */

    while (connect_server(s_ip) < 0) {
        retry_count++;
        printf("[Connect] 第 %d 次重连失败，%d 秒后重试...\n", retry_count, delay);
        sleep(delay);
        /* 指数退避：1 → 2 → 4 → 8 → 10(上限) */
        delay = delay * 2;
        if (delay > max_delay)
            delay = max_delay;
    }
}


static int v4l2_dev_init(const char *device)
{
    struct v4l2_capability cap = {0};

    /* 打开摄像头 */
    v4l2_fd = open(device, O_RDWR);
    if (0 > v4l2_fd) {
        fprintf(stderr, "open error: %s: %s\n", device, strerror(errno));
        return -1;
    }

    /* 查询设备功能 */
    ioctl(v4l2_fd, VIDIOC_QUERYCAP, &cap);

    /* 判断是否是视频采集设备 */
    if (!(V4L2_CAP_VIDEO_CAPTURE & cap.capabilities)) {
        fprintf(stderr, "Error: %s: No capture video device!\n", device);
        close(v4l2_fd);
        return -1;
    }

    return 0;
}

static void v4l2_enum_formats(void)
{
    struct v4l2_fmtdesc fmtdesc = {0};

    /* 枚举摄像头所支持的所有像素格式以及描述信息 */
    fmtdesc.index = 0;
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    while (0 == ioctl(v4l2_fd, VIDIOC_ENUM_FMT, &fmtdesc)) {

        // 将枚举出来的格式以及描述信息存放在数组中
        cam_fmts[fmtdesc.index].pixelformat = fmtdesc.pixelformat;
        strcpy(cam_fmts[fmtdesc.index].description, fmtdesc.description);
        printf("type:%s\n", cam_fmts[fmtdesc.index].description);
        fmtdesc.index++;
    }
    printf("\n");
}

static void v4l2_print_formats(void)
{
    struct v4l2_frmsizeenum frmsize = {0};
    struct v4l2_frmivalenum frmival = {0};
    int i;

    frmsize.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    frmival.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for (i = 0; cam_fmts[i].pixelformat; i++) {

        printf("format<0x%x>, description<%s>\n", cam_fmts[i].pixelformat,
                    cam_fmts[i].description);

        /* 枚举出摄像头所支持的所有视频采集分辨率 */
        frmsize.index = 0;
        frmsize.pixel_format = cam_fmts[i].pixelformat;
        frmival.pixel_format = cam_fmts[i].pixelformat;
        while (0 == ioctl(v4l2_fd, VIDIOC_ENUM_FRAMESIZES, &frmsize)) {

            printf("size<%d*%d> ",
                    frmsize.discrete.width,
                    frmsize.discrete.height);
            frmsize.index++;

            /* 获取摄像头视频采集帧率 */
            frmival.index = 0;
            frmival.width = frmsize.discrete.width;
            frmival.height = frmsize.discrete.height;
            while (0 == ioctl(v4l2_fd, VIDIOC_ENUM_FRAMEINTERVALS, &frmival)) {

                printf("<%dfps>", frmival.discrete.denominator /
                        frmival.discrete.numerator);
                frmival.index++;
            }
            printf("\n");
        }
        printf("\n");
    }
}

static int v4l2_set_format(void)
{
    struct v4l2_format fmt = {0};
    struct v4l2_streamparm streamparm = {0};

    /* 设置帧格式 */
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;//type类型
    fmt.fmt.pix.width = FRAME_HEIGHT;  //视频帧宽度
    fmt.fmt.pix.height = FRAME_WIDTH;   //视频帧高度
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;  //像素格式
    if (0 > ioctl(v4l2_fd, VIDIOC_S_FMT, &fmt)) {
        fprintf(stderr, "ioctl error: VIDIOC_S_FMT: %s\n", strerror(errno));
        return -1;
    }

    /*** 判断是否已经设置为我们要求的MJPEG像素格式
    如果没有设置成功表示该设备不支持MJPEG像素格式 */
    if (V4L2_PIX_FMT_MJPEG != fmt.fmt.pix.pixelformat) {
        fprintf(stderr, "Error: the device does not support MJPEG format!\n");
        return -1;
    }
    printf("成功设置为MJPEG格式\n");
    frm_width = fmt.fmt.pix.width;  //获取实际的帧宽度
    frm_height = fmt.fmt.pix.height;//获取实际的帧高度
    printf("视频帧大小<%d * %d>\n", frm_width, frm_height);

    /* 获取streamparm */
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(v4l2_fd, VIDIOC_G_PARM, &streamparm);

    /** 判断是否支持帧率设置 **/
    if (V4L2_CAP_TIMEPERFRAME & streamparm.parm.capture.capability) {
        streamparm.parm.capture.timeperframe.numerator = 1;
        streamparm.parm.capture.timeperframe.denominator = 30;//30fps
        if (0 > ioctl(v4l2_fd, VIDIOC_S_PARM, &streamparm)) {
            fprintf(stderr, "ioctl error: VIDIOC_S_PARM: %s\n", strerror(errno));
            return -1;
        }
    }

    return 0;
}

static int v4l2_init_buffer(void)
{
    struct v4l2_requestbuffers reqbuf = {0};
    struct v4l2_buffer buf = {0};

    /* 申请帧缓冲 */
    reqbuf.count = FRAMEBUFFER_COUNT;       //帧缓冲的数量
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    if (0 > ioctl(v4l2_fd, VIDIOC_REQBUFS, &reqbuf)) {
        fprintf(stderr, "ioctl error: VIDIOC_REQBUFS: %s\n", strerror(errno));
        return -1;
    }
    // 检查内核分配了几个缓冲区
    if (reqbuf.count < FRAMEBUFFER_COUNT) {
        printf("警告：只分配了 %d 个缓冲区！\n", reqbuf.count);
    }
    /* 建立内存映射 */
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    for (buf.index = 0; buf.index < FRAMEBUFFER_COUNT; buf.index++) {

        ioctl(v4l2_fd, VIDIOC_QUERYBUF, &buf);
        buf_infos[buf.index].length = buf.length;
        buf_infos[buf.index].start = mmap(NULL, buf.length,
                PROT_READ | PROT_WRITE, MAP_SHARED,
                v4l2_fd, buf.m.offset);
        if (MAP_FAILED == buf_infos[buf.index].start) {
            perror("mmap error");
            return -1;
        }
    }

    /* 入队 */
    for (buf.index = 0; buf.index < FRAMEBUFFER_COUNT; buf.index++) {

        if (0 > ioctl(v4l2_fd, VIDIOC_QBUF, &buf)) {
            fprintf(stderr, "ioctl error: VIDIOC_QBUF: %s\n", strerror(errno));
            return -1;
        }
    }

    return 0;
}

static int v4l2_stream_on(void)
{
    /* 打开摄像头、摄像头开始采集数据 */
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (0 > ioctl(v4l2_fd, VIDIOC_STREAMON, &type)) {
        fprintf(stderr, "ioctl error: VIDIOC_STREAMON: %s\n", strerror(errno));
        return -1;
    }

    return 0;
}

/* ============================================================================
 *  边缘端帧差法运动检测
 *  利用 libjpeg 将 MJPEG 帧解码为灰度缩略图，与上一帧做像素级差分，
 *  判断画面是否有明显运动。仅在检测到运动或心跳超时时才发帧给 PC 端，
 *  大幅降低网络带宽占用，体现端云协同的边缘计算价值。
 * ============================================================================ */

/* libjpeg 错误处理：遇到解码错误时通过 longjmp 跳出，避免程序崩溃 */
struct my_error_mgr {
    struct jpeg_error_mgr pub;          /* 标准 libjpeg 错误管理器 */
    jmp_buf setjmp_buffer;              /* longjmp 跳转目标 */
};

static void my_error_exit(j_common_ptr cinfo)
{
    struct my_error_mgr *myerr = (struct my_error_mgr *)cinfo->err;
    (*cinfo->err->output_message)(cinfo);   /* 打印错误信息 */
    longjmp(myerr->setjmp_buffer, 1);       /* 跳回调用点 */
}

/**
 * 将 JPEG 内存块解码为灰度缩略图
 *
 * 利用 libjpeg 的 DCT 域缩放功能 (scale_denom=4)，直接在解码阶段
 * 将 640×480 缩小到 160×120，避免全尺寸解码后再缩放的 CPU 开销。
 * 输出色彩空间设为 JCS_GRAYSCALE，省掉 YCbCr→RGB 转换。
 *
 * @param jpeg_data  JPEG 数据指针（V4L2 mmap 缓冲区）
 * @param jpeg_size  JPEG 数据字节数
 * @param gray_out   输出灰度缓冲区，大小 >= MOTION_WIDTH * MOTION_HEIGHT
 * @return 0 成功, -1 解码失败
 */
static int jpeg_to_gray(const unsigned char *jpeg_data, unsigned long jpeg_size,
                         unsigned char *gray_out)
{
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    JSAMPROW row_pointer[1];
    int row_stride;

    /* 设置错误处理 */
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        /* 解码出错，清理后返回 -1 */
        jpeg_destroy_decompress(&cinfo);
        return -1;
    }

    jpeg_create_decompress(&cinfo);

    /* 从内存读取 JPEG 数据（而非文件） */
    jpeg_mem_src(&cinfo, jpeg_data, jpeg_size);

    jpeg_read_header(&cinfo, TRUE);

    /* 请求灰度输出 + 1/4 DCT 缩放（640→160, 480→120） */
    cinfo.out_color_space = JCS_GRAYSCALE;
    cinfo.scale_num = 1;
    cinfo.scale_denom = 4;

    jpeg_start_decompress(&cinfo);

    row_stride = cinfo.output_width;    /* 灰度图每行字节数 = 宽度 */

    /* 逐行读取解码后的灰度像素 */
    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer[0] = gray_out + cinfo.output_scanline * row_stride;
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return 0;
}

/**
 * 帧差法运动检测
 *
 * 逐像素比较当前帧与上一帧灰度图的差异：
 *   对每个像素 i，若 |curr[i] - prev[i]| > MOTION_THRESHOLD，视为变化像素。
 *   变化像素占比超过 MOTION_RATIO 时，判定为有运动。
 *
 * @param curr      当前帧灰度图
 * @param prev      上一帧灰度图
 * @param total_px  像素总数 (MOTION_WIDTH * MOTION_HEIGHT)
 * @return 1 = 检测到运动, 0 = 画面静止
 */
static int detect_motion(const unsigned char *curr, const unsigned char *prev,
                          int total_px)
{
    int changed = 0;
    int threshold_count = (int)(total_px * MOTION_RATIO);
    int i;

    for (i = 0; i < total_px; i++) {
        int diff = (int)curr[i] - (int)prev[i];
        if (diff < 0) diff = -diff;     /* abs() */
        if (diff > MOTION_THRESHOLD) {
            changed++;
            /* 提前退出：已经超过阈值，无需继续遍历 */
            if (changed > threshold_count)
                return 1;
        }
    }
    return 0;
}

static void v4l2_read_data(void)
{
    struct v4l2_buffer buf = {0};

    /* 帧差法运动检测所需的灰度缓冲区
     * 注意：不使用 static，每次重连后重新初始化，
     * 避免拿断线前的旧灰度帧做对比产生误判 */
    unsigned char prev_gray[MOTION_WIDTH * MOTION_HEIGHT];
    unsigned char curr_gray[MOTION_WIDTH * MOTION_HEIGHT];
    int has_prev = 0;                       /* 是否已有上一帧（首帧无条件发送） */
    time_t last_send_time = 0;              /* 上次发送的时间戳（用于心跳帧计时） */
    int total_px = MOTION_WIDTH * MOTION_HEIGHT;
    unsigned long total_frames = 0;         /* 采集帧计数 */
    unsigned long sent_frames = 0;          /* 实际发送帧计数 */
    unsigned long skipped_frames = 0;       /* 跳过帧计数 */

    printf("[Motion] 边缘端帧差法运动检测已启用\n");
    printf("[Motion] 参数: 检测分辨率=%dx%d, 像素阈值=%d, 占比阈值=%.1f%%, 心跳=%ds\n",
           MOTION_WIDTH, MOTION_HEIGHT, MOTION_THRESHOLD,
           MOTION_RATIO * 100, HEARTBEAT_INTERVAL);

    for ( ; ; ) {
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(v4l2_fd, VIDIOC_DQBUF, &buf) < 0) {
            perror("出队失败 VIDIOC_DQBUF");
            continue;
        }
        total_frames++;

        uint32_t frame_size = buf.bytesused;
        const unsigned char *jpeg_data = (const unsigned char *)buf_infos[buf.index].start;
        int should_send = 0;
        const char *send_reason = "Unknown";

        /* ---------- 1. JPEG → 灰度缩略图 ---------- */
        if (jpeg_to_gray(jpeg_data, frame_size, curr_gray) < 0) {
            /* 解码失败：安全起见发送该帧（可能是关键画面） */
            fprintf(stderr, "[Motion] 灰度解码失败，无条件发送此帧\n");
            should_send = 1;
            send_reason = "DecodeErr";
        }
        /* ---------- 2. 首帧无条件发送 ---------- */
        else if (!has_prev) {
            memcpy(prev_gray, curr_gray, total_px);
            has_prev = 1;
            should_send = 1;
            send_reason = "FirstFrame";
        }
        /* ---------- 3. 帧差法运动判定 + 心跳超时 ---------- */
        else {
            int motion = detect_motion(curr_gray, prev_gray, total_px);
            time_t now = time(NULL);
            int elapsed = (int)(now - last_send_time);

            if (motion) {
                should_send = 1;
                send_reason = "Motion";
                /* 更新参考帧：运动期间持续跟踪最新画面 */
                memcpy(prev_gray, curr_gray, total_px);
            } else if (elapsed >= HEARTBEAT_INTERVAL) {
                should_send = 1;
                send_reason = "Heartbeat";
                memcpy(prev_gray, curr_gray, total_px);
            }
        }

        /* ---------- 4. 发送或跳过 ---------- */
        if (should_send) {
            /* TCP 粘包协议：[4字节帧大小] + [JPEG数据] */
            ssize_t ret1 = send(sockfd, &frame_size, sizeof(frame_size), 0);
            ssize_t ret2 = send(sockfd, jpeg_data, frame_size, 0);
            if (ret1 < 0 || ret2 < 0) {
                /* PC 端断开连接，send() 返回 -1 (SIGPIPE 已忽略，errno=EPIPE) */
                fprintf(stderr, "[Send] PC 端连接已断开 (errno=%d: %s)，停止发送\n",
                        errno, strerror(errno));
                /* 归还缓冲区后退出循环 */
                ioctl(v4l2_fd, VIDIOC_QBUF, &buf);
                break;
            }
            last_send_time = time(NULL);
            sent_frames++;
            printf("[%s] 发送一帧 | 缓冲区:%d, 大小:%u字节 | 统计: 发送%lu/采集%lu, 跳过%lu\n",
                   send_reason, buf.index, frame_size,
                   sent_frames, total_frames, skipped_frames);
        } else {
            skipped_frames++;
        }

        /* ---------- 5. 归还缓冲区 ---------- */
        if (ioctl(v4l2_fd, VIDIOC_QBUF, &buf) < 0) {
            perror("入队失败 VIDIOC_QBUF");
            break;
        }
    }
}


/* ============================================================================
 *  报警控制：通过 sysfs 控制蜂鸣器和 LED
 * ============================================================================ */

/**
 * 写入 sysfs 节点控制硬件开关
 * @param path   sysfs 路径（如 /sys/class/leds/beep/brightness）
 * @param value  "1" 开启，"0" 关闭
 */
static void sysfs_write(const char *path, const char *value)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) {
        /* 文件打不开可能是权限不足或路径不存在，仅打印警告不退出 */
        fprintf(stderr, "[Alarm] 无法打开 %s: %s\n", path, strerror(errno));
        return;
    }
    write(fd, value, strlen(value));
    close(fd);
}

/**
 * 设置报警状态：同时控制蜂鸣器和 LED
 * @param on  非零 = 开启报警，零 = 关闭报警
 */
static void alarm_set(int on)
{
    const char *val = on ? "1" : "0";
    sysfs_write(BEEP_SYSFS_PATH, val);
    sysfs_write(LED_SYSFS_PATH, val);
    printf("[Alarm] 报警状态: %s\n", on ? "开启" : "关闭");
}

/**
 * 报警指令接收线程函数
 *
 * 作为 TCP 服务端监听 ALARM_PORT，等待 PC 端连接后持续接收单字节指令：
 *   0x01 —— 开启蜂鸣器 + LED
 *   0x00 —— 关闭蜂鸣器 + LED
 *
 * 支持 PC 断开后自动回到等待重连状态。
 */
static void *alarm_listener_thread(void *arg)
{
    (void)arg;
    int server_fd, conn_fd;
    struct sockaddr_in addr = {0};
    unsigned char cmd;
    ssize_t n;

    /* 创建 TCP 服务端 */
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("[Alarm] socket error");
        return NULL;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(ALARM_PORT);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[Alarm] bind error");
        close(server_fd);
        return NULL;
    }

    if (listen(server_fd, 1) < 0) {
        perror("[Alarm] listen error");
        close(server_fd);
        return NULL;
    }

    printf("[Alarm] 报警指令监听线程已启动，端口 %d\n", ALARM_PORT);

    /* 外层循环：接受连接 → 内层循环：接收指令 → PC 断开后回到 accept */
    while (1) {
        printf("[Alarm] 等待 PC 端报警连接...\n");
        conn_fd = accept(server_fd, NULL, NULL);
        if (conn_fd < 0) {
            perror("[Alarm] accept error");
            continue;
        }
        printf("[Alarm] PC 端报警通道已连接\n");

        /* 设置接收超时：若 PC 端异常断开（如进程被杀、网络中断），
         * TCP 半开连接可能导致 recv() 永久阻塞。
         * 设置 5 秒超时，超时后检测到无数据即关闭报警并断开 */
        struct timeval tv;
        tv.tv_sec = 5;
        tv.tv_usec = 0;
        setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        int alarm_active = 0;       /* 跟踪当前报警状态 */
        int timeout_count = 0;      /* 连续超时计数 */

        /* 持续接收单字节报警指令 */
        while (1) {
            n = recv(conn_fd, &cmd, 1, 0);
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    /* recv 超时 */
                    timeout_count++;
                    if (alarm_active && timeout_count >= 2) {
                        /* 报警状态下连续超时 → PC 可能已异常断开，自动关闭报警 */
                        printf("[Alarm] 报警通道超时，自动关闭报警（安全措施）\n");
                        alarm_set(0);
                        alarm_active = 0;
                    }
                    continue;
                }
                /* 其他错误，断开 */
                printf("[Alarm] recv 错误: %s\n", strerror(errno));
                break;
            }
            if (n == 0) {
                /* PC 断开连接 */
                printf("[Alarm] PC 端报警通道断开\n");
                break;
            }

            timeout_count = 0;      /* 收到数据，重置超时计数 */

            if (cmd == 0x01) {
                alarm_set(1);                  /* 开启报警 */
                alarm_active = 1;
            } else if (cmd == 0x00) {
                alarm_set(0);                  /* 关闭报警 */
                alarm_active = 0;
            } else {
                printf("[Alarm] 收到未知指令: 0x%02X\n", cmd);
            }
        }

        close(conn_fd);
        /* 断开时确保关闭报警，防止蜂鸣器一直响 */
        alarm_set(0);
    }

    close(server_fd);
    return NULL;
}

int main(int argc, char *argv[])
{
    if (3 != argc) {
        fprintf(stderr, "Usage: %s <video_dev> <Server_Ip>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    /* 忽略 SIGPIPE 信号：PC 端断开连接后 send() 会触发 SIGPIPE，
     * 默认行为是直接杀死进程。忽略后 send() 会返回 -1 + errno=EPIPE，
     * 我们在发送逻辑中检查返回值并优雅处理。 */
    signal(SIGPIPE, SIG_IGN);

    /* 1. 启动报警指令接收线程（独立 TCP 服务端，监听 ALARM_PORT）
     *    该线程在后台运行，接收 PC 端发来的 0x01/0x00 指令控制蜂鸣器和 LED
     *    报警线程生命周期与进程一致，不受视频流重连影响 */
    pthread_t alarm_tid;
    if (pthread_create(&alarm_tid, NULL, alarm_listener_thread, NULL) != 0) {
        perror("创建报警线程失败");
        exit(EXIT_FAILURE);
    }
    pthread_detach(alarm_tid);
    printf("报警接收线程已启动\n");

    /* 2. 初始化摄像头（仅需一次，重连不影响摄像头状态）
     *    摄像头初始化失败是硬件问题，直接退出 */
    if (v4l2_dev_init(argv[1]))
        exit(EXIT_FAILURE);

    v4l2_enum_formats();
    v4l2_print_formats();

    if (v4l2_set_format())
        exit(EXIT_FAILURE);

    if (v4l2_init_buffer())
        exit(EXIT_FAILURE);

    if (v4l2_stream_on())
        exit(EXIT_FAILURE);

    printf("\n========================================\n");
    printf("  摄像头初始化完成，进入主循环\n");
    printf("  PC 端断开后将自动重连\n");
    printf("========================================\n\n");

    /* 3. 主循环：连接 → 采集发送 → 断开 → 关闭报警 → 重连
     *    摄像头始终在采集（内核驱动环形缓冲区），断线期间帧被丢弃，
     *    重连后立即发送最新帧，不会有旧画面积压。 */
    while (1) {
        /* 3a. 连接 PC 端（带指数退避重试） */
        connect_server_with_retry(argv[2]);

        /* 3b. 进入采集-发送循环（阻塞直到 PC 断开） */
        v4l2_read_data();

        /* 3c. PC 断开后，主动关闭报警（安全措施） */
        alarm_set(0);

        /* 3d. 关闭旧的 socket */
        if (sockfd >= 0) {
            close(sockfd);
            sockfd = -1;
        }

        printf("\n[Main] PC 端已断开，3 秒后尝试重连...\n\n");
        sleep(3);
    }

    /* 理论上不会执行到这里（Ctrl+C 终止进程） */
    exit(EXIT_SUCCESS);
}
