//
// VideoSource 实现。逐函数对照 Python 版 utils/detect.py：
//   open()          ↔ DataLoader.__init__（识别源类型、建通道）
//   read()          ↔ DataLoader.__next__（出帧 + flip + rotate + BGR转RGB，StopIteration = 返回 false）
//   decodeLoop()    ↔ VideoFrameDraw.run（后台按帧率抓帧、只留最新）
//   close()         ↔ DataLoader.__del__（release）
//
#include "VideoSource.h"
#include <opencv2/imgproc.hpp>      // cvtColor / flip / rotate
#include <opencv2/imgcodecs.hpp>   // imread
#include <chrono>

// ---------- open：识别类型 → 建通道 → 流式源起解码线程 ----------
bool VideoSource::open(const std::string& path) {
    close();                      // 防御：换源前先把旧源收干净，保证从干净状态开始

    if (path == "screen") {
        // 屏幕捕获预留位。将来用 QScreen::grabWindow 实现（必须在 GUI 线程抓屏），
        // 或者用 Win32 的 BitBlt。当前版本先返回失败。
        return false;
    }

    m_path = path;

    // 如果路径不为空，且path中找不在字符集合（0123456789）中的字符，找不到说明都是数字，那么isCameraIndex为true
    const bool isCameraIndex = (!path.empty() && (path.find_first_not_of("0123456789") == std::string::npos) );

    bool isOpened = false;
    if (isCameraIndex) {
        m_type = SourceType::Camera;
        // CAP_DSHOW：Windows 的 DirectShow 后端，摄像头启动快（Python 版同款）
        isOpened = m_cap.open(std::stoi(path), cv::CAP_DSHOW);
        if (isOpened) {
            m_fps = m_cap.get(cv::CAP_PROP_FPS);
            if (m_fps <= 1) m_fps = 30;   // 有些摄像头不报帧率，兜底 30
        } else {
            m_cap.release();          // 释放半开状态，保持对象干净
            return false;
        }

    } else {
        m_type = SourceType::File;
        // 显式指定 CAP_MSMF 后端：对 UTF-8（中文）路径友好；FFMPEG 后端在 Windows
        // 上打开含中文的路径会失败，且我们编译时没有带 FFmpeg
        isOpened = m_cap.open(path, cv::CAP_MSMF);
        if (isOpened) {
            m_fps = m_cap.get(cv::CAP_PROP_FPS);
        } else {
            // 视频打不开？再当图片试一次（imread 支持 jpg/png/bmp 等）
            cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
            if (!img.empty()) {
                m_isImage     = true;
                m_image       = img.clone();
                m_fps         = 0;
            } else {
                m_cap.release();          // 释放半开状态，保持对象干净
                return false;
            }
        }
    }

    m_opened = true;
    m_imageOver = false;

    // 流式源（摄像头/视频）起解码线程；图片源不需要线程
    if (!m_isImage) {
        m_FrameNum = 0;
        m_lastFrameNum = 0;
        m_videoCameraOver = false;
        m_decoding = true;
        m_decoder = std::thread(&VideoSource::decodeLoop, this);
    }
    return true;
}

// ---------- 解码线程：按视频自身帧率节奏抓帧，只保留最新一帧 ----------
void VideoSource::decodeLoop() {
    // 1 秒 = 1000 毫秒
    // 30fps = 1 秒 播放30帧
    // 每帧播放 1/30 秒 = 1000/30 毫秒  <==>  每帧间隔时长
    // 有些摄像头不报帧率，兜底 30帧 那间隔就是33毫秒
    const int delayMs = m_fps > 1 ? static_cast<int>(1000.0 / m_fps) : 33;

    int failCount = 0;    // 连续读帧失败计数
    while (m_decoding.load()) {
        cv::Mat f;
        const bool isreaded = m_cap.read(f) && !f.empty();

        if (!isreaded) {
            if (m_type == SourceType::Camera) {
                // 摄像头的读帧失败绝大多数是暂时性的（设备预热中、USB 抖动），
                // 不能当"流结束"。睡 100ms 重试；连续约 5 秒失败才认定设备真的没了。
                if (++failCount >= 50) {
                    m_videoCameraOver = true;
                    m_condVari.notify_all();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            // 视频文件不做重试：读失败就是读到尾了。改标志位，改条件变量，退出循环
            m_videoCameraOver = true;
            m_condVari.notify_all();
            break;
        }
        failCount = 0;      //重置失败帧数
        {
            // 锁内写入最新帧。clone() 是关键：cv::Mat 的等号是浅拷贝（共享像素内存），
            // 不 clone 的话下一轮 cap.read 会改写同一块内存，read() 就会读到"撕裂"的帧
            std::lock_guard<std::mutex> lk(m_mtx);
            m_latestFrame = f.clone();
            ++m_FrameNum;              // 帧号 +1：read() 用它区分"有没有新帧"
        }
        m_condVari.notify_all();        // 叫醒正在等帧的 read()

        // 在这里睡delayMs毫秒。
        // 检测线程消费不过来时（检测一帧的时长超过了delayMs毫秒），解码线程开始下一次循环把下一帧解码出来给到m_latestFrame并唤醒
        // 而之前检测线程在检测时解码出的帧就被丢帧
        // 检测线程消费的很快时，解码线程在这里睡delayMs毫秒就保证检测线程还是只能按照这个节奏去检测
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    }
}

// ---------- read：把"下一帧"交付给调用方（已翻转、已旋转、已转 RGB） ----------
bool VideoSource::read(cv::Mat& frame) {
    if (!m_opened) return false;

    cv::Mat out;
    if (m_isImage) {
        // 图片源：只交付一次，之后就是"结束"
        if (m_imageOver) return false;
        out = m_image.clone();
        m_imageOver = true;
    } else {
        // 持续等待，直到：有新帧 或 源结束 或 线程被要求关闭。
        // 不能"超时一次就当结束"——摄像头预热期第一帧经常超过 200ms，
        // 那是"暂时没帧"，不是"源结束"（摄像头的秒开秒关就是死在这）
        std::unique_lock<std::mutex> lk(m_mtx);
        while (m_FrameNum == m_lastFrameNum && m_decoding.load() && !m_videoCameraOver.load()) {
            m_condVari.wait_for(lk, std::chrono::milliseconds(200));
        }
        if (m_FrameNum == m_lastFrameNum) {
            // 走到这里且没有新帧：只可能是 eof 或已关闭 —— 真正的"没有更多帧"
            return false;
        }
        out = m_latestFrame.clone();  // 锁内拷出，出锁后再加工，缩小临界区
        m_lastFrameNum = m_FrameNum;    //
    }

    // 翻转
    const int flipIdx = m_flip.load();
    if (flipIdx != 0) {
        // 下拉框索引 → cv::flip 的 code：1水平 0垂直 -1双向
        // （这正是不能用索引直接当 cv 码的原因：索引 0 表示"不翻转"，而 flip 的 0 是垂直翻转）
        int code = 0;
        if (flipIdx == 1) code = 1;
        else if (flipIdx == 2) code = 0;
        else if (flipIdx == 3) code = -1;
        cv::flip(out, out, code);
    }
    // 旋转
    const int rotIdx = m_rotate.load();
    if (rotIdx != 0) {
        if (rotIdx == 1)      cv::rotate(out, out, cv::ROTATE_90_CLOCKWISE);
        else if (rotIdx == 2) cv::rotate(out, out, cv::ROTATE_90_COUNTERCLOCKWISE);
        else if (rotIdx == 3) cv::rotate(out, out, cv::ROTATE_180);
    }
    // BGR 转 RGB
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);

    frame = out;
    return true;
}

// ---------- close：三步收尾（可在任意线程调用，可重复调用） ----------
void VideoSource::close() {
    m_decoding = false;                       // ① 让解码线程退出循环
    m_condVari.notify_all();                        //    并叫醒可能睡在 read() 里的线程
    if (m_decoder.joinable())                 // ② 回收线程（joinable 判断防二次 join）
        m_decoder.join();
    if (m_cap.isOpened())                     // ③ 释放摄像头/视频通道
        m_cap.release();

    { std::lock_guard<std::mutex> lk(m_mtx); m_latestFrame.release(); }   // 清缓冲
    m_image.release();
    m_opened = false;
    m_isImage = false;
    m_imageOver = false;
    m_videoCameraOver = false;
}

VideoSource::~VideoSource() {
    close();      // RAII：就算调用方忘了 close，对象销毁时也会自动收工
}
