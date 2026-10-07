//
// VideoSource —— 数据源类：把"摄像头 / 视频文件 / 图片"统一成同一个取帧接口
//
// 对应 Python 版的 DataLoader + VideoFrameDraw（utils/detect.py）。
//
// 对外只暴露三个动词（类的使用者只需要知道这三个）：
//     open(path)  打开源：自动识别类型，成败有返回值
//     read(frame) 取下一帧（已经是 RGB、已经按界面设置翻转/旋转过），返回 false = 源结束
//     close()     收工：停解码线程、释放摄像头
//
// 内部结构（方案B）：
//   - 视频文件/摄像头：内部有一个"解码线程"，按视频自身帧率持续 grab，
//     只保留最新一帧到 m_latestFrame。read() 永远拿最新帧 —— 检测慢就丢帧，
//     播放速度不受推理速度拖累（这正是 Python 版 VideoFrameDraw 想做没做好的事）。
//   - 图片：open 时 imread 一次，read 第一次返回它、之后返回 false，不需要线程。
//
// 线程安全说明（哪些成员被谁碰）：
//   m_latestFrame/m_FrameNum   解码线程写、read() 读   → 必须在 mutex 内
//   m_decoding/m_videoCameraOver   两线程都读写            → atomic
//   m_flip/m_rotate       UI 线程写、read() 读    → atomic（热更新）
//   其余成员（m_cap/m_path/...）只在 open/close 时初始化，生命周期由调用方保证
//

#ifndef FIRESMOKEDETECTIONZCODE_VIDEOSOURCE_H
#define FIRESMOKEDETECTIONZCODE_VIDEOSOURCE_H

#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

// 输入源类型（和界面下拉框的 item data 对应）
enum class SourceType {
    Camera, File, Screen
};

class VideoSource {
public:
    VideoSource() = default;      // 无参构造：对象只是空壳，open() 才赋予灵魂
    ~VideoSource();               // 析构自动收工（RAII：忘调 close 也不泄漏）

    // 线程 + 锁成员让这个类天生不可拷贝，显式删除防止误用
    VideoSource(const VideoSource&) = delete;
    VideoSource& operator=(const VideoSource&) = delete;

    bool open(const std::string& path);   // 打开源：纯数字=摄像头，"screen"=屏幕(未实现)，其他=文件
    bool read(cv::Mat& frame);            // 取下一帧；返回 false = 没有更多帧了（≈ Python 的 StopIteration）
    void close();                         // 三步收尾：停线程 → join → 释放摄像头

    // 热更新参数（对应界面"翻转/旋转"下拉框，索引 0~3，0=不处理）
    // 存索引而不是直接存 cv 码的原因：cv::flip(-1) 和 cv::rotate(0) 都是合法码，会撞车
    void setFlip(int i)   { m_flip.store(i); }
    void setRotate(int i) { m_rotate.store(i); }

    // 查询接口
    bool isImage() const       { return m_isImage; }
    const std::string& path() const { return m_path; }

private:
    void decodeLoop();            // 解码线程入口（只有视频/摄像头源才有这个线程）

    // ---- 流式源的"最新帧"缓冲（解码线程写 / read 读，全程持锁）----
    cv::VideoCapture m_cap;               // OpenCV 的视频通道
    std::thread      m_decoder;           // 内部解码线程（HAS-A，不是继承）
    std::mutex       m_mtx;               // 保护 m_latestFrame / m_FrameNum
    std::condition_variable m_condVari;         // 解码线程解码好了唤醒检测线程read
    cv::Mat          m_latestFrame;       // 最新一帧（BGR 原始帧）
    uint64_t         m_FrameNum = 0;      // 帧号：解码线程每存一帧 +1
    uint64_t         m_lastFrameNum = 0;  // read() 上一次交付出去的帧号
    std::atomic_bool m_decoding{false};   // 解码线程开关
    std::atomic_bool m_videoCameraOver{false};        // 源已读完（视频到尾/图片已交付）

    std::atomic_int  m_flip{0};           // 翻转下拉框索引（热更新）
    std::atomic_int  m_rotate{0};         // 旋转下拉框索引（热更新）

    // ---- 图片源专用 ----
    cv::Mat  m_image;                     // imread 的结果
    bool     m_imageOver = false;        // 是否已经交付过（图片只出一次）

    // ---- 打开时确定、之后只读的属性 ----
    SourceType  m_type = SourceType::File;
    std::string m_path;
    bool   m_opened = false;
    bool   m_isImage = false;
    double m_fps = 0;
};

#endif //FIRESMOKEDETECTIONZCODE_VIDEOSOURCE_H
