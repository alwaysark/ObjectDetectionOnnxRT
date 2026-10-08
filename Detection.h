//
// Detection —— 检测线程（QThread 子类），整个程序的中枢调度器。
//
// run() 循环的三种状态：
//   无源待命  没有输入源，每 50ms 醒一次看有没有换源请求
//   有源预览  read 出帧直接发 frameReady，界面显示原始画面
//   检测中    在预览的基础上多跑一次推理，额外发 resultReady / targetDetected
//
// 依赖方向：MainWindow 认识所有人；Detection 只认识 VideoSource/
// YoloDetector/DetectionResult；后两者互不认识。
//
// 线程边界约定（本类的线程安全模型）：
//   - m_Source / m_Mode 两个 unique_ptr 只在检测线程（run 及其调用的函数）里
//     创建、使用、销毁 —— UI 线程永远不碰它们，所以没有指针竞态；
//   - UI 线程把"想干什么"投进请求信箱（mutex 保护）或热参数（atomic）；
//   - 检测线程每圈循环开头消费信箱、搬运热参数。
//

#ifndef FIRESMOKEDETECTIONZCODE_DETECTION_H
#define FIRESMOKEDETECTIONZCODE_DETECTION_H

#include "VideoSource.h"      // Detection 认识 VideoSource（依赖铁律）
#include "YoloDetector.h"     // 顺带带来 DetectionResult 结构
#include <QThread>
#include <QImage>
#include <QString>
#include <QMetaType>
#include <atomic>
#include <mutex>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

// 自定义类型要跨线程走信号（排队连接）时，必须先向 Qt 的元对象系统注册：
// 这两行是给 std::vector<DetectionResult> 发信号用的"报关手续"
Q_DECLARE_METATYPE(DetectionResult)
Q_DECLARE_METATYPE(std::vector<DetectionResult>)

// 检测配置：点"开始检测"时把界面控件打包成这个结构，经信箱送进检测线程
struct DetectConfig {
    QString modelPath;                                // 权重路径
    std::vector<std::string> classNames;              // 类别名（fire/smoke）
    float conf = 0.35f;                               // 置信度阈值
    float iou = 0.35f;                                // IOU 阈值
    bool drawBox = true;                              // 是否画锚框
    cv::Scalar boxColor{255, 0, 0};                   // 框颜色（RGB）
    bool recordVideo = false;                         // 是否录制检测视频
    int recordFps = 15;                               // 录制帧率
    QString saveDir;                                  // 录像保存目录
};

class Detection : public QThread {
    Q_OBJECT
public:
    explicit Detection(QObject* parent = nullptr);
    ~Detection() override;

    // ---- 检测线程开关 ----
    void StartThread();     // 启动检测线程
    void StopThread();      // 全停：停止推理并退出线程，关窗口时调用

    // ---- 检测开关 ----
    void BeginDetect()  { m_isDetecting = true; }
    void StopDetect()   { m_isDetecting = false; }

    // ---- 请求信箱：UI线程 和 检测线程 中转配置信息 ----
    void requestSource(SourceType type, const QString& path);   // 换输入源
    void requestDetect(const DetectConfig& cfg);                // 配置并(重)加载模型

    // ---- 热参数：UI线程写，检测线程 run()函数 每轮循环将这些参数传给VideoSource对象和YoloDetector对象 ----
    void setFlip(int i)       { m_flipIdx.store(i); }
    void setRotate(int i)     { m_rotateIdx.store(i); }

    void setConf(float v)     { m_conf.store(v); }
    void setIou(float v)      { m_iou.store(v); }
    void setDrawBox(bool b)   { m_drawBox.store(b); }
    void setDisplayFps(bool b){ m_showFps.store(b); }
    void setBoxColor(const cv::Scalar& rgb);                    // 多字段，用锁

signals:
    void frameReady(const QImage& img);                       // → 界面显示
    void resultReady(const std::vector<DetectionResult>& r);  // → 界面日志
    void targetDetected();                                    // → 警报（连续 10 帧检出任意目标）
    void errorOccurred(const QString& msg);                   // → 红字日志 + 按钮复位
    void sourceFinished();                                    // → 日志 + 按钮复位
    void sourceOpened(const QString& desc);                   // → 日志
    void infoMessage(const QString& msg);                     // → 一般提示（录像开始/保存等）

protected:
    void run() override;    // QThread 的线程体：三态循环都在这里

private:
    // 这两个函数只允许在检测线程内调用（操作 m_Source/m_Mode 的都在这里）
    void applySourceRequest();   // 消费换源信箱：销毁旧源、创建并打开新源
    void applyDetectRequest();   // 消费配置信箱：创建/重载 YoloDetector
    void stopRecord();           // 关录像文件

    // ---- 线程状态开关 ----
    std::atomic_bool m_isRunning{false};     // 线程存活（析构/关窗时置 false）
    std::atomic_bool m_isDetecting{false};   // 是否在推理（预览/检测的切换开关）

    // ---- 模型和源数据，只在检测线程里创建和销毁 ----
    std::unique_ptr<VideoSource>  m_Source;
    std::unique_ptr<YoloDetector> m_Mode;

    // ---- 换源请求信箱及其锁 ----
    std::mutex    m_srcMtx;
    SourceType    m_srcType = SourceType::File;
    QString       m_srcPath;
    std::atomic_bool m_srcRequested{false};

    // ---- 检测配置信箱及其锁 ----
    std::mutex    m_cfgMtx;
    DetectConfig  m_cfg;
    std::atomic_bool m_cfgRequested{false};

    // ---- 热参数：UI 写、run 读（单值用 atomic，多字段用锁）----
    std::atomic_int   m_flipIdx{0}, m_rotateIdx{0};
    std::atomic<float> m_conf{0.35f}, m_iou{0.35f};
    std::atomic_bool  m_drawBox{true}, m_showFps{true};
    std::mutex        m_colorMtx;
    cv::Scalar        m_boxColor{255, 0, 0};

    // ---- 警报连击计数（连续 10 帧检出任意目标响一次）----
    int m_alarmStreak = 0;

    // ---- 录像（全部只在检测线程使用，无竞态）----
    cv::VideoWriter m_writer;
    QString         m_recordPath;      // 当前录像文件路径
    bool            m_recordOn = false;
    int             m_recordFps = 15;
    QString         m_recordDir;
};

#endif //FIRESMOKEDETECTIONZCODE_DETECTION_H
