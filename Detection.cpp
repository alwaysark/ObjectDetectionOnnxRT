//
// Detection 实现。run() 的循环结构是全程序的心脏，建议对照
// Python 版 DetectThread.main() 的 for img, path in self.dataset 循环读。
//
#include "Detection.h"
#include <QDebug>
#include <QDateTime>
#include <opencv2/imgproc.hpp>     // cvtColor / putText
#include <chrono>

Detection::Detection(QObject* parent) : QThread(parent) {
    // 自定义类型走跨线程信号前要注册元类型（QueuedConnection 的"报关手续"），
    // 写一次即可，所有连接都受益
    qRegisterMetaType<DetectionResult>("DetectionResult");
    qRegisterMetaType<std::vector<DetectionResult>>("std::vector<DetectionResult>");
}

Detection::~Detection() {
    m_isDetecting = false;
    m_isRunning = false;     // run() 循环最多 200ms 内退出（read 的等待超时兜底）
    wait();                  // 等线程真正结束，防止对象销毁时线程还在跑
}

// ---------- 线程生命周期 ----------
void Detection::StartThread() {
    if (isRunning()) return;      // 防重入：QThread 重复 start 只是警告，但语义上也不该重启
    m_isRunning = true;
    start();                      // start() 内部会新开系统线程并调用 run()
}

void Detection::StopThread() {
    m_isDetecting = false;
    m_isRunning = false;
    // 不在这里直接碰 m_Source —— 它属于检测线程。run() 每圈最多 200ms 醒一次，
    // 看到 m_isRunning=false 就走收尾逻辑（关源、关录像）后自然退出
}

// ---------- 请求信箱（UI 线程调用） ----------
void Detection::requestSource(SourceType type, const QString& path) {
    std::lock_guard<std::mutex> lk(m_srcMtx);   // 信箱必须持锁投递（QString 非原子）
    m_srcType = type;
    m_srcPath = path;
    m_srcRequested = true;
}

void Detection::requestDetect(const DetectConfig& cfg) {
    std::lock_guard<std::mutex> lk(m_cfgMtx);
    m_cfg = cfg;
    m_cfgRequested = true;
}

void Detection::setBoxColor(const cv::Scalar& rgb) {
    std::lock_guard<std::mutex> lk(m_colorMtx);
    m_boxColor = rgb;
}



// ---------- 换源信箱的消费（仅检测线程调用） ----------
void Detection::applySourceRequest() {
    // 先把信箱内容取出来并清标志（持锁时间越短越好）
    QString path;
    {
        std::lock_guard<std::mutex> lk(m_srcMtx);
        path = m_srcPath;
        m_srcRequested = false;
    }

    if (path == "screen") {
        emit errorOccurred("屏幕捕获暂未实现（预留接口）");
        return;
    }

    // 旧源在这里被 make_unique 覆盖销毁。此刻本线程不在 read() 里，
    // 所以销毁是安全的 —— 这就是"换源由检测线程自己动手"的好处
    m_Source = std::make_unique<VideoSource>();
    if (!m_Source->open(path.toStdString())) {
        emit errorOccurred(QString("无法打开输入源：%1").arg(path));
        m_Source.reset();         // 回到"无源待命"状态
        return;
    }
    stopRecord();                 // 换源 = 上一段录像作结
    emit sourceOpened(QString::fromStdString(m_Source->path()));
}

// ---------- 检测配置信箱的消费（仅检测线程调用） ----------
void Detection::applyDetectRequest() {
    DetectConfig cfg;
    {
        std::lock_guard<std::mutex> lk(m_cfgMtx);
        cfg = m_cfg;
        m_cfgRequested = false;
    }

    // 与 Python start() 一致：每次开始都重建模型（模型不大，重建比增量配置省心）
    m_Mode = std::make_unique<YoloDetector>();
    QString err;
    if (!m_Mode->load(cfg.modelPath, &err)) {
        emit errorOccurred(QString("模型加载失败：%1").arg(err));
        m_Mode.reset();
        m_isDetecting = false;    // 加载失败就回到预览态，界面按钮由 errorOccurred 槽复位
        return;
    }
    m_Mode->setClasses(cfg.classNames);
    m_Mode->setConf(m_conf.load());
    m_Mode->setIou(m_iou.load());
    m_Mode->setDrawBox(m_drawBox.load());
    {
        std::lock_guard<std::mutex> lk(m_colorMtx);
        m_boxColor = cfg.boxColor;      // 框颜色是 3 个数的组合，走锁不走 atomic
    }
    m_Mode->setBoxColor(cfg.boxColor);

    // 录像配置
    m_recordOn = cfg.recordVideo;
    m_recordFps = cfg.recordFps;
    m_recordDir = cfg.saveDir;
}

// ---------- 关录像（仅检测线程调用） ----------
void Detection::stopRecord() {
    if (m_writer.isOpened()) {
        m_writer.release();
        emit infoMessage(QString("录像已保存：%1").arg(m_recordPath));
    }
}



// ---------- 线程体：循环里三种状态：无源待命、有源预览、检测中 ----------
void Detection::run() {
    using namespace std::chrono;
    auto fpsStart = steady_clock::now();
    int  fpsCount = 0;
    double shownFps = 0;

    while (m_isRunning.load()) {
        // ===== ① 消费换源请求 =====
        if (m_srcRequested.load()) applySourceRequest();

        // ===== ② 无源待命：睡 50ms 等请求，不空转烧 CPU =====
        if (!m_Source) {
            msleep(50);
            continue;
        }

        // ===== ③ 热参数搬运：把 UI 写的 atomic 搬给底层对象 =====
        m_Source->setFlip(m_flipIdx.load());
        m_Source->setRotate(m_rotateIdx.load());
        if (m_Mode) {
            m_Mode->setConf(m_conf.load());
            m_Mode->setIou(m_iou.load());
            m_Mode->setDrawBox(m_drawBox.load());
            cv::Scalar color;
            {
                std::lock_guard<std::mutex> lk(m_colorMtx);
                color = m_boxColor;
            }
            m_Mode->setBoxColor(color);
        }

        // ===== ④ 取帧（read 返回 false ≙ Python 的 StopIteration）=====
        cv::Mat frame;
        if (!m_Source->read(frame)) {
            emit sourceFinished();
            stopRecord();
            m_Source.reset();     // 回到无源待命（Python 版这里停整个线程，我们常驻更稳）
            continue;
        }

        // ===== ⑤ 预览 or 检测 =====
        if (m_isDetecting.load()) {
            if (m_cfgRequested.load()) applyDetectRequest();   // 开始/配置请求在这里落地
        }
        if (m_isDetecting.load() && m_Mode) {
            // 推理 + 画框（结果直接画在 frame 上，一份数据两用：显示和录制）
            std::vector<DetectionResult> results = m_Mode->detect(frame);

            // 警报：连续 10 帧检测到任意目标 → 发警报信号
            // （换模型/换类别后语义不变：只要 results 非空就算"有目标"，
            //   不写死任何类别名——哪类算危险是类别文件决定的，不是代码决定的）
            if (!results.empty()) {
                if (++m_alarmStreak >= 10) { emit targetDetected(); m_alarmStreak = 0; }
            } else {
                m_alarmStreak = 0;
            }

            emit resultReady(results);

            // 录制：第一帧到来时才知道画面尺寸，所以懒打开 VideoWriter
            if (m_recordOn && !m_writer.isOpened()) {
                const QString head = QDateTime::currentDateTime().toString("MM-dd HH-mm-ss");
                m_recordPath = m_recordDir + "/video_" + head + ".mp4";
                m_writer.open(m_recordPath.toStdString(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                              m_recordFps, cv::Size(frame.cols, frame.rows));
                if (m_writer.isOpened())
                    emit infoMessage("开始录制检测视频...");
                else
                    emit errorOccurred("录像文件创建失败：" + m_recordPath);
            }
            if (m_writer.isOpened()) {
                cv::Mat bgr;
                cv::cvtColor(frame, bgr, cv::COLOR_RGB2BGR);   // VideoWriter 要 BGR
                m_writer.write(bgr);
            }
        } else if (m_writer.isOpened()) {
            stopRecord();     // 停止检测时收尾录像文件
        }

        // ===== ⑥ FPS 统计与显示 =====
        ++fpsCount;
        const double elapsed = duration<double>(steady_clock::now() - fpsStart).count();
        if (elapsed >= 2.0) {          // 每 2 秒结算一次平均帧率（Python 同款周期）
            shownFps = fpsCount / elapsed;
            fpsCount = 0;
            fpsStart = steady_clock::now();
        }
        if (m_showFps.load() && !m_Source->isImage()) {
            const std::string fpsText = "FPS:" + std::to_string(static_cast<int>(shownFps));
            const int tf = 2;
            cv::putText(frame, fpsText, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), tf, cv::LINE_AA);
        }

        // ===== ⑦ 发帧给界面 =====
        // cv::Mat → QImage：指定每行字节数(step) 构造，然后 copy() 深拷贝。
        // 不 copy 的话 frame 复用缓冲后界面拿到的就是脏数据（悬垂指针的经典坑）
        QImage img(frame.data, frame.cols, frame.rows,
                   static_cast<int>(frame.step), QImage::Format_RGB888);
        emit frameReady(img.copy());
    }

    // ===== 线程退出前的收尾（此处仍在检测线程，可以安全销毁底层对象）=====
    stopRecord();
    m_Source.reset();
    m_Mode.reset();
}
