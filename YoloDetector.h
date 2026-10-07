//
// YoloDetector —— 推理类：加载 onnx 模型 → 前处理 → 推理 → 后处理(NMS) → 画框
//
// 对应 Python 版 utils/detect.py 的 YOLOv5 类（t='onnxruntime' 路线）。
//
// 推理引擎：onnxruntime 的 **C++ API**（onnxruntime_cxx_api.h）。
// 项目已迁移到 MSVC 工具链，与官方预编译包同语言体系，因此：
//   - 直接 #include 头文件 + 链接 onnxruntime.lib（隐式链接，加载器自动绑定）；
//   - Ort::Session/Ort::Env 等 RAII 类自动管理资源，不再手动 Release；
//   - 错误通过 Ort::Exception 异常上报，一次 try/catch 兜住全部调用；
//   - Session 路径参数是 wchar_t*，QString::toStdWString 直达中文路径。
// （旧方案"C API + LoadLibrary 动态加载"是 MinGW 时期的跨编译器做法，已留档。）
//
// 推理输出是 YOLOv5 导出的 onnx：[1, 25200, 7]
//   25200 = 三个尺度特征图的 anchor 格子总数
//   7     = 4 个框坐标(cx,cy,w,h) + 1 个目标置信度(obj) + 2 个类别分(fire, smoke)
//
// 顺手修了 Python 版的一个 bug：NMSBoxes 要求传 xywh（左上角+宽高），
// Python 版传的是 xyxy，导致框越叠越大；C++ 版从源头就按 xywh 存。
//

#ifndef FIRESMOKEDETECTIONZCODE_YOLODETECTOR_H
#define FIRESMOKEDETECTIONZCODE_YOLODETECTOR_H

#include <QString>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

// 检测结果数据载体：一个被检出的目标
struct DetectionResult {
    std::string label;    // 类别名，如 "fire" / "smoke"
    float       score;    // 置信度 0~1
    cv::Rect    box;      // 框（x, y, w, h），相对原图
};

class YoloDetector {
public:
    YoloDetector() = default;

    // 加载 onnx 模型。失败返回 false 并把原因写入 *err（不用异常，调用方好处理）
    bool load(const QString& onnxPath, QString* err = nullptr);
    bool isLoaded() const { return m_session != nullptr; }

    // 检测参数。conf/iou 会在检测线程里被频繁热更新，
    // setter 都做成一行内联——真正的跨线程保护由 Detection 层负责
    void setClasses(const std::vector<std::string>& names) { m_classNames = names; }
    void setConf(float v)     { m_conf = v; }
    void setIou(float v)      { m_iou = v; }
    void setDrawBox(bool b)   { m_drawBox = b; }
    void setBoxColor(const cv::Scalar& rgb) { m_boxColor = rgb; }   // 注意是 RGB 顺序

    // 对一帧做检测。frame 进出都是 RGB 格式；drawBox 打开时会把框和标签画在 frame 上
    std::vector<DetectionResult> detect(cv::Mat& frame);

private:
    void drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& dets) const;

    // ---- onnxruntime 对象（RAII 自动管理，无需手动 Release）----
    Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING, "FireSmokeDetection"};
    std::unique_ptr<Ort::Session> m_session;               // 每次 load 重建
    std::string m_inputName  = "images";                   // 从模型里读出来的输入/输出名
    std::string m_outputName = "output0";
    bool m_loaded = false;

    int    m_inputW = 640, m_inputH = 640;     // 模型输入尺寸（自动从模型读取）
    float  m_conf = 0.35f;                     // 置信度阈值
    float  m_iou = 0.35f;                      // NMS 的 IoU 阈值
    bool   m_drawBox = true;                   // 是否把框画到帧上
    int    m_thickness = 2;                    // 框线宽
    cv::Scalar m_boxColor{255, 0, 0};          // 框颜色（RGB 顺序，因为帧已经是 RGB）
    std::vector<std::string> m_classNames{"fire", "smoke"};   // 类别名兜底值
};

#endif //FIRESMOKEDETECTIONZCODE_YOLODETECTOR_H
