//
// YoloDetector —— 推理类：加载 onnx 模型 → 前处理 → 推理 → 后处理(NMS) → 画框
//
// 对应 Python 版 utils/detect.py 的 YOLOv5 类（t='cv2.dnn' 路线）。
//
// tiny 分支：推理引擎用 OpenCV 自带的 cv::dnn（readNetFromONNX），
// 不依赖 onnxruntime —— 前提是模型导出时不含 OpenCV 解析不了的算子
// （如 YOLOv5 新版导出中的 Floor；旧版导出或经 onnxsim 化简的模型没有）。
//
// 模型输出布局按 [1, 25200, 7] 解析（与 main 分支的 onnxruntime 版一致）：
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
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

// 检测结果数据载体：一个被检出的目标
// （Python 版把结果组织成 dict{label: {num, score[], pos[]}}，C++ 用扁平列表更直接）
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
    bool isLoaded() const { return !m_net.empty(); }

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

    // ---- 推理引擎：OpenCV dnn（隐式链接，无额外 DLL/加载代码）----
    cv::dnn::Net m_net;

    int    m_inputW = 640, m_inputH = 640;     // 模型输入尺寸（YOLOv5 默认 640；换尺寸模型要改这里）
    float  m_conf = 0.35f;                     // 置信度阈值
    float  m_iou = 0.35f;                      // NMS 的 IoU 阈值
    bool   m_drawBox = true;                   // 是否把框画到帧上
    int    m_thickness = 2;                    // 框线宽
    cv::Scalar m_boxColor{255, 0, 0};          // 框颜色（RGB 顺序，因为帧已经是 RGB）
    std::vector<std::string> m_classNames{"fire", "smoke"};   // 类别名兜底值
};

#endif //FIRESMOKEDETECTIONZCODE_YOLODETECTOR_H
