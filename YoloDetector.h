//
// YoloDetector —— YOLOv5 6.0 专用推理类：加载 onnx → 前处理 → 推理 → 后处理(NMS) → 画框
//
// 本项目只支持 YOLOv5 6.0 的导出格式（tiny 分支用 OpenCV 自带的 cv::dnn 推理，
// 不依赖 onnxruntime），输入/输出的格式都是固定约定：
//
//   输入  [1, 3, 640, 640] float32，RGB、0~1 归一化。输入尺寸写死 640（YOLOv5 默认），
//         换输入尺寸的模型不在支持范围内；
//   输出  终输出层固定叫 "output"，形状 [1, 25200, 5+类别数]：
//           25200 = (80×80 + 40×40 + 20×20) × 3 anchor，640 输入下恒定；
//           每行 = cx,cy,w,h（相对 640 输入的像素坐标）+ obj(0~1) + 各类别分(0~1)。
//           两类（fire/smoke）时即 [1, 25200, 7]；
//   换模型 = 用 yolov5-6.0 训练其他目标检测再导出：类别数可变（代码按 5+类别数 自适应），
//         输入尺寸和输出名都不变。导出时不能含 OpenCV 解析不了的算子
//         （如 YOLOv5 新版导出中的 Floor；6.0 导出或经 onnxsim 化简的模型没有）。
//
// 性能实测（Release 版 OpenCV，640 输入）：
//   N0.6 ≈ 48ms/帧（约 20 FPS）；M0.9 ≈ 257ms/帧（约 4 FPS）。
//   Debug 构建下推理慢 3 倍以上（N0.6 ≈ 157ms），觉得慢先查构建配置。
//

#ifndef FIRESMOKEDETECTIONZCODE_YOLODETECTOR_H
#define FIRESMOKEDETECTIONZCODE_YOLODETECTOR_H

#include <QString>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

// 检测结果数据
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

    // 检测参数。conf/iou 会在检测线程里被频繁热更新
    void setClasses(const std::vector<std::string>& names) { m_classNames = names; }
    void setConf(float v)     { m_conf = v; }
    void setIou(float v)      { m_iou = v; }
    void setDrawBox(bool b)   { m_drawBox = b; }
    void setBoxColor(const cv::Scalar& rgb) { m_boxColor = rgb; }   // 注意是 RGB 顺序

    // 检测
    std::vector<DetectionResult> detect(cv::Mat& frame);

private:
    void drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& dets) const;

    // ---- 推理引擎：OpenCV dnn（隐式链接，无额外 DLL/加载代码）----
    cv::dnn::Net m_net;

    float  m_conf = 0.35f;                     // 置信度阈值
    float  m_iou = 0.35f;                      // NMS 的 IoU 阈值
    bool   m_drawBox = true;                   // 是否把框画到帧上
    int    m_thickness = 2;                    // 框线宽
    cv::Scalar m_boxColor{255, 0, 0};          // 框颜色（RGB 顺序，因为帧已经是 RGB）
    std::vector<std::string> m_classNames{"fire", "smoke"};   // 类别名
};

#endif //FIRESMOKEDETECTIONZCODE_YOLODETECTOR_H
