//
// cv::dnn 推理版
// 推理：cv::dnn::readNetFromONNX + setInput/forward；
// 前处理/后处理/NMS/画框
//
#include "YoloDetector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>

// ---------- 加载模型 ----------
bool YoloDetector::load(const QString& onnxPath, QString* err) {
    try {
        // 解析模型结构。文件不存在/含 OpenCV 不支持的算子（如 Floor）会抛 cv::Exception
        m_net = cv::dnn::readNetFromONNX(onnxPath.toStdString());
    } catch (const cv::Exception& e) {
        if (err) *err = e.what();
        m_net = cv::dnn::Net();    // Net 没有 release()，赋默认空对象即"清空"
        return false;
    }
    // CPU 推理后端（OpenCV 自带，无额外依赖）
    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // yolov5-6.0 的终输出层固定叫 "output"。load 时校验一次存在性：
    // 名字对不上 forward("output") 就每帧抛异常，detect() 只能每帧静默空结果
    const std::vector<cv::String> outs = m_net.getUnconnectedOutLayersNames();
    for (const cv::String& n : outs)
        if (n == "output") return true;
    if (err) *err = "模型不是 yolov5-6.0 导出格式：找不到终输出层 \"output\"";
    m_net = cv::dnn::Net();
    return false;
}

// ---------- 检测一帧 ----------
std::vector<DetectionResult> YoloDetector::detect(cv::Mat& frame) {
    std::vector<DetectionResult> results;
    if (m_net.empty() || frame.empty()) return results;

    cv::Mat output;
    try {
        // ===== 前处理（OpenCV 代劳）=====
        // blobFromImage 一次做完四件事：缩放到 yolov5-6.0 输入固定 640×640、乘 1/255 归一化、HWC→NCHW、加 batch 维
        // swapRB 传 false： VideoSource::read已经把frame转为 RGB，模型要的就是 RGB。
        // 注意这里是拉伸缩放不是 letterbox：和下面 sx/sy 分轴缩放自洽。
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0,
                cv::Size(640, 640), cv::Scalar(), false, false, CV_32F);

        // ===== 推理 =====
        // 按名取终输出层 "output"。OpenCV 会裁掉该层不依赖的分支，
        // 所以取终输出时三个检测头各自的输出根本不用算。
        m_net.setInput(blob);
        output = m_net.forward("output");
    } catch (const cv::Exception& e) {
        return results;                    // 推理出错：放弃本帧，不逃逸到调用方
    }

    // yolov5-6.0 的输出固定是 [1, 25200, 5+类别数]。类别数因训练数据而异，
    // 只要求 attrs ≥ 6（至少 1 类），rows/attrs 都从输出里读，不写死
    if (output.dims != 3 || output.size[0] != 1 || output.size[2] < 6)
        return results;

    // batch 恒为 1，重排成 [25200, attrs] 的 2D 视图便于按行取（共享内存，无拷贝）
    const int rows  = output.size[1];
    const int attrs = output.size[2];
    const int classCount = attrs - 5;
    cv::Mat out2d = output.reshape(1, rows);

    // 框坐标是相对 640 输入的，按原图/输入的比例缩放回原图
    // （blobFromImage 是拉伸缩放，所以 x/y 要分轴各乘各的）
    const float sx = static_cast<float>(frame.cols) / 640;
    const float sy = static_cast<float>(frame.rows) / 640;

    std::vector<cv::Rect> boxes;           // xywh（NMSBoxes 要求的格式）
    std::vector<float>    scores;
    std::vector<int>      classIds;

    for (int i = 0; i < rows; ++i) {
        const float* p = out2d.ptr<float>(i);

        const float obj = p[4];            // 这格里"有没有东西"的置信度
        if (obj <= m_conf) continue;       // 第一道门槛：先筛掉大半候选

        // 在各类别分里挑最高的
        int bestClass = 0;
        float bestClassScore = p[5];
        for (int c = 1; c < classCount; ++c) {
            if (p[5 + c] > bestClassScore) { bestClassScore = p[5 + c]; bestClass = c; }
        }

        const float score = obj * bestClassScore;   // 最终得分 = obj × 类别分（YOLOv5 惯例）
        if (score <= m_conf) continue;              // 第二道门槛

        // 中心坐标 + 宽高，缩放回原图，再换算成左上角 xywh
        const float cx = p[0] * sx, cy = p[1] * sy;
        const float w  = p[2] * sx, h  = p[3] * sy;
        cv::Rect box(static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2),
                     static_cast<int>(w), static_cast<int>(h));

        boxes.push_back(box);
        scores.push_back(score);
        classIds.push_back(bestClass);
    }

    // ===== 非极大值抑制（NMS）：同一目标被相邻格子重复检出，只留分最高的 =====
    // 注意 cv::dnn::NMSBoxes 是类无关的：所有类别的框放在一起抑制，
    // fire/smoke 重叠时分数低的会被分数高的压掉（如需类内 NMS 要按类别分组各做一遍）
    std::vector<int> keep;
    if (!boxes.empty())
        cv::dnn::NMSBoxes(boxes, scores, m_conf, m_iou, keep);

    for (int idx : keep) {
        DetectionResult r;
        r.score = scores[idx];
        r.label = classIds[idx] < static_cast<int>(m_classNames.size())
                  ? m_classNames[classIds[idx]]
                  : "class_" + std::to_string(classIds[idx]);
        r.box = boxes[idx] & cv::Rect(0, 0, frame.cols, frame.rows);   // 裁掉出界部分
        results.push_back(r);
    }

    // 按置信度从高到低排，界面显示和画框都更直观
    std::sort(results.begin(), results.end(),
              [](const DetectionResult& a, const DetectionResult& b) { return a.score > b.score; });

    if (m_drawBox) drawDetections(frame, results);   // 框直接画在帧上，显示/录制共用这一份
    return results;
}

// ---------- 画框 + 标签（配色与 Python __formatResult 一致） ----------
void YoloDetector::drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& dets) const {
    // 线宽随图像尺寸自适应：小图细线、大图粗线，但不低于设定的 thickness
    const int lw = std::max(static_cast<int>(std::lround((frame.rows + frame.cols) / 2.0 * 0.003)),
                            m_thickness);
    const int tf = std::max(lw - 1, 1);                       // 文字笔画粗细
    // 文字颜色取框颜色的反色（255-x），保证在色块上看得清（Python 同款）
    const cv::Scalar txtColor(255 - m_boxColor[0], 255 - m_boxColor[1], 255 - m_boxColor[2]);

    for (const auto& r : dets) {
        cv::rectangle(frame, r.box, m_boxColor, lw, cv::LINE_AA);

        const std::string tag = r.label + " " + std::to_string(static_cast<int>(r.score * 100)) + "%";
        int baseLine = 0;
        const cv::Size ts = cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, lw / 3.0, tf, &baseLine);

        // 标签底色块：框上边放得下就放框外，放不下就塞框内
        const bool outside = r.box.y - ts.height >= 3;
        const cv::Point p2(r.box.x + ts.width,
                           outside ? r.box.y - ts.height - 3 : r.box.y + ts.height + 3);
        cv::rectangle(frame, cv::Point(r.box.x, r.box.y), p2, m_boxColor, -1, cv::LINE_AA);
        cv::putText(frame, tag,
                    cv::Point(r.box.x, outside ? r.box.y - 2 : r.box.y + ts.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, lw / 3.0, txtColor, tf, cv::LINE_AA);
    }
}
