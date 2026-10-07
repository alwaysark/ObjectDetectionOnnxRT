//
// YoloDetector 实现（onnxruntime C++ API 版）。
// 推理部分：Ort::Session；前处理/后处理/NMS/画框：OpenCV（与 Python YOLOv5 类逐步对应）。
//
// 顺手修了 Python 版的一个 bug：NMSBoxes 要求传 xywh（左上角+宽高），
// Python 版传的是 xyxy，导致框越叠越大；C++ 版从源头就按 xywh 存。
//
#include "YoloDetector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>        // cv::dnn::blobFromImage 造输入张量 + NMSBoxes
#include <algorithm>
#include <cmath>

// ---------- 加载模型 ----------
bool YoloDetector::load(const QString& onnxPath, QString* err) {
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(0);              // 0 = 默认线程数（≈物理核心数）
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Windows 上 Session 的路径参数是 wchar_t*，toStdWString 直达中文路径
        m_session = std::make_unique<Ort::Session>(
                m_env, onnxPath.toStdWString().c_str(), opts);

        // 从模型自动读输入尺寸（YOLOv5 640；模型写死多少就用多少）
        auto inShapeInfo = m_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto inShape = inShapeInfo.GetShape();
        if (inShape.size() == 4 && inShape[2] > 0 && inShape[3] > 0) {
            m_inputH = static_cast<int>(inShape[2]);
            m_inputW = static_cast<int>(inShape[3]);
        }

        // 从模型读输入/输出层的名字（YOLOv5 通常是 "images"/"output0"，不写死）
        Ort::AllocatorWithDefaultOptions alloc;
        m_inputName  = m_session->GetInputNameAllocated(0, alloc).get();
        m_outputName = m_session->GetOutputNameAllocated(0, alloc).get();
    } catch (const Ort::Exception& e) {
        if (err) *err = e.what();
        m_session.reset();
        return false;
    }
    m_loaded = true;
    return true;
}

// ---------- 检测一帧 ----------
std::vector<DetectionResult> YoloDetector::detect(cv::Mat& frame) {
    std::vector<DetectionResult> results;
    if (!m_session || frame.empty()) return results;

    std::vector<float> output;          // 模型输出的拷贝
    int64_t rows = 0, attrs = 0;

    try {
        // ===== 前处理（OpenCV 代劳）=====
        // blobFromImage 一次做完四件事：缩放到输入尺寸、乘 1/255 归一化、HWC→NCHW、加 batch 维
        // swapRB 传 false：因为 frame 已经是 RGB（VideoSource::read 保证），模型要的就是 RGB。
        // 如果哪天直接传 BGR 原始帧，这里要改成 true —— 通道顺序错了检测结果会莫名变差
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0,
                cv::Size(m_inputW, m_inputH), cv::Scalar(), false, false, CV_32F);

        // ===== 推理 =====
        const int64_t inShape[4] = {1, 3, m_inputH, m_inputW};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inTensor = Ort::Value::CreateTensor<float>(memInfo, blob.ptr<float>(),
                blob.total(), inShape, 4);

        const char* inNames[]  = { m_inputName.c_str() };
        const char* outNames[] = { m_outputName.c_str() };
        auto outputs = m_session->Run(Ort::RunOptions{nullptr},
                                      inNames, &inTensor, 1, outNames, 1);

        // ===== 取输出：[1, 25200, 7] 的 float 数组 =====
        float* outData = outputs.front().GetTensorMutableData<float>();
        auto shapeInfo = outputs.front().GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> dims = shapeInfo.GetShape();
        const size_t count = shapeInfo.GetElementCount();
        if (dims.size() != 3 || dims[2] < 6) return results;   // 形状不对就放弃本帧
        output.assign(outData, outData + count);
        rows  = dims[1];                       // 25200 个候选
        attrs = static_cast<int>(dims[2]);     // 每候选 7 个数
    } catch (const Ort::Exception& e) {
        return results;                        // 推理出错：放弃本帧，不逃逸到调用方
    }

    const int classCount = static_cast<int>(attrs) - 5;   // 7-5 = 2 类（fire/smoke）

    // 框坐标是相对模型输入的，要按原图/输入的比例缩放回原图
    const float sx = static_cast<float>(frame.cols) / m_inputW;
    const float sy = static_cast<float>(frame.rows) / m_inputH;

    std::vector<cv::Rect> boxes;              // xywh（NMSBoxes 要求的格式）
    std::vector<float>    scores;
    std::vector<int>      classIds;

    for (int64_t i = 0; i < rows; ++i) {
        const float* p = output.data() + static_cast<size_t>(i) * attrs;

        const float obj = p[4];               // 这格里"有没有东西"的置信度
        if (obj <= m_conf) continue;          // 第一道门槛：先筛掉大半候选

        // 在两个类别分里挑最高的
        int bestClass = 0;
        float bestClassScore = p[5];
        for (int c = 1; c < classCount; ++c) {
            if (p[5 + c] > bestClassScore) { bestClassScore = p[5 + c]; bestClass = c; }
        }

        const float score = obj * bestClassScore;   // 最终得分 = obj × 类别分（YOLOv5 惯例）
        if (score <= m_conf) continue;              // 第二道门槛

        // 中心坐标 + 宽高，按比例缩放回原图，再换算成左上角 xywh
        const float cx = p[0] * sx, cy = p[1] * sy;
        const float w  = p[2] * sx, h  = p[3] * sy;
        cv::Rect box(static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2),
                     static_cast<int>(w), static_cast<int>(h));

        boxes.push_back(box);
        scores.push_back(score);
        classIds.push_back(bestClass);
    }

    // ===== 非极大值抑制（NMS）：同一目标被相邻格子重复检出，只留最好的 =====
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

    if (m_drawBox) drawDetections(frame, results);   // 直接把框画在帧上（Python 同款做法）
    return results;
}

// ---------- 画框 + 标签（公式与 Python __formatResult 一致） ----------
void YoloDetector::drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& dets) const {
    // 线宽随图像尺寸自适应：小图细线、大图粗线，但不小于设定的 thickness
    const int lw = std::max(static_cast<int>(std::lround((frame.rows + frame.cols) / 2.0 * 0.003)),
                            m_thickness);
    const int tf = std::max(lw - 1, 1);                       // 字体笔画粗细
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
