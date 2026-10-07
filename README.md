# ObjectDetectionOnnxRT

> **当前分支：tiny** —— 推理引擎为 OpenCV 自带 dnn（无需 onnxruntime）；main 分支使用 onnxruntime。两分支的模型/类别文件通用。

**实时目标检测桌面应用** —— Qt6 界面 + OpenCV 解码与后处理 + OpenCV dnn 推理，
三线程架构，模型与类别文件可随时更换（不限于烟雾火焰，任何 YOLOv5 格式的 onnx 检测模型均可）。

> 源自个人毕设"烟雾火焰检测"（PyQt5 + onnxruntime）的 C++/Qt6 完整重构版：
> 检测目标从烟雾火焰泛化为**任意可检测类别**，推理后端为 OpenCV dnn（tiny 分支，无需 onnxruntime）。

## 功能

- 检测视频文件 / 图片中的目标，实时显示带锚框与置信度的画面
- 模型（.onnx）与类别文件（.txt）可随时更换，输入尺寸从模型自动读取
- 置信度 / IOU 阈值、翻转 / 旋转、锚框颜色——开始检测时生效
- 警报：连续 10 帧检出任意目标 → 警报音（5 秒冷却，运行中可开关）
- 录制检测视频（mp4）、截图、日志保存
- 界面配置持久化（QSettings），下次启动自动恢复
- 检测进行中锁定输入页；关窗确认 + 线程安全退出

## 架构：三线程 + 请求信箱

```
主线程(GUI)                     检测线程                      解码线程(VideoSource 内部)
──────────                     ─────────                     ─────────────────────
点"开始检测"：打包 DetectConfig
  ── requestDetect ──► 信箱
点"输入方式/文件" ── requestSource ──► 信箱
滑块/复选框 ── setConf/setFlip ──► atomic 热参数
                                 run() 三态循环：
                                 ① 消费信箱（换源/换模型）
                                 ② read(frame) ◄───────── 解码线程按视频 fps 抓帧
                                 ③ 推理 + 画框                            只保留最新一帧
                                 ④ frameReady/resultReady ── 排队信号 ──► 主线程槽更新界面
```

设计要点：

- **谁用谁创建**：VideoSource / YoloDetector 对象只在检测线程内创建和销毁，
  UI 线程永不持有它们的指针——跨线程只递"数据请求"（mutex 信箱）和"参数"（atomic），无指针竞态
- **最新帧缓冲 + 帧号**：解码线程按视频自身帧率持续抓帧、只保留最新一帧；
  推理慢时自动丢帧保实时，推理快时 `read()` 被节拍器限速防快进
- **识别与决策分层**：模型只回答"这帧有什么"；"什么触发警报"是可配置的业务规则

## 技术栈

| 组件 | 用途 |
|---|---|
| Qt 6.8.3 (msvc2022_64) | 界面、信号槽、QThread、QSoundEffect |
| OpenCV 4.13 (videoio/imgproc/dnn) | 视频解码、前处理 blob、NMS、画框 |
| OpenCV dnn（tiny 分支内置） | ONNX 模型推理 |
| C++17 / CMake / Ninja | 构建 |

## 构建

环境要求：**Visual Studio 2022（MSVC）+ Qt 6.8.3 msvc2022_64 + OpenCV 4.13 MSVC 版 + onnxruntime 1.20.1**。

1. 修改 `CMakeLists.txt` 中三处本地路径为你的安装位置：
   - `Qt6_DIR`（Qt msvc 版）
   - `OpenCV_DIR`（OpenCV MSVC 版 build 目录）
2. CLion 打开项目（选用 Visual Studio 工具链），或命令行：

```bat
call "<VS路径>\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B cmake-build-debug-msvc -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug-msvc
```

构建完成后 `need/`（模型等资源）与全部依赖 DLL 会自动拷贝到 exe 旁边。

## 使用

1. 输入页选择视频/图片、onnx 模型（自动配对同名类别 txt，如 `m0.9.onnx` → `m0.9.txt`）
2. 输出页调整阈值、锚框、录制选项
3. 点"开始检测"；点"停止"回到预览态，视频播完自动结束

模型放置：将 onnx 模型放入 `need/`（构建时自动同步到 exe 旁边），类别 txt 首行格式如 `1 fire`（序号会被自动剥除）。

## 模型与导出

本项目按 **YOLOv5 检测头输出布局** `[1, 25200, 4+1+nc]`（候选×(框+obj置信度+类别分)）解析结果。
更换模型时必须保持该布局（YOLOv5 系列 onnx 均可；YOLOv8/11 输出布局不同，需改后处理）。

### 附带模型 `need/best.onnx`

| 项 | 值 |
|---|---|
| 来源 | `best.pt`（YOLOv5 6.0 训练，2 类：fire / smoke） |
| 导出工具 | YOLOv5 v6.0 官方 `export.py` |
| 导出命令 | `python export.py --weights best.pt --include onnx --imgsz 640 --simplify --opset 12` |
| 精度 | FP16（220 个权重张量由 FP32 转换，体积 176MB→88MB；cv::dnn CPU 推理自动以 FP32 计算，精度无损） |
| 输出 | 4 个输出层（3 个检测头中间结果 + 最终 `output`）；程序已自动适配多输出模型 |

`--simplify`（onnxsim 图化简）是关键：它折叠了 YOLOv5 新版导出中的 `Floor` 算子——
**未经化简的模型会被 OpenCV dnn 拒绝加载**（main 分支的 onnxruntime 则无此限制）。

### 换用自己的模型

```bash
git clone -b v6.0 --depth 1 https://github.com/ultralytics/yolov5.git
cd yolov5
python export.py --weights <你的.pt> --include onnx --imgsz 640 --simplify --opset 12
```

导出后把 onnx 放入 `need/`，并准备同名类别 txt（每行一个类别，或 `序号 类别名` 格式）。

### main 分支（onnxruntime）的导出差异

onnxruntime 对算子的兼容性远好于 OpenCV dnn，未化简、含 Floor、甚至 YOLOv8 模型都能加载；
但 YOLOv8 输出布局不同，仍需改后处理解析。

## 项目结构

```
├── main.cpp              # 入口
├── MainWindow.h/cpp      # 界面层：控件、槽、配置持久化
├── Detection.h/cpp       # 检测线程：三态循环、请求信箱、警报计数、录像
├── VideoSource.h/cpp     # 数据源：解码线程、最新帧缓冲、翻转/旋转
├── YoloDetector.h/cpp    # 推理：blobFromImage → forward → NMS → 画框
└── need/                 # 模型、类别文件、测试视频、警报音
```

## 已知边界

- 屏幕捕获未实现（open 中预留了接口）
- 视频源为实时模式：推理慢于视频帧率时丢帧采样（画面跳跃但时间线对齐）；
  如需逐帧全覆盖分析，可参考架构说明去掉解码线程
- 更换模型类别后，警报对所有检出目标生效
