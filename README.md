# ObjectDetectionOnnxRT

> **当前分支：tiny** —— 推理引擎为 OpenCV 自带 dnn（无需 onnxruntime）；main 分支使用 onnxruntime。

**实时目标检测桌面应用** —— Qt6 界面 + OpenCV 解码与后处理 + OpenCV dnn 推理，三线程架构。

> 源自个人毕设"烟雾火焰检测"（PyQt5 + onnxruntime）的 C++/Qt6 完整重构版。

## ⚠ 模型支持范围：仅限 YOLOv5 6.0 导出的 onnx

**本项目的推理代码是按 YOLOv5 6.0 的导出格式写死的专用推理，不是通用 ONNX 检测器。**

| 约定项 | 固定值 |
|---|---|
| 输入尺寸 | **640×640**（代码写死，拉伸缩放） |
| 终输出层名 | **`output`**（load 时校验，名字不对直接拒绝加载） |
| 输出形状 | `[1, 25200, 5+类别数]`，25200 = (80×80+40×40+20×20)×3 anchor |
| 每行含义 | `cx, cy, w, h`（相对 640 输入）+ obj 置信度 + 各类别分 |

由此：

- ✅ **可以换的模型**：用 **YOLOv5 v6.0** 训练**其他类别的目标检测**再导出——类别数可变（后处理按 `5+类别数` 自适应），输入尺寸、输出名都不变；
- ❌ **不能换的**：其他输入尺寸、其他版本（YOLOv5 新版 / YOLOv8 / YOLOv11 等）的模型。
  换版本 = 输出布局、输出名、anchor 结构全变（如 YOLOv8 是 `[1, 4+nc, 8400]`、DFL 解码），**必须重写推理与后处理代码**。

附带模型 `need/` 下三个（N0.6 / M0.9 / L0.9）均为 YOLOv5 6.0 训练的 fire/smoke 两类模型，
导出命令：`python export.py --weights best.pt --include onnx --imgsz 640 --simplify --opset 12`。
`--simplify`（onnxsim）是关键：它折叠了新版导出中的 `Floor` 算子——**未经化简的模型会被 OpenCV dnn 拒绝加载**（main 分支的 onnxruntime 无此限制）。

## 功能

- 检测视频文件 / 图片 / 摄像头中的目标，实时显示带锚框与置信度的画面
- 模型（.onnx）与类别文件（.txt）可随时更换（限 YOLOv5 6.0 格式，见上文）
- 置信度 / IOU 阈值、翻转 / 旋转、锚框颜色运行中即时生效（atomic 热参数）
- 警报：连续 10 帧检出任意目标 → 警报音（5 秒冷却，运行中可开关）
- 录制检测视频（mp4）、截图、日志保存
- 界面配置持久化（QSettings），下次启动自动恢复
- 检测进行中锁定输入页；关窗确认 + 线程安全退出

## 已知问题：多目标互相"覆盖"（NMS 类无关）

当前用 `cv::dnn::NMSBoxes` 做非极大值抑制，它是**类无关**的：所有类别的候选框放在
同一个池子里互相抑制。表现是——**两个不同类别的目标如果位置重叠**（烟雾火焰场景
里很常见：烟和火本就长在一起），置信度低的那个框会被置信度高的压掉，
界面上"少画了一个框"、日志里"少报了一类"。

正确做法是**类内 NMS**：按类别分组、各组各做一遍 NMS（YOLOv5 官方做法是给框坐标
整体平移 `classId × 大常数` 再统一 NMS，效果等价）。tiny 分支尚未修，留作已知问题。

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

## 性能参考

实测（Release 版 OpenCV，640 输入，CPU 推理）：
N0.6 ≈ 48ms/帧（约 20 FPS）；M0.9 ≈ 257ms/帧（约 4 FPS）。
**Debug 构建下推理慢 3 倍以上（N0.6 ≈ 157ms），觉得慢先查构建配置。**

## 技术栈

| 组件 | 用途 |
|---|---|
| Qt 6.8.3 (msvc2022_64) | 界面、信号槽、QThread、QSoundEffect |
| OpenCV 4.13 (videoio/imgproc/dnn) | 视频解码、前处理 blob、NMS、画框 |
| OpenCV dnn（tiny 分支内置） | ONNX 模型推理（CPU） |
| C++17 / CMake / Ninja | 构建 |

## 构建

环境要求：**Visual Studio 2022（MSVC）+ Qt 6.8.3 msvc2022_64 + OpenCV 4.13 MSVC 版**
（tiny 分支不需要 onnxruntime）。

1. 修改 `CMakeLists.txt` 中两处本地路径为你的安装位置：
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

1. 输入页选择摄像头 / 视频 / 图片，选择 onnx 模型（自动配对同名类别 txt，如 `M0.9.onnx` → `M0.9.txt`）
2. 输出页调整阈值、锚框、录制选项
3. 点"开始检测"；点"停止"回到预览态，视频播完自动结束

模型放置：将 onnx 模型放入 `need/`（构建时自动同步到 exe 旁边），类别 txt 每行一个类别，
或 `序号 类别名` 格式（序号会被自动剥除）。

### 换用自己的模型（限 YOLOv5 6.0）

```bash
git clone -b v6.0 --depth 1 https://github.com/ultralytics/yolov5.git
cd yolov5
python export.py --weights <你的.pt> --include onnx --imgsz 640 --simplify --opset 12
```

导出后把 onnx 放入 `need/`，并准备同名类别 txt。

## 项目结构

```
├── main.cpp              # 入口
├── MainWindow.h/cpp      # 界面层：控件、槽、配置持久化
├── Detection.h/cpp       # 检测线程：三态循环、请求信箱、警报计数、录像
├── VideoSource.h/cpp     # 数据源：解码线程、最新帧缓冲、翻转/旋转
├── YoloDetector.h/cpp    # 推理：blobFromImage → forward("output") → NMS → 画框
└── need/                 # 模型、类别文件、测试视频、警报音
```

## 已知边界

- 多目标 NMS 互相覆盖（类无关 NMS），见上文"已知问题"
- 屏幕捕获未实现（open 中预留了接口）
- 视频源为实时模式：推理慢于视频帧率时丢帧采样（画面跳跃但时间线对齐）；
  如需逐帧全覆盖分析，可参考架构说明去掉解码线程
- 更换模型类别后，警报对所有检出目标生效
