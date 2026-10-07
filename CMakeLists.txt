cmake_minimum_required(VERSION 3.16)
project(ObjectDetectionOnnxRT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# 本项目仅支持 MSVC 工具链（Qt 6.8.3 msvc2022_64 + 官方 OpenCV MSVC 预编译版）
if (NOT MSVC)
    message(FATAL_ERROR "本项目仅支持 MSVC 工具链：请在 CLion 的 CMake 配置文件中选用 Visual Studio 工具链")
endif ()

# 锁定 Qt 6.8.3 MSVC 套件（FORCE 覆盖旧缓存/PATH 带进来的其他 Qt）
set(Qt6_DIR "D:/ProgramFiles/Qt/6.8.3/msvc2022_64/lib/cmake/Qt6" CACHE PATH "" FORCE)
list(APPEND CMAKE_PREFIX_PATH "D:/ProgramFiles/Qt/6.8.3/msvc2022_64")

# 源码为 UTF-8；/Zc:__cplusplus 为 Qt6+MSVC 标配；NOMINMAX 防 windows.h 的 min/max 宏污染 std::min/max
add_compile_options(/utf-8 /Zc:__cplusplus)
add_compile_definitions(NOMINMAX)

find_package(Qt6 COMPONENTS
        Core
        Gui
        Widgets
        Multimedia      # QSoundEffect 播放警报音用
        REQUIRED)

# OpenCV（官方 MSVC 预编译版，opencv_world 单库）
set(OpenCV_DIR "D:/Libraries/opencv-4.13.0-msvc/opencv/build")
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs highgui dnn)

# onnxruntime（官方 MSVC 预编译包，YoloDetector 运行时动态加载）
set(ORT_DIR "D:/Libraries/onnxruntime-win-x64-1.20.1")

add_executable(ObjectDetectionOnnxRT
        main.cpp
        MainWindow.cpp
        MainWindow.h
        Detection.h
        Detection.cpp
        VideoSource.cpp
        VideoSource.h
        YoloDetector.cpp
        YoloDetector.h
)

target_link_libraries(ObjectDetectionOnnxRT
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Multimedia
        ${OpenCV_LIBS}
        "${ORT_DIR}/lib/onnxruntime.lib")

target_include_directories(ObjectDetectionOnnxRT PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        "${ORT_DIR}/include")

if (WIN32)
    # 从 Qt6Core_DIR 反推 Qt 插件目录
    get_filename_component(QT_PLUGINS_DIR "${Qt6Core_DIR}/../../../plugins" ABSOLUTE)

    # 拷贝 Qt 运行库 DLL（Multimedia 依赖 Network 也要，否则加载期直接起不来）
    foreach (QT_LIB Core Gui Widgets Multimedia Network)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:Qt6::${QT_LIB}>"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endforeach (QT_LIB)

    # 拷贝 OpenCV 运行库（opencv_world 单 DLL）
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:opencv_world>"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>")

    # 拷贝 MSVC 运行库（脱离开发环境双击运行必需）
    file(GLOB CRT_DLLS
            "D:/ProgramFiles/VisualStudio/MicrosoftVisualStudio18/Community/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT/*.dll")
    foreach (CRT_DLL ${CRT_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CRT_DLL}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endforeach (CRT_DLL)

    # Debug 配置还需要调试版 CRT，否则 CLion 调试时 exe 起不来
    file(GLOB CRTD_DLLS
            "D:/ProgramFiles/VisualStudio/MicrosoftVisualStudio18/Community/VC/Redist/MSVC/*/x64/Microsoft.VC*.DebugCRT/*.dll")
    foreach (CRT_DLL ${CRTD_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CRT_DLL}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endforeach (CRT_DLL)
    if (EXISTS "D:/Windows Kits/10/bin/10.0.28000.0/x64/ucrt/ucrtbased.dll")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "D:/Windows Kits/10/bin/10.0.28000.0/x64/ucrt/ucrtbased.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endif ()

    # 拷贝平台插件 qwindows（GUI 程序运行必需；Debug 构建加载 d 后缀的调试版插件）
    foreach (QW_NAME qwindows.dll qwindowsd.dll)
        if (EXISTS "${QT_PLUGINS_DIR}/platforms/${QW_NAME}")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/platforms/")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${QT_PLUGINS_DIR}/platforms/${QW_NAME}"
                    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/platforms/")
        endif ()
    endforeach (QW_NAME)

    # 拷贝资源目录 need/（模型、测试视频、警报音）到 exe 旁边，程序按相对路径取用
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/need"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/need")
endif ()

# 拷贝 onnxruntime DLL（YoloDetector 运行时动态加载）
foreach (ORT_DLL onnxruntime.dll onnxruntime_providers_shared.dll)
    if (EXISTS "${ORT_DIR}/lib/${ORT_DLL}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_DIR}/lib/${ORT_DLL}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endif ()
endforeach ()

# 拷贝 Multimedia 音频后端插件（QSoundEffect 播警报必需；Debug 构建加载 d 后缀插件）
foreach (MM_NAME windowsmediaplugin.dll windowsmediaplugind.dll)
    if (EXISTS "${QT_PLUGINS_DIR}/multimedia/${MM_NAME}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/multimedia/")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${QT_PLUGINS_DIR}/multimedia/${MM_NAME}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/multimedia/")
    endif ()
endforeach (MM_NAME)
