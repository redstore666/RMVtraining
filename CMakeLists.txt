cmake_minimum_required(VERSION 3.16)
project(RMVTraining LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_WEB_DEBUG "Build the scalar Web Debug dashboard" ON)
option(ENABLE_WEB_SYSTEM "Build the web system view (tests/web_system, test_camera_system)" ON)
option(BUILD_TESTING "Build the starter-project tests" ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc calib3d highgui dnn)
find_package(Threads REQUIRED)

# OpenVINO 运行时（神经网络四点检测路线，ArmorDetectorNet 依赖）。
# 安装：python3 -m pip install openvino
# 然后用 -DOpenVINO_DIR=<venv>/lib/python3.X/site-packages/openvino/cmake 指向配置目录。
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_subdirectory(serialPort)

if(ENABLE_WEB_DEBUG)
    add_subdirectory(web_debug)
endif()

# 系统视图用的 webdebug 副本（多一条 /frame.jpg 图片通道）；与 web_debug 互不影响。
if(ENABLE_WEB_SYSTEM)
    add_subdirectory(tests/web_system)
endif()

# MVS SDK 路径变量（放在 add_library 之前）
set(MVS_SDK_ROOT "/opt/MVS" CACHE PATH "MVS SDK install root")

add_library(algorithm_support STATIC
    armor/src/BulletModel.cpp
    armor/src/KalmanFilter.cpp
)
target_include_directories(algorithm_support PUBLIC armor/include)
target_link_libraries(algorithm_support PUBLIC Eigen3::Eigen)

# 旋转车体运动模型（任务 4）：只依赖 Eigen，与相机/串口/OpenVINO 解耦，
# 便于离线跑合成真值收敛测试。
add_library(rotating_armor_model STATIC
    student/src/RotatingArmorModel.cpp
)
target_include_directories(rotating_armor_model PUBLIC student/include)
target_link_libraries(rotating_armor_model PUBLIC Eigen3::Eigen)

# 生产 EKF（任务 4 第 2 步）：状态/协方差/生命周期 + 首帧锁定契约，
# 数学公式全部来自 rotating_armor_model，本库不复制任何公式。
add_library(rotating_armor_ekf STATIC
    student/src/RotatingArmorEkf.cpp
    student/src/RotatingArmorTracker.cpp
    student/src/AimGeometry.cpp
    student/src/MultiTargetTracker.cpp
)
target_include_directories(rotating_armor_ekf PUBLIC
    student/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(rotating_armor_ekf PUBLIC rotating_armor_model Eigen3::Eigen opencv_core)

# EKF 俯视图（Top-Down）渲染器：把 EKF 结果画成 tmp/example.jpg 风格的调试图
# （红点车心 + 四块蓝色模型板 + 绿色实测板 + 瞄准射线）。只读 EKF 结果、不碰跟踪
# 逻辑，也不进 training_core —— 主程序 rmv_training 的依赖与行为保持不变。
add_library(topdown_view STATIC
    src/TopDownView.cpp
)
target_link_libraries(topdown_view PUBLIC rotating_armor_ekf opencv_imgproc)

# 实验性视图代码（卡片/折线拼图、迷你俯视图）。全部放在 tests/ 下：
# 这部分只在调试工具里用，不进 training_core，主程序 rmv_training 不依赖它。
add_library(system_view STATIC
    tests/SystemView.cpp
)
target_include_directories(system_view PUBLIC tests)
target_link_libraries(system_view PUBLIC Eigen3::Eigen opencv_core opencv_imgproc)

# 相机叠加层（检测框/坐标轴/PnP 面板/EKF 面板）：两个实时工具共用一份画法。
add_library(camera_overlay STATIC
    tests/CameraOverlay.cpp
)
target_include_directories(camera_overlay PUBLIC tests)
target_link_libraries(camera_overlay PUBLIC training_core opencv_core opencv_imgproc)

add_library(training_core STATIC
    src/AppSupport.cpp
    src/AutoAimTypes.cpp
    student/src/StudentTasks.cpp
    student/src/ArmorDetectorNet.cpp
)
target_include_directories(training_core
    PUBLIC
    include
    student/include
    serialPort/include
    ${OpenCV_INCLUDE_DIRS}
    ${MVS_SDK_ROOT}/include # ← 新增：MVS SDK 头文件
)
target_link_libraries(training_core
    PUBLIC
    serialPort
    rotating_armor_ekf # ← 新增：armor_solve 之后的 EKF 预测要用（含模型层）
    ${OpenCV_LIBS}
    openvino::runtime
    MvCameraControl # ← 新增：MVS SDK 库
)
target_link_directories(training_core
    PUBLIC
    ${MVS_SDK_ROOT}/lib/64 # ← 新增：库搜索路径
)

# OpenVINO 运行库目录（pip 包内 libs/，与 cmake/ 同级）：RPATH 传导给所有可执行文件，
# 保证 dlopen 能找到 libopenvino.so 与 CPU 插件。
get_filename_component(OV_LIBS_DIR "${OpenVINO_DIR}/../libs" ABSOLUTE)
set_target_properties(training_core PROPERTIES
    INTERFACE_BUILD_RPATH "${OV_LIBS_DIR}")
link_directories("${OV_LIBS_DIR}")

add_executable(rmv_training main.cpp)
target_include_directories(rmv_training PRIVATE web_debug/include)
target_link_libraries(rmv_training
    PRIVATE
    training_core
    algorithm_support
    Threads::Threads
)

set_target_properties(rmv_training PROPERTIES
    BUILD_RPATH "/opt/MVS/lib/64")

# 与 test_camera 同理：让 dlopen 能按 RPATH 找到 libFormatConversion.so。
target_link_options(rmv_training PRIVATE "-Wl,--disable-new-dtags")

if(ENABLE_WEB_DEBUG)
    target_link_libraries(rmv_training PRIVATE web_debug)
endif()

# 上车程序的迷你俯视图网页（main.cpp 的 RMV_SYSTEM_WEB 块）：渲染/推流纯被动，
# 与生产 webdebug 的 web_port 错开（实际端口 = web_port+1，默认 8081）。
if(ENABLE_WEB_SYSTEM)
    target_link_libraries(rmv_training PRIVATE system_web system_view ${OpenCV_LIBS})
    target_include_directories(rmv_training PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_compile_definitions(rmv_training PRIVATE RMV_SYSTEM_WEB=1)
endif()

# 精简版（上车测试）：合成回归/单元测试套件留在 test 分支，本分支只保留
# 上车运行与现场调试所需的两个工具（test_camera_live / test_camera_system）
# 和标定工具（calib_capture / calibrate_intrinsics）。

# ============ 标定工具 ============

# 标定图像采集：预览 + 空格保存（需要相机，运行方式同 test_camera）。
add_executable(calib_capture tests/calib_capture.cpp)
target_link_libraries(calib_capture PRIVATE training_core ${OpenCV_LIBS})
target_link_options(calib_capture PRIVATE "-Wl,--disable-new-dtags")
set_target_properties(calib_capture PROPERTIES
    BUILD_RPATH "/opt/MVS/lib/64")

# 内参标定脚本：纯 OpenCV，不依赖相机（可离线对图像目录运行）。
add_executable(calibrate_intrinsics tools/calibrate_intrinsics.cpp)
target_link_libraries(calibrate_intrinsics PRIVATE ${OpenCV_LIBS})
target_include_directories(calibrate_intrinsics PRIVATE ${OpenCV_INCLUDE_DIRS})

# 实时相机取流 + 装甲板检测（相机必须已连接）。
add_executable(test_camera_live tests/test_camera_live.cpp)
# topdown_view：EKF 俯视图渲染（与 test_camera_playback 的 --topdown 共用同一份实现）。
target_link_libraries(test_camera_live PRIVATE training_core topdown_view camera_overlay
                      ${OpenCV_LIBS})

# 系统视图（网页版）：遥测 + 右下角迷你俯视图都推到浏览器，不画相机画面。
# 与 test_camera_live 完全独立，互不影响（第一阶段验收仍用 test_camera_live）。
add_executable(test_camera_system tests/test_camera_system.cpp)
target_link_libraries(test_camera_system PRIVATE training_core camera_overlay system_view
                      ${OpenCV_LIBS})
# MVS SDK 的 ConvertPixelType 内部 dlopen libFormatConversion.so：dlopen 只认
# 老式 DT_RPATH（RUNPATH 不认）——没有它就报 0x8000000C"像素转换失败"
# （test_camera_live 有 RPATH 所以能跑，本 target 曾漏配）。与 rmv_training 同修。
target_link_options(test_camera_system PRIVATE "-Wl,--disable-new-dtags")
# 正弦小陀螺 ω(t)=A·sin(Ωt+φ)+b 在线拟合（Ceres，参考用户 task2_fit 项目）。
# 找不到 Ceres 时静默跳过（在线拟合改走无 Ceres 的粗扫+线性 LS 降级路径）。
find_package(Ceres QUIET)
if(Ceres_FOUND)
    target_link_libraries(test_camera_system PRIVATE ${CERES_LIBRARIES})
    target_include_directories(test_camera_system PRIVATE ${CERES_INCLUDE_DIRS})
    target_compile_definitions(test_camera_system PRIVATE HAVE_CERES=1)
    message(STATUS "test_camera_system: Ceres ${Ceres_VERSION} 已链接（正弦拟合可用）")
endif()
if(ENABLE_WEB_SYSTEM)
    target_link_libraries(test_camera_system PRIVATE system_web)
endif()
target_include_directories(test_camera_live PRIVATE ${OpenCV_INCLUDE_DIRS})

# 同 test_camera：让 dlopen 能按 RPATH 找到 libFormatConversion.so。
target_link_options(test_camera_live PRIVATE "-Wl,--disable-new-dtags")
set_target_properties(test_camera_live PROPERTIES
    BUILD_RPATH "/opt/MVS/lib/64")