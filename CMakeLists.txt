cmake_minimum_required(VERSION 3.16)
project(RMVTraining LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_WEB_DEBUG "Build the scalar Web Debug dashboard" ON)
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

if(BUILD_TESTING)
    enable_testing()
    # 虚拟电控：用伪终端代替主控板，无硬件也能跑完整链路（见 tests/sim_controller.cpp）
    add_executable(sim_controller tests/sim_controller.cpp)
    target_include_directories(sim_controller PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/serialPort/include)
    target_link_libraries(sim_controller PRIVATE util)

    add_executable(test_scaffold tests/test_scaffold.cpp)
    target_link_libraries(test_scaffold PRIVATE training_core)
    add_test(NAME scaffold_safety COMMAND test_scaffold)
    add_test(NAME cli_help COMMAND rmv_training --help)

    # 独立相机取流测试：直接调用 get_pic()，不依赖串口。
    add_executable(test_camera tests/test_camera.cpp)
    target_link_libraries(test_camera PRIVATE training_core ${OpenCV_LIBS})

    # SDK 内部用 dlopen 加载 libFormatConversion.so：dlopen 只认老式 DT_RPATH，
    # 所以用 --disable-new-dtags 写入 RPATH 而非 RUNPATH。
    target_link_options(test_camera PRIVATE "-Wl,--disable-new-dtags")
    set_target_properties(test_camera PROPERTIES
        BUILD_RPATH "/opt/MVS/lib/64")
    add_test(NAME camera_grab COMMAND test_camera)

    # 旋转车体运动模型（任务 4）：
    #   ekf_model_unit        —— F/H/Q 的自洽性（解析 Jacobian vs 有限差分）
    #   ekf_model_convergence —— 合成真值收敛 + 默认 Q 防回退（生产 EKF）
    #   ekf_init_contract     —— 首帧锁定契约：有/无板面法向、reset 与可观测性
    #   ekf_aim_geometry     —— 云台/世界坐标系约定与互逆关系
    #   ekf_tracker          —— 锁定/收敛/换板/遮挡/超时/离群/换目标
    #   ekf_pipeline         —— 观测 -> ekf_predict -> applyPrediction -> 协议字段
    add_executable(test_rotating_armor_model tests/test_rotating_armor_model.cpp)
    target_link_libraries(test_rotating_armor_model PRIVATE rotating_armor_model)
    add_test(NAME ekf_model_unit COMMAND test_rotating_armor_model)

    add_executable(test_rotating_armor_convergence tests/test_rotating_armor_convergence.cpp)
    target_link_libraries(test_rotating_armor_convergence PRIVATE rotating_armor_ekf)
    add_test(NAME ekf_model_convergence COMMAND test_rotating_armor_convergence)

    add_executable(test_rotating_armor_ekf_init tests/test_rotating_armor_ekf_init.cpp)
    target_link_libraries(test_rotating_armor_ekf_init PRIVATE rotating_armor_ekf)
    add_test(NAME ekf_init_contract COMMAND test_rotating_armor_ekf_init)

    #   任务 4 粘合层与端到端管线。
    add_executable(test_aim_geometry tests/test_aim_geometry.cpp)
    target_link_libraries(test_aim_geometry PRIVATE rotating_armor_ekf)
    add_test(NAME ekf_aim_geometry COMMAND test_aim_geometry)

    add_executable(test_rotating_armor_tracker tests/test_rotating_armor_tracker.cpp)
    target_link_libraries(test_rotating_armor_tracker PRIVATE rotating_armor_ekf)
    add_test(NAME ekf_tracker COMMAND test_rotating_armor_tracker)

    #   多目标：两辆车各自一个 EKF，靠径向一致性把板分给不同的车。
    add_executable(test_multi_target tests/test_multi_target.cpp)
    target_link_libraries(test_multi_target PRIVATE rotating_armor_ekf)
    add_test(NAME ekf_multi_target COMMAND test_multi_target)

    #   管线测试走真实的 ekf_predict + applyPrediction，所以必须链 training_core。
    add_executable(test_ekf_pipeline tests/test_ekf_pipeline.cpp)
    target_link_libraries(test_ekf_pipeline PRIVATE training_core)
    add_test(NAME ekf_pipeline COMMAND test_ekf_pipeline)

    # 俯视图冒烟测试：合成一圈旋转车体，确认渲染器能跑完并落盘
    # （不开窗口、不连相机，几秒内完成；只验证"能画、不崩、状态收敛"）。
    add_test(NAME ekf_viewer_smoke
             COMMAND test_ekf_viewer --no-show --frames 40
                     --out ${CMAKE_CURRENT_BINARY_DIR}/topdown_smoke)
    set_tests_properties(ekf_viewer_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "phase=tracking")
endif()

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

# ============ 任务 2 配套工具（装甲板检测）============

# 装甲板检测离线测试：调用 armor_detect，跑 tests/data/ 里的合成图。
add_executable(test_detect tests/test_detect.cpp)
target_link_libraries(test_detect PRIVATE training_core ${OpenCV_LIBS})
target_include_directories(test_detect PRIVATE ${OpenCV_INCLUDE_DIRS})

# 检测耗时基准：把 detect() 拆成 预处理/推理/解码 三段，并对比模型尺寸与
# OpenVINO 线程数、性能提示（优化 detect 时间前先量清楚时间花在哪）。
add_executable(bench_detect tests/bench_detect.cpp)
target_link_libraries(bench_detect PRIVATE training_core ${OpenCV_LIBS})
target_include_directories(bench_detect PRIVATE ${OpenCV_INCLUDE_DIRS})

# 位姿解算离线测试：armor_detect → armor_solve，打印位姿（单位外参开发配置）。
add_executable(test_solve tests/test_solve.cpp)
target_link_libraries(test_solve PRIVATE training_core ${OpenCV_LIBS})
target_include_directories(test_solve PRIVATE ${OpenCV_INCLUDE_DIRS})

# 实拍批量评估：对目录内所有图片跑检测并输出检出率（实拍调参/回归用）。
add_executable(test_batch tests/test_batch.cpp)
target_link_libraries(test_batch PRIVATE training_core ${OpenCV_LIBS})
target_include_directories(test_batch PRIVATE ${OpenCV_INCLUDE_DIRS})

# 离线回放：把录制的视频喂进完整链路（检测 -> PnP -> 可选 EKF），用于在没有
# 相机现场时评估视觉识别质量，以及用真实序列做 EKF 关联/锁定/预测的离线调参。
add_executable(test_camera_playback tests/test_camera_playback.cpp)
target_link_libraries(test_camera_playback PRIVATE training_core topdown_view ${OpenCV_LIBS})
if(ENABLE_WEB_DEBUG)
    target_link_libraries(test_camera_playback PRIVATE web_debug)
endif()
target_include_directories(test_camera_playback PRIVATE ${OpenCV_INCLUDE_DIRS})

# EKF 俯视图演示：合成真值绕云台旋转，实时看状态收敛与实测板对齐
# （不连相机、不需要模型文件和视频；--out 可在无显示环境导出 PNG）。
add_executable(test_ekf_viewer tests/test_ekf_viewer.cpp)
target_link_libraries(test_ekf_viewer PRIVATE topdown_view ${OpenCV_LIBS})
target_include_directories(test_ekf_viewer PRIVATE ${OpenCV_INCLUDE_DIRS})

# 实时相机取流 + 装甲板检测（相机必须已连接）。
add_executable(test_camera_live tests/test_camera_live.cpp)
target_link_libraries(test_camera_live PRIVATE training_core ${OpenCV_LIBS})
target_include_directories(test_camera_live PRIVATE ${OpenCV_INCLUDE_DIRS})

# 同 test_camera：让 dlopen 能按 RPATH 找到 libFormatConversion.so。
target_link_options(test_camera_live PRIVATE "-Wl,--disable-new-dtags")
set_target_properties(test_camera_live PROPERTIES
    BUILD_RPATH "/opt/MVS/lib/64")