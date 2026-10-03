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

add_library(training_core STATIC
    src/AppSupport.cpp
    src/AutoAimTypes.cpp
    student/src/StudentTasks.cpp
    student/src/ArmorClassifier.cpp
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
    ${OpenCV_LIBS}
    MvCameraControl # ← 新增：MVS SDK 库
)
target_link_directories(training_core
    PUBLIC
    ${MVS_SDK_ROOT}/lib/64 # ← 新增：库搜索路径
)

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

# ============ 任务 2 配套工具（数字分类 / 装甲板检测）============

# 数字分类器独立测试：模型 + 预处理链路是否工作。
add_executable(test_classifier tests/test_classifier.cpp)
target_link_libraries(test_classifier PRIVATE training_core ${OpenCV_LIBS})

# 装甲板检测离线测试：调用 armor_detect，跑 tests/data/ 里的合成图。
add_executable(test_detect tests/test_detect.cpp)
target_link_libraries(test_detect PRIVATE training_core ${OpenCV_LIBS})
target_include_directories(test_detect PRIVATE ${OpenCV_INCLUDE_DIRS})

# 分类器烟雾测试（ctest -R classifier 运行）。
add_test(NAME classifier_sticker_1
    COMMAND test_classifier ${CMAKE_SOURCE_DIR}/tests/data/sticker_1.png
            --onnx ${CMAKE_SOURCE_DIR}/models/armor_cls_64.onnx --expect 1)
add_test(NAME classifier_sticker_5
    COMMAND test_classifier ${CMAKE_SOURCE_DIR}/tests/data/sticker_5.png
            --onnx ${CMAKE_SOURCE_DIR}/models/armor_cls_64.onnx --expect 5)
add_test(NAME classifier_sticker_9neg
    COMMAND test_classifier ${CMAKE_SOURCE_DIR}/tests/data/sticker_9neg.jpg
            --onnx ${CMAKE_SOURCE_DIR}/models/armor_cls_64.onnx --expect 9neg)
