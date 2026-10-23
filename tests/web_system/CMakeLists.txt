# webdebug 的副本（tests/web_system/src/SystemWebLogger.cpp），只给 test_camera_system 用。
#
# 与 web_debug 的区别：
#  * 类名/宏名都加了 System 前缀（SystemWebLogger / START_SYSTEM_WEB / SYSTEM_LOG），
#    两套代码可以在同一个进程里共存也不会撞符号；
#  * 多一条 GET /frame.jpg（迷你俯视图的 JPEG 通道）；
#  * 端口由调用方给（test_camera_system 默认 8081，与生产程序的 8080 错开）。
#
# httplib.h / json.hpp 直接复用 web_debug/3rdparty 的头文件，不复制大文件。

add_library(system_web STATIC
    src/SystemWebLogger.cpp
)

target_include_directories(system_web PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # httplib.h、json.hpp 是 header-only，只需要 include 路径。
    ${CMAKE_SOURCE_DIR}/web_debug/3rdparty
)

# WEB_SYSTEM 打开整套实现；WEB_SYSTEM_STATIC_DIR 让 set_mount_point 找到网页，
# 不依赖进程的当前工作目录。
target_compile_definitions(system_web PUBLIC
    WEB_SYSTEM
    WEB_SYSTEM_STATIC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/static"
)

find_package(Threads REQUIRED)
target_link_libraries(system_web PUBLIC Threads::Threads)
