# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.0)

# 项目名称
project(detect_memory)

# C++版本
set(CMAKE_CXX_STANDARD 20)

# 包含include目录，里面都是头文件
include_directories(
        ${cppocrlite_SOURCE_DIR}/include
)

# 将 src 目录下的所有源文件加入后面的变量之中
file(GLOB DIR_EXE_SRCS ./*.cpp)

# 构建可执行文件
add_executable(${PROJECT_NAME} ${DIR_EXE_SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE /usr/local/onnxruntime/include ${OpenCV_INCLUDE_DIRS})

# 指定目标链接的库
target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        cppocrlite
        onnxruntime
        ${OpenCV_LIBS}
        ${OpenMP_CXX_LIB_NAMES}
        dl
        rt
)
