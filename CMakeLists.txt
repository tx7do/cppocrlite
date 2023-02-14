cmake_minimum_required(VERSION 3.16)

project(cppocrlite)

set(CMAKE_CXX_STANDARD 11)

# 设置动态库的输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# 设置静态库的输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_definitions(-DUNICODE -D_UNICODE)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions("-Wall -g -O0")
else ()
    add_definitions("-Wall")
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# source
file(GLOB OCR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(OCR_COMPILE_CODE ${OCR_SRC})

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "OpenMP Not Found!")
endif ()

# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif (OpenCV_FOUND)

# OnnxRuntime
set(OnnxRuntime_INCLUDE_DIRS, "/opt/onnxruntime/include")
set(OnnxRuntime_LIBS, "/opt/onnxruntime/lib")
message(STATUS "OnnxRuntime_LIBS: ${OnnxRuntime_LIBS}")
message(STATUS "OnnxRuntime_INCLUDE_DIRS: ${OnnxRuntime_INCLUDE_DIRS}")

message(STATUS "OCR_COMPILE_CODE: ${OCR_COMPILE_CODE}")

add_library(cppocrlite SHARED ${OCR_COMPILE_CODE})

target_include_directories(cppocrlite PRIVATE /opt/onnxruntime/include ${OnnxRuntime_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

target_link_libraries(cppocrlite ${OnnxRuntime_LIBS} ${OpenCV_LIBS} ${OpenMP_CXX_LIB_NAMES})