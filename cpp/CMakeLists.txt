cmake_minimum_required(VERSION 3.16)
project(classify CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(dxrt REQUIRED)          # /usr/local/lib/cmake/dxrt
find_package(OpenCV REQUIRED)

add_executable(classify main.cpp)
target_link_libraries(classify PRIVATE dxrt::dxrt ${OpenCV_LIBS})
