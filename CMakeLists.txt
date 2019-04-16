cmake_minimum_required(VERSION 2.6)
project(match)

find_package(OpenCV 3.4.0 REQUIRED)
message(${OpenCV_VERSION})
message(${OpenCV_LIBRARIES})

set(CMAke_CXX_FLAGS -std=c++11)

add_executable(${PROJECT_NAME} main.cpp match.cpp gms.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})