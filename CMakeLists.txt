cmake_minimum_required(VERSION 2.6)
project(match)

find_package(OpenCV 3.4.0 REQUIRED)
message(${OpenCV_VERSION})
message(${OpenCV_LIBRARIES})

set(CMAke_CXX_FLAGS -std=c++11)

set(SRC src/main.cpp
        src/gms.cpp
        src/match.cpp
        src/tool.cpp)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})