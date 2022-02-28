cmake_minimum_required(VERSION 3.19)
project(vo_nono)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# external libraries
find_package(OpenCV REQUIRED)

# vo_nono
add_library(vo_nono SHARED include/vo_nono/system.h src/system.cpp src/frontend.cpp include/vo_nono/frontend.h include/vo_nono/types.h src/frame.cpp include/vo_nono/frame.h include/vo_nono/util.h src/util.cpp include/vo_nono/camera.h include/vo_nono/map.h include/vo_nono/point.h src/point.cpp)
target_include_directories(vo_nono PUBLIC include/)
target_link_libraries(vo_nono ${OpenCV_LIBS})

add_subdirectory(app)