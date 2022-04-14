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
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Threads REQUIRED)

# vo_nono
add_library(vo_nono SHARED include/vo_nono/system.h src/system.cpp src/frontend.cpp include/vo_nono/frontend.h include/vo_nono/types.h src/frame.cpp include/vo_nono/frame.h include/vo_nono/util.h src/util.cpp include/vo_nono/camera.h include/vo_nono/map.h include/vo_nono/point.h src/point.cpp include/vo_nono/config.h src/config.cpp src/motion.cpp include/vo_nono/pnp.h src/pnp.cpp include/vo_nono/keypoint/match.h src/match.cpp include/vo_nono/keypoint/geometry.h include/vo_nono/optimize_graph.h src/optimize_graph.cpp src/map.cpp include/vo_nono/keypoint/triangulate.h src/triangulate.cpp include/vo_nono/keypoint/epipolar.h)
target_compile_options(vo_nono PRIVATE -Wall -Werror)
target_include_directories(vo_nono PUBLIC include/)
target_link_libraries(vo_nono ${OpenCV_LIBS})
target_link_libraries(vo_nono yaml-cpp)
target_link_libraries(vo_nono Threads::Threads)
target_link_libraries(vo_nono ${CERES_LIBRARIES})

add_subdirectory(app)