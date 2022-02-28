add_executable(app_match match/main.cpp)
target_link_libraries(app_match PRIVATE vo_nono)
target_link_libraries(app_match PRIVATE ${OpenCV_LIBS})