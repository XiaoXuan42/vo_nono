add_executable(app_match match/main.cpp)
target_link_libraries(app_match PRIVATE vo_nono)
target_link_libraries(app_match PRIVATE ${OpenCV_LIBS})

add_executable(app_tum tum/main.cpp tum/tum.h tum/tum.cpp)
target_link_libraries(app_tum PRIVATE vo_nono)
target_link_libraries(app_tum PRIVATE ${OpenCV_LIBS})

add_executable(app_test test/main.cpp)
target_link_libraries(app_test PRIVATE ${OpenCV_LIBS})
target_link_libraries(app_test PRIVATE vo_nono)