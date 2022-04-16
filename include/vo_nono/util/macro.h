#ifndef VO_NONO_MACRO_H
#define VO_NONO_MACRO_H

#include <iostream>

#define assert_float_eq(A, B) \
    assert(fabs((double) (A) - (double) (B)) < 0.00001);
#define unimplemented() assert(false);
#define float_eq_zero(A) (fabs((double) (A)) < 0.00001)

#define TIME_IT(CODE, msg)                                                     \
    do {                                                                       \
        std::chrono::steady_clock::time_point t1 =                             \
                std::chrono::steady_clock::now();                              \
        { CODE; }                                                              \
        std::chrono::steady_clock::time_point t2 =                             \
                std::chrono::steady_clock::now();                              \
        std::chrono::duration<double> time_used =                              \
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 - \
                                                                          t1); \
        std::cout << (msg) << time_used.count() << " seconds." << std::endl;   \
    } while (0);

#ifdef NDEBUG
#define log_debug(contents) ;
#define log_debug_line(contents) ;
#define log_debug_pos(contents) ;
#else
#include <chrono>
#include <iostream>

#define log_debug(contents) (std::cout << contents)

#define log_debug_line(contents) (std::cout << contents << std::endl)

#define log_debug_pos(contents) \
    (std::cout << __FUNCTION__ << ";" << __LINE__ << std::endl << contents)

#endif

#endif//VO_NONO_MACRO_H
