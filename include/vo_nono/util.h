#ifndef VO_NONO_UTIL_H
#define VO_NONO_UTIL_H

#include <cmath>
#include <cstdint>
#include <ctime>
#include <random>

namespace vo_nono {
constexpr double EPS = 0.001;
uint64_t rand64();
}// namespace vo_nono

#define assert_float_eq(A, B) \
    assert(fabs((double) (A) - (double) (B)) < vo_nono::EPS);
#define unimplemented() assert(false);
#define float_eq_zero(A) (fabs((double) (A)) < vo_nono::EPS)

#ifdef NDEBUG
#define log_debug(contents) ;
#define log_debug_pos(contents) ;
#else
#include <iostream>

#define log_debug(contents) \
    (std::cout << contents << std::endl)

#define log_debug_pos(contents) \
    (std::cout << __FUNCTION__ << ";" << __LINE__ << std::endl << contents)
#endif

#endif//VO_NONO_UTIL_H
