#ifndef VO_NONO_UTIL_H
#define VO_NONO_UTIL_H

#include <cstdint>
#include <ctime>
#include <cmath>
#include <random>

namespace vo_nono {
    constexpr double EPS = 0.001;
    uint64_t rand64();
}

#define assert_float_eq(A, B) assert(fabs((double)(A) - (double)(B)) < vo_nono::EPS);
#define unimplemented() assert(false);
#define float_eq_zero(A) (fabs((double)(A)) < vo_nono::EPS)

#endif//VO_NONO_UTIL_H
