#ifndef VO_NONO_UTIL_H
#define VO_NONO_UTIL_H

#include <cmath>
#include <cstdint>
#include <ctime>
#include <opencv2/core.hpp>
#include <random>

namespace vo_nono {
constexpr double EPS = 0.001;
uint64_t rand64();

// from https://gist.github.com/shubh-agrawal/76754b9bfb0f4143819dbd146d15d4c8
template<typename T>
void rotation_mat_to_quaternion(const cv::Mat &R, T Q[]) {
    double trace = R.at<T>(0, 0) + R.at<T>(1, 1) + R.at<T>(2, 2);

    if (trace > 0.0) {
        T s = sqrt(trace + 1.0);
        Q[3] = (s * 0.5);
        s = 0.5 / s;
        Q[0] = ((R.at<T>(2, 1) - R.at<T>(1, 2)) * s);
        Q[1] = ((R.at<T>(0, 2) - R.at<T>(2, 0)) * s);
        Q[2] = ((R.at<T>(1, 0) - R.at<T>(0, 1)) * s);
    }

    else {
        int i = R.at<T>(0, 0) < R.at<T>(1, 1)
                        ? (R.at<T>(1, 1) < R.at<T>(2, 2) ? 2 : 1)
                        : (R.at<T>(0, 0) < R.at<T>(2, 2) ? 2 : 0);
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;

        T s = sqrt(R.at<T>(i, i) - R.at<T>(j, j) - R.at<T>(k, k) + 1.0);
        Q[i] = s * 0.5;
        s = 0.5 / s;

        Q[3] = (R.at<T>(k, j) - R.at<T>(j, k)) * s;
        Q[j] = (R.at<T>(j, i) + R.at<T>(i, j)) * s;
        Q[k] = (R.at<T>(k, i) + R.at<T>(i, k)) * s;
    }
}

cv::Mat quaternion_to_rotation_mat(const float Q[]);
}// namespace vo_nono

#define assert_float_eq(A, B) \
    assert(fabs((double) (A) - (double) (B)) < vo_nono::EPS);
#define unimplemented() assert(false);
#define float_eq_zero(A) (fabs((double) (A)) < vo_nono::EPS)

#ifdef NDEBUG
#define log_debug_line(contents) ;
#define log_debug_pos(contents) ;
#else
#include <iostream>

#define log_debug(contents) (std::cout << contents)

#define log_debug_line(contents) (std::cout << contents << std::endl)

#define log_debug_pos(contents) \
    (std::cout << __FUNCTION__ << ";" << __LINE__ << std::endl << contents)
#endif

#endif//VO_NONO_UTIL_H
