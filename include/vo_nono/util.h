#ifndef VO_NONO_UTIL_H
#define VO_NONO_UTIL_H

#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <random>

namespace vo_nono {
constexpr double EPS = 0.001;
uint64_t rand64();

template<typename T>
class Histogram {
public:
    Histogram(int N, std::function<int(T)> indexer)
        : m_indexer(indexer),
          m_boxes(N, 0),
          m_topK(0) {}

    void insert_element(T val) {
        int index = m_indexer(val);
        assert(index >= 0 && index < (int) m_boxes.size());
        m_boxes[index] += 1;
    }

    void cal_topK(int k) {
        m_topK = -1;
        std::vector<int> buffer(k, -1);
        for (auto val : m_boxes) {
            if (val > buffer[k - 1]) { buffer[k - 1] = val; }
            int cur = k - 1;
            while (cur > 0) {
                if (buffer[cur] > buffer[cur - 1]) {
                    int tmp = buffer[cur];
                    buffer[cur] = buffer[cur - 1];
                    buffer[cur - 1] = tmp;
                    cur -= 1;
                } else {
                    break;
                }
            }
        }
        m_topK = buffer[k - 1];
    }

    bool is_topK(T val) const {
        int index = m_indexer(val);
        assert(index >= 0 && index < (int) m_boxes.size());
        return m_boxes[index] >= m_topK;
    }

private:
    std::function<int(T)> m_indexer;
    std::vector<int> m_boxes;
    int m_topK;
};

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

void angle_axis_to_rotation_mat(const double angle_axis[3],
                                double result[3][3]);

template<typename T, typename U>
[[nodiscard]] inline std::vector<T> filter_by_mask(
        const std::vector<T> &targets, const std::vector<U> &mask) {
    assert(targets.size() == mask.size());
    std::vector<T> res;
    res.reserve(targets.size());
    for (int i = 0; i < (int) targets.size(); ++i) {
        if (mask[i]) { res.push_back(targets[i]); }
    }
    return res;
}
}// namespace vo_nono

#define assert_float_eq(A, B) \
    assert(fabs((double) (A) - (double) (B)) < vo_nono::EPS);
#define unimplemented() assert(false);
#define float_eq_zero(A) (fabs((double) (A)) < vo_nono::EPS)

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

#endif//VO_NONO_UTIL_H
