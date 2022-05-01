#ifndef VO_NONO_FILTER_H
#define VO_NONO_FILTER_H

#include "vo_nono/camera.h"
#include <opencv2/core.hpp>

namespace vo_nono {
class InvDepthFilter {
public:
    InvDepthFilter() : dir_(cv::Mat::zeros(3, 1, CV_32F)) {}
    bool filter(const cv::Mat &o0_cw, const cv::Mat &Rcw0, const cv::Mat &o1_cw,
                const cv::Mat &coord);
    [[nodiscard]] double get_variance() const { return var_; }
    [[nodiscard]] cv::Mat get_coord(const cv::Mat &o_cw,
                                    const cv::Mat &Rcw) const {
        return -o_cw + Rcw.t() * dir_ / mean_;
    }
    [[nodiscard]] int get_cnt() const { return cnt_; }
    void set_information(const Camera &camera, const cv::Point2f pt,
                         double basic_var) {
        dir_ = cv::Mat::zeros(3, 1, CV_32F);
        dir_.at<float>(0) = (pt.x - camera.cx()) / camera.fx();
        dir_.at<float>(1) = (pt.y - camera.cy()) / camera.fy();
        dir_.at<float>(2) = 1.0f;
        dir_ /= cv::norm(dir_);
        basic_var_ = basic_var;
    }
    [[nodiscard]] bool relative_error_less(double th) const {
        if (std::isinf(th)) {
            return true;
        } else if (mean_ - 2 * sqrt_var_ <= 0) {
            return false;
        }
        double max_d = 1.0 / (mean_ - 2 * sqrt_var_);
        double mean_d = 1.0 / mean_;
        return (max_d - mean_d) <= th * mean_d;
    }

private:
    int cnt_ = 0;
    double mean_ = 1.0;
    double var_ = 1.0;
    double sqrt_var_ = 1.0;
    double basic_var_ = 1;
    cv::Mat dir_;
};
}
#endif//VO_NONO_FILTER_H
