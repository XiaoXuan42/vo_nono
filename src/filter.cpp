#include "vo_nono/util/filter.h"

namespace vo_nono {
bool InvDepthFilter::filter(const cv::Mat &o0_cw, const cv::Mat &Rcw0,
                            const cv::Mat &o1_cw, const cv::Mat &coord) {
    assert(coord.type() == CV_32F);
    cv::Mat t0 = coord + o0_cw;
    cv::Mat t1 = coord + o1_cw;
    double t0_square = t0.dot(t0);
    double t1_square = t1.dot(t1);
    double cos2 = t0.dot(t1);
    cos2 = (cos2 * cos2) / (t0_square * t1_square);
    double cur_var =
            t1_square / (t0_square * t0_square * (1.0 - cos2)) * basic_var_;
    double cur_d = 1.0 / std::sqrt(t0_square);

    if (cnt_ >= 3 && std::abs(cur_d - mean_) > 2 * sqrt_var_) { return false; }

    double update_mean = (var_ * cur_d + cur_var * mean_) / (cur_var + var_);
    double update_var = (var_ * cur_var) / (cur_var + var_);
    mean_ = update_mean;
    var_ = update_var;
    sqrt_var_ = std::sqrt(var_);

    dir_ = (dir_ * float(cnt_) + Rcw0 * t0 / cv::norm(t0)) / float(cnt_ + 1);
    dir_ /= cv::norm(dir_);
    cnt_ += 1;
    return true;
}
}