#ifndef VO_NONO_MOTION_H
#define VO_NONO_MOTION_H

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace vo_nono {
class MotionPredictor {
public:
    MotionPredictor() {
        cur_ = 0;
        inform_cnt_ = 0;
    }
    void predict_pose(double time, cv::Mat& Rcw, cv::Mat& tcw) const;
    void inform_pose(const cv::Mat& new_Rcw, const cv::Mat& new_tcw,
                     double time);
    [[nodiscard]] bool is_available() const { return inform_cnt_ >= 2; }

private:
    Eigen::Vector3f t_[2];
    Eigen::Quaterniond q_[2];
    double time_[2]{};
    int cur_;
    int inform_cnt_;
};
}// namespace vo_nono

#endif//VO_NONO_MOTION_H
