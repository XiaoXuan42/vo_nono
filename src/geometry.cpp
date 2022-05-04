#include "vo_nono/util/geometry.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

namespace vo_nono {
std::array<double, 3> Geometry::rotation_mat_to_angle_axis(const cv::Mat &R) {
    assert(R.type() == CV_32F);
    assert(R.rows == 3);
    assert(R.cols == 3);
    cv::Mat R64;
    R.convertTo(R64, CV_64F);
    Eigen::Matrix3d R1;
    cv::cv2eigen(R64, R1);
    Eigen::AngleAxisd angle_axis(R1);
    Eigen::Vector3d angle_axis_vec = angle_axis.angle() * angle_axis.axis();

    return std::array<double, 3>{angle_axis_vec(0), angle_axis_vec(1),
                                 angle_axis_vec(2)};
}
}// namespace vo_nono