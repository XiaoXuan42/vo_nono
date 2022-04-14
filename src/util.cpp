#include "vo_nono/util.h"

namespace vo_nono {
uint64_t rand64() {
    static thread_local std::random_device rd_dv;
    static thread_local std::mt19937 rd{rd_dv()};
    return rd();
}

cv::Mat quaternion_to_rotation_mat(const float Q[]) {
    // R = vv^T + s^2I + 2sv^ + (v^)^2
    const float s = Q[3];
    cv::Mat v(3, 1, CV_32F);
    v.at<float>(0, 0) = Q[0];
    v.at<float>(0, 1) = Q[1];
    v.at<float>(0, 2) = Q[2];
    cv::Mat v_vert = cv::Mat::zeros(3, 3, CV_32F);
    v_vert.at<float>(0, 1) = -Q[2];
    v_vert.at<float>(0, 2) = Q[1];
    v_vert.at<float>(1, 0) = Q[2];
    v_vert.at<float>(1, 2) = -Q[0];
    v_vert.at<float>(2, 0) = -Q[1];
    v_vert.at<float>(2, 1) = Q[0];

    cv::Mat res = v * v.t() + s * s * cv::Mat::eye(3, 3, CV_32F) +
                  2 * s * v_vert + v_vert * v_vert;
    return res;
}

void angle_axis_to_rotation_mat(const double angle_axis[3],
                                double result[3][3]) {
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat R;
    for (int i = 0; i < 3; ++i) { rvec.at<double>(i) = angle_axis[i]; }
    cv::Rodrigues(rvec, R);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { result[i][j] = R.at<double>(i, j); }
    }
}
}// namespace vo_nono
