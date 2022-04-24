#include "vo_nono/motion.h"

#include <cmath>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "vo_nono/util/geometry.h"

namespace vo_nono {
void MotionPredictor::inform_pose(const cv::Mat& new_Rcw,
                                  const cv::Mat& new_tcw, double time) {
    assert(new_tcw.type() == CV_32F);
    assert(new_tcw.cols == 1);
    assert(new_tcw.rows == 3);
    assert(new_Rcw.type() == CV_32F);
    assert(new_Rcw.cols == 3);
    assert(new_Rcw.rows == 3);
    cv::cv2eigen(new_tcw, t_[cur_]);
    Eigen::Matrix3f rcw;
    cv::cv2eigen(new_Rcw, rcw);
    q_[cur_] = Eigen::Quaternionf(rcw);
    q_[cur_].normalize();
    time_[cur_] = time;
    cur_ ^= 1;
    inform_cnt_ += 1;
}

void MotionPredictor::predict_pose(double time, cv::Mat& Rcw,
                                   cv::Mat& tcw) const {
    // linear velocity
    assert(is_available());
    const int other = cur_;
    const int cur = cur_ ^ 1;
    double c1 = time - time_[other];
    double c2 = time_[cur] - time;
    double invDiv = 1.0 / (time_[cur] - time_[other]);
    Eigen::Vector3f predict_tcw =
            ((float) c1 * t_[cur] + (float) c2 * t_[other]) * invDiv;
    cv::eigen2cv(predict_tcw, tcw);
    assert(tcw.type() == CV_32F);
    assert(tcw.cols == 1);
    assert(tcw.rows == 3);

    // slerp
    Eigen::Quaternionf predict_q;
    auto q1 = q_[other], q2 = q_[cur];
    double ang = acos(q_[other].dot(q_[cur]));
    if (ang < 0.001) {
        predict_q.coeffs() = q2.coeffs() * c1 + q1.coeffs() * c2;
    } else {
        double scaledT = c1 / (time_[cur] - time_[other]);
        double c3 = sin((1 - scaledT) * ang), c4 = sin(scaledT * ang);
        predict_q.coeffs() = q2.coeffs() * c4 + q1.coeffs() * c3;
    }
    predict_q.normalize();
    auto predict_rcw = predict_q.toRotationMatrix();
    cv::eigen2cv(predict_rcw, Rcw);
    assert(Rcw.type() == CV_32F);
    assert(Rcw.cols == 3);
    assert(Rcw.rows == 3);
}
}// namespace vo_nono