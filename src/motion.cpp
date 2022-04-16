#include "vo_nono/motion.h"

#include <cmath>

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
    m_t[m_cur] = new_tcw.clone();
    Geometry::rotation_mat_to_quaternion<float>(new_Rcw, m_q[m_cur]);
    m_time[m_cur] = time;
    m_cur ^= 1;
    m_inform_cnt += 1;
}

void MotionPredictor::predict_pose(double time, cv::Mat& Rcw,
                                   cv::Mat& tcw) const {
    // linear velocity
    assert(is_available());
    const int other = m_cur;
    const int cur = m_cur ^ 1;
    double c1 = time - m_time[other];
    double c2 = m_time[cur] - time;
    double invDiv = 1.0 / (m_time[cur] - m_time[other]);
    tcw = ((float) c1 * m_t[cur] + (float) c2 * m_t[other]) * invDiv;

    // slerp
    float q[4];
    const float *q1 = m_q[other], *q2 = m_q[cur];
    double ang = acos((double) (q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] +
                                q1[3] * q2[3]));
    if (ang < 0.001) {
        for (int i = 0; i < 4; ++i) {
            q[i] = q2[i] * (float) c1 + q1[i] * (float) c2;
        }
    } else {
        double scaledT = c1 / (m_time[cur] - m_time[other]);
        double c3 = sin((1 - scaledT) * ang), c4 = sin(scaledT * ang);
        for (int i = 0; i < 4; ++i) {
            q[i] = q2[i] * (float) c4 + q1[i] * (float) c3;
        }
    }
    auto q_length = (float) sqrt(
            (double) (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]));
    for (int i = 0; i < 4; ++i) { q[i] /= q_length; }
    Rcw = Geometry::quaternion_to_rotation_mat(q);
}
}// namespace vo_nono