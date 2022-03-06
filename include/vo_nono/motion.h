#ifndef VO_NONO_MOTION_H
#define VO_NONO_MOTION_H

#include <opencv2/core.hpp>

namespace vo_nono {
class MotionPredictor {
public:
    MotionPredictor() { m_cur = 0; m_inform_cnt = 0; }
    void predict_pose(double time, cv::Mat& Rcw, cv::Mat& tcw) const;
    void inform_pose(const cv::Mat& new_Rcw, const cv::Mat& new_tcw,
                     double time);
    [[nodiscard]] bool is_available() const { return m_inform_cnt >= 2; }

private:
    cv::Mat m_t[2];
    float m_q[2][4]{};
    double m_time[2]{};
    int m_cur;
    int m_inform_cnt;
};
}// namespace vo_nono

#endif//VO_NONO_MOTION_H
