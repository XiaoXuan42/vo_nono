#ifndef VO_NONO_EPIPOLAR_H
#define VO_NONO_EPIPOLAR_H

#include <opencv2/core.hpp>

namespace vo_nono {
class Epipolar
{
public:
    static cv::Mat compute_essential(const cv::Mat &R21, const cv::Mat &T21) {
        assert(R21.type() == CV_32F);
        assert(T21.type() == CV_32F);
        cv::Mat t_hat = cv::Mat::zeros(3, 3, CV_32F);
        float t1 = T21.at<float>(0), t2 = T21.at<float>(1), t3 = T21.at<float>(2);
        t_hat.at<float>(0, 1) = -t3;
        t_hat.at<float>(0, 2) = t2;
        t_hat.at<float>(1, 0) = t3;
        t_hat.at<float>(1, 2) = -t1;
        t_hat.at<float>(2, 0) = -t2;
        t_hat.at<float>(2, 1) = t1;
        return t_hat * R21;
    }
};
}

#endif//VO_NONO_EPIPOLAR_H
