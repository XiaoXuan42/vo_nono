#ifndef VO_NONO_EPIPOLAR_H
#define VO_NONO_EPIPOLAR_H

#include <opencv2/core.hpp>

namespace vo_nono {
class Epipolar {
public:
    static cv::Mat compute_essential(const cv::Mat &R21, const cv::Mat &T21) {
        assert(R21.type() == CV_32F);
        assert(T21.type() == CV_32F);
        cv::Mat t_hat = cv::Mat::zeros(3, 3, CV_32F);
        float t1 = T21.at<float>(0), t2 = T21.at<float>(1),
              t3 = T21.at<float>(2);
        t_hat.at<float>(0, 1) = -t3;
        t_hat.at<float>(0, 2) = t2;
        t_hat.at<float>(1, 0) = t3;
        t_hat.at<float>(1, 2) = -t1;
        t_hat.at<float>(2, 0) = -t2;
        t_hat.at<float>(2, 1) = t1;
        return t_hat * R21;
    }

    static double epipolar_line_dis(const cv::Mat &camera_intrinsic,
                                    const cv::Mat &ess, const cv::Point2f &pt1,
                                    const cv::Point2f &pt2) {
        cv::Mat inv_cam_intrinsic = camera_intrinsic.inv();
        cv::Mat fundamental = inv_cam_intrinsic.t() * ess * inv_cam_intrinsic;
        cv::Mat mat1 = cv::Mat::zeros(3, 1, CV_32F),
                mat2 = cv::Mat::zeros(3, 1, CV_32F);
        mat1.at<float>(0, 0) = pt1.x;
        mat1.at<float>(1, 0) = pt1.y;
        mat1.at<float>(2, 0) = 1.0f;
        mat2.at<float>(0, 0) = pt2.x;
        mat2.at<float>(1, 0) = pt2.y;
        mat2.at<float>(2, 0) = 1.0f;
        cv::Mat l = fundamental * mat1;
        cv::Mat mat_res = mat2.t() * l;
        double diff = std::abs(double(mat_res.at<float>(0)));
        double l_denominator =
                std::sqrt(l.dot(l) - l.at<float>(2) * l.at<float>(2));
        return diff / l_denominator;
    }

    static double epipolar_line_dis(const cv::Mat &camera_intrinsic,
                                    const cv::Mat &R21, const cv::Mat &T21,
                                    const cv::Point2f &pt1,
                                    const cv::Point2f &pt2) {
        cv::Mat ess = compute_essential(R21, T21);
        return epipolar_line_dis(camera_intrinsic, ess, pt1, pt2);
    }
};
}// namespace vo_nono

#endif//VO_NONO_EPIPOLAR_H
