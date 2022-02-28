#ifndef VO_NONO_CAMERA_H
#define VO_NONO_CAMERA_H

#include <opencv2/core.hpp>
#include <utility>

namespace vo_nono {
    class Camera {
    public:
        Camera(): m_mat(cv::Mat::eye(3, 3, CV_32F)) {}
        explicit Camera(cv::Mat o_mat): m_mat(std::move(o_mat)) {}

        [[nodiscard]] const cv::Mat &get_intrinsic_mat() const {
            return m_mat;
        }
    private:
        cv::Mat m_mat;
    };
}

#endif //VO_NONO_CAMERA_H
