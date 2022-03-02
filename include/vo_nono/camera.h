#ifndef VO_NONO_CAMERA_H
#define VO_NONO_CAMERA_H

#include <opencv2/core.hpp>
#include <utility>

namespace vo_nono {
class Camera {
public:
    Camera() : m_mat(cv::Mat::eye(3, 3, CV_32F)) {}
    explicit Camera(cv::Mat o_mat) : m_mat(std::move(o_mat)) {}

    void set_dist_coeff(std::vector<float> new_dist_coeff) {
        m_dist_coeff = std::move(new_dist_coeff);
    }

    [[nodiscard]] const cv::Mat& get_intrinsic_mat() const { return m_mat; }
    [[nodiscard]] const std::vector<float>& get_dist_coeff() const {
        return m_dist_coeff;
    }

private:
    cv::Mat m_mat;
    std::vector<float> m_dist_coeff;
};
}// namespace vo_nono

#endif//VO_NONO_CAMERA_H
