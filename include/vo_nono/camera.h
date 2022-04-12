#ifndef VO_NONO_CAMERA_H
#define VO_NONO_CAMERA_H

#include <opencv2/core.hpp>
#include <utility>

namespace vo_nono {
class Camera {
public:
    Camera()
        : m_mat(cv::Mat::eye(3, 3, CV_32F)),
          m_width(640.0),
          m_height(480.0) {}
    Camera(float width, float height)
        : m_mat(cv::Mat::eye(3, 3, CV_32F)),
          m_width(width),
          m_height(height) {}
    explicit Camera(cv::Mat o_mat, float width, float height)
        : m_mat(std::move(o_mat)),
          m_width(width),
          m_height(height) {}

    void set_dist_coeff(std::vector<float> new_dist_coeff) {
        m_dist_coeff = std::move(new_dist_coeff);
    }

    [[nodiscard]] cv::Mat get_intrinsic_mat() const { return m_mat.clone(); }
    [[nodiscard]] const std::vector<float>& get_dist_coeff() const {
        return m_dist_coeff;
    }
    [[nodiscard]] float get_width() const { return m_width; }
    [[nodiscard]] float get_height() const { return m_height; }
    [[nodiscard]] float fx() const {
        assert(m_mat.type() == CV_32F);
        return m_mat.at<float>(0, 0);
    }
    [[nodiscard]] float fy() const {
        assert(m_mat.type() == CV_32F);
        return m_mat.at<float>(1, 1);
    }
    [[nodiscard]] float cx() const {
        assert(m_mat.type() == CV_32F);
        return m_mat.at<float>(0, 2);
    }
    [[nodiscard]] float cy() const {
        assert(m_mat.type() == CV_32F);
        return m_mat.at<float>(1, 2);
    }

private:
    cv::Mat m_mat;
    std::vector<float> m_dist_coeff;
    float m_width;
    float m_height;
};
}// namespace vo_nono

#endif//VO_NONO_CAMERA_H
