#ifndef VO_NONO_CAMERA_H
#define VO_NONO_CAMERA_H

#include <opencv2/core.hpp>
#include <utility>

namespace vo_nono {
class Camera {
public:
    Camera()
        : mat_(cv::Mat::eye(3, 3, CV_32F)),
          width_(640.0),
          height_(480.0) {}
    Camera(float width, float height)
        : mat_(cv::Mat::eye(3, 3, CV_32F)),
          width_(width),
          height_(height) {}
    explicit Camera(cv::Mat o_mat, float width, float height)
        : mat_(std::move(o_mat)),
          width_(width),
          height_(height) {}

    void set_dist_coeff(std::vector<float> new_dist_coeff) {
        dist_coeff_ = std::move(new_dist_coeff);
    }

    [[nodiscard]] cv::Mat get_intrinsic_mat() const { return mat_.clone(); }
    [[nodiscard]] const std::vector<float>& get_dist_coeff() const {
        return dist_coeff_;
    }
    [[nodiscard]] float get_width() const { return width_; }
    [[nodiscard]] float get_height() const { return height_; }
    [[nodiscard]] float fx() const {
        assert(mat_.type() == CV_32F);
        return mat_.at<float>(0, 0);
    }
    [[nodiscard]] float fy() const {
        assert(mat_.type() == CV_32F);
        return mat_.at<float>(1, 1);
    }
    [[nodiscard]] float cx() const {
        assert(mat_.type() == CV_32F);
        return mat_.at<float>(0, 2);
    }
    [[nodiscard]] float cy() const {
        assert(mat_.type() == CV_32F);
        return mat_.at<float>(1, 2);
    }

private:
    cv::Mat mat_;
    std::vector<float> dist_coeff_;
    float width_;
    float height_;
};
}// namespace vo_nono

#endif//VO_NONO_CAMERA_H
