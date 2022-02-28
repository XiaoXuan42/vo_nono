#ifndef VO_NONO_FRAME_H
#define VO_NONO_FRAME_H

#include <memory>
#include <opencv2/core/core.hpp>
#include <utility>
#include <vector>

namespace vo_nono {
struct Frame {
    cv::Mat m_image;
    cv::Mat descriptor;
    std::vector<cv::KeyPoint> kpts;

    Frame() = default;
    explicit Frame(cv::Mat image) : m_image(std::move(image)) {}
};
}// namespace vo_nono

#endif//VO_NONO_FRAME_H
