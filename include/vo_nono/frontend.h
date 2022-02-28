#ifndef VO_NONO_FRONTEND_H
#define VO_NONO_FRONTEND_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <utility>

#include "vo_nono/camera.h"
#include "vo_nono/frame.h"
#include "vo_nono/types.h"

namespace vo_nono {
class Frontend {
public:
    enum class State : int {
        Start,
        Initializing,
        Tracking,
    };

    void get_image(const cv::Mat &image, vo_time_t t);

    static void detect_and_compute(const cv::Mat &image,
                                   std::vector<cv::KeyPoint> &kpts,
                                   cv::Mat &dscpts);

    static std::vector<cv::DMatch> match_descriptor(const cv::Mat &dscpt1,
                                                    const cv::Mat &dscpt2);

    static void compute_essential_mat(
            const std::vector<std::pair<cv::Point2f, cv::Point2f>> &pts,
            cv::Mat &W, cv::Mat &U, cv::Mat &Vt);

    static double assess_essential_mat(
            const cv::Mat &U, const cv::Mat &Vt,
            const std::vector<std::pair<cv::Point2f, cv::Point2f>> &pts);

    [[nodiscard]] State get_state() const { return m_state; }

    explicit Frontend(Camera camera)
        : m_state(State::Start),
          m_camera(std::move(camera)) {}

private:
    void initialize(const cv::Mat &image);
    cv::Mat get_proj_mat(const cv::Mat &Rcw, const cv::Mat &t);

private:
    Camera m_camera;
    Frame m_prev_frame;
    Frame m_cur_frame;
    State m_state;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
