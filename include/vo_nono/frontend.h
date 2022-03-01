#ifndef VO_NONO_FRONTEND_H
#define VO_NONO_FRONTEND_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <utility>

#include "vo_nono/camera.h"
#include "vo_nono/frame.h"
#include "vo_nono/map.h"
#include "vo_nono/types.h"

namespace vo_nono {
struct FrontendConfig {
    Camera camera;
};

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

    explicit Frontend(const FrontendConfig &config)
        : m_state(State::Start),
          m_camera(config.camera) {}

private:
    void initialize(const cv::Mat &image, vo_time_t time);
    void tracking(const cv::Mat &image, vo_time_t time);
    cv::Mat get_proj_mat(const cv::Mat &Rcw, const cv::Mat &t);
    void _finish_tracking(const cv::Mat &new_tri_res,
                          const std::vector<cv::DMatch> &matches);
    void insert_map_points(std::vector<vo_uptr<MapPoint>> &points) {
        if (m_map) { m_map->insert_map_points(points); }
    }

private:
    FrontendConfig m_config;
    Camera m_camera;
    State m_state;

    vo_ptr<Frame> m_keyframe;
    vo_ptr<Frame> m_cur_frame;
    vo_ptr<Map> m_map;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
