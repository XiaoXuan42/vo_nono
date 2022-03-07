#ifndef VO_NONO_FRONTEND_H
#define VO_NONO_FRONTEND_H

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <utility>

#include "vo_nono/camera.h"
#include "vo_nono/feature.h"
#include "vo_nono/frame.h"
#include "vo_nono/map.h"
#include "vo_nono/motion.h"
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

    void get_image(const cv::Mat &image, double t);

    static void detect_and_compute(const cv::Mat &image,
                                   std::vector<cv::KeyPoint> &kpts,
                                   cv::Mat &dscpts);

    static std::vector<cv::DMatch> match_descriptor(const cv::Mat &dscpt1,
                                                    const cv::Mat &dscpt2);

    static std::vector<cv::DMatch> filter_matches(
            const std::vector<cv::DMatch> &matches,
            const std::vector<cv::KeyPoint> &kpt1,
            const std::vector<cv::KeyPoint> &kpt2, double ransac_th = 1.0);

    [[nodiscard]] State get_state() const { return m_state; }

    explicit Frontend(const FrontendConfig &config,
                      vo_ptr<Map> p_map = vo_ptr<Map>())
        : m_config(config),
          m_camera(config.camera),
          m_state(State::Start),
          m_map(std::move(p_map)) {
        log_debug_line("Frontend camera intrinsic matrix:\n"
                       << m_camera.get_intrinsic_mat());
    }

private:
    void filter_triangulate_points(const cv::Mat &tri, const cv::Mat &Rcw1,
                                   const cv::Mat &tcw1, const cv::Mat &Rcw2,
                                   const cv::Mat &tcw2,
                                   const std::vector<cv::Point2f> &kpts1,
                                   const std::vector<cv::Point2f> &kpts2,
                                   std::vector<bool> &inliers,
                                   float thresh_square = 1.0);

    void initialize(const cv::Mat &image, double t);
    void tracking(const cv::Mat &image, double t);
    int track_with_motion(const size_t cnt_pt_th);
    bool track_with_keyframe(bool b_estimate_valid);
    void _finish_tracking(const cv::Mat &new_tri_res,
                          const std::vector<cv::DMatch> &matches,
                          const std::vector<bool> &inliers);
    void _try_switch_keyframe(size_t new_pt, size_t old_pt);
    void insert_map_points(std::vector<vo_uptr<MapPoint>> &points) {
        if (m_map) { m_map->insert_map_points(points); }
    }


private:
    FrontendConfig m_config;
    Camera m_camera;
    State m_state;

    vo_ptr<Frame> m_keyframe;
    vo_ptr<Frame> m_cur_frame;
    vo_ptr<Frame> m_last_frame;
    vo_ptr<Map> m_map;

    MotionPredictor m_motion_pred;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
