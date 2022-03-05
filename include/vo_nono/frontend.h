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

    void get_image(const cv::Mat &image, double t);

    static void detect_and_compute(const cv::Mat &image,
                                   std::vector<cv::KeyPoint> &kpts,
                                   cv::Mat &dscpts);

    static std::vector<cv::DMatch> match_descriptor(const cv::Mat &dscpt1,
                                                    const cv::Mat &dscpt2);

    static std::vector<cv::DMatch> filter_matches(
            const std::vector<cv::DMatch> &matches,
            const std::vector<cv::KeyPoint> &kpt1,
            const std::vector<cv::KeyPoint> &kpt2);

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
    static std::vector<cv::DMatch> _filter_matches_by_mask(
            const std::vector<cv::DMatch> &matches,
            const std::vector<unsigned char> &mask) {
        std::vector<cv::DMatch> res;
        res.reserve(matches.size());
        for (int i = 0; i < (int) matches.size(); ++i) {
            if (mask[i]) { res.push_back(matches[i]); }
        }
        return res;
    }

    void filter_triangulate_points(const cv::Mat &tri, const cv::Mat &Rcw1,
                                   const cv::Mat &tcw1, const cv::Mat &Rcw2,
                                   const cv::Mat &tcw2,
                                   const std::vector<cv::Point2f> &kpts1,
                                   const std::vector<cv::Point2f> &kpts2,
                                   std::vector<bool> &inliers,
                                   float thresh_square = 1.0);

    void initialize(const cv::Mat &image, double time);
    void tracking(const cv::Mat &image, double time);
    cv::Mat get_proj_mat(const cv::Mat &Rcw, const cv::Mat &t);
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
    vo_ptr<Map> m_map;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
