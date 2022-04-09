#ifndef VO_NONO_FRONTEND_H
#define VO_NONO_FRONTEND_H

#include <list>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <unordered_map>
#include <utility>

#include "vo_nono/camera.h"
#include "vo_nono/frame.h"
#include "vo_nono/keypoint/match.h"
#include "vo_nono/map.h"
#include "vo_nono/motion.h"
#include "vo_nono/types.h"

namespace vo_nono {
struct FrontendConfig {};

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
                                   cv::Mat &dscpts, int nfeatures);

    [[nodiscard]] State get_state() const { return m_state; }

    explicit Frontend(const FrontendConfig &config, const Camera &camera,
                      vo_ptr<Map> p_map = vo_ptr<Map>())
        : m_config(config),
          m_camera(camera),
          m_state(State::Start),
          m_map(std::move(p_map)) {
        log_debug_line("Frontend camera intrinsic matrix:\n"
                       << m_camera.get_intrinsic_mat());
    }

private:
    int match_with_keyframe(int match_cnt);
    int initialize(const cv::Mat &image, double t);
    bool tracking(const cv::Mat &image, double t);
    int track_with_match();
    int track_with_local_points();

private:
    static constexpr int CNT_INIT_MATCHES = 500;
    static constexpr int CNT_KEY_PTS = 1000;
    static constexpr int CNT_MATCHES = 200;
    static constexpr int CNT_MIN_MATCHES = 20;

    FrontendConfig m_config;
    const Camera &m_camera;
    State m_state;

    FrameInfo m_keyframe_info;
    FrameInfo m_curframe_info;
    vo_uptr<ORBMatcher> m_matcher;
    std::vector<cv::DMatch> m_matches;
    std::vector<bool> m_matches_inlier;

    vo_ptr<Map> m_map;

    MotionPredictor m_motion_pred;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
