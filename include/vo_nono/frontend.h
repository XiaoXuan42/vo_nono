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
          m_map(std::move(p_map)),
          mb_new_key_frame(false) {
        log_debug_line("Frontend camera intrinsic matrix:\n"
                       << m_camera.get_intrinsic_mat());
    }

private:
    int match_with_keyframe(int match_cnt);
    int initialize(const cv::Mat &image);
    bool tracking(const cv::Mat &image, double t);
    int track_by_match_with_keyframe();
    int track_by_local_points();
    int triangulate_with_keyframe();

private:
    static constexpr int CNT_INIT_MATCHES = 500;
    static constexpr int CNT_KEY_PTS = 1000;
    static constexpr int CNT_MATCHES = 200;
    static constexpr int CNT_MIN_MATCHES = 20;

    void show_keyframe_curframe_match(const std::vector<cv::DMatch> &matches,
                                      const std::string &prefix) const;

    FrontendConfig m_config;
    const Camera &m_camera;
    State m_state;

    vo_ptr<Frame> m_keyframe;
    vo_ptr<Frame> m_curframe;
    vo_uptr<ORBMatcher> m_matcher;
    std::vector<cv::DMatch> m_matches;
    std::vector<bool> m_matches_inlier;

    vo_ptr<Map> m_map;

    MotionPredictor m_motion_pred;
    bool mb_new_key_frame;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
