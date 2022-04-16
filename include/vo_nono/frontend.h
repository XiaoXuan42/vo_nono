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
    void reset_state() {
        mb_new_key_frame = false;
        mb_match_good = false;
        mb_track_good = false;
        m_keyframe_matches.clear();
    }

    std::vector<cv::DMatch> match_frame(const vo_ptr<Frame> &ref_frame,
                                        int match_cnt);
    int initialize(const cv::Mat &image);
    bool tracking(const cv::Mat &image, double t);
    int track_by_match(const vo_ptr<Frame> &ref_frame,
                       const std::vector<cv::DMatch> &matches, float ransac_th);
    int track_by_projection(const std::vector<vo_ptr<MapPoint>> &points,
                            float r_th, float ransac_th);
    int track_by_projection_frame(const vo_ptr<Frame> &ref_frame);
    int track_by_projection_local_map();
    int triangulate_with_keyframe();

private:
    static constexpr int CNT_INIT_MATCHES = 500;
    static constexpr int CNT_KEY_PTS = 1000;
    static constexpr int CNT_MATCHES = 200;
    static constexpr int CNT_MIN_MATCHES = 30;

    void show_cur_frame_match(const vo_ptr<Frame> &ref_frame,
                                      const std::vector<cv::DMatch> &matches,
                                      const std::string &prefix) const;

    FrontendConfig m_config;
    const Camera &m_camera;
    State m_state;

    vo_ptr<Frame> m_keyframe;
    vo_ptr<Frame> m_cur_frame;
    vo_ptr<Frame> m_prev_frame;
    vo_uptr<ORBMatcher> m_matcher;

    std::vector<cv::DMatch> m_keyframe_matches;
    std::unordered_map<vo_ptr<MapPoint>, int> m_points_seen;

    vo_ptr<Map> m_map;

    MotionPredictor m_motion_pred;
    bool mb_new_key_frame;

    bool mb_track_good;
    bool mb_match_good;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
