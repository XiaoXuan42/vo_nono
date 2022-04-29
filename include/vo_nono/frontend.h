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

    void detect_and_compute(const cv::Mat &image,
                            std::vector<cv::KeyPoint> &kpts, cv::Mat &dscpts,
                            int nfeatures);

    [[nodiscard]] State get_state() const { return state_; }

    explicit Frontend(const FrontendConfig &config, const Camera &camera,
                      vo_ptr<Map> p_map = vo_ptr<Map>())
        : config_(config),
          camera_(camera),
          state_(State::Start),
          map_(std::move(p_map)) {
        log_debug_line("Frontend camera intrinsic matrix:\n"
                       << camera_.get_intrinsic_mat());
    }

private:
    void reset_state() {
        b_new_keyframe_ = false;
        b_match_good_ = false;
        b_track_good_ = false;
        keyframe_matches_.clear();
        cnt_inlier_direct_match_ = 0;
        cnt_inlier_proj_match_ = 0;
    }

    int initialize(const cv::Mat &image);
    bool tracking(const cv::Mat &image, double t);
    int track_by_match(const vo_ptr<Frame> &ref_frame,
                       const std::vector<cv::DMatch> &matches, float ransac_th);
    int track_by_projection(const std::vector<vo_ptr<MapPoint>> &points,
                            float r_th, float ransac_th);
    void need_new_keyframe();

private:
    static constexpr int CNT_INIT_MATCHES = 500;
    static constexpr int CNT_KEY_PTS = 1000;
    static constexpr int CNT_MATCHES = 500;
    static constexpr int CNT_MATCH_MIN_MATCHES = 30;
    static constexpr int CNT_TRACKING_MIN_MATCHES = 30;

    FrontendConfig config_;
    const Camera &camera_;
    State state_;

    vo_ptr<Frame> keyframe_;
    vo_ptr<Frame> curframe_;
    vo_uptr<ORBMatcher> matcher_;

    std::vector<cv::DMatch> keyframe_matches_;
    std::vector<cv::DMatch> init_matches_;
    std::vector<cv::DMatch> direct_matches_;
    std::unordered_map<vo_ptr<MapPoint>, int> points_seen_;

    vo_ptr<Map> map_;

    MotionPredictor motion_pred_;
    bool b_new_keyframe_ = false;

    bool b_track_good_ = false;
    bool b_match_good_ = false;
    int cnt_inlier_direct_match_ = 0;
    int cnt_inlier_proj_match_ = 0;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
