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
#include "vo_nono/util/filter.h"

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
        b_match_good_ = false;
        b_track_good_ = false;
        keyframe_matches_.clear();
        cnt_inlier_direct_match_ = 0;
        cnt_inlier_proj_match_ = 0;
    }

    int initialize(const cv::Mat &image);
    bool tracking(const cv::Mat &image, double t);
    void tracking_with_keyframe();
    void relocalization();
    int track_by_match(const vo_ptr<Frame> &ref_frame,
                       const std::vector<cv::DMatch> &matches, float ransac_th);
    int track_by_projection(const std::vector<vo_ptr<MapPoint>> &points,
                            float r_th, float ransac_th);
    void new_keyframe();
    void triangulate_and_set(const std::vector<cv::DMatch> &matches);
    std::vector<cv::DMatch> filter_match(const std::vector<cv::DMatch> &matches,
                                         double epi_th);
    void _set_keyframe(const vo_ptr<Frame> &keyframe);
    void _update_points_location(const std::vector<cv::Mat> &tri_res,
                                 const std::vector<cv::DMatch> &matches);
    void _associate_points(const std::vector<cv::DMatch> &matches,
                           double rel_th);
    void insert_local_frame(const vo_ptr<Frame> &frame) {
        local_map_.local_frames.push_back(frame);
        map_->insert_frame(frame);
    }
    cv::Mat _get_local_map_point_coord(int index);
    void _add_observation(const cv::Mat &Rcw, const cv::Mat &tcw,
                             const cv::Point2f &pixel, int index);

private:
    static constexpr int CNT_KEYPTS = 1000;
    static constexpr int CNT_MATCHES = 1000;
    static constexpr int CNT_TRACK_MIN_MATCHES = 30;

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

    bool b_track_good_ = false;
    bool b_match_good_ = false;
    int cnt_inlier_direct_match_ = 0;
    int cnt_inlier_proj_match_ = 0;

private:
    struct LocalMap {
        std::vector<InvDepthFilter> filters;
        std::vector<cv::Mat> tri_mats;
        std::vector<bool> own_points;
        std::list<vo_ptr<Frame>> local_frames;
    };

    LocalMap local_map_;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
