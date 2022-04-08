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
                                   cv::Mat &dscpts, int nfeatures);

    static std::vector<cv::DMatch> filter_matches(
            const std::vector<cv::DMatch> &matches,
            const std::vector<cv::KeyPoint> &kpt1,
            const std::vector<cv::KeyPoint> &kpt2, int topK = 3);

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
    static int filter_triangulate_points(cv::Mat &tri, const cv::Mat &Rcw1,
                                         const cv::Mat &tcw1,
                                         const cv::Mat &Rcw2,
                                         const cv::Mat &tcw2,
                                         const std::vector<cv::Point2f> &pts1,
                                         const std::vector<cv::Point2f> &pts2,
                                         std::vector<bool> &inliers,
                                         double grad_th);
    static void filter_match_with_kpts(const std::vector<cv::KeyPoint> &kpts1,
                                       const std::vector<cv::KeyPoint> &kpts2,
                                       std::vector<unsigned char> &mask,
                                       int topK);

    int match_between_frames(const vo_ptr<Frame> &ref_frame,
                             std::vector<cv::DMatch> &matches, int match_cnt);
    int initialize(const cv::Mat &image, double t);
    bool tracking(const cv::Mat &image, double t);
    int track_with_match(const std::vector<cv::DMatch> &matches,
                         const vo_ptr<Frame> &ref_frame);
    int track_with_local_points();
    int set_new_map_points(const vo_ptr<Frame> &ref_frame,
                           const cv::Mat &new_tri_res,
                           const std::vector<cv::DMatch> &matches,
                           const std::vector<bool> &inliers);
    void select_new_keyframe(const vo_ptr<Frame> &new_keyframe);

    void _triangulate_with_match(const std::vector<cv::DMatch> &matches,
                                 const vo_ptr<Frame> &ref_frame);

private:
    static constexpr int CNT_INIT_MATCHES = 500;
    static constexpr int CNT_KEY_PTS = 1000;
    static constexpr int CNT_MATCHES = 200;
    static constexpr int CNT_MIN_MATCHES = 20;

    FrontendConfig m_config;
    Camera m_camera;
    State m_state;

    vo_ptr<Frame> m_keyframe;
    vo_ptr<Frame> m_cur_frame;
    vo_ptr<Frame> m_last_frame;
    vo_ptr<ORBMatcher> m_matcher;

    vo_ptr<Map> m_map;

    MotionPredictor m_motion_pred;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
