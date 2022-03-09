#ifndef VO_NONO_FRONTEND_H
#define VO_NONO_FRONTEND_H

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <utility>

#include "vo_nono/camera.h"
#include "vo_nono/frame.h"
#include "vo_nono/map.h"
#include "vo_nono/motion.h"
#include "vo_nono/types.h"

namespace vo_nono {
struct FrontendConfig {
    Camera camera;
};

class ReprojRes;
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

    static std::vector<cv::DMatch> match_descriptor(const cv::Mat &dscpt1,
                                                    const cv::Mat &dscpt2,
                                                    float soft_dis_th,
                                                    float hard_dis_th,
                                                    int expect_cnt);

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
    void filter_triangulate_points(const cv::Mat &tri, const cv::Mat &Rcw1,
                                   const cv::Mat &tcw1, const cv::Mat &Rcw2,
                                   const cv::Mat &tcw2,
                                   const std::vector<cv::Point2f> &pts1,
                                   const std::vector<cv::Point2f> &pts2,
                                   std::vector<bool> &inliers,
                                   float thresh_square = 1.0);

    void initialize(const cv::Mat &image, double t);
    void tracking(const cv::Mat &image, double t);
    void reproj_with_motion(ReprojRes &proj_res);
    void reproj_with_keyframe(ReprojRes &proj_res);
    void reproj_pose_estimate(ReprojRes &proj_res, float reproj_th);
    void triangulate_with_keyframe(const ReprojRes &proj_res);
    void set_new_map_points(const cv::Mat &new_tri_res,
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
