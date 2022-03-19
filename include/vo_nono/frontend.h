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
                                   double ang_cos_th = 0.99999999);

    void initialize(const cv::Mat &image, double t);
    bool tracking(const cv::Mat &image, double t);
    void reproj_with_motion(ReprojRes &proj_res);
    void reproj_with_local_points(ReprojRes &proj_res);
    void reproj_pose_estimate(ReprojRes &proj_res, float reproj_th);
    int track_with_match(const vo_ptr<Frame> &o_frame);
    int triangulate(const vo_ptr<Frame> &ref_frame, ReprojRes &proj_res);
    int set_new_map_points(const vo_ptr<Frame> &ref_frame,
                           const cv::Mat &new_tri_res,
                           const std::vector<cv::DMatch> &matches,
                           const std::vector<bool> &inliers);
    void insert_map_points(std::vector<vo_ptr<MapPoint>> &points) {
        if (m_map) { m_map->insert_map_points(points); }
    }
    void select_new_keyframe(const vo_ptr<Frame> &new_keyframe);
    void set_local_map_point(const vo_ptr<MapPoint> &map_pt) {
        vo_id_t id = map_pt->get_id();
        auto iter = m_local_points.find(id);
        if (iter == m_local_points.end()) {
            m_local_points[id] = std::make_pair(1, map_pt);
        } else {
            iter->second.first += 1;
        }
    }
    void unset_local_map_point(vo_id_t map_pt_id) {
        auto iter = m_local_points.find(map_pt_id);
        assert(iter != m_local_points.end());
        if (iter->second.first == 1) {
            m_local_points.erase(iter);
        } else {
            iter->second.first -= 1;
        }
    }

private:
    static constexpr int CNT_MAX_WINDOW_FRAMES = 5;
    static constexpr int CNT_INIT_KEY_PTS = 1000;
    static constexpr int CNT_TRACK_KEY_PTS = 1000;

    FrontendConfig m_config;
    Camera m_camera;
    State m_state;

    vo_ptr<Frame> m_keyframe;
    vo_ptr<Frame> m_cur_frame;
    vo_ptr<Frame> m_last_frame;
    std::list<vo_ptr<Frame>> m_window_frame;
    std::unordered_map<vo_id_t, std::pair<int, vo_ptr<MapPoint>>>
            m_local_points;

    vo_ptr<Map> m_map;

    MotionPredictor m_motion_pred;
};
}// namespace vo_nono

#endif//VO_NONO_FRONTEND_H
