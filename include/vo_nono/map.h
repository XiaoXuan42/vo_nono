#ifndef VO_NONO_MAP_H
#define VO_NONO_MAP_H

#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <thread>
#include <unordered_set>
#include <vector>

#include "vo_nono/frame.h"
#include "vo_nono/keypoint/bow.h"
#include "vo_nono/keypoint/triangulate.h"
#include "vo_nono/point.h"
#include "vo_nono/util/geometry.h"
#include "vo_nono/util/queue.h"

namespace vo_nono {
struct FrameMessage {
    vo_ptr<Frame> frame;
    std::vector<cv::DMatch> match_with_keyframe;
    bool is_keyframe;

    FrameMessage(vo_ptr<Frame> pframe, std::vector<cv::DMatch> matches,
                 bool b_keyframe)
        : frame(std::move(pframe)),
          match_with_keyframe(std::move(matches)),
          is_keyframe(b_keyframe) {}
};

class Map;
class LocalMap;

class InvDepthFilter {
public:
    InvDepthFilter() : dir_(cv::Mat::zeros(3, 1, CV_32F)) {}
    bool filter(const cv::Mat &o0_cw, const cv::Mat &Rcw0, const cv::Mat &o1_cw,
                const cv::Mat &coord);
    [[nodiscard]] double get_variance() const { return var_; }
    [[nodiscard]] cv::Mat get_coord(const cv::Mat &o_cw,
                                    const cv::Mat &Rcw) const {
        return -o_cw + Rcw.t() * dir_ / mean_;
    }
    [[nodiscard]] int get_cnt() const { return cnt_; }
    void set_information(const Camera &camera, const cv::Point2f pt,
                         double basic_var) {
        dir_ = cv::Mat::zeros(3, 1, CV_32F);
        dir_.at<float>(0) = (pt.x - camera.cx()) / camera.fx();
        dir_.at<float>(1) = (pt.y - camera.cy()) / camera.fy();
        dir_.at<float>(2) = 1.0f;
        dir_ /= cv::norm(dir_);
        basic_var_ = basic_var;
    }
    [[nodiscard]] bool relative_error_less(double th) const {
        if (std::isinf(th)) {
            return true;
        } else if (mean_ - 2 * sqrt_var_ <= 0) {
            return false;
        }
        double max_d = 1.0 / (mean_ - 2 * sqrt_var_);
        double mean_d = 1.0 / mean_;
        return (max_d - mean_d) <= th * mean_d;
    }

private:
    int cnt_ = 0;
    double mean_ = 1.0;
    double var_ = 1.0;
    double sqrt_var_ = 1.0;
    double basic_var_ = 1;
    cv::Mat dir_;
};

class LocalMap {
public:
    LocalMap() = delete;
    void initialize(const vo_ptr<Frame> &keyframe, const vo_ptr<Frame> &frame,
                    const std::vector<cv::DMatch> &matches);
    void insert_frame(const FrameMessage &message);
    void set_keyframe(const vo_ptr<Frame> &keyframe);

private:
    explicit LocalMap(Map *map);
    void triangulate_with_keyframe(const std::vector<cv::DMatch> &matches,
                                   double th);

    Map *map_;
    vo_ptr<Frame> keyframe_;
    vo_ptr<Frame> curframe_;
    const Camera camera_;
    std::vector<InvDepthFilter> filters_;
    std::vector<bool> own_point_;

    friend class Map;
};

class Map {
public:
    explicit Map(Camera camera, const char *voc_path)
        : camera_(std::move(camera)),
          bow_database_(voc_path),
          local_map_(new LocalMap(this)),
          b_shutdown_(false),
          b_global_ba_(false) {
        t_global_ba_ = std::thread(&Map::global_bundle_adjustment, this);
    }
    explicit Map(const Map &) = delete;
    ~Map() { shutdown(); }

    using Trajectory = std::vector<std::pair<double, cv::Mat>>;

    void global_bundle_adjustment();

    [[nodiscard]] Trajectory get_trajectory() {
        Trajectory trajectory;
        trajectory.reserve(frames_.size());
        for (const vo_ptr<Frame> &frame : frames_) {
            trajectory.emplace_back(
                    std::make_pair(frame->get_time(), frame->get_pose()));
        }
        return trajectory;
    }

    std::vector<vo_ptr<MapPoint>> get_local_map_points() {
        std::unordered_set<vo_id_t> id_book;
        std::vector<vo_ptr<MapPoint>> result;
        int cnt = 0;
        for (auto iter = keyframes_.rbegin(); iter != keyframes_.rend();
             ++iter) {
            std::vector<vo_ptr<MapPoint>> frame_pts =
                    (*iter)->get_all_map_pts();
            for (auto &map_pt : frame_pts) {
                if (!id_book.count(map_pt->get_id())) {
                    id_book.insert(map_pt->get_id());
                    result.push_back(map_pt);
                }
            }
            cnt += 1;
            if (cnt >= 5) { break; }
        }
        return result;
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(map_global_mutex);
            b_global_ba_ = true;
            b_shutdown_ = true;
            cv_global_ba_.notify_all();
        }
        if (t_global_ba_.joinable()) { t_global_ba_.join(); }
    }

    void initialize(const vo_ptr<Frame> &keyframe, const vo_ptr<Frame> &frame,
                    const std::vector<cv::DMatch> &matches) {
        frames_.push_back(keyframe);
        frames_.push_back(frame);
        insert_key_frame(keyframe);
        local_map_->initialize(keyframe, frame, matches);
    }

    void insert_frame(const FrameMessage &message) {
        frames_.push_back(message.frame);
        local_map_->insert_frame(message);
        if (message.is_keyframe) {
            local_map_->set_keyframe(message.frame);
            insert_key_frame(message.frame);
        }
    }

    std::mutex map_global_mutex;

private:
    void insert_key_frame(const vo_ptr<Frame> &frame) {
        log_debug_line("Switch keyframe: " << frame->get_id());
        keyframes_.push_back(frame);
        b_global_ba_ = true;
        cv_global_ba_.notify_all();
    }

    void _global_bundle_adjustment(std::unique_lock<std::mutex> &lock);

    const Camera camera_;
    BowDataBase bow_database_;
    vo_uptr<LocalMap> local_map_;
    std::vector<vo_ptr<Frame>> keyframes_;
    std::vector<vo_ptr<Frame>> frames_;


    std::condition_variable cv_global_ba_;

    bool b_shutdown_;
    bool b_global_ba_;
    std::thread t_global_ba_;

    friend class LocalMap;
};
}// namespace vo_nono

#endif//VO_NONO_MAP_H
