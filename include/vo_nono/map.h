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
class Map {
public:
    explicit Map(Camera camera, const char *voc_path)
        : camera_(std::move(camera)),
          bow_database_(voc_path),
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

    void insert_frame(const vo_ptr<Frame> &frame) {
        frames_.push_back(frame);
    }

    void insert_key_frame(const vo_ptr<Frame> &frame) {
        log_debug_line("Switch keyframe: " << frame->get_id());
        keyframes_.push_back(frame);
        b_global_ba_ = true;
        cv_global_ba_.notify_all();
    }
    std::mutex map_global_mutex;

private:
    void _global_bundle_adjustment(std::unique_lock<std::mutex> &lock);

    const Camera camera_;
    BowDataBase bow_database_;
    std::vector<vo_ptr<Frame>> keyframes_;
    std::vector<vo_ptr<Frame>> frames_;

    std::condition_variable cv_global_ba_;

    bool b_shutdown_;
    bool b_global_ba_;
    std::thread t_global_ba_;
};
}// namespace vo_nono

#endif//VO_NONO_MAP_H
