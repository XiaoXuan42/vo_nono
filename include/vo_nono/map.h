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

class LocalMap {
public:
    LocalMap() = delete;
    void insert_frame(const FrameMessage &message);
    void clear() {
        keyframe_.reset();
        curframe_.reset();
        points_seen_.clear();
    }

private:
    explicit LocalMap(Map *map);
    void triangulate_with_keyframe(const std::vector<cv::DMatch> &matches);

    Map *map_;
    vo_ptr<Frame> keyframe_;
    vo_ptr<Frame> curframe_;
    const Camera camera_;
    std::unordered_map<vo_ptr<MapPoint>, int> points_seen_;

    friend class Map;
};

class Map {
public:
    explicit Map(Camera camera)
        : mr_camera(std::move(camera)),
          local_map_(new LocalMap(this)),
          mb_shutdown(false),
          mb_global_ba(false) {
        mt_global_ba = std::thread(&Map::global_bundle_adjustment, this);
    }
    explicit Map(const Map &) = delete;
    ~Map() { shutdown(); }

    using Trajectory = std::vector<std::pair<double, cv::Mat>>;

    void global_bundle_adjustment();

    [[nodiscard]] Trajectory get_trajectory() {
        Trajectory trajectory;
        trajectory.reserve(m_frames.size());
        for (const vo_ptr<Frame> &frame : m_frames) {
            trajectory.emplace_back(
                    std::make_pair(frame->time, frame->get_pose()));
        }
        return trajectory;
    }

    std::vector<vo_ptr<MapPoint>> get_local_map_points() {
        std::unordered_set<vo_id_t> id_book;
        std::vector<vo_ptr<MapPoint>> result;
        int cnt = 0;
        for (auto iter = m_keyframes.rbegin(); iter != m_keyframes.rend();
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
            mb_global_ba = true;
            mb_shutdown = true;
            m_global_ba_cv.notify_all();
        }
        if (mt_global_ba.joinable()) { mt_global_ba.join(); }
    }

    void initialize(const vo_ptr<Frame> &keyframe,
                    const vo_ptr<Frame> &other_frame) {
        local_map_->clear();
        m_frames.push_back(keyframe);
        insert_key_frame(keyframe);
        if (other_frame) { m_frames.push_back(other_frame); }
    }
    void insert_frame(const FrameMessage &message);

    std::mutex map_global_mutex;

private:
    void insert_key_frame(const vo_ptr<Frame> &frame) {
        log_debug_line("Switch keyframe: " << frame->id);
        m_keyframes.push_back(frame);
        mb_global_ba = true;
        m_global_ba_cv.notify_all();
    }

    void _global_bundle_adjustment(std::unique_lock<std::mutex> &lock);

    const Camera mr_camera;
    vo_uptr<LocalMap> local_map_;
    std::vector<vo_ptr<Frame>> m_keyframes;
    std::vector<vo_ptr<Frame>> m_frames;
    std::unordered_map<vo_id_t, vo_ptr<MapPoint>> m_id_to_map_pt;


    std::condition_variable m_global_ba_cv;

    bool mb_shutdown;
    bool mb_global_ba;
    std::thread mt_global_ba;

    friend class LocalMap;
};
}// namespace vo_nono

#endif//VO_NONO_MAP_H
