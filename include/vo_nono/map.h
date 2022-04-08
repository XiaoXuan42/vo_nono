#ifndef VO_NONO_MAP_H
#define VO_NONO_MAP_H

#include <memory>
#include <opencv2/core.hpp>
#include <unordered_set>
#include <vector>

#include "vo_nono/point.h"

namespace vo_nono {
class Map {
public:
    using Trajectory = std::vector<std::pair<double, cv::Mat>>;
    void insert_key_frame(const vo_ptr<Frame> &frame) {
        m_keyframes.push_back(frame);
    }
    void insert_frame(const vo_ptr<Frame> &frame) { m_frames.push_back(frame); }

    [[nodiscard]] Trajectory get_trajectory() const {
        Trajectory trajectory;
        trajectory.reserve(m_frames.size());
        for (const vo_ptr<Frame> &frame : m_frames) {
            trajectory.emplace_back(
                    std::make_pair(frame->get_time(), frame->get_pose()));
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

private:
    std::vector<vo_ptr<MapPoint>> m_points;
    std::vector<vo_ptr<Frame>> m_keyframes;
    std::vector<vo_ptr<Frame>> m_frames;
};
}// namespace vo_nono

#endif//VO_NONO_MAP_H
