#ifndef VO_NONO_MAP_H
#define VO_NONO_MAP_H

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "vo_nono/point.h"

namespace vo_nono {
class Map {
public:
    using Trajectory = std::vector<std::pair<double, cv::Mat>>;
    void insert_map_points(const std::vector<vo_ptr<MapPoint>> &points) {
        m_points.reserve(m_points.size() +
                         std::distance(points.begin(), points.end()));
        m_points.insert(m_points.end(), std::make_move_iterator(points.begin()),
                        std::make_move_iterator(points.end()));
    }

    void insert_key_frame(const vo_ptr<Frame> &frame) {
        m_frames.push_back(frame);
    }

    [[nodiscard]] Trajectory get_trajectory()
            const {
        Trajectory trajectory;
        trajectory.reserve(m_frames.size());
        for (const vo_ptr<Frame> &frame : m_frames) {
            trajectory.emplace_back(
                    std::make_pair(frame->get_time(), frame->get_pose()));
        }
        return trajectory;
    }

private:
    std::vector<vo_ptr<MapPoint>> m_points;
    std::vector<vo_ptr<Frame>> m_frames;
};
}// namespace vo_nono

#endif//VO_NONO_MAP_H
