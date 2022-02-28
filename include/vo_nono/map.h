#ifndef VO_NONO_MAP_H
#define VO_NONO_MAP_H

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "vo_nono/point.h"

namespace vo_nono {
class Map {
public:
    void insert_map_points(std::vector<vo_uptr<MapPoint>> &points) {
        m_points.reserve(m_points.size() +
                         std::distance(points.begin(), points.end()));
        m_points.insert(m_points.end(), std::make_move_iterator(points.begin()),
                        std::make_move_iterator(points.end()));
    }

private:
    std::vector<vo_uptr<MapPoint>> m_points;
};
}// namespace vo_nono

#endif//VO_NONO_MAP_H
