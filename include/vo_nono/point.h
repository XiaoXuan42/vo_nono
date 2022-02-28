#ifndef VO_NONO_POINT_H
#define VO_NONO_POINT_H

#include <opencv2/core.hpp>

#include "vo_nono/types.h"

namespace vo_nono {
class MapPoint {
public:
    static MapPoint create_map_point(float x, float y, float z);

    [[nodiscard]] vo_id_t get_id() const { return m_id; }

private:
    static vo_id_t map_point_id_cnt;
    MapPoint(vo_id_t id, float x, float y, float z): m_id(id), m_coord(x, y, z) {}

    vo_id_t m_id;
    cv::Matx31f m_coord;
};
}

#endif//VO_NONO_POINT_H
