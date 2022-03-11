#include "vo_nono/point.h"

namespace vo_nono {
vo_id_t MapPoint::map_point_id_cnt = 0;

MapPoint MapPoint::create_map_point(float x, float y, float z, const cv::Mat &desc) {
    vo_id_t old_id = map_point_id_cnt;
    map_point_id_cnt += 1;
    return MapPoint(old_id, x, y, z, desc);
}

}