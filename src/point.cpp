#include "vo_nono/point.h"

namespace vo_nono {
vo_id_t MapPoint::map_point_id_cnt = 0;

MapPoint MapPoint::create_map_point(float x, float y, float z,
                                    const cv::Mat &desc, int pyramid_level) {
    vo_id_t old_id = map_point_id_cnt;
    map_point_id_cnt += 1;
    return MapPoint(old_id, x, y, z, desc, pyramid_level);
}

MapPoint MapPoint::create_map_point(const cv::Mat &coord, const cv::Mat &desc,
                                    int pyramid_level) {
    assert(coord.type() == CV_32F);
    assert(coord.rows == 3);
    assert(coord.cols == 1);
    return create_map_point(coord.at<float>(0), coord.at<float>(1),
                            coord.at<float>(2), desc, pyramid_level);
}
}// namespace vo_nono