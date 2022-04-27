#ifndef VO_NONO_POINT_H
#define VO_NONO_POINT_H

#include <memory>
#include <opencv2/core.hpp>

#include "vo_nono/types.h"

namespace vo_nono {
class Frame;
class MapPoint {
public:
    static MapPoint create_map_point(float x, float y, float z,
                                     const cv::Mat& desc, int pyramid_level);
    static MapPoint create_map_point(const cv::Mat& coord, const cv::Mat& desc,
                                     int pyramid_level);

    [[nodiscard]] vo_id_t get_id() const { return id_; }
    [[nodiscard]] int get_pyramid_level() const { return pyramid_level_; }

    [[nodiscard]] cv::Mat get_coord() const {
        assert(coord_.type() == CV_32F);
        assert(coord_.cols == 1);
        assert(coord_.rows == 3);
        return coord_;
    }

    void set_coord(const cv::Mat& coord) {
        assert(coord.type() == CV_32F);
        assert(coord.cols == 1);
        assert(coord.rows == 3);
        coord_ = coord;
    }

    [[nodiscard]] const cv::Mat& get_desc() const { return desc_; }

private:
    static vo_id_t map_point_id_cnt;
    MapPoint(vo_id_t id, float x, float y, float z, cv::Mat desc,
             int pyramid_level)
        : id_(id),
          desc_(std::move(desc)),
          pyramid_level_(pyramid_level) {
        coord_ = cv::Mat(3, 1, CV_32F);
        coord_.at<float>(0, 0) = x;
        coord_.at<float>(1, 0) = y;
        coord_.at<float>(2, 0) = z;
    }

    vo_id_t id_;
    cv::Mat coord_;
    cv::Mat desc_;
    int pyramid_level_;
};
}// namespace vo_nono

#endif//VO_NONO_POINT_H
