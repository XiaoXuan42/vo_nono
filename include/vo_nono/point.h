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
                                     const cv::Mat& desc);
    static MapPoint create_map_point(const cv::Mat& coord, const cv::Mat& desc);

    [[nodiscard]] vo_id_t get_id() const { return m_id; }

    [[nodiscard]] cv::Mat get_coord() const {
        assert(m_coord.type() == CV_32F);
        assert(m_coord.cols == 1);
        assert(m_coord.rows == 3);
        return m_coord;
    }

    void set_coord(const cv::Mat& coord) {
        assert(coord.type() == CV_32F);
        assert(coord.cols == 1);
        assert(coord.rows == 3);
        m_coord = coord;
    }

    [[nodiscard]] const cv::Mat& get_desc() const { return m_desc; }

private:
    static vo_id_t map_point_id_cnt;
    MapPoint(vo_id_t id, float x, float y, float z, cv::Mat desc)
        : m_id(id),
          m_desc(std::move(desc)) {
        m_coord = cv::Mat(3, 1, CV_32F);
        m_coord.at<float>(0, 0) = x;
        m_coord.at<float>(1, 0) = y;
        m_coord.at<float>(2, 0) = z;
    }

    vo_id_t m_id;
    cv::Mat m_coord;
    cv::Mat m_desc;
    std::vector<std::weak_ptr<Frame>> m_visible_frames;
};
}// namespace vo_nono

#endif//VO_NONO_POINT_H
