#ifndef VO_NONO_FRAME_H
#define VO_NONO_FRAME_H

#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vo_nono/types.h"
#include "vo_nono/util.h"

namespace vo_nono {
class Frame {
public:
    static Frame create_frame(cv::Mat descriptor,
                              std::vector<cv::KeyPoint> kpts, vo_time_t time);

    [[nodiscard]] vo_id_t get_id() const { return m_id; }
    [[nodiscard]] vo_time_t get_time() const { return m_time; }
    [[nodiscard]] cv::Mat get_dscpt_array() const { return m_descriptor; }
    [[nodiscard]] const std::vector<cv::KeyPoint> &get_kpts() const {
        return m_kpts;
    }

    void set_Rcw(const cv::Mat &Rcw) {
        assert(Rcw.rows == 3);
        assert(Rcw.cols == 3);
        assert_float_eq(cv::determinant(Rcw), 1.0f);
        m_Rcw = Rcw;
    }
    void set_Tcw(const cv::Mat &Tcw) {
        assert(Tcw.rows == 3);
        assert(Tcw.cols == 1);
        m_Tcw = Tcw;
    }
    [[nodiscard]] cv::Mat get_Rcw() const { return m_Rcw; }
    [[nodiscard]] cv::Mat get_Tcw() const { return m_Tcw; }

    void set_pt(int i, vo_id_t id, float x, float y, float z) {
        assert(m_pt_id.count(i) == 0);
        m_pt_id[i] = PointCache(id, x, y, z);
    }

private:
    static vo_id_t frame_id_cnt;

    Frame(vo_id_t id, cv::Mat descriptor, std::vector<cv::KeyPoint> kpts,
          vo_time_t time)
        : m_id(id),
          m_descriptor(std::move(descriptor)),
          m_kpts(std::move(kpts)),
          m_time(time) {}

    vo_id_t m_id;
    cv::Mat m_descriptor;
    std::vector<cv::KeyPoint> m_kpts;
    vo_time_t m_time;

    cv::Mat m_Rcw;
    cv::Mat m_Tcw;

    struct PointCache {
        vo_id_t id;
        cv::Matx31f coord;
        PointCache(vo_id_t id, float x, float y, float z)
            : id(id),
              coord(x, y, z) {}
    };
    // from index of m_kpts to points' ID and coordinate
    std::unordered_map<int, PointCache> m_pt_id;
};
}// namespace vo_nono

#endif//VO_NONO_FRAME_H
