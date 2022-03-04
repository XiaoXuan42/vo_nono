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
    cv::Mat img;

private:
    struct PointCache {
        vo_id_t id;
        cv::Matx31f coord;
        PointCache(vo_id_t id, float x, float y, float z)
            : id(id),
              coord(x, y, z) {}
    };

public:
    static Frame create_frame(cv::Mat descriptor,
                              std::vector<cv::KeyPoint> kpts, double time,
                              cv::Mat Rcw = cv::Mat::eye(3, 3, CV_32F),
                              cv::Mat Tcw = cv::Mat::zeros(3, 1, CV_32F));

    [[nodiscard]] vo_id_t get_id() const { return m_id; }
    [[nodiscard]] double get_time() const { return m_time; }
    [[nodiscard]] cv::Mat get_dscpts() const { return m_descriptor; }
    [[nodiscard]] const std::vector<cv::KeyPoint> &get_kpts() const {
        return m_kpts;
    }
    [[nodiscard]] size_t get_set_cnt() const { return m_set_cnt; }

    void set_Rcw(const cv::Mat &Rcw) {
        assert(Rcw.rows == 3);
        assert(Rcw.cols == 3);
        assert_float_eq(cv::determinant(Rcw), 1.0f);
        if (Rcw.type() == CV_32F) {
            m_Rcw = Rcw;
        } else if (Rcw.type() == CV_64F) {
            Rcw.convertTo(m_Rcw, CV_32F);
        } else {
            assert(false);
        }
        assert(m_Rcw.type() == CV_32F);
    }

    void set_Tcw(const cv::Mat &Tcw) {
        assert(Tcw.rows == 3);
        assert(Tcw.cols == 1);
        if (Tcw.type() == CV_32F) {
            m_Tcw = Tcw;
        } else if (Tcw.type() == CV_64F) {
            Tcw.convertTo(m_Tcw, CV_32F);
        } else {
            assert(false);
        }
        assert(m_Tcw.type() == CV_32F);
    }

    void set_pose(const cv::Mat &Rcw, const cv::Mat &Tcw) {
        set_Rcw(Rcw);
        set_Tcw(Tcw);
    }

    [[nodiscard]] cv::Mat get_Rcw() const { return m_Rcw.clone(); }
    [[nodiscard]] cv::Mat get_Tcw() const { return m_Tcw.clone(); }
    [[nodiscard]] cv::Mat get_pose() const {
        cv::Mat pose = cv::Mat::zeros(3, 4, CV_32F);
        m_Rcw.copyTo(pose.rowRange(0, 3).colRange(0, 3));
        m_Tcw.copyTo(pose.rowRange(0, 3).col(3));
        return pose;
    }

    void set_pt(int i, vo_id_t id, float x, float y, float z) {
        assert(m_pt_id.count(i) == 0);
        m_pt_id.insert({i, PointCache(id, x, y, z)});
        m_set_cnt += 1;
    }

    cv::Matx31f get_pt_coord(int i) const {
        assert(m_pt_id.count(i) != 0);
        return m_pt_id.at(i).coord;
    }

    vo_id_t get_pt_id(int i) const {
        assert(m_pt_id.count(i) != 0);
        return m_pt_id.at(i).id;
    }

    bool is_pt_set(int i) const { return m_pt_id.count(i) != 0; }

private:
    static vo_id_t frame_id_cnt;

    Frame(vo_id_t id, cv::Mat descriptor, std::vector<cv::KeyPoint> kpts,
          double time, cv::Mat Rcw, cv::Mat Tcw)
        : m_id(id),
          m_descriptor(std::move(descriptor)),
          m_kpts(std::move(kpts)),
          m_time(time),
          m_Rcw(std::move(Rcw)),
          m_Tcw(std::move(Tcw)),
          m_set_cnt(0) {}

    vo_id_t m_id;
    cv::Mat m_descriptor;
    std::vector<cv::KeyPoint> m_kpts;
    double m_time;

    cv::Mat m_Rcw;
    cv::Mat m_Tcw;

    // from index of m_kpts to points' ID and coordinate
    std::unordered_map<int, PointCache> m_pt_id;
    size_t m_set_cnt;
};
}// namespace vo_nono

#endif//VO_NONO_FRAME_H
