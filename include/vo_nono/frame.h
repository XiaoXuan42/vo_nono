#ifndef VO_NONO_FRAME_H
#define VO_NONO_FRAME_H

#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vo_nono/camera.h"
#include "vo_nono/point.h"
#include "vo_nono/types.h"
#include "vo_nono/util.h"

namespace vo_nono {
class Frame {
public:
    cv::Mat img;

public:
    static Frame create_frame(cv::Mat descriptor,
                              std::vector<cv::KeyPoint> kpts,
                              const Camera &camera, double time,
                              cv::Mat Rcw = cv::Mat::eye(3, 3, CV_32F),
                              cv::Mat Tcw = cv::Mat::zeros(3, 1, CV_32F));

    [[nodiscard]] vo_id_t get_id() const { return m_id; }
    [[nodiscard]] double get_time() const { return m_time; }
    [[nodiscard]] cv::Mat get_descs() const { return m_descriptor; }
    [[nodiscard]] const std::vector<cv::KeyPoint> &get_kpts() const {
        return m_kpts;
    }
    // keypoints that already has corresponding map point
    [[nodiscard]] size_t get_set_cnt() const { return m_pt_mappt.size(); }
    [[nodiscard]] size_t get_kpt_cnt() const { return m_kpts.size(); }

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

    void set_map_pt(int i, const std::shared_ptr<MapPoint> &pt) {
        assert(m_pt_mappt.count(i) == 0);
        m_pt_mappt.insert({i, pt});
    }
    std::shared_ptr<MapPoint> get_map_pt(int i) const {
        assert(m_pt_mappt.count(i) != 0);
        return m_pt_mappt.at(i);
    }
    bool is_pt_set(int i) const { return m_pt_mappt.count(i) != 0; }

    cv::KeyPoint get_kpt_by_index(int i) const {
        assert(i < (int) m_kpts.size());
        return m_kpts[i];
    }
    cv::Mat get_desc_by_index(int i) const {
        assert(m_descriptor.rows > i);
        return m_descriptor.row(i);
    }
    int local_match(const cv::Mat &desc, const cv::Point2f &pos, double &dis,
                    const float dist_th);

private:
    constexpr static int WIDTH_TOTAL_GRID = 20;
    constexpr static int HEIGHT_TOTAL_GRID = 20;

    static vo_id_t frame_id_cnt;

    inline int get_grid_width_id(float x) const {
        return int(x / m_width_per_grid);
    }

    inline int get_grid_height_id(float y) const {
        return int(y / m_height_per_grid);
    }

    inline int get_grid_id(float x, float y) const {
        int w_id = get_grid_width_id(x), h_id = get_grid_height_id(y);
        assert(w_id <= WIDTH_TOTAL_GRID);
        assert(h_id <= HEIGHT_TOTAL_GRID);
        return h_id * WIDTH_TOTAL_GRID + w_id;
    }

    Frame(vo_id_t id, cv::Mat descriptor, std::vector<cv::KeyPoint> kpts,
          double time, cv::Mat Rcw, cv::Mat Tcw, float height, float width)
        : m_id(id),
          m_descriptor(std::move(descriptor)),
          m_kpts(std::move(kpts)),
          m_time(time),
          m_Rcw(std::move(Rcw)),
          m_Tcw(std::move(Tcw)),
          m_height(height),
          m_width(width),
          m_height_per_grid(std::ceil(height / HEIGHT_TOTAL_GRID)),
          m_width_per_grid(std::ceil(width / WIDTH_TOTAL_GRID)),
          m_grid_to_index((HEIGHT_TOTAL_GRID + 1) * (WIDTH_TOTAL_GRID + 1)) {
        for (size_t i = 0; i < m_kpts.size(); ++i) {
            int cur_id = get_grid_id(m_kpts[i].pt.x, m_kpts[i].pt.y);
            m_grid_to_index[cur_id].push_back((int) i);
        }
    }

    vo_id_t m_id;
    cv::Mat m_descriptor;
    std::vector<cv::KeyPoint> m_kpts;
    double m_time;

    cv::Mat m_Rcw;
    cv::Mat m_Tcw;

    // from index of m_kpts to map points
    std::unordered_map<int, std::shared_ptr<MapPoint>> m_pt_mappt;

    // accelerate local search
    float m_height, m_width;
    const float m_height_per_grid, m_width_per_grid;
    std::vector<std::vector<int>> m_grid_to_index;
};
}// namespace vo_nono

#endif//VO_NONO_FRAME_H
