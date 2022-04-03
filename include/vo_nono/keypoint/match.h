/*
#ifndef VO_NONO_MATCH_H
#define VO_NONO_MATCH_H

#include <opencv2/calib3d.hpp>

#include "vo_nono/camera.h"
#include "vo_nono/frame.h"
#include "vo_nono/point.h"

namespace vo_nono {
struct ProjMatch {
    int index;
    cv::Mat coord3d;
    cv::Point2f img_pt;
};

class ORBMatcher {
public:
    ORBMatcher(vo_ptr<Frame> p_frame, const cv::Mat &proj_mat,
               float total_width, float total_height)
        : m_width_per_grid(total_width / WIDTH_TOTAL_GRID),
          m_height_per_grid(total_height / HEIGHT_TOTAL_GRID),
          mp_frame(std::move(p_frame)),
          m_proj_mat(proj_mat) {
        assert(proj_mat.type() == CV_32F);
        assert(proj_mat.rows == 3);
        assert(proj_mat.cols == 4);
        for (size_t i = 0; i < mp_frame->get_kpts().size(); ++i) {
            cv::KeyPoint cur_kpt = mp_frame->get_kpt_by_index(int(i));
            int cur_id = get_grid_id(cur_kpt.pt.x, cur_kpt.pt.y);
            if (m_pyramid_grids.size() <= size_t(cur_kpt.octave)) {
                m_pyramid_grids.resize(cur_kpt.octave + 1);
            }
            m_pyramid_grids[cur_kpt.octave].grid[cur_id].push_back(int(i));
        }
    }

    void set_proj_mat(const cv::Mat &proj_mat) { m_proj_mat = proj_mat; }
    void match_by_projection(const std::vector<vo_ptr<MapPoint>> &map_points,
                             float dist_th,
                             std::vector<ProjMatch> &proj_matches);

private:
    constexpr static int WIDTH_TOTAL_GRID = 20;
    constexpr static int HEIGHT_TOTAL_GRID = 20;
    struct KeyPointGrid {
        std::vector<int> grid[WIDTH_TOTAL_GRID * HEIGHT_TOTAL_GRID];
    };

    [[nodiscard]] inline int get_grid_width_id(float x) const {
        return int(x / m_width_per_grid);
    }

    [[nodiscard]] inline int get_grid_height_id(float y) const {
        return int(y / m_height_per_grid);
    }

    [[nodiscard]] inline int get_grid_id(float x, float y) const {
        int w_id = get_grid_width_id(x), h_id = get_grid_height_id(y);
        assert(w_id <= WIDTH_TOTAL_GRID);
        assert(h_id <= HEIGHT_TOTAL_GRID);
        return h_id * WIDTH_TOTAL_GRID + w_id;
    }

    const float m_width_per_grid, m_height_per_grid;
    vo_ptr<Frame> mp_frame;
    std::vector<KeyPointGrid> m_pyramid_grids;
    cv::Mat m_proj_mat;
};
}// namespace vo_nono

#endif//VO_NONO_MATCH_H
*/
