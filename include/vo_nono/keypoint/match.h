#ifndef VO_NONO_MATCH_H
#define VO_NONO_MATCH_H

#include <opencv2/calib3d.hpp>
#include <utility>

#include "vo_nono/camera.h"
#include "vo_nono/frame.h"
#include "vo_nono/point.h"

namespace vo_nono {
struct ProjMatch {
    int index;
    cv::Matx31f coord3d;
    cv::Point2f img_pt;
    vo_ptr<MapPoint> p_map_pt;

    ProjMatch(int o_index, const cv::Matx31f &o_coord3d,
              const cv::Point2f &o_img_pt, vo_ptr<MapPoint> op_map_pt)
        : index(o_index),
          coord3d(o_coord3d),
          img_pt(o_img_pt),
          p_map_pt(std::move(op_map_pt)) {}
};

class ORBMatcher {
public:
    ORBMatcher(vo_ptr<Frame> p_frame, const Camera &camera)
        : m_total_width(camera.get_width()),
          m_total_height(camera.get_height()),
          m_width_per_grid(
                  std::ceil((camera.get_width() + WIDTH_TOTAL_GRID - 1.0f) /
                            WIDTH_TOTAL_GRID)),
          m_height_per_grid(
                  std::ceil((camera.get_height() + HEIGHT_TOTAL_GRID - 1.0f) /
                            HEIGHT_TOTAL_GRID)),
          mp_frame(std::move(p_frame)),
          m_camera_intrinsic(camera.get_intrinsic_mat()) {
        for (size_t i = 0; i < mp_frame->get_kpts().size(); ++i) {
            cv::KeyPoint cur_kpt = mp_frame->get_kpt_by_index(int(i));
            if (m_pyramid_grids.size() <= size_t(cur_kpt.octave)) {
                m_pyramid_grids.resize(cur_kpt.octave + 1);
            }
            int width_id = get_grid_width_id(cur_kpt.pt.x);
            int height_id = get_grid_height_id(cur_kpt.pt.y);
            m_pyramid_grids[cur_kpt.octave].grid[height_id][width_id].push_back(
                    int(i));
        }
    }

    std::vector<ProjMatch> match_by_projection(
            const std::vector<vo_ptr<MapPoint>> &map_points, float dist_th);
    std::vector<cv::DMatch> match_descriptor_bf(const cv::Mat &o_descpt,
                                                float soft_dis_th,
                                                float hard_dis_th,
                                                int expect_cnt);

private:
    constexpr static int WIDTH_TOTAL_GRID = 64;
    constexpr static int HEIGHT_TOTAL_GRID = 48;
    constexpr static int MAX_DESC_DIS = 100;

    struct KeyPointGrid {
        std::vector<int> grid[HEIGHT_TOTAL_GRID][WIDTH_TOTAL_GRID];
    };

    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    static int orb_distance(const cv::Mat &a, const cv::Mat &b) {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();
        int dist = 0;
        for (int i = 0; i < 8; i++, pa++, pb++) {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }
        return dist;
    }

    [[nodiscard]] inline int get_grid_width_id(float x) const {
        return int(x / m_width_per_grid);
    }

    [[nodiscard]] inline int get_grid_height_id(float y) const {
        return int(y / m_height_per_grid);
    }

    const float m_total_width, m_total_height;
    const float m_width_per_grid, m_height_per_grid;
    vo_ptr<Frame> mp_frame;
    std::vector<KeyPointGrid> m_pyramid_grids;
    cv::Mat m_camera_intrinsic;
};
}// namespace vo_nono

#endif//VO_NONO_MATCH_H
