#include "vo_nono/frame.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace vo_nono {
vo_id_t Frame::frame_id_cnt = 0;

Frame Frame::create_frame(cv::Mat descriptor, std::vector<cv::KeyPoint> kpts,
                          const Camera& camera, double time, cv::Mat Rcw,
                          cv::Mat Tcw) {
    vo_id_t old_id = frame_id_cnt;
    frame_id_cnt += 1;
    return Frame(old_id, std::move(descriptor), std::move(kpts), time,
                 std::move(Rcw), std::move(Tcw), camera.get_height(),
                 camera.get_width());
}

int Frame::local_match(const cv::KeyPoint& other_pt, const cv::Mat& desc,
                       const cv::Point2f& pos, const float dist_th) {
    if (pos.x < 0 || pos.x > m_width || pos.y < 0 || pos.y > m_height) {
        return -1;
    }
    const float dist_th_square = dist_th * dist_th;
    float x_left_most = pos.x - dist_th, x_right_most = pos.x + dist_th;
    float y_top_most = pos.y - dist_th, y_bottom_most = pos.y + dist_th;
    x_left_most = std::max(0.0f, x_left_most);
    x_right_most = std::min(x_right_most, m_width);
    y_top_most = std::max(0.0f, y_top_most);
    y_bottom_most = std::min(y_bottom_most, m_height);
    int least_grid_id = get_grid_id(x_left_most, y_top_most);
    int max_grid_id = get_grid_id(x_right_most, y_bottom_most);

    double second_min_dis = std::numeric_limits<double>::max();
    double min_dis = std::numeric_limits<double>::max();
    int min_index = -1;
    for (int i = least_grid_id; i <= max_grid_id; ++i) {
        for (auto index : m_grid_to_index[i]) {
            const float dx = pos.x - m_kpts[index].pt.x;
            const float dy = pos.y - m_kpts[index].pt.y;
            if (dx * dx + dy * dy < dist_th_square) {
                double cur_dis = cv::norm(desc, m_descriptor.row(index),
                                          cv::NORM_HAMMING);
                if (cur_dis < min_dis) {
                    second_min_dis = min_dis;
                    min_dis = cur_dis;
                    min_index = index;
                }
            }
        }
    }
    // ratio test
    if (min_dis >= 20 || min_dis >= second_min_dis * 0.8) { return -1; }
    return min_index;
}
}// namespace vo_nono
