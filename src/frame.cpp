#include "vo_nono/frame.h"

#include <limits>
#include <utility>

namespace vo_nono {
vo_id_t Frame::frame_id_cnt = 0;

Frame Frame::create_frame(cv::Mat descriptor, std::vector<cv::KeyPoint> kpts,
                          double time, cv::Mat Rcw, cv::Mat Tcw) {
    vo_id_t old_id = frame_id_cnt;
    frame_id_cnt += 1;
    return Frame(old_id, std::move(descriptor), std::move(kpts), time,
                 std::move(Rcw), std::move(Tcw));
}

int Frame::local_match(const cv::KeyPoint& other_pt, const cv::Mat& desc,
                       const cv::Point2f& pos, const float dist_th) {
    const float dist_th_square = dist_th * dist_th;

    double min_dis = std::numeric_limits<double>::max();
    int min_index = -1;
    for (int i = 0; i < (int) m_kpts.size(); ++i) {
        const float dx = pos.x - m_kpts[i].pt.x;
        const float dy = pos.y - m_kpts[i].pt.y;
        if (dx * dx + dy * dy < dist_th_square) {
            double cur_dis =
                    cv::norm(desc, m_descriptor.row(i), cv::NORM_HAMMING);
            if (cur_dis < min_dis) {
                min_dis = cur_dis;
                min_index = i;
            }
        }
    }
    return min_index;
}
}// namespace vo_nono
