#include "vo_nono/frame.h"

#include <algorithm>
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
}// namespace vo_nono
