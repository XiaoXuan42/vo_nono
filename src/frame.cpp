#include "vo_nono/frame.h"

namespace vo_nono {
Frame::vo_id_t Frame::frame_id_cnt = 0;

Frame Frame::create_frame(cv::Mat descriptor, std::vector<cv::KeyPoint> kpts, vo_time_t time) {
    vo_id_t old_id = frame_id_cnt;
    frame_id_cnt += 1;
    return Frame(old_id, std::move(descriptor), std::move(kpts), time);
}
}
