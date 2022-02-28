#include "vo_nono/system.h"

namespace vo_nono {
void VoSystem::get_image(const cv::Mat &image, vo_time_t t) {
    m_frontend.get_image(image, t);
}
}// namespace vo_nono