#include "vo_nono/system.h"

namespace vo_nono {
void System::get_image(const cv::Mat &image, double t) {
    m_frontend.get_image(image, t);
}
}// namespace vo_nono