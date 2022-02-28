#ifndef VO_NONO_SYSTEM_H
#define VO_NONO_SYSTEM_H

#include <opencv2/core/core.hpp>

#include "vo_nono/frontend.h"
#include "vo_nono/types.h"

namespace vo_nono {
class VoSystem {
public:
    void get_image(const cv::Mat &image, vo_time_t t);

private:
    Frontend m_frontend;
};
}// namespace vo_nono

#endif//VO_NONO_SYSTEM_H
