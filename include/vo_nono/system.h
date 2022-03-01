#ifndef VO_NONO_SYSTEM_H
#define VO_NONO_SYSTEM_H

#include <opencv2/core/core.hpp>

#include "vo_nono/frontend.h"
#include "vo_nono/types.h"

namespace vo_nono {
struct SystemConfig {
    FrontendConfig frontend_config;
};

class System {
public:
    void get_image(const cv::Mat& image, vo_time_t t);

    explicit System(const SystemConfig& config)
        : m_config(config),
          m_frontend(config.frontend_config) {}

private:
    SystemConfig m_config;
    Frontend m_frontend;
};
}// namespace vo_nono

#endif//VO_NONO_SYSTEM_H
