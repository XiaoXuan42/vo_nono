#ifndef VO_NONO_SYSTEM_H
#define VO_NONO_SYSTEM_H

#include <opencv2/core/core.hpp>

#include "vo_nono/frontend.h"
#include "vo_nono/map.h"
#include "vo_nono/types.h"

namespace vo_nono {
struct SystemConfig {
    FrontendConfig frontend_config;
    Camera camera;
};

class System {
public:
    explicit System(const SystemConfig& config)
        : m_config(config),
          m_camera(config.camera),
          m_map(std::make_shared<Map>(m_camera)),
          m_frontend(config.frontend_config, m_camera, m_map) {}

    void get_image(const cv::Mat& image, double t);

    [[nodiscard]] Map::Trajectory get_trajectory() const {
        return m_map->get_trajectory();
    }

private:
    SystemConfig m_config;
    Camera m_camera;
    vo_ptr<Map> m_map;
    Frontend m_frontend;
};
}// namespace vo_nono

#endif//VO_NONO_SYSTEM_H
