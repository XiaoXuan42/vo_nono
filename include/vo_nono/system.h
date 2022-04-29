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
    std::string voc_path;
};

class System {
public:
    explicit System(const SystemConfig& config)
        : config_(config),
          camera_(config.camera),
          map_(std::make_shared<Map>(camera_, config.voc_path.c_str())),
          frontend_(config.frontend_config, camera_, map_) {}

    void get_image(const cv::Mat& image, double t);

    [[nodiscard]] Map::Trajectory get_trajectory() const {
        return map_->get_trajectory();
    }

private:
    SystemConfig config_;
    Camera camera_;
    vo_ptr<Map> map_;
    Frontend frontend_;
};
}// namespace vo_nono

#endif//VO_NONO_SYSTEM_H
