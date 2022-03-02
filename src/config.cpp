#include "vo_nono/config.h"

#include <yaml-cpp/yaml.h>

#include <opencv2/core.hpp>

#include "vo_nono/camera.h"

namespace vo_nono {
namespace {
FrontendConfig frontend_config_from_yaml_node(const YAML::Node &node) {
    FrontendConfig res = FrontendConfig();
    if (node["Camera"]) {
        const auto camera_node = node["Camera"];
        cv::Mat camera_mat = cv::Mat::zeros(3, 3, CV_32F);
        camera_mat.at<float>(0, 0) = camera_node["fx"].as<float>();
        camera_mat.at<float>(1, 1) = camera_node["fy"].as<float>();
        camera_mat.at<float>(2, 2) = 1.0f;
        camera_mat.at<float>(0, 2) = camera_node["cx"].as<float>();
        camera_mat.at<float>(1, 2) = camera_node["cy"].as<float>();
        res.camera = Camera(camera_mat);

        if (node["dist"]) {
            res.camera.set_dist_coeff(node["dist"].as<std::vector<float>>());
        }
    }
    return res;
}

SystemConfig sysconf_from_yaml_node(const YAML::Node &node) {
    SystemConfig res = SystemConfig();
    if (node["frontend"]) {
        res.frontend_config = frontend_config_from_yaml_node(node["frontend"]);
    }
    return res;
}
}// namespace
SystemConfig YamlConfig::generate_sysconf_from_str(const std::string &str) {
    YAML::Node node = YAML::Load(str);
    return sysconf_from_yaml_node(node);
}

FrontendConfig YamlConfig::generate_frontend_conf_from_str(
        const std::string &str) {
    YAML::Node node = YAML::Load(str);
    return frontend_config_from_yaml_node(node);
}
}// namespace vo_nono