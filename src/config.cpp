#include "vo_nono/config.h"

#include <yaml-cpp/yaml.h>

#include <opencv2/core.hpp>

#include "vo_nono/camera.h"

namespace vo_nono {
namespace {
Camera camera_from_yaml_node(const YAML::Node &camera_node) {
    cv::Mat camera_mat = cv::Mat::zeros(3, 3, CV_32F);
    camera_mat.at<float>(0, 0) = camera_node["fx"].as<float>();
    camera_mat.at<float>(1, 1) = camera_node["fy"].as<float>();
    camera_mat.at<float>(2, 2) = 1.0f;
    camera_mat.at<float>(0, 2) = camera_node["cx"].as<float>();
    camera_mat.at<float>(1, 2) = camera_node["cy"].as<float>();
    auto width = camera_node["width"].as<float>();
    auto height = camera_node["height"].as<float>();
    auto camera_res = Camera(camera_mat, width, height);

    if (camera_node["dist"]) {
        camera_res.set_dist_coeff(camera_node["dist"].as<std::vector<float>>());
    }
    return camera_res;
}

FrontendConfig frontend_config_from_yaml_node(const YAML::Node &node) {
    FrontendConfig res = FrontendConfig();
    return res;
}

SystemConfig sysconf_from_yaml_node(const YAML::Node &node) {
    SystemConfig res = SystemConfig();
    if (node["frontend"]) {
        res.frontend_config = frontend_config_from_yaml_node(node["frontend"]);
    }
    if (node["Camera"]) { res.camera = camera_from_yaml_node(node["Camera"]); }
    if (node["vocabulary"]) {
        res.voc_path = node["vocabulary"].as<std::string>();
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