#include <fstream>
#include <string>

#include "vo_nono/types.h"
#include "vo_nono/config.h"
#include "vo_nono/frontend.h"

#include "tum.h"

int main(int argc, const char *argv[]) {
    if (argc != 3) { return 0; }
    std::ifstream stream(argv[1]);
    std::stringstream sstream;
    if (!stream.is_open()) { return 0; }
    sstream << stream.rdbuf();

    std::string yaml_config = sstream.str();
    vo_nono::FrontendConfig config =
            vo_nono::YamlConfig().generate_frontend_conf_from_str(yaml_config);
    vo_nono::Frontend frontend = vo_nono::Frontend(config);

    TumDataBase database(argv[2]);
    while (!database.is_end()) {
        double cur_time = database.cur_time();
        cv::Mat img = database.cur_image_gray();
        frontend.get_image(img, vo_nono::vo_time_from(cur_time));
    }

    return 0;
}