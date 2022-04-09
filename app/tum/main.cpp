#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

#include "tum.h"
#include "vo_nono/config.h"
#include "vo_nono/system.h"
#include "vo_nono/types.h"

#define OUTPUT_REDIR

#ifdef OUTPUT_REDIR
#include <cstdio>
#endif

int main(int argc, const char *argv[]) {
    if (argc < 3) { return 0; }
#ifdef OUTPUT_REDIR
    if (argc == 5) { freopen(argv[4], "w", stdout); }
#endif
    std::ifstream stream(argv[1]);
    std::stringstream sstream;
    if (!stream.is_open()) { return 0; }
    sstream << stream.rdbuf();

    std::string yaml_config = sstream.str();
    vo_nono::SystemConfig config =
            vo_nono::YamlConfig().generate_sysconf_from_str(yaml_config);
    vo_nono::System system = vo_nono::System(config);

    TumDataBase database(argv[2]);
    int frame_id = 0;
    while (!database.is_end()) {
        double cur_time = database.cur_time();
        cv::Mat img = database.cur_image_gray();
        database.next();
        std::chrono::steady_clock::time_point t1 =
                std::chrono::steady_clock::now();
        system.get_image(img, cur_time);
        std::chrono::steady_clock::time_point t2 =
                std::chrono::steady_clock::now();
        double time_used =
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                          t1)
                        .count();
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(6);
        std::cout << "Frame " << frame_id << "(time: " << cur_time << ") cost "
                  << time_used << " seconds." << std::endl;
        std::cout << "---------------------------------------------------------"
                     "----"
                  << std::endl;
        if (frame_id >= 160) { break; }
        frame_id += 1;
    }

    if (argc >= 4) {
        std::vector<std::pair<double, cv::Mat>> trajectory =
                system.get_trajectory();
        std::cout << "Save Trajectory to " << argv[3] << std::endl;
        TumDataBase::trajectory_to_tum(trajectory, argv[3]);
    }
#ifdef OUTPUT_REDIR
    if (argc >= 5) { fclose(stdout); }
#endif

    return 0;
}