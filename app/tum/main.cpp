#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>

#include "tum.h"
#include "vo_nono/config.h"
#include "vo_nono/system.h"
#include "vo_nono/types.h"

#define OUTPUT_REDIR

#ifdef OUTPUT_REDIR
#include <cstdio>
#endif

void live_stream(const char *database_path) {
    TumDataBase database(database_path);
    int frame_id = 0;
    while (!database.is_end()) {
        cv::Mat img = database.cur_image_gray();
        database.next();
        std::string title = "live stream";
        std::cerr << frame_id << std::endl;
        cv::imshow(title, img);
        cv::waitKey(100);
        frame_id += 1;
    }
}

void tum(const char *config_path, const char *database_path,
         const char *traj_path, int max_frame) {
    std::ifstream stream(config_path);
    std::stringstream sstream;
    if (!stream.is_open()) { return; }
    sstream << stream.rdbuf();

    std::string yaml_config = sstream.str();
    vo_nono::SystemConfig config =
            vo_nono::YamlConfig().generate_sysconf_from_str(yaml_config);
    vo_nono::System system = vo_nono::System(config);

    TumDataBase database(database_path);
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
        frame_id += 1;
        if (frame_id >= max_frame) { break; }
    }

    if (traj_path) {
        std::vector<std::pair<double, cv::Mat>> trajectory =
                system.get_trajectory();
        std::cout << "Save Trajectory to " << traj_path << std::endl;
        TumDataBase::trajectory_to_tum(trajectory, traj_path);
    }
}

int main(int argc, const char *argv[]) {
    if (argc < 3) { return 0; }
#ifdef OUTPUT_REDIR
    if (argc == 5) { freopen(argv[4], "w", stdout); }
#endif
    const char *config_path = argv[1], *database_path = argv[2],
               *traj_path = nullptr;
    if (argc >= 4) { traj_path = argv[3]; }
    tum(config_path, database_path, traj_path, 130);

    //live_stream(argv[2]);
#ifdef OUTPUT_REDIR
    if (argc >= 5) { fclose(stdout); }
#endif

    return 0;
}