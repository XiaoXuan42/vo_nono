#include "tum.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <sstream>

TumDataBase::TumDataBase(const std::string& base_path)
    : m_base_path(base_path) {
    m_cur = 0;
    std::ifstream stream(base_path + "/rgb.txt");
    if (!stream.is_open()) { return; }
    std::string str;
    while (std::getline(stream, str)) {
        if (str.length() == 0 || str[0] == '#') { continue; }
        std::stringstream ss(str);
        double timestamp;
        std::string path;
        ss >> timestamp >> path;
        m_timestamps.push_back(timestamp);
        m_img_paths.push_back(path);
    }
}

bool TumDataBase::is_end() const { return m_cur >= m_timestamps.size(); }

cv::Mat TumDataBase::cur_image_gray() const {
    std::string cur_path = m_base_path + "/" + m_img_paths[m_cur];
    cv::Mat img = cv::imread(cur_path, cv::IMREAD_GRAYSCALE);
    return img;
}

double TumDataBase::cur_time() const { return m_timestamps[m_cur]; }

void TumDataBase::next() { m_cur += 1; }

void TumDataBase::trajectory_to_tum(
        const std::vector<std::pair<double, cv::Mat>>& trajectory,
        const char* path) {
    std::ofstream o_stream(path);
    if (!o_stream.is_open()) { return; }
    o_stream << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    for (const auto& time_pose : trajectory) {
        double time = time_pose.first;
        cv::Mat pose = time_pose.second;
        cv::Mat t(3, 1, CV_32F), R(3, 3, CV_32F), R64(3, 3, CV_64F);
        double quat[4];
        pose.rowRange(0, 3).col(3).copyTo(t);
        pose.rowRange(0, 3).colRange(0, 3).copyTo(R);
        R.convertTo(R64, CV_64F);
        std::cout << R64 << std::endl;
        get_quaternion(R64, quat);
        o_stream << time << " ";
        for (int i = 0; i < 3; ++i) { o_stream << t.at<float>(i, 0) << " "; }
        for (int i = 0; i < 4; ++i) {
            o_stream << quat[i];
            if (i != 3) { o_stream << " "; }
        }
        o_stream << std::endl;
    }
}