#include "tum.h"

#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

TumDataBase::TumDataBase(const std::string& base_path)
    : m_base_path(base_path) {
    m_cur = 0;
    std::ifstream stream(base_path);
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

bool TumDataBase::is_end() const {
    return m_cur >= m_timestamps.size();
}

cv::Mat TumDataBase::cur_image_gray() const {
    std::string cur_path = m_base_path + "/" + m_img_paths[m_cur];
    cv::Mat img = cv::imread(cur_path, cv::IMREAD_GRAYSCALE);
    return img;
}

double TumDataBase::cur_time() const {
    return m_timestamps[m_cur];
}

void TumDataBase::next() {
    m_cur += 1;
}