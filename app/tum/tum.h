#ifndef VO_NONO_TUM_H
#define VO_NONO_TUM_H

#include <string>
#include <map>

#include <opencv2/core.hpp>

class TumDataBase {
public:
    TumDataBase(const std::string &base_path);
    [[nodiscard]] bool is_end() const;
    cv::Mat cur_image_gray() const;
    double cur_time() const;
    void next();

private:
    std::string m_base_path;
    std::vector<double> m_timestamps;
    std::vector<std::string> m_img_paths;
    int m_cur;
};

#endif//VO_NONO_TUM_H
