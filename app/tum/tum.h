#ifndef VO_NONO_TUM_H
#define VO_NONO_TUM_H

#include <string>
#include <map>

#include <opencv2/core.hpp>
#include "vo_nono/types.h"

class TumDataBase {
public:
    explicit TumDataBase(const std::string &base_path);
    [[nodiscard]] bool is_end() const;
    [[nodiscard]] cv::Mat cur_image_gray() const;
    [[nodiscard]] double cur_time() const;
    [[nodiscard]] std::string cur_path() const {
        return m_img_paths[m_cur];
    }
    void next();

    static void trajectory_to_tum(const std::vector<std::pair<double, cv::Mat>> &trajectory, const char *path);

private:
    // from https://gist.github.com/shubh-agrawal/76754b9bfb0f4143819dbd146d15d4c8
    // R: double
    static void get_quaternion(cv::Mat R, double Q[])
    {
        double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);

        if (trace > 0.0)
        {
            double s = sqrt(trace + 1.0);
            Q[3] = (s * 0.5);
            s = 0.5 / s;
            Q[0] = ((R.at<double>(2,1) - R.at<double>(1,2)) * s);
            Q[1] = ((R.at<double>(0,2) - R.at<double>(2,0)) * s);
            Q[2] = ((R.at<double>(1,0) - R.at<double>(0,1)) * s);
        }

        else
        {
            int i = R.at<double>(0,0) < R.at<double>(1,1) ? (R.at<double>(1,1) < R.at<double>(2,2) ? 2 : 1) : (R.at<double>(0,0) < R.at<double>(2,2) ? 2 : 0);
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;

            double s = sqrt(R.at<double>(i, i) - R.at<double>(j,j) - R.at<double>(k,k) + 1.0);
            Q[i] = s * 0.5;
            s = 0.5 / s;

            Q[3] = (R.at<double>(k,j) - R.at<double>(j,k)) * s;
            Q[j] = (R.at<double>(j,i) + R.at<double>(i,j)) * s;
            Q[k] = (R.at<double>(k,i) + R.at<double>(i,k)) * s;
        }
    }
private:
    std::string m_base_path;
    std::vector<double> m_timestamps;
    std::vector<std::string> m_img_paths;
    int m_cur;
};

#endif//VO_NONO_TUM_H
