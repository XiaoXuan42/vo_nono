#ifndef VO_NONO_PNP_H
#define VO_NONO_PNP_H

#include <cstdint>
#include <vector>

#include "vo_nono/camera.h"

namespace vo_nono {
enum PNP_RANSAC : uint32_t {
    CV_PNP_RANSAC,
    VO_NONO_PNP_RANSAC,
};
void pnp_ransac(const std::vector<cv::Matx31f>& coords,
                const std::vector<cv::Point2f>& img_pts, const Camera& camera,
                int iter_cnt, float proj_th2, cv::Mat& Rcw, cv::Mat& tcw,
                std::vector<bool>& is_inlier, PNP_RANSAC method);

void pnp_optimize_proj_err(const std::vector<cv::Matx31f>& coords,
                           const std::vector<cv::Point2f>& img_pts,
                           const Camera& camera,
                           cv::Mat &Rcw,
                           cv::Mat &tcw);
}// namespace vo_nono

#endif//VO_NONO_PNP_H
