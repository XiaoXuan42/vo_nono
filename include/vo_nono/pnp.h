#ifndef VO_NONO_PNP_H
#define VO_NONO_PNP_H

#include <cstdint>
#include <vector>

#include "vo_nono/camera.h"

namespace vo_nono {
class PnP {
public:
    static void pnp_ransac(const std::vector<cv::Matx31f>& coords,
                           const std::vector<cv::Point2f>& img_pts,
                           const Camera& camera, int iter_cnt, float proj_th,
                           cv::Mat& Rcw, cv::Mat& tcw,
                           std::vector<bool>& is_inlier, double th = 5.99);

    static void pnp_ransac_cv(const std::vector<cv::Matx31f>& coords,
                              const std::vector<cv::Point2f>& img_pts,
                              const Camera& camera, int iter_cnt, float proj_th,
                              cv::Mat& Rcw, cv::Mat& tcw,
                              std::vector<bool>& is_inlier);

    static void pnp_optimize_cv(const std::vector<cv::Matx31f>& coords,
                                const std::vector<cv::Point2f>& img_pts,
                                const Camera& camera, cv::Mat& Rcw,
                                cv::Mat& tcw);

    static std::vector<bool> pnp_by_optimize(
            const std::vector<cv::Matx31f>& coords,
            const std::vector<cv::Point2f>& img_pts, const Camera& camera,
            cv::Mat& Rcw, cv::Mat& tcw);
};
}// namespace vo_nono

#endif//VO_NONO_PNP_H
