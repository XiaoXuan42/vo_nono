#ifndef VO_NONO_TRIANGULATE_H
#define VO_NONO_TRIANGULATE_H

#include <opencv2/core.hpp>

#include "vo_nono/frame.h"

namespace vo_nono {
class Triangulator {
public:
    static cv::Mat triangulate(const cv::Mat &proj1, const cv::Mat &proj2, const cv::Point2f &pixel1, const cv::Point2f &pixel2);
    static cv::Mat triangulate(const std::vector<cv::Mat> &proj, const std::vector<cv::Point2f> &pixels);
    static void triangulate(const cv::Mat &proj1, const cv::Mat &proj2,
                            const std::vector<cv::Point2f> &img_pt1,
                            const std::vector<cv::Point2f> &img_pt2,
                            std::vector<cv::Mat> &result);

    static bool is_triangulate_inlier(const cv::Mat &Rcw1, const cv::Mat &tcw1,
                                      const cv::Mat &Rcw2, const cv::Mat &tcw2,
                                      const cv::Mat &tri_res, double grad_th);

    static int filter_triangulate(const cv::Mat &Rcw1, const cv::Mat &tcw1,
                                  const cv::Mat &Rcw2, const cv::Mat &tcw2,
                                  const std::vector<cv::Mat> &tri_res,
                                  std::vector<bool> &is_inlier, double grad_th);

    static int triangulate_and_filter_frames(
            const Frame *frame1, const Frame *frame2,
            const cv::Mat &cam_intrinsic_mat,
            const std::vector<cv::DMatch> &matches,
            std::vector<cv::Mat> &tri_result, std::vector<bool> &is_inlier,
            double grad_th);

private:
    constexpr static double EPS = 0.001;
};
}// namespace vo_nono

#endif//VO_NONO_TRIANGULATE_H
