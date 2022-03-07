#ifndef VO_NONO_FEATURE_H
#define VO_NONO_FEATURE_H

#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "vo_nono/camera.h"
#include "vo_nono/frame.h"
#include "vo_nono/types.h"
#include "vo_nono/util.h"

namespace vo_nono {
static inline void filter_match_pts(const std::vector<cv::Point2f> &pts1,
                                    const std::vector<cv::Point2f> &pts2,
                                    std::vector<unsigned char> &mask,
                                    double ransac_th = 1.0) {
    cv::findFundamentalMat(pts1, pts2, mask, cv::FM_RANSAC, ransac_th, 0.99);
    assert(mask.size() == pts1.size());
}

void filter_match_key_pts(const std::vector<cv::KeyPoint> &kpts1,
                          const std::vector<cv::KeyPoint> &kpts2,
                          std::vector<unsigned char> &mask,
                          double ransac_th = 1.0);

struct ReprojRes {
    vo_id_t frame_id;
    vo_id_t map_point_id;
    int point_index;
    double desc_dis;
    cv::Matx31f coord;
};

// book: image point index of right frame -> map point id and descriptor distance
void reproj_points_from_frame(const vo_ptr<Frame> &left_frame,
                              const vo_ptr<Frame> &right_frame,
                              const Camera &camera,
                              std::map<int, ReprojRes> &book);

void pnp_from_reproj_res(const vo_ptr<Frame> &frame, const Camera &camera,
                         const std::map<int, ReprojRes> &book,
                         std::vector<int> &img_pt_index,
                         std::vector<vo_id_t> &map_pt_ids,
                         std::vector<cv::Matx31f> &map_pt_coords,
                         std::vector<int> &inliers, cv::Mat &Rcw, cv::Mat &tcw,
                         bool use_init = false, double reproj_error = 1.0);
}// namespace vo_nono

#endif//VO_NONO_FEATURE_H
