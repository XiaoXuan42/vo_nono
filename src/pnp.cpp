#include "vo_nono/pnp.h"

#include <opencv2/calib3d.hpp>

#include "vo_nono/util.h"

namespace vo_nono {
inline void _pnp_ransac_select(const std::vector<cv::Point2f>& img_pts,
                               int iter_cnt,
                               std::vector<std::vector<int>>& res) {
    assert(img_pts.size() > 4);
    res.resize(iter_cnt);
    const size_t total_sz = img_pts.size();
    for (int i = 0; i < iter_cnt; ++i) {
        res[i] = std::vector<int>(4);
        for (int j = 0; j < 4; ++j) {
            int local_violate_cnt = 0;
            while (true) {
                int propose = int(rand64() % total_sz);
                bool is_same = false, is_near = false;
                cv::Point2f cur_pt = img_pts[propose];
                for (int k = 0; k < j; ++k) {
                    if (res[i][k] == propose) {
                        is_same = true;
                        break;
                    }
                    if (local_violate_cnt < 5) {
                        cv::Point2f diff_pt = cur_pt - img_pts[res[i][k]];
                        float diff_square =
                                diff_pt.x * diff_pt.x + diff_pt.y * diff_pt.y;
                        if (diff_square < 64) {
                            local_violate_cnt += 1;
                            is_near = true;
                            break;
                        }
                    }
                }
                if (!is_same && !is_near) {
                    res[i][j] = propose;
                    break;
                }
            }
        }
    }
}

void _vo_nono_pnp_ransac(const std::vector<cv::Matx31f>& coords,
                         const std::vector<cv::Point2f>& img_pts,
                         const Camera& camera, int iter_cnt, float proj_th2,
                         cv::Mat& Rcw, cv::Mat& tcw,
                         std::vector<bool>& is_inlier) {
    assert(img_pts.size() > 4);
    std::vector<std::vector<int>> rd_selects;
    _pnp_ransac_select(img_pts, iter_cnt, rd_selects);

    cv::Mat init_rvec, init_tcw = tcw.clone();
    cv::Rodrigues(Rcw, init_rvec);
    int best_cnt_inliers = 0;
    cv::Mat best_rvec;

    for (auto& sel : rd_selects) {
        std::vector<cv::Matx31f> sample_coords;
        std::vector<cv::Point2f> sample_img_pts;
        cv::Mat cur_rvec = init_rvec.clone(), cur_tcw = init_tcw.clone();
        assert(sel.size() == 4);
        for (int i = 0; i < 4; ++i) {
            sample_coords.push_back(coords[sel[i]]);
            sample_img_pts.push_back(img_pts[sel[i]]);
        }
        cv::solvePnP(sample_coords, sample_img_pts, camera.get_intrinsic_mat(),
                     std::vector<double>(), cur_rvec, cur_tcw, false,
                     cv::SOLVEPNP_P3P);
        std::vector<cv::Point2f> res_img_pts;
        cv::projectPoints(coords, cur_rvec, cur_tcw, camera.get_intrinsic_mat(),
                          std::vector<double>(), res_img_pts);
        int cnt_inlier = 0;
        std::vector<bool> cur_is_inlier(img_pts.size(), false);
        for (int i = 0; i < (int) img_pts.size(); ++i) {
            cv::Point2f diff_pt = img_pts[i] - res_img_pts[i];
            float diff_square = diff_pt.x * diff_pt.x + diff_pt.y * diff_pt.y;
            if (diff_square < proj_th2) {
                cnt_inlier += 1;
                cur_is_inlier[i] = true;
            }
        }
        if (cnt_inlier > best_cnt_inliers) {
            best_cnt_inliers = cnt_inlier;
            is_inlier = std::move(cur_is_inlier);
            best_rvec = cur_rvec;
            tcw = cur_tcw;
        }
    }
    cv::Rodrigues(best_rvec, Rcw);
    if (Rcw.type() != CV_32F) { Rcw.convertTo(Rcw, CV_32F); }
    if (tcw.type() != CV_32F) { tcw.convertTo(tcw, CV_32F); }
}

void _cv_pnp_ransac(const std::vector<cv::Matx31f>& coords,
                    const std::vector<cv::Point2f>& img_pts,
                    const Camera& camera, int iter_cnt, float proj_th,
                    cv::Mat& Rcw, cv::Mat& tcw, std::vector<bool>& is_inlier) {
    std::vector<int> inlier_index;
    cv::Mat rvec;
    cv::Rodrigues(Rcw, rvec);
    rvec.convertTo(rvec, CV_64F);
    tcw.convertTo(tcw, CV_64F);
    cv::solvePnPRansac(coords, img_pts, camera.get_intrinsic_mat(),
                       std::vector<double>(), rvec, tcw, true, iter_cnt,
                       proj_th, 0.99, inlier_index);
    is_inlier.resize(coords.size());
    size_t cur_inlier = 0;
    for (int i = 0; i < (int) coords.size(); ++i) {
        if (cur_inlier < inlier_index.size() && inlier_index[cur_inlier] == i) {
            is_inlier[i] = true;
        } else {
            is_inlier[i] = false;
        }
        while (cur_inlier < inlier_index.size() &&
               i >= inlier_index[cur_inlier]) {
            cur_inlier += 1;
        }
    }
    cv::Rodrigues(rvec, Rcw);
    if (Rcw.type() != CV_32F) { Rcw.convertTo(Rcw, CV_32F); }
    if (tcw.type() != CV_32F) { tcw.convertTo(tcw, CV_32F); }
}

void pnp_ransac(const std::vector<cv::Matx31f>& coords,
                const std::vector<cv::Point2f>& img_pts, const Camera& camera,
                int iter_cnt, float proj_th2, cv::Mat& Rcw, cv::Mat& tcw,
                std::vector<bool>& is_inlier, PNP_RANSAC method) {
    switch (method) {
        case CV_PNP_RANSAC:
            _cv_pnp_ransac(coords, img_pts, camera, iter_cnt, proj_th2, Rcw, tcw,
                           is_inlier);
            break;
        case VO_NONO_PNP_RANSAC:
            _vo_nono_pnp_ransac(coords, img_pts, camera, iter_cnt, proj_th2, Rcw,
                                tcw, is_inlier);
            break;
        default:
            unimplemented();
    }
}

void pnp_optimize_proj_err(const std::vector<cv::Matx31f>& coords,
                           const std::vector<cv::Point2f>& img_pts,
                           const Camera& camera, cv::Mat& Rcw, cv::Mat& tcw) {
    cv::Mat rvec;
    cv::Rodrigues(Rcw, rvec);
    rvec.convertTo(rvec, CV_64F);
    tcw.convertTo(tcw, CV_64F);
    cv::solvePnPRefineLM(coords, img_pts, camera.get_intrinsic_mat(),
                         std::vector<double>(), rvec, tcw);
    cv::Rodrigues(rvec, Rcw);
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);
}
}// namespace vo_nono