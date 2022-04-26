#include "vo_nono/keypoint/match.h"

#include <iostream>

#include "vo_nono/util/geometry.h"
#include "vo_nono/util/histogram.h"
#include "vo_nono/util/util.h"

namespace vo_nono {
std::vector<ProjMatch> ORBMatcher::match_by_projection(
        const std::vector<vo_ptr<MapPoint>> &map_points, float r_th) {
    if (!mb_space_hash) { space_hash(); }
    std::vector<cv::Matx31f> coord3ds;
    std::vector<cv::Point2f> img_pts;
    std::vector<vo_ptr<MapPoint>> map_pts;
    std::vector<double> distances;
    std::unordered_map<int, int> book;

    int proj_exceed = 0, in_image = 0, no_near = 0, collide = 0;
    cv::Mat proj_mat = Geometry::get_proj_mat(m_camera_intrinsic, m_Rcw, m_tcw);
    for (auto &map_pt : map_points) {
        cv::Point2f proj_img_pt =
                Geometry::project_euclid3d(proj_mat, map_pt->get_coord());
        auto coord = cv::Matx31f(map_pt->get_coord());
        if (proj_img_pt.x >= 0 && proj_img_pt.x < m_total_width &&
            proj_img_pt.y >= 0 && proj_img_pt.y < m_total_height) {
            in_image += 1;

            int min_width_id =
                    get_grid_width_id(std::max(0.0f, proj_img_pt.x - r_th));
            int max_width_id = get_grid_width_id(
                    std::min(m_total_width, proj_img_pt.x + r_th));
            int min_height_id =
                    get_grid_height_id(std::max(0.0f, proj_img_pt.y - r_th));
            int max_height_id = get_grid_height_id(
                    std::min(m_total_height, proj_img_pt.y + r_th));

            int best_id = -1;
            int best_dis = std::numeric_limits<int>::max();
            for (int i = min_height_id; i <= max_height_id; ++i) {
                for (int j = min_width_id; j <= max_width_id; ++j) {
                    int min_pyramid_level =
                            std::max(map_pt->get_pyramid_level() - 1, 0);
                    int max_pyramid_level =
                            std::min(map_pt->get_pyramid_level() + 1,
                                     int(m_pyramid_grids.size()) - 1);
                    for (int k = min_pyramid_level; k <= max_pyramid_level;
                         ++k) {
                        auto &level_grid = m_pyramid_grids[k];
                        for (auto index : level_grid.grid[i][j]) {
                            cv::KeyPoint kpt = kpts[index];
                            if (std::fabs(kpt.pt.x - proj_img_pt.x) > r_th ||
                                std::fabs(kpt.pt.y - proj_img_pt.y) > r_th) {
                                continue;
                            }
                            int cur_dis = orb_distance(descriptors.row(index),
                                                       map_pt->get_desc());
                            if (cur_dis < best_dis) {
                                best_id = index;
                                best_dis = cur_dis;
                            }
                        }
                    }
                }
            }

            if (best_id < 0) {
                no_near += 1;
                continue;
            }
            if (best_dis > MAX_DESC_DIS) {
                proj_exceed += 1;
                continue;
            }
            if (book.count(best_id)) {
                assert(book[best_id] < int(distances.size()));
                if (distances[book[best_id]] < best_dis) {
                    collide += 1;
                    continue;
                }
            }

            book[best_id] = int(distances.size());
            distances.push_back(best_dis);
            coord3ds.push_back(coord);
            img_pts.push_back(kpts[best_id].pt);
            map_pts.push_back(map_pt);
        }
    }

    log_debug_line(in_image << " projected inside image with " << proj_exceed
                            << " points distance too far, " << no_near
                            << " points no near point, " << collide
                            << " collided.");

    std::vector<ProjMatch> proj_matches;
    for (auto pair : book) {
        int frame_index = pair.first;
        int cur_index = pair.second;
        assert(cur_index < int(distances.size()));
        proj_matches.emplace_back(ProjMatch(frame_index, coord3ds[cur_index],
                                            img_pts[cur_index],
                                            map_pts[cur_index]));
    }
    log_debug_line("Total projected " << proj_matches.size() << " points.");
    return proj_matches;
}

std::vector<cv::DMatch> ORBMatcher::filter_match_by_dis(
        const std::vector<cv::DMatch> &matches, float soft_th, float hard_th,
        int cnt) {
    std::vector<bool> mask(matches.size(), true);
    int bin[256];
    memset(bin, 0, sizeof(bin));
    for (size_t i = 0; i < mask.size(); ++i) {
        bin[(int) std::floor(matches[i].distance)] += 1;
    }
    int total_cnt = 0;
    int soft_th_int = (int) soft_th;
    int hard_th_int = (int) hard_th;
    int final_th = soft_th_int;
    for (int i = 0; i < (int) mask.size(); ++i) {
        total_cnt += bin[i];
        if (i >= soft_th_int && total_cnt >= cnt) {
            final_th = int(i);
            break;
        } else if (i >= hard_th_int) {
            final_th = hard_th_int;
            break;
        }
    }
    for (size_t i = 0; i < mask.size(); ++i) {
        if (int(matches[i].distance) > final_th) { mask[i] = false; }
    }

    return filter_by_mask(matches, mask);
}

std::vector<cv::DMatch> ORBMatcher::match_descriptor_bf(const cv::Mat &o_descpt,
                                                        float soft_dis_th,
                                                        float hard_dis_th,
                                                        int expect_cnt) const {
    assert(hard_dis_th >= soft_dis_th);
    std::vector<cv::DMatch> matches;
    auto matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
    matcher.match(o_descpt, descriptors, matches);
    return filter_match_by_dis(matches, soft_dis_th, hard_dis_th, expect_cnt);
}

std::vector<cv::DMatch> ORBMatcher::filter_match_by_rotation_consistency(
        const std::vector<cv::DMatch> &matches,
        const std::vector<cv::KeyPoint> &kpts1,
        const std::vector<cv::KeyPoint> &kpts2, const int topK) {
    auto ang_diff_index = [](double diff_ang) {
        if (diff_ang < 0) { diff_ang += 360; }
        return (int) (diff_ang / 3.6);
    };
    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    Histogram<double> histo(101, ang_diff_index);
    std::vector<double> ang_diff;
    for (auto &match : matches) {
        double diff = kpts1[match.queryIdx].angle - kpts2[match.trainIdx].angle;
        histo.insert_element(diff);
        ang_diff.push_back(diff);

        pts1.push_back(kpts1[match.queryIdx].pt);
        pts2.push_back(kpts2[match.trainIdx].pt);
    }
    std::vector<bool> mask(matches.size(), true);
    histo.cal_topK(topK);
    for (int i = 0; i < (int) matches.size(); ++i) {
        if (!histo.is_topK(ang_diff[i])) { mask[i] = false; }
    }
    return filter_by_mask(matches, mask);
}

void ORBMatcher::filter_match_by_ess(const cv::Mat &Ess,
                                     const cv::Mat &camera_intrinsic,
                                     const std::vector<cv::Point2f> &pts1,
                                     const std::vector<cv::Point2f> &pts2,
                                     double th, std::vector<bool> &mask) {
    assert(pts1.size() == pts2.size());
    cv::Mat inv_cam_intrinsic = camera_intrinsic.inv();
    cv::Mat l_mat = Ess * inv_cam_intrinsic;
    mask.resize(pts1.size(), true);
    for (int i = 0; i < int(pts1.size()); ++i) {
        cv::Mat mat1 = cv::Mat::zeros(3, 1, CV_32F),
                mat2 = cv::Mat::zeros(3, 1, CV_32F);
        mat1.at<float>(0, 0) = pts1[i].x;
        mat1.at<float>(1, 0) = pts1[i].y;
        mat1.at<float>(2, 0) = 1.0f;
        mat2.at<float>(0, 0) = pts2[i].x;
        mat2.at<float>(1, 0) = pts2[i].y;
        mat2.at<float>(2, 0) = 1.0f;
        cv::Mat l = l_mat * mat1;
        cv::Mat pt2 = inv_cam_intrinsic * mat2;
        cv::Mat mat_res = pt2.t() * l;
        double diff = std::abs(double(mat_res.at<float>(0)));
        double l_norm = cv::norm(l);
        if (diff > th * l_norm) { mask[i] = false; }
    }
}
}// namespace vo_nono
