#include "vo_nono/keypoint/match.h"

#include <iostream>

#include "vo_nono/keypoint/epipolar.h"
#include "vo_nono/util/geometry.h"
#include "vo_nono/util/histogram.h"
#include "vo_nono/util/util.h"

namespace vo_nono {
int ORBMatcher::match_in_rec(const cv::Point2f &pixel, const cv::Mat &dscpt,
                             float r_th, int pyramid_level, float lowe,
                             float max_d) {
    if (!mb_space_hash) { space_hash(); }
    assert(pixel.x >= 0 && pixel.x < m_total_width && pixel.y >= 0 &&
           pixel.y < m_total_height);
    int min_width_id = get_grid_width_id(std::max(0.0f, pixel.x - r_th));
    int max_width_id =
            get_grid_width_id(std::min(m_total_width, pixel.x + r_th));
    int min_height_id = get_grid_height_id(std::max(0.0f, pixel.y - r_th));
    int max_height_id =
            get_grid_height_id(std::min(m_total_height, pixel.y + r_th));

    int best_id = -1;
    int best_dis = std::numeric_limits<int>::max();
    int second_dis = std::numeric_limits<int>::max();
    for (int i = min_height_id; i <= max_height_id; ++i) {
        for (int j = min_width_id; j <= max_width_id; ++j) {
            int min_pyramid_level = 0, max_pyramid_level = 0;
            // if pyramid level is negative, then search in all layer of pyramids
            if (pyramid_level < 0) {
                min_pyramid_level = 0;
                max_pyramid_level = int(m_pyramid_grids.size()) - 1;
            } else {
                min_pyramid_level = std::max(pyramid_level - 1, 0);
                max_pyramid_level = std::min(pyramid_level + 1,
                                             int(m_pyramid_grids.size()) - 1);
            }

            for (int k = min_pyramid_level; k <= max_pyramid_level; ++k) {
                auto &level_grid = m_pyramid_grids[k];
                for (auto index : level_grid.grid[i][j]) {
                    cv::KeyPoint kpt = kpts[index];
                    if (std::fabs(kpt.pt.x - pixel.x) > r_th ||
                        std::fabs(kpt.pt.y - pixel.y) > r_th) {
                        continue;
                    }
                    int cur_dis = orb_distance(descriptors.row(index), dscpt);
                    if (cur_dis < best_dis) {
                        best_id = index;
                        second_dis = best_dis;
                        best_dis = cur_dis;
                    }
                }
            }
        }
    }
    if (double(best_dis) > double(max_d)) {
        return -1;
    } else if (double(best_dis) > double(second_dis) * double(lowe)) {
        return -1;
    }
    return best_id;
}

std::vector<ProjMatch> ORBMatcher::match_by_projection(
        const std::vector<vo_ptr<MapPoint>> &map_points, float r_th) {
    if (!mb_space_hash) { space_hash(); }
    std::vector<cv::Matx31f> coord3ds;
    std::vector<cv::Point2f> img_pts;
    std::vector<vo_ptr<MapPoint>> map_pts;
    std::vector<double> distances;
    std::unordered_map<int, int> book;

    cv::Mat proj_mat = Geometry::get_proj_mat(m_camera_intrinsic, m_Rcw, m_tcw);
    for (auto &map_pt : map_points) {
        // in front of camera
        cv::Mat coord_cam =
                Geometry::transform_coord(m_Rcw, m_tcw, map_pt->get_coord());
        if (coord_cam.at<float>(2) < 0) { continue; }

        cv::Point2f proj_pixel =
                Geometry::hm2d_to_euclid2d(m_camera_intrinsic * coord_cam);
        if (proj_pixel.x < 0 || proj_pixel.x >= m_total_width ||
            proj_pixel.y < 0 || proj_pixel.y >= m_total_height) {
            continue;
        }

        auto coord = cv::Matx31f(map_pt->get_coord());
        int best_id = match_in_rec(proj_pixel, map_pt->get_desc(), r_th,
                                   map_pt->get_pyramid_level(), 0.8, 64);
        if (best_id >= 0) {
            int best_dis =
                    orb_distance(map_pt->get_desc(), descriptors.row(best_id));
            if (book.count(best_id)) {
                assert(book[best_id] < int(distances.size()));
                if (distances[book[best_id]] < best_dis) { continue; }
            }

            book[best_id] = int(distances.size());
            distances.push_back(best_dis);
            coord3ds.push_back(coord);
            img_pts.push_back(kpts[best_id].pt);
            map_pts.push_back(map_pt);
        }
    }
    std::vector<ProjMatch> proj_matches;
    for (auto pair : book) {
        int frame_index = pair.first;
        int cur_index = pair.second;
        assert(cur_index < int(distances.size()));
        proj_matches.emplace_back(ProjMatch(frame_index, coord3ds[cur_index],
                                            img_pts[cur_index],
                                            map_pts[cur_index]));
    }
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

std::vector<cv::DMatch> ORBMatcher::match_descriptor_bf(
        const cv::Mat &o_descpt) const {
    std::vector<cv::DMatch> matches;
    auto matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
    matcher.match(o_descpt, descriptors, matches);
    return matches;
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
    mask.resize(pts1.size(), true);
    for (int i = 0; i < int(pts1.size()); ++i) {
        double dis = Epipolar::epipolar_line_dis(camera_intrinsic, Ess, pts1[i],
                                                 pts2[i]);
        if (dis > th) { mask[i] = false; }
    }
}
}// namespace vo_nono
