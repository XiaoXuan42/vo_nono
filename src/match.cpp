#include "vo_nono/keypoint/match.h"

#include <iostream>

#include "vo_nono/geometry.h"
#include "vo_nono/util.h"

namespace vo_nono {
std::vector<ProjMatch> ORBMatcher::match_by_projection(
        const std::vector<vo_ptr<MapPoint>> &map_points, float dist_th) {
    assert(mb_space_hash);
    std::vector<cv::Matx31f> coord3ds;
    std::vector<cv::Point2f> img_pts;
    std::vector<vo_ptr<MapPoint>> map_pts;
    std::vector<double> distances;
    std::unordered_map<int, int> book;

    int proj_exceed = 0, in_image = 0, no_near = 0, collide = 0;
    cv::Mat proj_mat = get_proj_mat(m_camera_intrinsic, m_Rcw, m_tcw);
    for (auto &map_pt : map_points) {
        cv::Point2f proj_img_pt = hm2d_to_euclid2d(
                proj_mat * euclid3d_to_hm3d(map_pt->get_coord()));
        auto coord = cv::Matx31f(map_pt->get_coord());
        if (proj_img_pt.x >= 0 && proj_img_pt.x < m_total_width &&
            proj_img_pt.y >= 0 && proj_img_pt.y < m_total_height) {
            in_image += 1;

            int min_width_id =
                    get_grid_width_id(std::max(0.0f, proj_img_pt.x - dist_th));
            int max_width_id = get_grid_width_id(
                    std::min(m_total_width, proj_img_pt.x + dist_th));
            int min_height_id =
                    get_grid_height_id(std::max(0.0f, proj_img_pt.y - dist_th));
            int max_height_id = get_grid_height_id(
                    std::min(m_total_height, proj_img_pt.y + dist_th));

            int best_id = -1;
            int best_dis = std::numeric_limits<int>::max();
            for (int i = min_height_id; i <= max_height_id; ++i) {
                for (int j = min_width_id; j <= max_width_id; ++j) {
                    for (auto &level_grid : m_pyramid_grids) {
                        for (auto index : level_grid.grid[i][j]) {
                            cv::KeyPoint kpt = kpts[index];
                            if (std::fabs(kpt.pt.x - proj_img_pt.x) > dist_th ||
                                std::fabs(kpt.pt.y - proj_img_pt.y) > dist_th) {
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

namespace {
std::vector<cv::DMatch> _filter_match_by_dis_th(std::vector<cv::DMatch> matches,
                                                float soft_dis_th,
                                                float hard_dis_th,
                                                int expect_cnt) {
    std::vector<bool> mask(matches.size(), true);
    int bin[256];
    memset(bin, 0, sizeof(bin));
    for (size_t i = 0; i < mask.size(); ++i) {
        bin[(int) std::floor(matches[i].distance)] += 1;
    }
    int total_cnt = 0;
    int soft_th_int = (int) soft_dis_th;
    int hard_th_int = (int) hard_dis_th;
    int final_th = soft_th_int;
    for (int i = 0; i < (int) mask.size(); ++i) {
        total_cnt += bin[i];
        if (i >= soft_th_int && total_cnt >= expect_cnt) {
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

    matches = filter_by_mask(matches, mask);
    return matches;
}
}// namespace
std::vector<cv::DMatch> ORBMatcher::match_descriptor_bf(const cv::Mat &o_descpt,
                                                        float soft_dis_th,
                                                        float hard_dis_th,
                                                        int expect_cnt) const {
    assert(hard_dis_th >= soft_dis_th);
    std::vector<cv::DMatch> matches;
    auto matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
    matcher.match(o_descpt, descriptors, matches);
    return _filter_match_by_dis_th(std::move(matches), soft_dis_th, hard_dis_th,
                                   expect_cnt);
}

void ORBMatcher::filter_match_rotation_consistency(
        const std::vector<cv::KeyPoint> &kpts1,
        const std::vector<cv::KeyPoint> &kpts2,
        std::vector<unsigned char> &mask, const int topK) {
    assert(kpts1.size() == kpts2.size());
    auto ang_diff_index = [](double diff_ang) {
        if (diff_ang < 0) { diff_ang += 360; }
        return (int) (diff_ang / 3.6);
    };
    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(kpts1.size());
    pts2.reserve(kpts2.size());
    Histogram<double> histo(101, ang_diff_index);
    std::vector<double> ang_diff;
    for (int i = 0; i < (int) kpts1.size(); ++i) {
        double diff = kpts1[i].angle - kpts2[i].angle;
        histo.insert_element(diff);
        ang_diff.push_back(diff);

        pts1.push_back(kpts1[i].pt);
        pts2.push_back(kpts2[i].pt);
    }
    mask = std::vector<unsigned char>(pts1.size(), 1);
    histo.cal_topK(topK);
    for (int i = 0; i < (int) kpts1.size(); ++i) {
        if (!histo.is_topK(ang_diff[i])) { mask[i] = 0; }
    }
}
}// namespace vo_nono
