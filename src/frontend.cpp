#include "vo_nono/frontend.h"

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <unordered_set>
#include <vector>

#include "vo_nono/keypoint/epipolar.h"
#include "vo_nono/keypoint/triangulate.h"
#include "vo_nono/pnp.h"
#include "vo_nono/point.h"
#include "vo_nono/util/constants.h"
#include "vo_nono/util/debug.h"
#include "vo_nono/util/macro.h"
#include "vo_nono/util/util.h"

namespace vo_nono {
// detect feature points and compute descriptors
void Frontend::detect_and_compute(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &kpts,
                                  cv::Mat &dscpts, int nfeatures) {
    cv::Ptr orb_detector = cv::ORB::create(nfeatures, 1.2f);
    orb_detector->detectAndCompute(image, cv::Mat(), kpts, dscpts);
    // undistort keypoints
    std::vector<cv::Point2f> pts, res_pts;
    for (auto &kpt : kpts) { pts.push_back(kpt.pt); }
    cv::undistortPoints(pts, res_pts, camera_.get_intrinsic_mat(),
                        camera_.get_dist_coeff(), cv::noArray(),
                        camera_.get_intrinsic_mat());
    for (int i = 0; i < int(kpts.size()); ++i) { kpts[i].pt = res_pts[i]; }
}

void Frontend::get_image(const cv::Mat &image, double t) {
    reset_state();

    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptor;
    detect_and_compute(image, kpts, descriptor, CNT_KEYPTS);
    matcher_ =
            std::make_unique<ORBMatcher>(ORBMatcher(kpts, descriptor, camera_));
    curframe_ =
            std::make_shared<Frame>(Frame::create_frame(descriptor, kpts, t));
    curframe_->image = image;

    std::unique_lock<std::mutex> lock(map_->map_global_mutex);
    bool b_succ = false;
    if (state_ == State::Start) {
        state_ = State::Initializing;
        keyframe_ = curframe_;
        b_succ = true;
    } else if (state_ == State::Initializing) {
        int init_state = initialize();
        if (init_state == 0) {
            state_ = State::Tracking;
            map_->insert_frame(keyframe_);
            insert_local_frame(curframe_);
            b_succ = true;
        } else if (init_state == -1) {
            // not enough matches
            keyframe_ = curframe_;
        }
    } else if (state_ == State::Tracking) {
        if (tracking(image, t)) {
            insert_local_frame(curframe_);
            b_succ = true;
        }
        new_keyframe();
    } else {
        unimplemented();
    }

    if (b_succ) {
        motion_pred_.inform_pose(curframe_->get_Rcw(), curframe_->get_Tcw(), t);
    }
}

int Frontend::initialize() {
    keyframe_matches_ =
            matcher_->match_descriptor_bf(keyframe_->get_descriptors());
    init_matches_ = ORBMatcher::filter_match_by_dis(keyframe_matches_, 8, 32,
                                                    CNT_MATCHES);
    init_matches_ = ORBMatcher::filter_match_by_rotation_consistency(
            init_matches_, keyframe_->get_keypoints(),
            curframe_->get_keypoints(), 3);

    if (init_matches_.size() < 100) { return -1; }
    std::vector<cv::Point2f> matched_pt1, matched_pt2;
    for (auto &match : init_matches_) {
        matched_pt1.push_back(keyframe_->get_pixel_pt(match.queryIdx));
        matched_pt2.push_back(curframe_->get_pixel_pt(match.trainIdx));
    }
    std::vector<unsigned char> mask_ess;
    std::vector<unsigned char> mask_homo;
    cv::Mat Ess;
    cv::Mat H;
    Ess = cv::findEssentialMat(matched_pt1, matched_pt2,
                               camera_.get_intrinsic_mat(), cv::RANSAC, 0.999,
                               2.0, 1000, mask_ess);
    H = cv::findHomography(matched_pt1, matched_pt2, cv::RANSAC, 2.0,
                           mask_homo);
    auto ess_matches = filter_by_mask(init_matches_, mask_ess);
    auto h_matches = filter_by_mask(init_matches_, mask_homo);
    double e_score = _compute_ess_score(Ess, ess_matches);
    double h_score = _compute_h_score(H, h_matches);
    if (e_score > h_score) {
        init_matches_ = std::move(ess_matches);
    } else {
        init_matches_ = std::move(h_matches);
    }
    if (init_matches_.size() < 100) { return -1; }
    matched_pt1.clear();
    matched_pt2.clear();
    for (auto &match : init_matches_) {
        matched_pt1.push_back(keyframe_->get_pixel_pt(match.queryIdx));
        matched_pt2.push_back(curframe_->get_pixel_pt(match.trainIdx));
    }

    // recover pose
    cv::Mat Rcw, tcw;
    if (e_score > h_score) {
        // recover from ess
        cv::recoverPose(Ess, matched_pt1, matched_pt2,
                        camera_.get_intrinsic_mat(), Rcw, tcw);
    } else {
        // recover from homography
        std::vector<int> indices;
        std::vector<cv::Mat> rotations, translations, normals;
        cv::decomposeHomographyMat(H, camera_.get_intrinsic_mat(), rotations,
                                   translations, normals);
        cv::filterHomographyDecompByVisibleRefpoints(
                rotations, normals, matched_pt1, matched_pt2, indices);
        if (indices.size() > 1 || indices.empty()) {
            return -2;
        } else {
            Rcw = rotations[indices[0]];
            tcw = translations[indices[0]];
        }
    }
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);
    curframe_->set_Rcw(Rcw);
    curframe_->set_Tcw(tcw);

    // triangulate points
    std::vector<cv::Mat> triangulate_result;
    std::vector<bool> inliers;
    int cnt_new_pt = Triangulator::triangulate_and_filter_frames(
            keyframe_.get(), curframe_.get(), camera_.get_intrinsic_mat(),
            init_matches_, triangulate_result, inliers, 1000);
    if (cnt_new_pt < 40) { return -2; }

    double scale = 3;
    for (int i = 0; i < int(triangulate_result.size()); ++i) {
        if (inliers[i]) {
            assert(triangulate_result[i].at<float>(2) > 0);
            scale += triangulate_result[i].at<float>(2);
        }
    }
    scale /= cnt_new_pt;
    tcw /= scale;
    curframe_->set_Tcw(tcw);

    for (auto &tri : triangulate_result) { tri /= scale; }
    init_matches_ = filter_by_mask(init_matches_, inliers);
    triangulate_result = filter_by_mask(triangulate_result, inliers);
    _set_keyframe(keyframe_);
    _update_points_location(init_matches_, 1000);
    _associate_points(init_matches_, 2);
    return 0;
}

double Frontend::_compute_ess_score(const cv::Mat &ess,
                                    const std::vector<cv::DMatch> &matches) {
    double e_score = 0;
    cv::Mat cam_inv = camera_.get_intrinsic_mat().inv();
    cam_inv.convertTo(cam_inv, ess.type());
    cv::Mat fundamental1 = cam_inv.t() * ess * cam_inv;
    cv::Mat fundamental2 = cam_inv.t() * ess.t() * cam_inv;
    for (auto &match : matches) {
        double dis1 = Epipolar::epipolar_line_dis(
                fundamental1, keyframe_->get_pixel_pt(match.queryIdx),
                curframe_->get_pixel_pt(match.trainIdx));
        dis1 *= dis1;
        e_score += chi2_2_5 - dis1;
        double dis2 = Epipolar::epipolar_line_dis(
                fundamental2, curframe_->get_pixel_pt(match.trainIdx),
                keyframe_->get_pixel_pt(match.queryIdx));
        dis2 *= dis2;
        e_score += chi2_2_5 - dis2;
    }
    assert(!matches.empty());
    return e_score / double(matches.size());
}

double Frontend::_compute_h_score(const cv::Mat &H,
                                  const std::vector<cv::DMatch> &matches) {
    double h_score = 0;
    cv::Mat H_inv = H.inv();
    float h[3][3], h_inv[3][3];
    assert(H.type() == CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            h[i][j] = (float) H.at<double>(i, j);
            h_inv[i][j] = (float) H_inv.at<double>(i, j);
        }
    }

    for (auto &match : matches) {
        cv::Point2f pt1 = keyframe_->get_pixel_pt(match.queryIdx);
        cv::Point2f pt2 = curframe_->get_pixel_pt(match.trainIdx);
        float inv1_2 = 1.0f / (h[2][0] * pt1.x + h[2][1] * pt1.y + h[2][2]);
        float x1_2 = (h[0][0] * pt1.x + h[0][1] * pt1.y + h[0][2]) * inv1_2;
        float y1_2 = (h[1][0] * pt1.x + h[1][1] * pt1.y + h[1][2]) * inv1_2;
        double dis1 = (pt2.x - x1_2) * (pt2.x - x1_2) +
                      (pt2.y - y1_2) * (pt2.y - y1_2);
        h_score += chi2_2_5 - dis1;

        float inv2_1 = 1.0f / (h_inv[2][0] * pt2.x + h_inv[2][1] * pt2.y +
                               h_inv[2][2]);
        float x2_1 = (h_inv[0][0] * pt2.x + h_inv[0][1] * pt2.y + h_inv[0][2]) *
                     inv2_1;
        float y2_1 = (h_inv[1][0] * pt2.x + h_inv[1][1] * pt2.y + h_inv[1][2]) *
                     inv2_1;
        double dis2 = (pt1.x - x2_1) * (pt1.x - x2_1) +
                      (pt1.y - y2_1) * (pt1.y - y2_1);
        h_score += chi2_2_5 - dis2;
    }
    assert(!matches.empty());
    return h_score / double(matches.size());
}

bool Frontend::tracking(const cv::Mat &image, double t) {
    cv::Mat motion_Rcw, motion_Tcw;
    motion_pred_.predict_pose(t, motion_Rcw, motion_Tcw);
    curframe_->set_Rcw(motion_Rcw);
    curframe_->set_Tcw(motion_Tcw);

    keyframe_matches_ =
            matcher_->match_descriptor_bf(keyframe_->get_descriptors());
    tracking_with_keyframe();
    if (cnt_inlier_direct_match_ < CNT_TRACK_MIN_MATCHES) {
        //auto local_points = map_->get_local_map_points();
        //cnt_inlier_proj_match_ = track_by_projection(local_points, 20, 10);
    } else {
        // project to get more matches
        //project_keyframe();
        b_match_good_ = true;
    }
    if (std::max(cnt_inlier_proj_match_, cnt_inlier_direct_match_) >=
        CNT_TRACK_MIN_MATCHES) {
        b_track_good_ = true;
    }
    if (b_track_good_) { b_track_good_ = triangulate_and_set(direct_matches_); }

    log_debug_line("Track good: " << b_track_good_);
    log_debug_line("Match good: " << b_match_good_);
    log_debug_line("Set " << curframe_->get_cnt_map_pt() << " points:\n"
                          << curframe_->get_Rcw() << std::endl
                          << curframe_->get_Tcw() << std::endl);
    log_debug_line("Keyframe " << keyframe_->get_id() << " has "
                               << keyframe_->get_cnt_map_pt()
                               << " map points.");
    return b_track_good_;
}

void Frontend::tracking_with_keyframe() {
    direct_matches_ = ORBMatcher::filter_match_by_dis(keyframe_matches_, 8, 64,
                                                      CNT_MATCHES);
    direct_matches_ = ORBMatcher::filter_match_by_rotation_consistency(
            direct_matches_, keyframe_->get_keypoints(),
            curframe_->get_keypoints(), 3);
    direct_match_inliers_ = track_by_match(keyframe_, direct_matches_, 6);
    assert(direct_match_inliers_.size() == direct_matches_.size());
    auto inlier_matches =
            filter_by_mask(direct_matches_, direct_match_inliers_);
    for (auto &match : inlier_matches) {
        track_curframe_keyframe_[match.trainIdx] = match.queryIdx;
        track_keyframe_curframe_[match.queryIdx] = match.trainIdx;
    }
    cnt_inlier_direct_match_ = cnt_inliers_from_mask(direct_match_inliers_);
}

void Frontend::project_keyframe() {
    cv::Mat proj_mat =
            Geometry::get_proj_mat(camera_.get_intrinsic_mat(),
                                   curframe_->get_Rcw(), curframe_->get_Tcw());
    std::vector<cv::DMatch> extended_matches = _match_keyframe_by_proj(2);
    for (auto &pair : track_keyframe_curframe_) {
        extended_matches.emplace_back(pair.first, pair.second, 1.0);
    }
    cv::Mat Rcw = curframe_->get_Rcw().clone(),
            tcw = curframe_->get_Tcw().clone();
    auto mask = _pose_from_match_by_optimize(extended_matches, Rcw, tcw);
    curframe_->set_pose(Rcw, tcw);
    track_keyframe_curframe_.clear();
    track_curframe_keyframe_.clear();
    extended_matches = filter_by_mask(extended_matches, mask);
    for (auto &match : extended_matches) {
        track_keyframe_curframe_[match.queryIdx] = match.trainIdx;
        track_curframe_keyframe_[match.trainIdx] = match.queryIdx;
    }
}

std::vector<bool> Frontend::_pose_from_match_by_optimize(
        const std::vector<cv::DMatch> &matches, cv::Mat &Rcw, cv::Mat &tcw) {
    std::vector<bool> mask;
    std::vector<cv::Matx31f> pt_coords;
    std::vector<cv::Point2f> pixels;
    for (auto &match : matches) {
        assert(keyframe_->is_index_set(match.queryIdx));
        pt_coords.push_back(keyframe_->get_map_pt(match.queryIdx)->get_coord());
        pixels.push_back(curframe_->get_pixel_pt(match.trainIdx));
    }
    mask = PnP::pnp_by_optimize(pt_coords, pixels, camera_, Rcw, tcw);
    return mask;
}

void Frontend::relocalization() {
    // case1: not enough (true) matches between keyframe and curframe.
}

std::vector<bool> Frontend::track_by_match(
        const vo_ptr<Frame> &ref_frame, const std::vector<cv::DMatch> &matches,
        float ransac_th) {
    std::vector<cv::Matx31f> pt_coords;
    std::vector<cv::Point2f> img_pts;
    std::vector<cv::DMatch> old_matches;
    std::vector<bool> inliers1, inliers2;
    std::vector<bool> cur_mask;

    for (int i = 0; i < int(matches.size()); ++i) {
        if (ref_frame->is_index_set(matches[i].queryIdx)) {
            old_matches.push_back(matches[i]);
            pt_coords.push_back(
                    ref_frame->get_map_pt(matches[i].queryIdx)->get_coord());
            img_pts.push_back(curframe_->get_pixel_pt(matches[i].trainIdx));
            cur_mask.push_back(true);
        } else {
            cur_mask.push_back(false);
        }
    }
    if (old_matches.size() < 10) { return cur_mask; }

    log_debug_line("Total match: " << matches.size());
    log_debug_line("Old match: " << old_matches.size());
    int cnt_inlier = 0;
    cv::Mat Rcw = curframe_->get_Rcw(), tcw = curframe_->get_Tcw();
    PnP::pnp_ransac(pt_coords, img_pts, camera_, 100, ransac_th, Rcw, tcw,
                    inliers1);
    assert(inliers1.size() == old_matches.size());
    for (int i = 0; i < (int) old_matches.size(); ++i) {
        if (inliers1[i] && !curframe_->is_index_set(old_matches[i].trainIdx)) {
            cnt_inlier += 1;
        } else {
            inliers1[i] = false;
        }
    }
    cur_mask = mask_chaining(cur_mask, inliers1);
    if (cnt_inlier < 10) { return cur_mask; }

    old_matches = filter_by_mask(old_matches, inliers1);
    log_debug_line("Old match after pnp: " << old_matches.size());
    pt_coords = filter_by_mask(pt_coords, inliers1);
    img_pts = filter_by_mask(img_pts, inliers1);
    inliers2 = PnP::pnp_by_optimize(pt_coords, img_pts, camera_, Rcw, tcw);
    curframe_->set_Rcw(Rcw);
    curframe_->set_Tcw(tcw);
    log_debug_line("Match after optimize: " << cnt_inliers_from_mask(inliers2));
    cur_mask = mask_chaining(cur_mask, inliers2);
    assert(cnt_inliers_from_mask(cur_mask) == cnt_inliers_from_mask(inliers2));
    return cur_mask;
}

int Frontend::track_by_projection(const std::vector<vo_ptr<MapPoint>> &points,
                                  float r_th, float ransac_th) {
    int cnt_proj_match = 0;
    std::unordered_set<vo_id_t> map_pt_set;
    std::vector<ProjMatch> proj_matches;
    std::vector<cv::Matx31f> pt_coords, inlier_coords;
    std::vector<cv::Point2f> img_pts, inlier_img_pts;
    std::vector<int> inlier_proj_index;
    std::vector<bool> is_inliers, is_inliers2;
    cv::Mat Rcw = curframe_->get_Rcw(), tcw = curframe_->get_Tcw();
    matcher_->set_estimate_pose(Rcw, tcw);

    proj_matches = matcher_->match_by_projection(points, r_th);
    for (auto &proj_match : proj_matches) {
        pt_coords.push_back(proj_match.coord3d);
        img_pts.push_back(proj_match.img_pt);
    }
    PnP::pnp_ransac(pt_coords, img_pts, camera_, 100, ransac_th, Rcw, tcw,
                    is_inliers);

    assert(proj_matches.size() == is_inliers.size());
    for (int i = 0; i < int(is_inliers.size()); ++i) {
        if (is_inliers[i] && !curframe_->is_index_set(proj_matches[i].index)) {
            inlier_coords.push_back(pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
            inlier_proj_index.push_back(i);
            cnt_proj_match += 1;
        }
    }
    if (cnt_proj_match < 10) { return cnt_proj_match; }
    PnP::pnp_by_optimize(inlier_coords, inlier_img_pts, camera_, Rcw, tcw);

    cv::Mat proj_mat =
            Geometry::get_proj_mat(camera_.get_intrinsic_mat(), Rcw, tcw);
    is_inliers2.resize(inlier_coords.size());
    for (int i = 0; i < int(inlier_coords.size()); ++i) {
        cv::Mat mat_coord = cv::Mat(inlier_coords[i]);
        double err2 = Geometry::reprojection_err2(proj_mat, mat_coord,
                                                  inlier_img_pts[i]);
        if (err2 < chi2_2_5) {
            is_inliers2[i] = true;
        } else {
            is_inliers2[i] = false;
        }
    }

    cnt_proj_match = 0;
    for (int i = 0; i < int(is_inliers2.size()); ++i) {
        if (is_inliers2[i]) {
            int proj_index = inlier_proj_index[i];
            curframe_->set_map_pt(proj_matches[proj_index].index,
                                  proj_matches[proj_index].p_map_pt);
            cnt_proj_match += 1;
        }
    }
    curframe_->set_Rcw(Rcw);
    curframe_->set_Tcw(tcw);
    log_debug_line("Pose estimate using " << is_inliers2.size()
                                          << " projection with "
                                          << cnt_proj_match << " map points.");
    return cnt_proj_match;
}

void Frontend::new_keyframe() {
    bool is_keyframe = false;
    vo_ptr<Frame> candidate;
    for (auto iter = local_map_.local_frames.rbegin();
         iter != local_map_.local_frames.rend(); ++iter) {
        if (*iter == keyframe_) { break; }
        if ((*iter)->get_cnt_map_pt() >= 100) {
            candidate = *iter;
            break;
        }
    }
    if (!candidate || candidate == keyframe_) { return; }

    if (curframe_->get_cnt_map_pt() < cnt_inlier_direct_match_) {
        is_keyframe = true;
    } else if (curframe_->get_cnt_map_pt() <
               0.4 * keyframe_->get_cnt_map_pt()) {
        is_keyframe = true;
    } else if (!b_track_good_) {
        is_keyframe = true;
    }
    if (is_keyframe) {
        // project from map to get more matches
        vo_uptr<ORBMatcher> candidate_matcher = std::make_unique<ORBMatcher>(
                ORBMatcher(candidate->get_keypoints(),
                           candidate->get_descriptors(), camera_));
        candidate_matcher->set_estimate_pose(candidate->get_Rcw(),
                                             candidate->get_Tcw());
        auto local_points = map_->get_local_map_points();
        auto proj_matches =
                candidate_matcher->match_by_projection(local_points, 2);
        int proj_cnt = 0;
        for (auto &proj_match : proj_matches) {
            if (!candidate->is_index_set(proj_match.index)) {
                candidate->set_map_pt(proj_match.index, proj_match.p_map_pt);
                proj_cnt += 1;
            }
        }
        log_debug_line("Projected " << proj_cnt << " map points.");
        log_debug_line("Switch keyframe: " << candidate->get_id() << " set "
                                           << candidate->get_cnt_map_pt()
                                           << " points.");
        _set_keyframe(candidate);
    }
}

bool Frontend::triangulate_and_set(const std::vector<cv::DMatch> &matches) {
    std::vector<cv::DMatch> valid_matches = filter_match_cv(matches, 6);
    _update_points_location(valid_matches, 1000);
    _associate_points(valid_matches, 3);
    return true;
}

std::vector<cv::DMatch> Frontend::filter_match(
        const std::vector<cv::DMatch> &matches, double epi_th) {
    std::vector<cv::DMatch> valid_match;
    std::vector<bool> mask;
    std::vector<cv::Point2f> pts1, pts2;
    cv::Mat R21, t21;
    Geometry::relative_pose(keyframe_->get_Rcw(), keyframe_->get_Tcw(),
                            curframe_->get_Rcw(), curframe_->get_Tcw(), R21,
                            t21);
    cv::Mat ess = Epipolar::compute_essential(R21, t21);
    for (auto &match : matches) {
        pts1.push_back(keyframe_->feature_points[match.queryIdx]->keypoint.pt);
        pts2.push_back(curframe_->feature_points[match.trainIdx]->keypoint.pt);
    }
    ORBMatcher::filter_match_by_ess(ess, camera_.get_intrinsic_mat(), pts1,
                                    pts2, epi_th, mask);
    valid_match = filter_by_mask(matches, mask);
    return valid_match;
}

std::vector<cv::DMatch> Frontend::filter_match_cv(
        const std::vector<cv::DMatch> &matches, double epi_th) {
    std::vector<cv::Point2f> pt1, pt2;
    for (auto &match : matches) {
        pt1.push_back(keyframe_->get_pixel_pt(match.queryIdx));
        pt2.push_back(curframe_->get_pixel_pt(match.trainIdx));
    }
    std::vector<unsigned char> mask;
    cv::findFundamentalMat(pt1, pt2, cv::FM_RANSAC, epi_th, 0.999, 100, mask);
    return filter_by_mask(matches, mask);
}

void Frontend::_update_points_location(const std::vector<cv::DMatch> &matches,
                                       double tri_grad_th) {
    for (auto &match : matches) {
        int keyframe_index = match.queryIdx;
        int curframe_index = match.trainIdx;
        if (local_map_.own_points[keyframe_index]) {
            _add_observation(curframe_->get_Rcw(), curframe_->get_Tcw(),
                             curframe_->get_pixel_pt(curframe_index),
                             keyframe_index, tri_grad_th);
            if (keyframe_->is_index_set(keyframe_index)) {
                keyframe_->get_map_pt(keyframe_index)
                        ->set_coord(_get_local_map_point_coord(keyframe_index));
            }
        }
    }
}

void Frontend::_associate_points(const std::vector<cv::DMatch> &matches,
                                 int least_obs) {
    track_keyframe_curframe_.clear();
    track_curframe_keyframe_.clear();
    cv::Mat proj_mat =
            Geometry::get_proj_mat(camera_.get_intrinsic_mat(),
                                   curframe_->get_Rcw(), curframe_->get_Tcw());
    for (auto &match : matches) {
        int keyframe_index = match.queryIdx;
        int curframe_index = match.trainIdx;
        if (!keyframe_->is_index_set(keyframe_index)) {
            assert(local_map_.own_points[keyframe_index]);
            if ((int) local_map_.point_infos[keyframe_index]
                        .observations.size() >= least_obs) {
                cv::Mat coord = _get_local_map_point_coord(keyframe_index);
                auto new_pt =
                        std::make_shared<MapPoint>(MapPoint::create_map_point(
                                coord,
                                keyframe_->feature_points[keyframe_index]
                                        ->descriptor,
                                keyframe_->feature_points[keyframe_index]
                                        ->keypoint.octave));
                new_pt->associate_feature_point(
                        keyframe_->feature_points[keyframe_index]);
                keyframe_->set_map_pt(keyframe_index, new_pt);
            }
        }
        if (keyframe_->is_index_set(keyframe_index) &&
            !curframe_->is_index_set(curframe_index)) {
            auto pt = keyframe_->get_map_pt(keyframe_index);
            cv::Mat pt_coord_cam = Geometry::transform_coord(
                    curframe_->get_Rcw(), curframe_->get_Tcw(),
                    pt->get_coord());
            if (pt_coord_cam.at<float>(2) > 0) {
                cv::Point2f proj_pixel = Geometry::hm2d_to_euclid2d(
                        camera_.get_intrinsic_mat() * pt_coord_cam);
                cv::Point2f diff =
                        proj_pixel - curframe_->get_pixel_pt(curframe_index);
                double err2 = diff.x * diff.x + diff.y * diff.y;
                if (err2 < chi2_2_5) {
                    curframe_->set_map_pt(curframe_index, pt);
                    track_keyframe_curframe_[keyframe_index] = curframe_index;
                    track_curframe_keyframe_[curframe_index] = keyframe_index;
                }
            }
        }
    }
}

void Frontend::_set_keyframe(const vo_ptr<Frame> &keyframe) {
    local_map_.local_frames.clear();
    local_map_.point_infos.clear();
    local_map_.point_infos =
            std::vector<LocalMap::PointInfo>(keyframe->feature_points.size());
    local_map_.own_points = std::vector(keyframe->feature_points.size(), false);

    double basic_var = std::min(camera_.fx(), camera_.fy());
    basic_var = 1.0 / (basic_var * basic_var);
    for (int i = 0; i < (int) keyframe->feature_points.size(); ++i) {
        if (!keyframe->is_index_set(i)) {
            local_map_.own_points[i] = true;
            local_map_.point_infos[i].filter.set_information(
                    camera_, keyframe->get_pixel_pt(i), basic_var);
            _add_observation(keyframe->get_Rcw(), keyframe->get_Tcw(),
                             keyframe->get_pixel_pt(i), i, 1000);
        } else {
            keyframe->get_map_pt(i)->associate_feature_point(
                    keyframe->feature_points[i]);
        }
    }

    local_map_.local_frames.push_back(keyframe);
    keyframe_ = keyframe;
    map_->insert_keyframe(keyframe);
}

cv::Mat Frontend::_get_local_map_point_coord(int index) {
    assert(local_map_.own_points[index]);
    return local_map_.point_infos[index].coord;
}

bool Frontend::_add_observation(const cv::Mat &Rcw, const cv::Mat &tcw,
                                const cv::Point2f &pixel, int index,
                                double tri_grad_th) {
    if (local_map_.point_infos[index].observations.empty()) {
        local_map_.point_infos[index].observations.emplace_back(Rcw, tcw,
                                                                pixel);
        return true;
    }
    cv::Mat proj_mat1 =
            Geometry::get_proj_mat(camera_.get_intrinsic_mat(),
                                   keyframe_->get_Rcw(), keyframe_->get_Tcw());
    cv::Mat proj_mat2 =
            Geometry::get_proj_mat(camera_.get_intrinsic_mat(), Rcw, tcw);
    cv::Mat tri_res = Triangulator::triangulate(
            proj_mat1, proj_mat2, keyframe_->get_pixel_pt(index), pixel);

    local_map_.point_infos[index].filter.filter(keyframe_->get_Tcw(),
                                                keyframe_->get_Rcw(),
                                                curframe_->get_Tcw(), tri_res);

    bool ok = true;
    for (auto &obs : local_map_.point_infos[index].observations) {
        // parallax test
        if (!Triangulator::is_triangulate_inlier(obs.Rcw, obs.tcw, Rcw, tcw,
                                                 tri_res, tri_grad_th)) {
            ok = false;
            break;
        }
        // reprojection error test
        cv::Mat cur_proj_mat = Geometry::get_proj_mat(
                camera_.get_intrinsic_mat(), obs.Rcw, obs.tcw);
        double err2 =
                Geometry::reprojection_err2(cur_proj_mat, tri_res, obs.pixel);
        if (err2 > chi2_2_5) {
            ok = false;
            break;
        }
    }
    // update state
    if (ok) {
        local_map_.point_infos[index].observations.emplace_back(Rcw, tcw,
                                                                pixel);
        // update map points from multiple observations
        std::vector<cv::Mat> projs;
        std::vector<cv::Point2f> pixels;
        for (auto &obs : local_map_.point_infos[index].observations) {
            cv::Mat cur_proj_mat = Geometry::get_proj_mat(
                    camera_.get_intrinsic_mat(), obs.Rcw, obs.tcw);
            projs.push_back(cur_proj_mat);
            pixels.push_back(obs.pixel);
        }
        local_map_.point_infos[index].coord =
                Triangulator::triangulate(projs, pixels);
    }
    return ok;
}

std::vector<cv::DMatch> Frontend::_match_keyframe_by_proj(float r_th) {
    std::vector<cv::DMatch> new_matches;
    std::vector<bool> collision_mask;
    std::unordered_map<int, int> matched_index_book;
    for (int i = 0; i < int(keyframe_->feature_points.size()); ++i) {
        if (keyframe_->is_index_set(i)) {
            auto map_pt = keyframe_->get_map_pt(i);
            cv::Point2f pixel;
            if (!Geometry::project_euclid3d_in_front(
                        camera_.get_intrinsic_mat(), curframe_->get_Rcw(),
                        curframe_->get_Tcw(), map_pt->get_coord(), pixel)) {
                continue;
            }
            if (camera_.inside_image(pixel)) {
                int matched_index = matcher_->match_in_rec(
                        pixel, map_pt->get_desc(), r_th,
                        map_pt->get_pyramid_level(), 0.8, 64);
                if (matched_index >= 0 &&
                    !track_curframe_keyframe_.count(matched_index)) {
                    assert(!curframe_->is_index_set(matched_index));
                    if (!matched_index_book.count(matched_index)) {
                        new_matches.emplace_back(i, matched_index, 1);
                        matched_index_book[matched_index] =
                                int(collision_mask.size());
                        collision_mask.push_back(true);
                    } else {
                        collision_mask[matched_index_book[matched_index]] =
                                false;
                    }
                }
            }
        }
    }
    new_matches = filter_by_mask(new_matches, collision_mask);
    return new_matches;
}
}// namespace vo_nono