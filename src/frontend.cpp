#include "vo_nono/frontend.h"

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_set>
#include <vector>

#include "vo_nono/keypoint/epipolar.h"
#include "vo_nono/keypoint/triangulate.h"
#include "vo_nono/pnp.h"
#include "vo_nono/point.h"
#include "vo_nono/util.h"

namespace vo_nono {
namespace {
[[maybe_unused]] void show_matches(vo_id_t left_id, vo_id_t right_id,
                                   const cv::Mat &img1, const cv::Mat &img2,
                                   const std::vector<cv::KeyPoint> &kpts1,
                                   const std::vector<cv::KeyPoint> &kpts2,
                                   const std::vector<cv::DMatch> &matches,
                                   const std::string &prefix) {
    cv::Mat outimg;
    std::string title = prefix + " " + std::to_string(left_id) + " match " +
                        std::to_string(right_id);
    cv::drawMatches(img1, kpts1, img2, kpts2, matches, outimg);
    cv::imshow(title, outimg);
    cv::waitKey(0);
}

[[maybe_unused]] void show_image(const cv::Mat &image,
                                 const std::string &title) {
    cv::imshow(title, image);
    cv::waitKey(0);
}
}// namespace

// detect feature points and compute descriptors
void Frontend::detect_and_compute(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &kpts,
                                  cv::Mat &dscpts, int nfeatures) {
    cv::Ptr orb_detector = cv::ORB::create(nfeatures, 1.2f);
    orb_detector->detectAndCompute(image, cv::Mat(), kpts, dscpts);
}

std::vector<cv::DMatch> Frontend::match_frame(const vo_ptr<Frame> &ref_frame,
                                              int match_cnt) {
    std::vector<cv::DMatch> matches;
    matches = m_matcher->match_descriptor_bf(ref_frame->descriptor, 8, 30,
                                             match_cnt);

    std::vector<cv::KeyPoint> match_kpt1, match_kpt2;
    match_kpt1.reserve(matches.size());
    match_kpt2.reserve(matches.size());
    for (auto &match : matches) {
        match_kpt1.push_back(ref_frame->kpts[match.queryIdx]);
        match_kpt2.push_back(m_cur_frame->kpts[match.trainIdx]);
    }
    std::vector<unsigned char> mask;
    m_matcher->filter_match_by_rotation_consistency(match_kpt1, match_kpt2,
                                                    mask, 3);
    matches = filter_by_mask(matches, mask);
    return matches;
}

void Frontend::get_image(const cv::Mat &image, double t) {
    reset_state();

    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptor;
    detect_and_compute(image, kpts, descriptor, CNT_KEY_PTS);
    m_matcher = std::make_unique<ORBMatcher>(
            ORBMatcher(kpts, descriptor, m_camera));
    m_cur_frame =
            std::make_shared<Frame>(Frame::create_frame(descriptor, kpts, t));
    m_cur_frame->image = image;

    std::unique_lock<std::mutex> lock(m_map->map_global_mutex);
    bool b_succ = false;
    if (m_state == State::Start) {
        m_state = State::Initializing;
        m_keyframe = m_cur_frame;
        b_succ = true;
    } else if (m_state == State::Initializing) {
        int init_state = initialize(image);
        if (init_state == 0) {
            m_state = State::Tracking;
            m_map->insert_frame(m_keyframe);
            m_map->insert_frame(m_cur_frame);
            m_map->insert_key_frame(m_keyframe);
            b_succ = true;
        } else if (init_state == -1) {
            // not enough matches
            m_keyframe = m_cur_frame;
        }
    } else if (m_state == State::Tracking) {
        if (tracking(image, t)) {
            m_map->insert_frame(m_cur_frame);
            if (mb_new_key_frame) {
                m_map->insert_key_frame(m_cur_frame);
                m_keyframe = m_cur_frame;
                mb_new_key_frame = false;
            }
            b_succ = true;
        }
    } else {
        unimplemented();
    }

    if (b_succ) {
        m_prev_frame = m_cur_frame;
        m_motion_pred.inform_pose(m_cur_frame->get_Rcw(),
                                  m_cur_frame->get_Tcw(), t);
    }
}

int Frontend::initialize(const cv::Mat &image) {
    std::vector<cv::DMatch> matches;
    matches = m_matcher->match_descriptor_bf(m_keyframe->descriptor, 8, 15,
                                             CNT_INIT_MATCHES);

    if (matches.size() < 10) { return -1; }
    std::vector<cv::Point2f> matched_pt1, matched_pt2;
    for (auto &match : matches) {
        matched_pt1.push_back(m_keyframe->kpts[match.queryIdx].pt);
        matched_pt2.push_back(m_cur_frame->kpts[match.trainIdx].pt);
    }
    std::vector<unsigned char> mask;
    cv::Mat Ess;
    TIME_IT(Ess = cv::findEssentialMat(matched_pt1, matched_pt2,
                                       m_camera.get_intrinsic_mat(), cv::RANSAC,
                                       0.999, 1.0, 1000, mask),
            "Find essential mat cost ");
    // filter outliers
    matches = filter_by_mask(matches, mask);
    if (matches.size() < 50) { return -1; }
    matched_pt1.clear();
    matched_pt2.clear();
    for (auto &match : matches) {
        matched_pt1.push_back(m_keyframe->kpts[match.queryIdx].pt);
        matched_pt2.push_back(m_cur_frame->kpts[match.trainIdx].pt);
    }

    cv::Mat Rcw, tcw;
    cv::recoverPose(Ess, matched_pt1, matched_pt2, m_camera.get_intrinsic_mat(),
                    Rcw, tcw);
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);
    m_cur_frame->set_Rcw(Rcw);
    m_cur_frame->set_Tcw(tcw);

    // triangulate points
    std::vector<cv::Mat> triangulate_result;
    std::vector<bool> inliers;
    int cnt_new_pt = Triangulator::triangulate_and_filter_frames(
            m_keyframe.get(), m_cur_frame.get(), m_camera.get_intrinsic_mat(),
            matches, triangulate_result, inliers, 1000);
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
    m_cur_frame->set_Tcw(tcw);

    for (auto &tri_pt : triangulate_result) { tri_pt /= scale; }

    assert(triangulate_result.size() == matches.size());
    for (int i = 0; i < int(triangulate_result.size()); ++i) {
        if (inliers[i]) {
            auto new_pt = std::make_shared<MapPoint>(MapPoint::create_map_point(
                    triangulate_result[i],
                    m_keyframe->descriptor.row(matches[i].queryIdx)));
            m_keyframe->set_map_pt(matches[i].queryIdx, new_pt);
            m_cur_frame->set_map_pt(matches[i].trainIdx, new_pt);
        }
    }

    log_debug_line("Initialize with R: " << std::endl
                                         << Rcw << std::endl
                                         << "T: " << std::endl
                                         << tcw << std::endl
                                         << cnt_new_pt << " new map points.");
    return 0;
}

bool Frontend::tracking(const cv::Mat &image, double t) {
    cv::Mat motion_Rcw, motion_Tcw;
    m_motion_pred.predict_pose(t, motion_Rcw, motion_Tcw);
    m_cur_frame->set_Rcw(motion_Rcw);
    m_cur_frame->set_Tcw(motion_Tcw);

    m_keyframe_matches = match_frame(m_keyframe, CNT_MATCHES);
    int cnt_match = track_by_match(m_keyframe, m_keyframe_matches, 6);
    int cnt_proj_match = 0;
    if (cnt_match < CNT_MIN_MATCHES) {
        cnt_proj_match = track_by_projection_local_map();
    } else {
        mb_match_good = true;
    }
    if (std::max(cnt_proj_match, cnt_match) >= CNT_MIN_MATCHES) {
        mb_track_good = true;
    }

    if (!mb_match_good && mb_track_good) {
        if (double(cnt_match) < 0.2 * double(m_keyframe->get_cnt_map_pt())) {
            mb_new_key_frame = true;
        }
    }

    if (mb_track_good) { triangulate_with_keyframe(); }

    log_debug_line("Track good: " << mb_track_good);
    log_debug_line("Match good: " << mb_match_good);
    log_debug_line("Match " << cnt_match << ". Project " << cnt_proj_match
                            << ". Set " << m_cur_frame->get_cnt_map_pt()
                            << " map points.");
    log_debug_line(m_cur_frame->get_cnt_map_pt()
                   << ":\n"
                   << m_cur_frame->get_Rcw() << std::endl
                   << m_cur_frame->get_Tcw() << std::endl);
    return mb_track_good;
}

int Frontend::track_by_match(const vo_ptr<Frame> &ref_frame,
                             const std::vector<cv::DMatch> &matches,
                             float ransac_th) {
    std::vector<cv::Matx31f> pt_coords;
    std::vector<cv::Point2f> img_pts;
    std::vector<cv::DMatch> old_matches;
    std::unordered_map<int, int> origin_index;

    for (int i = 0; i < int(matches.size()); ++i) {
        if (ref_frame->is_index_set(matches[i].queryIdx)) {
            int cur_old_index = int(matches.size());
            origin_index[cur_old_index] = i;
            old_matches.push_back(matches[i]);
            pt_coords.push_back(
                    ref_frame->get_map_pt(matches[i].queryIdx)->get_coord());
            img_pts.push_back(m_cur_frame->kpts[matches[i].trainIdx].pt);
        }
    }
    log_debug_line("Match with frame " << ref_frame->id << " "
                                       << old_matches.size() << " old match. "
                                       << matches.size() - old_matches.size()
                                       << " new match.");

    if (old_matches.size() < CNT_MIN_MATCHES) {
        return int(old_matches.size());
    }

    int cnt_inlier = 0;
    std::vector<bool> inliers;
    std::vector<cv::Matx31f> inlier_coords;
    std::vector<cv::Point2f> inlier_img_pts;
    cv::Mat Rcw = m_cur_frame->get_Rcw(), tcw = m_cur_frame->get_Tcw();
    PnP::pnp_ransac(pt_coords, img_pts, m_camera, 100, ransac_th, Rcw, tcw,
                    inliers);
    assert(inliers.size() == pt_coords.size());
    assert(old_matches.size() == pt_coords.size());
    for (int i = 0; i < (int) pt_coords.size(); ++i) {
        if (inliers[i] && !m_cur_frame->is_index_set(old_matches[i].trainIdx)) {
            cnt_inlier += 1;
            auto p_point = ref_frame->get_map_pt(old_matches[i].queryIdx);
            m_cur_frame->set_map_pt(old_matches[i].trainIdx, p_point);
        } else {
            inliers[i] = false;
        }
    }
    log_debug_line(cnt_inlier << " inliers after pnp ransac");
    if (cnt_inlier < 10) { return int(cnt_inlier); }

    for (int i = 0; i < (int) pt_coords.size(); ++i) {
        if (inliers[i]) {
            inlier_coords.push_back(pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
        }
    }
    PnP::pnp_by_optimize(inlier_coords, inlier_img_pts, m_camera, Rcw, tcw);
    m_cur_frame->set_Rcw(Rcw);
    m_cur_frame->set_Tcw(tcw);
    return cnt_inlier;
}

int Frontend::track_by_projection(const std::vector<vo_ptr<MapPoint>> &points,
                                  float r_th, float ransac_th) {
    int cnt_proj_match = 0;
    std::unordered_set<vo_id_t> map_pt_set;
    std::vector<ProjMatch> proj_matches;
    std::vector<cv::Matx31f> map_pt_coords, inlier_coords;
    std::vector<cv::Point2f> img_pts, inlier_img_pts;
    std::vector<bool> is_inliers;
    cv::Mat Rcw = m_cur_frame->get_Rcw(), tcw = m_cur_frame->get_Tcw();
    m_matcher->set_estimate_pose(Rcw, tcw);

    TIME_IT(proj_matches = m_matcher->match_by_projection(points, r_th),
            "projection match cost ");

    for (auto &proj_match : proj_matches) {
        map_pt_coords.push_back(proj_match.coord3d);
        img_pts.push_back(proj_match.img_pt);
    }

    if (map_pt_coords.size() < CNT_MIN_MATCHES) {
        return int(map_pt_coords.size());
    }

    TIME_IT(PnP::pnp_ransac(map_pt_coords, img_pts, m_camera, 100, ransac_th,
                            Rcw, tcw, is_inliers),
            "projection pnp cost ");

    assert(proj_matches.size() == is_inliers.size());
    for (int i = 0; i < int(is_inliers.size()); ++i) {
        if (is_inliers[i]) { cnt_proj_match += 1; }
    }

    log_debug_line("Projection ransac with " << proj_matches.size()
                                             << " points and " << cnt_proj_match
                                             << " matches.");
    /*

    if (m_cur_frame->id >= 180) {
        std::vector<cv::DMatch> dmatches;
        dmatches.clear();
        for (int i = 0; i < int(is_inliers.size()); ++i) {
            if (is_inliers[i]) {
                for (int j = 0; j < int(m_keyframe->kpts.size()); ++j) {
                    if (m_keyframe->is_index_set(j) &&
                        m_keyframe->get_map_pt(j)->get_id() ==
                                proj_matches[i].p_map_pt->get_id()) {
                        dmatches.emplace_back(
                                cv::DMatch(j, proj_matches[i].index, 10));
                    }
                }
            }
        }
        show_cur_frame_match(m_keyframe, dmatches, "Filtered projection ");
    }
*/

    if (cnt_proj_match < CNT_MIN_MATCHES) { return cnt_proj_match; }
    for (int i = 0; i < int(is_inliers.size()); ++i) {
        if (is_inliers[i] &&
            !m_cur_frame->is_index_set(proj_matches[i].index)) {
            inlier_coords.push_back(map_pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
            m_cur_frame->set_map_pt(proj_matches[i].index,
                                    proj_matches[i].p_map_pt);
        }
    }
    PnP::pnp_by_optimize(inlier_coords, inlier_img_pts, m_camera, Rcw, tcw);
    m_cur_frame->set_Rcw(Rcw);
    m_cur_frame->set_Tcw(tcw);
    log_debug_line("Pose estimate using "
                   << is_inliers.size() << " projection with " << cnt_proj_match
                   << " map points.");
    return cnt_proj_match;
}

int Frontend::track_by_projection_frame(const vo_ptr<Frame> &ref_frame) {
    std::vector<vo_ptr<MapPoint>> points = ref_frame->get_all_map_pts();
    int proj_res = track_by_projection(points, 30, 2);
    return proj_res;
}

int Frontend::track_by_projection_local_map() {
    std::vector<vo_ptr<MapPoint>> local_pts = m_map->get_local_map_points();
    return track_by_projection(local_pts, 20, 15);
}

int Frontend::triangulate_with_keyframe() {
    std::vector<cv::DMatch> valid_match;
    std::vector<bool> mask;
    std::vector<cv::Point2f> pts1, pts2;
    cv::Mat R21, t21;
    relative_pose(m_keyframe->get_Rcw(), m_keyframe->get_Tcw(),
                  m_cur_frame->get_Rcw(), m_cur_frame->get_Tcw(), R21, t21);
    cv::Mat ess = Epipolar::compute_essential(R21, t21);
    for (auto &match : m_keyframe_matches) {
        pts1.push_back(m_keyframe->kpts[match.queryIdx].pt);
        pts2.push_back(m_cur_frame->kpts[match.trainIdx].pt);
    }
    ORBMatcher::filter_match_by_ess(ess, m_camera.get_intrinsic_mat(), pts1,
                                    pts2, 0.01, mask);
    valid_match = filter_by_mask(m_keyframe_matches, mask);
    log_debug_line("Before ess check: " << m_keyframe_matches.size()
                                        << " after: " << valid_match.size());

    std::vector<bool> tri_inliers;
    std::vector<cv::Mat> tri_results;
    Triangulator::triangulate_and_filter_frames(
            m_keyframe.get(), m_cur_frame.get(), m_camera.get_intrinsic_mat(),
            valid_match, tri_results, tri_inliers, 1000);

    assert(valid_match.size() == tri_inliers.size());
    int cnt_succ = 0;
    for (int i = 0; i < int(tri_inliers.size()); ++i) {
        if (tri_inliers[i]) {
            if (!m_cur_frame->is_index_set(valid_match[i].trainIdx)) {
                vo_ptr<MapPoint> target_pt;
                if (m_keyframe->is_index_set(valid_match[i].queryIdx)) {
                    target_pt = m_keyframe->get_map_pt(valid_match[i].queryIdx);
                } else {
                    target_pt = std::make_shared<MapPoint>(
                            MapPoint::create_map_point(
                                    tri_results[i],
                                    m_keyframe->descriptor.row(
                                            valid_match[i].queryIdx)));
                    m_keyframe->set_map_pt(valid_match[i].queryIdx, target_pt);
                    cnt_succ += 1;
                }
                m_cur_frame->set_map_pt(valid_match[i].trainIdx, target_pt);
            }
        }
    }
    log_debug_line("Triangulated " << cnt_succ << " points.");
    return cnt_succ;
}

void Frontend::show_cur_frame_match(const vo_ptr<Frame> &ref_frame,
                                    const std::vector<cv::DMatch> &matches,
                                    const std::string &prefix) const {
    show_matches(ref_frame->id, m_cur_frame->id, ref_frame->image,
                 m_cur_frame->image, ref_frame->kpts, m_cur_frame->kpts,
                 matches, prefix);
}

}// namespace vo_nono