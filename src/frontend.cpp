#include "vo_nono/frontend.h"

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_set>
#include <vector>

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

int Frontend::match_with_keyframe(int match_cnt) {
    m_matches = m_matcher->match_descriptor_bf(m_keyframe->descriptor, 8, 30,
                                               match_cnt);
    log_debug_line("Original match: " << m_matches.size());
    std::vector<cv::KeyPoint> match_kpt1, match_kpt2;
    match_kpt1.reserve(m_matches.size());
    match_kpt2.reserve(m_matches.size());
    for (auto &match : m_matches) {
        match_kpt1.push_back(m_keyframe->kpts[match.queryIdx]);
        match_kpt2.push_back(m_curframe->kpts[match.trainIdx]);
    }
    std::vector<unsigned char> mask;
    m_matcher->filter_match_rotation_consistency(match_kpt1, match_kpt2, mask,
                                                 3);
    m_matches = filter_by_mask(m_matches, mask);
    m_matches_inlier = std::vector(m_matches.size(), true);
    return int(m_matches.size());
}

void Frontend::get_image(const cv::Mat &image, double t) {
    reset();
    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptor;
    detect_and_compute(image, kpts, descriptor, CNT_KEY_PTS);
    m_matcher = std::make_unique<ORBMatcher>(
            ORBMatcher(kpts, descriptor, m_camera));
    m_curframe =
            std::make_shared<Frame>(Frame::create_frame(descriptor, kpts, t));
    m_curframe->image = image;

    std::unique_lock<std::mutex> lock(m_map->map_global_mutex);
    bool b_succ = false;
    if (m_state == State::Start) {
        m_state = State::Initializing;
        m_keyframe = m_curframe;
        b_succ = true;
    } else if (m_state == State::Initializing) {
        int init_state = initialize(image, t);
        if (init_state == 0) {
            m_state = State::Tracking;
            m_map->insert_frame(m_keyframe);
            m_map->insert_frame(m_curframe);
            m_map->insert_key_frame(m_keyframe);
            b_succ = true;
        } else if (init_state == -1) {
            b_succ = false;
        } else if (init_state == -2) {
            b_succ = false;
        } else {
            unimplemented();
        }
    } else if (m_state == State::Tracking) {
        if (tracking(image, t)) {
            m_map->insert_frame(m_curframe);
            if (mb_new_key_frame) {
                m_map->insert_key_frame(m_curframe);
                m_keyframe = m_curframe;
            }
            b_succ = false;
        }
    } else {
        unimplemented();
    }

    if (b_succ) {
        m_motion_pred.inform_pose(m_curframe->get_Rcw(), m_curframe->get_Tcw(),
                                  t);
    }
}

int Frontend::initialize(const cv::Mat &image, double t) {
    m_matches = m_matcher->match_descriptor_bf(m_keyframe->descriptor, 8, 15,
                                               CNT_INIT_MATCHES);

    std::vector<cv::Point2f> matched_pt1, matched_pt2;
    for (auto &match : m_matches) {
        matched_pt1.push_back(m_keyframe->kpts[match.queryIdx].pt);
        matched_pt2.push_back(m_curframe->kpts[match.trainIdx].pt);
    }

    // todo: less than 8 matched points?
    // todo: findEssentialMat hyper parameters
    std::vector<unsigned char> mask;
    cv::Mat Ess;
    TIME_IT(Ess = cv::findEssentialMat(matched_pt1, matched_pt2,
                                       m_camera.get_intrinsic_mat(), cv::RANSAC,
                                       0.999, 0.5, mask),
            "Find essential mat cost ");
    // filter outliers
    m_matches = filter_by_mask(m_matches, mask);
    if (m_matches.size() < 50) { return -1; }
    matched_pt1.clear();
    matched_pt2.clear();
    for (auto &match : m_matches) {
        matched_pt1.push_back(m_keyframe->kpts[match.queryIdx].pt);
        matched_pt2.push_back(m_curframe->kpts[match.trainIdx].pt);
    }

    cv::Mat Rcw, tcw;
    cv::recoverPose(Ess, matched_pt1, matched_pt2, Rcw, tcw);
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);
    m_curframe->set_Rcw(Rcw);
    m_curframe->set_Tcw(tcw);

    // triangulate points
    std::vector<cv::Mat> triangulate_result;
    std::vector<bool> inliers;
    int cnt_new_pt = Triangulator::triangulate_and_filter_frames(
            m_keyframe.get(), m_curframe.get(), m_camera.get_intrinsic_mat(),
            m_matches, triangulate_result, inliers, 10000);
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
    m_curframe->set_Tcw(tcw);

    for (auto &tri_pt : triangulate_result) { tri_pt /= scale; }

    assert(triangulate_result.size() == m_matches.size());
    for (int i = 0; i < int(triangulate_result.size()); ++i) {
        if (inliers[i]) {
            auto new_pt = std::make_shared<MapPoint>(MapPoint::create_map_point(
                    triangulate_result[i],
                    m_keyframe->descriptor.row(m_matches[i].queryIdx)));
            m_keyframe->set_map_pt(m_matches[i].queryIdx, new_pt);
            m_curframe->set_map_pt(m_matches[i].trainIdx, new_pt);
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
    m_curframe->set_Rcw(motion_Rcw);
    m_curframe->set_Tcw(motion_Tcw);

    std::vector<bool> tri_inliers_keyframe;
    match_with_keyframe(CNT_MATCHES);
    m_matches_inlier = std::vector<bool>(m_matches.size(), true);

    bool b_track_good = false, b_keyframe_good = false;
    int cnt_keyframe_match = track_by_match_with_keyframe();
    int cnt_proj_match = 0;
    if (cnt_keyframe_match < CNT_MIN_MATCHES) {
        m_matcher->space_hash();
        cnt_proj_match = track_by_local_points();
    } else {
        b_keyframe_good = true;
    }
    if (std::max(cnt_proj_match, cnt_keyframe_match) >= CNT_MIN_MATCHES) {
        b_track_good = true;
    }

    if (b_track_good) { triangulate_with_keyframe(); }

    if (!b_keyframe_good && b_track_good) {
        if (double(cnt_keyframe_match) <
            0.2 * double(m_keyframe->get_cnt_map_pt())) {
            mb_new_key_frame = true;
        }
    }

    log_debug_line("Track good: " << b_track_good);
    log_debug_line("Keyframe good: " << b_keyframe_good);
    log_debug_line("Match " << cnt_keyframe_match << ". Project "
                            << cnt_proj_match << ". Set "
                            << m_curframe->get_cnt_map_pt() << " map points.");
    log_debug_line(m_curframe->get_cnt_map_pt()
                   << ":\n"
                   << m_curframe->get_Rcw() << std::endl
                   << m_curframe->get_Tcw() << std::endl);
    return b_track_good;
}

int Frontend::track_by_match_with_keyframe() {
    std::vector<cv::Matx31f> pt_coords;
    std::vector<cv::Point2f> img_pts;
    std::vector<cv::DMatch> old_match;
    std::unordered_map<int, int> origin_index;

    for (int i = 0; i < int(m_matches.size()); ++i) {
        if (m_keyframe->is_index_set(m_matches[i].queryIdx)) {
            int cur_old_index = int(old_match.size());
            origin_index[cur_old_index] = i;
            old_match.push_back(m_matches[i]);
            pt_coords.push_back(
                    m_keyframe->get_map_pt(m_matches[i].queryIdx)->get_coord());
            img_pts.push_back(m_curframe->kpts[m_matches[i].trainIdx].pt);
        }
    }
    log_debug_line(old_match.size()
                   << " old match. " << m_matches.size() - old_match.size()
                   << " new match.");

    if (old_match.size() < CNT_MIN_MATCHES) { return int(old_match.size()); }

    std::vector<bool> inliers;
    std::vector<cv::Matx31f> inlier_coords;
    std::vector<cv::Point2f> inlier_img_pts;
    cv::Mat Rcw = m_curframe->get_Rcw(), tcw = m_curframe->get_Tcw();
    pnp_ransac(pt_coords, img_pts, m_camera, 100, 10, Rcw, tcw, inliers,
               PNP_RANSAC::VO_NONO_PNP_RANSAC);
    assert(inliers.size() == pt_coords.size());
    assert(old_match.size() == pt_coords.size());
    int cnt_inlier = 0;
    std::vector<cv::DMatch> dmatches;
    for (int i = 0; i < (int) pt_coords.size(); ++i) {
        if (inliers[i] && !m_curframe->is_index_set(old_match[i].trainIdx)) {
            cnt_inlier += 1;
            auto p_point = m_keyframe->get_map_pt(old_match[i].queryIdx);
            m_curframe->set_map_pt(old_match[i].trainIdx, p_point);
            dmatches.push_back(old_match[i]);
        } else {
            m_matches_inlier[origin_index[i]] = false;
        }
    }
    log_debug_line(cnt_inlier << " inliers after pnp ransac");
    if (m_curframe->id >= 180) {
        //show_keyframe_curframe_match(dmatches, "Filtered match ");
    }

    if (cnt_inlier < CNT_MIN_MATCHES / 2) { return int(cnt_inlier); }

    for (int i = 0; i < (int) pt_coords.size(); ++i) {
        if (inliers[i]) {
            inlier_coords.push_back(pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
        }
    }
    pnp_optimize_proj_err(inlier_coords, inlier_img_pts, m_camera, Rcw, tcw);
    m_curframe->set_Rcw(Rcw);
    m_curframe->set_Tcw(tcw);
    return cnt_inlier;
}

int Frontend::track_by_local_points() {
    int cnt_proj_match = 0;
    std::vector<vo_ptr<MapPoint>> local_map_pts = m_map->get_local_map_points();
    std::unordered_set<vo_id_t> map_pt_set;
    std::vector<ProjMatch> proj_matches;
    std::vector<cv::Matx31f> map_pt_coords, inlier_coords;
    std::vector<cv::Point2f> img_pts, inlier_img_pts;
    std::vector<bool> is_inliers;
    cv::Mat Rcw = m_curframe->get_Rcw(), tcw = m_curframe->get_Tcw();
    m_matcher->set_estimate_pose(Rcw, tcw);

    TIME_IT(proj_matches = m_matcher->match_by_projection(local_map_pts, 5.0f),
            "projection match cost ");

    std::vector<cv::DMatch> dmatches;
    for (auto &proj_match : proj_matches) {
        map_pt_coords.push_back(proj_match.coord3d);
        img_pts.push_back(proj_match.img_pt);
        for (int i = 0; i < int(m_keyframe->kpts.size()); ++i) {
            if (m_keyframe->is_index_set(i) &&
                m_keyframe->get_map_pt(i)->get_id() ==
                        proj_match.p_map_pt->get_id()) {
                dmatches.emplace_back(cv::DMatch(i, proj_match.index, 10));
            }
        }
    }
    //show_keyframe_curframe_match(dmatches, "projection match ");

    if (map_pt_coords.size() < CNT_MIN_MATCHES) {
        return int(map_pt_coords.size());
    }

    TIME_IT(pnp_ransac(map_pt_coords, img_pts, m_camera, 100, 10, Rcw, tcw,
                       is_inliers, PNP_RANSAC::VO_NONO_PNP_RANSAC),
            "projection pnp cost ");

    assert(proj_matches.size() == is_inliers.size());
    for (int i = 0; i < int(is_inliers.size()); ++i) {
        if (is_inliers[i]) { cnt_proj_match += 1; }
    }

    log_debug_line("Projection ransac with " << proj_matches.size()
                                             << " points and " << cnt_proj_match
                                             << " matches.");

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
    //show_keyframe_curframe_match(dmatches, "Filtered projection ");

    if (cnt_proj_match < CNT_MIN_MATCHES) { return cnt_proj_match; }
    for (int i = 0; i < int(is_inliers.size()); ++i) {
        if (is_inliers[i] && !m_curframe->is_index_set(proj_matches[i].index)) {
            inlier_coords.push_back(map_pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
            m_curframe->set_map_pt(proj_matches[i].index,
                                   proj_matches[i].p_map_pt);
        }
    }
    pnp_optimize_proj_err(inlier_coords, inlier_img_pts, m_camera, Rcw, tcw);
    m_curframe->set_Rcw(Rcw);
    m_curframe->set_Tcw(tcw);
    log_debug_line("Pose estimate using "
                   << is_inliers.size() << " projection with " << cnt_proj_match
                   << " map points.");
    return cnt_proj_match;
}

int Frontend::triangulate_with_keyframe() {
    std::vector<bool> tri_inliers;
    std::vector<cv::Mat> tri_results;
    int cnt_succ = Triangulator::triangulate_and_filter_frames(
            m_keyframe.get(), m_curframe.get(), m_camera.get_intrinsic_mat(),
            m_matches, tri_results, tri_inliers, 10000);

    for (int i = 0; i < int(tri_inliers.size()); ++i) {
        if (tri_inliers[i]) {
            if (!m_keyframe->is_index_set(m_matches[i].queryIdx) &&
                !m_curframe->is_index_set(m_matches[i].trainIdx)) {
                cnt_succ -= 1;
                auto new_pt =
                        std::make_shared<MapPoint>(MapPoint::create_map_point(
                                tri_results[i],
                                m_keyframe->descriptor.row(
                                        m_matches[i].queryIdx)));
                m_keyframe->set_map_pt(m_matches[i].queryIdx, new_pt);
                m_curframe->set_map_pt(m_matches[i].trainIdx, new_pt);
            }
        }
    }
    return cnt_succ;
}

void Frontend::show_keyframe_curframe_match(
        const std::vector<cv::DMatch> &matches,
        const std::string &prefix) const {
    show_matches(m_keyframe->id, m_curframe->id, m_keyframe->image,
                 m_curframe->image, m_keyframe->kpts, m_curframe->kpts, matches,
                 prefix);
}

}// namespace vo_nono