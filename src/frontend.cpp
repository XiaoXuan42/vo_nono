#include "vo_nono/frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vo_nono/pnp.h"
#include "vo_nono/point.h"
#include "vo_nono/util.h"

namespace vo_nono {
namespace {
[[maybe_unused]] void show_matches(const vo_ptr<Frame> &left_frame,
                                   const vo_ptr<Frame> &right_frame,
                                   const std::vector<cv::DMatch> &matches) {
    cv::Mat outimg;
    std::string title = std::to_string(left_frame->get_id()) + " match " +
                        std::to_string(right_frame->get_id());
    cv::drawMatches(left_frame->img, left_frame->get_kpts(), right_frame->img,
                    right_frame->get_kpts(), matches, outimg);
    cv::imshow(title, outimg);
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

std::vector<cv::DMatch> Frontend::filter_matches(
        const std::vector<cv::DMatch> &matches,
        const std::vector<cv::KeyPoint> &kpt1,
        const std::vector<cv::KeyPoint> &kpt2, int topK) {
    std::vector<cv::KeyPoint> match_kp1, match_kp2;
    for (auto &match : matches) {
        match_kp1.push_back(kpt1[match.queryIdx]);
        match_kp2.push_back(kpt2[match.trainIdx]);
    }

    std::vector<unsigned char> mask;
    filter_match_with_kpts(match_kp1, match_kp2, mask, topK);
    return filter_by_mask(matches, mask);
}

int Frontend::filter_triangulate_points(cv::Mat &tri, const cv::Mat &Rcw1,
                                        const cv::Mat &tcw1,
                                        const cv::Mat &Rcw2,
                                        const cv::Mat &tcw2,
                                        const std::vector<cv::Point2f> &pts1,
                                        const std::vector<cv::Point2f> &pts2,
                                        std::vector<bool> &inliers,
                                        double grad_th) {
    assert(tri.cols == (int) pts1.size());
    assert(tri.cols == (int) pts2.size());
    int cnt_inlier = 0;
    const int total_pts = tri.cols;
    inliers.resize(total_pts);
    for (int i = 0; i < total_pts; ++i) {
        tri.col(i) /= tri.at<float>(3, i);
        if (!std::isfinite(tri.at<float>(0, i)) ||
            !std::isfinite(tri.at<float>(1, i)) ||
            !std::isfinite(tri.at<float>(2, i))) {
            inliers[i] = false;
            continue;
        }
        cv::Mat coord = tri.col(i).rowRange(0, 3);
        // depth must be positive
        cv::Mat coord_c1 = Rcw1 * coord + tcw1;
        cv::Mat coord_c2 = Rcw2 * coord + tcw2;
        if (coord_c1.at<float>(2, 0) < EPS || coord_c2.at<float>(2, 0) < EPS) {
            inliers[i] = false;
            continue;
        }
        // compute parallax
        cv::Mat op1 = (coord + tcw1);// (coord - (-tcw1) = coord + tcw1)
        op1 /= cv::norm(op1);
        cv::Mat op2 = coord + tcw2;
        op2 /= cv::norm(op2);
        double sin_theta3 = cv::norm(op1.cross(op2));
        if (grad_th * sin_theta3 * sin_theta3 < 1.0) {
            inliers[i] = false;
            continue;
        }
        inliers[i] = true;
        cnt_inlier += 1;
    }
    return cnt_inlier;
}

void Frontend::filter_match_with_kpts(const std::vector<cv::KeyPoint> &kpts1,
                                      const std::vector<cv::KeyPoint> &kpts2,
                                      std::vector<unsigned char> &mask,
                                      const int topK) {
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

int Frontend::match_between_frames(const vo_ptr<Frame> &ref_frame,
                                   std::vector<cv::DMatch> &matches,
                                   int match_cnt) {
    matches = m_matcher->match_descriptor_bf(ref_frame->get_descs(), 8, 30,
                                             match_cnt);
    std::vector<cv::KeyPoint> match_kpt1, match_kpt2;
    match_kpt1.reserve(matches.size());
    match_kpt2.reserve(matches.size());
    for (auto &match : matches) {
        match_kpt1.push_back(ref_frame->get_kpt_by_index(match.queryIdx));
        match_kpt2.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx));
    }
    std::vector<unsigned char> mask;
    filter_match_with_kpts(match_kpt1, match_kpt2, mask, 3);
    matches = filter_by_mask(matches, mask);
    return int(matches.size());
}

void Frontend::get_image(const cv::Mat &image, double t) {
    bool ok = false;
    m_last_frame = m_cur_frame;

    std::vector<cv::KeyPoint> kpts;
    cv::Mat dscpts;
    detect_and_compute(image, kpts, dscpts, CNT_KEY_PTS);
    m_cur_frame = std::make_shared<Frame>(Frame::create_frame(dscpts, kpts, t));
    m_cur_frame->img = image;
    m_matcher = std::make_shared<ORBMatcher>(ORBMatcher(m_cur_frame, m_camera));

    if (m_state == State::Start) {
        assert(!m_keyframe);
        m_keyframe = m_cur_frame;
        m_state = State::Initializing;
        ok = true;
    } else if (m_state == State::Initializing) {
        assert(m_keyframe);
        int init_state = initialize(image, t);

        if (init_state == 0) {
            m_state = State::Tracking;
            ok = true;
        } else if (init_state == -1) {
            m_keyframe = m_cur_frame;
            ok = false;
        } else if (init_state == -2) {
            ok = false;
        } else {
            unimplemented();
        }
    } else if (m_state == State::Tracking) {
        assert(m_keyframe);
        if (tracking(image, t)) {
            m_map->insert_frame(m_cur_frame);
            ok = true;
        } else {
            m_cur_frame = m_last_frame;
        }
    } else {
        unimplemented();
    }
    assert(m_cur_frame);
    if (ok) {
        m_motion_pred.inform_pose(m_cur_frame->get_Rcw(),
                                  m_cur_frame->get_Tcw(),
                                  m_cur_frame->get_time());
    }
}

int Frontend::initialize(const cv::Mat &image, double t) {
    std::vector<cv::DMatch> matches = m_matcher->match_descriptor_bf(
            m_keyframe->get_descs(), 8, 15, CNT_INIT_MATCHES);

    const std::vector<cv::KeyPoint> &prev_kpts = m_keyframe->get_kpts();
    std::vector<cv::Point2f> matched_pt1, matched_pt2;
    for (auto &match : matches) {
        matched_pt1.push_back(prev_kpts[match.queryIdx].pt);
        matched_pt2.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
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
    matches = filter_by_mask(matches, mask);
    if (matches.size() < 50) { return -1; }
    matched_pt1.clear();
    matched_pt2.clear();
    for (auto &match : matches) {
        matched_pt1.push_back(prev_kpts[match.queryIdx].pt);
        matched_pt2.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
    }

    cv::Mat Rcw, tcw;
    cv::recoverPose(Ess, matched_pt1, matched_pt2, Rcw, tcw);
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);

    // triangulate points
    cv::Mat tri_res;
    cv::Mat proj_mat1 =
            get_proj_mat(m_camera.get_intrinsic_mat(), m_keyframe->get_Rcw(),
                         m_keyframe->get_Tcw());
    cv::Mat proj_mat2 = get_proj_mat(m_camera.get_intrinsic_mat(), Rcw, tcw);
    cv::triangulatePoints(proj_mat1, proj_mat2, matched_pt1, matched_pt2,
                          tri_res);
    std::vector<bool> inliers;
    int cnt_new_pt = filter_triangulate_points(
            tri_res, m_keyframe->get_Rcw(), m_keyframe->get_Tcw(), Rcw, tcw,
            matched_pt1, matched_pt2, inliers, 10000);
    if (cnt_new_pt < 40) { return -2; }

    double scale = 3;
    for (int i = 0; i < tri_res.cols; ++i) {
        if (inliers[i]) {
            assert(tri_res.at<float>(2, i) > 0);
            scale += tri_res.at<float>(2, i);
        }
    }
    scale /= cnt_new_pt;
    tcw /= scale;
    m_cur_frame->set_pose(Rcw, tcw);
    for (int i = 0; i < tri_res.cols; ++i) {
        tri_res.rowRange(0, 3).col(i) /= scale;
    }

    log_debug_line("Initialize with R: " << std::endl
                                         << Rcw << std::endl
                                         << "T: " << std::endl
                                         << tcw << std::endl
                                         << cnt_new_pt << " new map points.");
    set_new_map_points(m_keyframe, tri_res, matches, inliers);
    select_new_keyframe(m_keyframe);
    return 0;
}

bool Frontend::tracking(const cv::Mat &image, double t) {
    // todo: relocalization(both model fails)
    // todo: camera dist coeff?
    assert(m_keyframe);
    cv::Mat motion_Rcw, motion_Tcw;
    m_motion_pred.predict_pose(m_cur_frame->get_time(), motion_Rcw, motion_Tcw);
    m_cur_frame->set_pose(motion_Rcw, motion_Tcw);

    std::vector<cv::DMatch> match_keyframe;
    std::vector<bool> tri_inliers_keyframe;
    match_between_frames(m_keyframe, match_keyframe, CNT_MATCHES);
    bool b_track_good = false, b_keyframe_good = false;
    int cnt_keyframe_match = track_with_match(match_keyframe, m_keyframe);
    int cnt_proj_match = 0;
    if (cnt_keyframe_match < CNT_MIN_MATCHES) {
        // show_matches(m_keyframe, m_cur_frame, match_keyframe);
        cnt_proj_match = track_with_local_points();
    } else {
        b_keyframe_good = true;
    }
    if (std::max(cnt_proj_match, cnt_keyframe_match) >= CNT_MIN_MATCHES) {
        b_track_good = true;
    }

    if (b_track_good) { _triangulate_with_match(match_keyframe, m_keyframe); }
    if (b_track_good && !b_keyframe_good &&
        m_cur_frame->get_id() > m_keyframe->get_id() + 5) {
        select_new_keyframe(m_cur_frame);
    }

    log_debug_line("Track good: " << b_track_good);
    log_debug_line("Keyframe good: " << b_keyframe_good);
    log_debug_line(m_cur_frame->get_id()
                   << ":\n"
                   << m_cur_frame->get_Rcw() << std::endl
                   << m_cur_frame->get_Tcw() << std::endl);
    log_debug_line("Frame has " << m_cur_frame->get_cnt_map_pt()
                                << " points set.");
    return b_track_good;
}

int Frontend::track_with_match(const std::vector<cv::DMatch> &matches,
                               const vo_ptr<Frame> &ref_frame) {
    std::vector<cv::Matx31f> pt_coords;
    std::vector<cv::Point2f> img_pts;
    std::vector<cv::Point2f> new_img_pt1, new_img_pt2;
    std::vector<cv::DMatch> new_match, old_match;
    for (auto &match : matches) {
        if (ref_frame->is_pt_set(match.queryIdx)) {
            old_match.push_back(match);
            pt_coords.push_back(
                    ref_frame->get_map_pt(match.queryIdx)->get_coord());
            img_pts.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
        } else {
            new_match.push_back(match);
            new_img_pt1.push_back(
                    ref_frame->get_kpt_by_index(match.queryIdx).pt);
            new_img_pt2.push_back(
                    m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
        }
    }
    log_debug_line(old_match.size()
                   << " old match. " << new_match.size() << " new match.");

    if (old_match.size() < CNT_MIN_MATCHES) { return int(old_match.size()); }

    std::vector<bool> inliers;
    std::vector<cv::Matx31f> inlier_coords;
    std::vector<cv::Point2f> inlier_img_pts;
    cv::Mat Rcw = m_cur_frame->get_Rcw().clone(),
            tcw = m_cur_frame->get_Tcw().clone();
    pnp_ransac(pt_coords, img_pts, m_camera, 100, 2, Rcw, tcw, inliers,
               PNP_RANSAC::VO_NONO_PNP_RANSAC);
    assert(inliers.size() == pt_coords.size());
    size_t cnt_inlier = 0;
    for (int i = 0; i < (int) pt_coords.size(); ++i) {
        if (inliers[i]) { cnt_inlier += 1; }
    }
    log_debug_line(cnt_inlier << " inliers after pnp ransac");

    if (cnt_inlier < CNT_MIN_MATCHES / 2) { return int(cnt_inlier); }

    bool should_set = cnt_inlier >= CNT_MIN_MATCHES;
    for (int i = 0; i < (int) pt_coords.size(); ++i) {
        if (inliers[i]) {
            inlier_coords.push_back(pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);

            if (should_set && !m_cur_frame->is_pt_set(old_match[i].trainIdx)) {
                m_cur_frame->set_map_pt(
                        old_match[i].trainIdx,
                        ref_frame->get_map_pt(old_match[i].queryIdx));
            }
        }
    }
    pnp_optimize_proj_err(inlier_coords, inlier_img_pts, m_camera, Rcw, tcw);
    m_cur_frame->set_pose(Rcw, tcw);
    return cnt_inlier;
}

int Frontend::track_with_local_points() {
    int cnt_proj_match = 0;
    std::vector<vo_ptr<MapPoint>> local_map_pts = m_map->get_local_map_points();
    std::unordered_set<vo_id_t> map_pt_set;
    std::vector<ProjMatch> proj_matches;
    std::vector<cv::Matx31f> map_pt_coords, inlier_coords;
    std::vector<cv::Point2f> img_pts, inlier_img_pts;
    std::vector<bool> is_inliers;
    cv::Mat Rcw = m_cur_frame->get_Rcw(), tcw = m_cur_frame->get_Tcw();

    TIME_IT(proj_matches = m_matcher->match_by_projection(local_map_pts, 5.0f),
            "projection match cost ");

    std::vector<cv::DMatch> dmatches;
    for (auto &proj_match : proj_matches) {
        map_pt_coords.push_back(proj_match.coord3d);
        img_pts.push_back(proj_match.img_pt);
        for (int i = 0; i < int(m_keyframe->get_cnt_kpt()); ++i) {
            if (m_keyframe->is_pt_set(i) &&
                m_keyframe->get_map_pt(i)->get_id() ==
                        proj_match.p_map_pt->get_id()) {
                dmatches.emplace_back(i, proj_match.index, 1.0f);
            }
        }
    }
    // show_matches(m_keyframe, m_cur_frame, dmatches);
    if (map_pt_coords.size() < CNT_MIN_MATCHES) {
        return int(map_pt_coords.size());
    }

    TIME_IT(pnp_ransac(map_pt_coords, img_pts, m_camera, 100, 2, Rcw, tcw,
                       is_inliers, PNP_RANSAC::VO_NONO_PNP_RANSAC),
            "projection pnp cost ");

    for (int i = 0; i < int(is_inliers.size()); ++i) {
        if (is_inliers[i]) { cnt_proj_match += 1; }
    }
    if (cnt_proj_match < CNT_MIN_MATCHES) { return cnt_proj_match; }
    for (int i = 0; i < int(is_inliers.size()); ++i) {
        if (is_inliers[i]) {
            inlier_coords.push_back(map_pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
            m_cur_frame->set_map_pt(proj_matches[i].index,
                                    proj_matches[i].p_map_pt);
        }
    }
    pnp_optimize_proj_err(inlier_coords, inlier_img_pts, m_camera, Rcw, tcw);
    m_cur_frame->set_pose(Rcw, tcw);
    log_debug_line("Pose estimate using "
                   << is_inliers.size() << " projection with " << cnt_proj_match
                   << " map points.");
    return cnt_proj_match;
}

void Frontend::_triangulate_with_match(const std::vector<cv::DMatch> &matches,
                                       const vo_ptr<Frame> &ref_frame) {
    std::vector<cv::Point2f> img_pt1, img_pt2;
    std::vector<cv::DMatch> valid_matches;
    cv::Mat tri_res;
    std::vector<bool> inliers;
    for (auto &match : matches) {
        if (!ref_frame->is_pt_set(match.queryIdx) &&
            !m_cur_frame->is_pt_set(match.trainIdx)) {
            img_pt1.push_back(ref_frame->get_kpt_by_index(match.queryIdx).pt);
            img_pt2.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
            valid_matches.push_back(match);
        }
    }
    if (img_pt1.empty()) { return; }
    cv::Mat proj_mat1 =
            get_proj_mat(m_camera.get_intrinsic_mat(), ref_frame->get_Rcw(),
                         ref_frame->get_Tcw());
    cv::Mat proj_mat2 =
            get_proj_mat(m_camera.get_intrinsic_mat(), m_cur_frame->get_Rcw(),
                         m_cur_frame->get_Tcw());
    cv::triangulatePoints(proj_mat1, proj_mat2, img_pt1, img_pt2, tri_res);

    filter_triangulate_points(tri_res, ref_frame->get_Rcw(),
                              ref_frame->get_Tcw(), m_cur_frame->get_Rcw(),
                              m_cur_frame->get_Tcw(), img_pt1, img_pt2, inliers,
                              10000);
    set_new_map_points(ref_frame, tri_res, valid_matches, inliers);
    /*    int total_new_mpt =
            set_new_map_points(ref_frame, tri_res, valid_matches, inliers);
    log_debug_line("Create " << total_new_mpt << " new map points.");*/
}

int Frontend::set_new_map_points(const vo_ptr<Frame> &ref_frame,
                                 const cv::Mat &new_tri_res,
                                 const std::vector<cv::DMatch> &matches,
                                 const std::vector<bool> &inliers) {
    assert(new_tri_res.cols == (int) matches.size());
    assert(inliers.size() == matches.size());
    std::vector<vo_ptr<MapPoint>> new_points;
    int total_new_pt = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (!inliers[i]) { continue; }
        cv::Mat cur_point = new_tri_res.col((int) i);
        float z_val = cur_point.at<float>(3);
        if (!float_eq_zero(z_val) && std::isfinite(z_val)) {
            // not infinite point
            cur_point /= z_val;
            float x = cur_point.at<float>(0);
            float y = cur_point.at<float>(1);
            float z = cur_point.at<float>(2);

            vo_ptr<MapPoint> cur_map_pt = std::make_shared<MapPoint>(
                    MapPoint::create_map_point(x, y, z,
                                               m_cur_frame->get_desc_by_index(
                                                       matches[i].trainIdx)));
            ref_frame->set_map_pt(matches[i].queryIdx, cur_map_pt);
            m_cur_frame->set_map_pt(matches[i].trainIdx, cur_map_pt);
            new_points.push_back(cur_map_pt);
            total_new_pt += 1;
        }
    }
    return total_new_pt;
}

void Frontend::select_new_keyframe(const vo_ptr<Frame> &new_keyframe) {
    m_keyframe = new_keyframe;
    m_map->insert_key_frame(m_keyframe);
    log_debug_line("New key frame: " << m_keyframe->get_id());
}
}// namespace vo_nono