#include "vo_nono/frontend.h"

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <vector>

namespace vo_nono {
namespace {
inline cv::Mat pt2_to_homogeneous_mat(const cv::Point2f &pt) {
    cv::Mat res(3, 1, CV_32F);
    res.at<float>(0, 0) = pt.x;
    res.at<float>(1, 0) = pt.y;
    res.at<float>(2, 0) = 1.0f;
    return res;
}
}// namespace

// detect feature points and compute descriptors
void Frontend::detect_and_compute(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &kpts,
                                  cv::Mat &dscpts) {
    static cv::Ptr<cv::ORB> orb_detector;
    if (!orb_detector) { orb_detector = cv::ORB::create(); }
    orb_detector->detectAndCompute(image, cv::Mat(), kpts, dscpts);
}

std::vector<cv::DMatch> Frontend::match_descriptor(const cv::Mat &dscpt1,
                                                   const cv::Mat &dscpt2) {
    static const int MAXIMAL_MATCH_COUNT = 50;

    std::vector<cv::DMatch> matches;
    auto matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
    matcher.match(dscpt1, dscpt2, matches);

    if (matches.size() > MAXIMAL_MATCH_COUNT) {
        std::sort(matches.begin(), matches.end(),
                  [](cv::DMatch &match1, cv::DMatch &match2) {
                      return match1.distance < match2.distance;
                  });
        matches.resize(MAXIMAL_MATCH_COUNT);
    }
    return matches;
}

// essential matrix
void Frontend::compute_essential_mat(
        const std::vector<std::pair<cv::Point2f, cv::Point2f>> &pts, cv::Mat &W,
        cv::Mat &U, cv::Mat &Vt) {
    int sz = pts.size();
    cv::Mat A = cv::Mat(sz, 9, CV_32F);

    // (u2u1, u2v1, u2, v2u1, v2v1, v2, u1, v1, 1)
    for (int i = 0; i < sz; ++i) {
        const float u1 = pts[i].first.x, v1 = pts[i].first.y;
        const float u2 = pts[i].second.x, v2 = pts[i].second.y;
        A.at<float>(i, 0) = u2 * u1;
        A.at<float>(i, 1) = u2 * v1;
        A.at<float>(i, 2) = u2;
        A.at<float>(i, 3) = v2 * u1;
        A.at<float>(i, 4) = v2 * v1;
        A.at<float>(i, 5) = v2;
        A.at<float>(i, 6) = u1;
        A.at<float>(i, 7) = v1;
        A.at<float>(i, 8) = 1.0f;
    }

    // solve Ax = 0 to get E
    cv::SVDecomp(A, W, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat E = Vt.row(8).reshape(0, 3);
    // project to the space of essential matrix
    cv::SVDecomp(E, W, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    float sing_val = (W.at<float>(0) + W.at<float>(1)) / 2.0f;
    W.at<float>(0) = sing_val;
    W.at<float>(1) = sing_val;
    W.at<float>(2) = 0.0f;
}

double Frontend::assess_essential_mat(
        const cv::Mat &U, const cv::Mat &Vt,
        const std::vector<std::pair<cv::Point2f, cv::Point2f>> &pts) {
    const cv::Mat A = U.col(0) * Vt.row(0) + U.col(1) + Vt.row(1);
    constexpr float scale = 0.01;
    // (scale, scale, 1)^T
    const cv::Mat err_vec = cv::Mat(std::vector<float>{scale, scale, 1.0f});

    size_t sz = pts.size();
    cv::Mat tmp_err_mat;
    double score = 0.0;
    for (size_t i = 0; i < sz; ++i) {
        cv::Mat pt1 = pt2_to_homogeneous_mat(pts[i].first);
        cv::Mat pt2 = pt2_to_homogeneous_mat(pts[i].second);
        tmp_err_mat = pt2.t() * A * pt1;
        double true_err = tmp_err_mat.at<float>(0, 0);

        tmp_err_mat = pt2.t() * A * err_vec;
        double expect_max_err = tmp_err_mat.at<float>(0, 0);
        tmp_err_mat = err_vec.t() * A * pt1;
        expect_max_err += tmp_err_mat.at<float>(0, 0);

        if (true_err < expect_max_err) {
            score += (expect_max_err - true_err) / expect_max_err;
        }
    }
    return score;
}

void Frontend::get_image(const cv::Mat &image, vo_time_t t) {
    if (m_state == State::Start) {
        m_prev_frame = Frame(image);
        m_state = State::Initializing;
    } else if (m_state == State::Initializing) {
        assert(!m_prev_frame.m_image.empty());
        initialize(image);
    } else if (m_state == State::Tracking) {
        assert(false);
    } else {
        assert(false);
    }
}

void Frontend::initialize(const cv::Mat &image) {
    m_cur_frame = Frame(image);

    detect_and_compute(m_prev_frame.m_image, m_prev_frame.kpts,
                       m_prev_frame.descriptor);
    detect_and_compute(image, m_cur_frame.kpts, m_cur_frame.descriptor);
    std::vector<cv::DMatch> matches =
            match_descriptor(m_prev_frame.descriptor, m_cur_frame.descriptor);

    std::vector<cv::Point2f> matched_pt1, matched_pt2;
    for (auto &match : matches) {
        matched_pt1.push_back(m_prev_frame.kpts[match.queryIdx].pt);
        matched_pt2.push_back(m_cur_frame.kpts[match.trainIdx].pt);
    }

    // todo: less than 8 matched points?
    // todo: normalize scale?
    // todo: filter inliners
    cv::Mat Ess = cv::findEssentialMat(matched_pt1, matched_pt2,
                                       m_camera.get_intrinsic_mat());
    cv::Mat R, t;
    cv::recoverPose(Ess, matched_pt1, matched_pt2, R, t);

    // triangulate points
    cv::Mat tri_res;
    cv::Mat proj_mat1 = get_proj_mat(cv::Mat::eye(3, 3, CV_32F ),
                                     cv::Mat::zeros(3, 1, CV_32F));
    cv::Mat proj_mat2 = get_proj_mat(R, t);
    cv::triangulatePoints(proj_mat1, proj_mat2, matched_pt1, matched_pt2,
                          tri_res);
}

cv::Mat Frontend::get_proj_mat(const cv::Mat &Rcw, const cv::Mat &t) {
    cv::Mat proj = cv::Mat::zeros(3, 4, CV_32F);
    proj.rowRange(0, 3).colRange(0, 3) = Rcw;
    proj.rowRange(0, 3).col(3) = t;
    proj = m_camera.get_intrinsic_mat() * proj;
    return proj;
}

}// namespace vo_nono