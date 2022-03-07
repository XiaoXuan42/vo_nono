#include "vo_nono/feature.h"

#include <map>

#include "vo_nono/camera.h"

namespace vo_nono {
void filter_match_key_pts(const std::vector<cv::KeyPoint>& kpts1,
                          const std::vector<cv::KeyPoint>& kpts2,
                          std::vector<unsigned char>& mask, double ransac_th) {
    assert(kpts1.size() == kpts2.size());
    auto ang_diff_index = [](double diff_ang) {
        if (diff_ang < 0) { diff_ang += 360; }
        return (int) (diff_ang / 3.6);
    };
    Histogram<double> histo(101, ang_diff_index);
    std::vector<cv::Point2f> pt21;
    std::vector<cv::Point2f> pt22;
    std::vector<double> ang_diff;
    for (int i = 0; i < (int) kpts1.size(); ++i) {
        double diff = kpts1[i].angle - kpts2[i].angle;
        pt21.push_back(kpts1[i].pt);
        pt22.push_back(kpts2[i].pt);
        histo.insert_element(diff);
        ang_diff.push_back(diff);
    }
    filter_match_pts(pt21, pt22, mask, ransac_th);
    assert(mask.size() == kpts1.size());
    histo.cal_topK(3);
    for (int i = 0; i < (int) kpts1.size(); ++i) {
        if (!histo.is_topK(ang_diff[i])) { mask[i] = 0; }
    }
}

void reproj_points_from_frame(const vo_ptr<Frame>& left_frame,
                              const vo_ptr<Frame>& right_frame,
                              const Camera& camera,
                              std::map<int, ReprojRes>& book) {
    std::vector<int> match_left_id;
    std::vector<cv::KeyPoint> left_img_kpts;
    std::vector<cv::Mat> left_img_descs;
    std::vector<cv::Point2f> proj_right_img_pts;
    cv::Mat proj_mat =
            get_proj_mat(camera.get_intrinsic_mat(), right_frame->get_Rcw(),
                         right_frame->get_Tcw());
    for (int i = 0; i < (int) left_frame->get_kpts().size(); ++i) {
        if (left_frame->is_pt_set(i)) {
            cv::Mat coord = left_frame->get_pt_3dcoord(i);
            cv::Mat hm_coord(4, 1, CV_32F);
            coord.copyTo(hm_coord.rowRange(0, 3));
            hm_coord.at<float>(3, 0) = 1.0f;

            cv::Mat hm_img_pt = proj_mat * hm_coord;
            const float scale = hm_img_pt.at<float>(0, 2);
            if (!std::isfinite(scale)) { continue; }
            hm_img_pt /= scale;
            float img_x = hm_img_pt.at<float>(0, 0),
                  img_y = hm_img_pt.at<float>(1, 0);
            if (!std::isfinite(img_x) || img_x < 0 ||
                img_x > camera.get_width() || !std::isfinite(img_y) ||
                img_y < 0 || img_y > camera.get_height()) {
                continue;
            }
            proj_right_img_pts.emplace_back(img_x, img_y);
            left_img_descs.push_back(left_frame->get_pt_desc(i));
            left_img_kpts.push_back(left_frame->get_pt_keypt(i));
            match_left_id.push_back(i);
        }
    }

    std::vector<cv::KeyPoint> right_img_kpts(match_left_id.size());
    std::vector<bool> inliers(match_left_id.size());
    std::vector<double> desc_dis(match_left_id.size());
    std::vector<int> match_right_id(match_left_id.size());
    std::map<int, int> local_book;
    assert(proj_right_img_pts.size() == left_img_kpts.size());
    assert(left_img_kpts.size() == left_img_descs.size());
    assert(match_left_id.size() == left_img_kpts.size());
    for (int i = 0; i < (int) match_left_id.size(); ++i) {
        int match_id =
                right_frame->local_match(left_img_kpts[i], left_img_descs[i],
                                         proj_right_img_pts[i], 9.0f);
        if (match_id < 0) {
            inliers[i] = false;
        } else {
            double cur_dis = cv::norm(left_img_descs[i],
                                      right_frame->get_pt_desc(match_id));
            if (local_book.count(match_id)) {
                double prev_dis = desc_dis[local_book[match_id]];
                if (prev_dis <= cur_dis) {
                    inliers[i] = false;
                    continue;
                } else {
                    inliers[local_book[match_id]] = false;
                }
            }
            if (book.count(match_id)) {
                double prev_dis = book[match_id].desc_dis;
                if (prev_dis <= cur_dis) {
                    inliers[i] = false;
                    continue;
                }
            }
            inliers[i] = true;
            desc_dis[i] = cur_dis;
            match_right_id[i] = match_id;
            right_img_kpts[i] = right_frame->get_pt_keypt(match_id);
            local_book[match_id] = i;
        }
    }

    // filter matches
    match_left_id = filter_by_mask(match_left_id, inliers);
    match_right_id = filter_by_mask(match_right_id, inliers);
    right_img_kpts = filter_by_mask(right_img_kpts, inliers);
    left_img_kpts = filter_by_mask(left_img_kpts, inliers);
    desc_dis = filter_by_mask(desc_dis, inliers);
    assert(match_left_id.size() == match_right_id.size());

    if (match_left_id.size() < 10) { return; }
    std::vector<unsigned char> inliers2;
    filter_match_key_pts(left_img_kpts, right_img_kpts, inliers2);
    // right_img_kpts = filter_by_mask(right_img_kpts, inliers2);
    match_left_id = filter_by_mask(match_left_id, inliers2);
    match_right_id = filter_by_mask(match_right_id, inliers2);
    for (int i = 0; i < (int) match_right_id.size(); ++i) {
        book[match_right_id[i]] = ReprojRes{
                .frame_id = left_frame->get_id(),
                .map_point_id = left_frame->get_pt_id(match_left_id[i]),
                .point_index = match_left_id[i],
                .desc_dis = desc_dis[i],
                .coord = left_frame->get_pt_3dcoord(match_left_id[i]),
        };
    }
}

void pnp_from_reproj_res(const vo_ptr<Frame>& frame, const Camera& camera,
                         const std::map<int, ReprojRes>& book,
                         std::vector<int>& img_pt_index,
                         std::vector<vo_id_t>& map_pt_ids,
                         std::vector<cv::Matx31f>& map_pt_coords,
                         std::vector<int>& inliers, cv::Mat& Rcw, cv::Mat& tcw,
                         bool use_init, double reproj_error) {
    img_pt_index.clear();
    map_pt_ids.clear();
    map_pt_coords.clear();
    img_pt_index.reserve(book.size());
    map_pt_ids.reserve(book.size());
    map_pt_coords.reserve(book.size());

    std::vector<cv::Point2f> img_pts;
    img_pts.reserve(book.size());

    inliers.clear();
    for (auto& pair : book) {
        img_pt_index.push_back(pair.first);
        img_pts.push_back(frame->get_pt_keypt(pair.second.point_index).pt);
        map_pt_ids.push_back(pair.second.map_point_id);
        map_pt_coords.push_back(pair.second.coord);
    }
    cv::Mat rvec, tmp_tcw;
    if (use_init) {
        assert(Rcw.type() == CV_32F);
        assert(tcw.type() == CV_32F);
        cv::Mat tmp_Rcw;
        Rcw.convertTo(tmp_Rcw, CV_64F);
        cv::Rodrigues(tmp_Rcw, rvec);
        tcw.convertTo(tmp_tcw, CV_64F);
    }
    cv::solvePnPRansac(map_pt_coords, img_pts, camera.get_intrinsic_mat(),
                       std::vector<float>(), rvec, tmp_tcw, use_init, 100,
                       reproj_error, 0.99, inliers);
    cv::Rodrigues(rvec, Rcw);
    Rcw.convertTo(Rcw, CV_32F);
    tmp_tcw.convertTo(tcw, CV_32F);
}
}// namespace vo_nono