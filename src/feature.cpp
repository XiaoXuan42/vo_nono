#include "vo_nono/feature.h"

#include <map>

#include "vo_nono/camera.h"

namespace vo_nono {
void reproj_points_from_frame(const vo_ptr<Frame> &left_frame,
                              const vo_ptr<Frame> &right_frame,
                              const Camera &camera,
                              std::map<int, ReprojRes> &book) {
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
        };
    }
}
}// namespace vo_nono