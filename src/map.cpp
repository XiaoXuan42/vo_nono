#include "vo_nono/map.h"

#include <memory>

#include "vo_nono/camera.h"
#include "vo_nono/keypoint/epipolar.h"
#include "vo_nono/keypoint/match.h"
#include "vo_nono/util/util.h"

namespace vo_nono {
LocalMap::LocalMap(Map *map) : map_(map), camera_(map->mr_camera) {}
void LocalMap::triangulate_with_keyframe(
        const std::vector<cv::DMatch> &matches) {
    std::vector<cv::DMatch> valid_match;
    std::vector<bool> mask;
    std::vector<cv::Point2f> pts1, pts2;
    cv::Mat R21, t21;
    Geometry::relative_pose(keyframe_->get_Rcw(), keyframe_->get_Tcw(),
                            curframe_->get_Rcw(), curframe_->get_Tcw(), R21,
                            t21);
    cv::Mat ess = Epipolar::compute_essential(R21, t21);
    for (auto &match : matches) {
        pts1.push_back(keyframe_->kpts[match.queryIdx].pt);
        pts2.push_back(curframe_->kpts[match.trainIdx].pt);
    }
    ORBMatcher::filter_match_by_ess(ess, camera_.get_intrinsic_mat(), pts1,
                                    pts2, 0.01, mask);
    valid_match = filter_by_mask(matches, mask);
    log_debug_line("Before ess check: " << matches.size()
                                        << " after: " << valid_match.size());

    std::vector<bool> tri_inliers;
    std::vector<cv::Mat> tri_results;
    Triangulator::triangulate_and_filter_frames(
            keyframe_.get(), curframe_.get(), camera_.get_intrinsic_mat(),
            valid_match, tri_results, tri_inliers, 10000);

    assert(valid_match.size() == tri_inliers.size());
    int cnt_succ = 0;
    for (int i = 0; i < int(tri_inliers.size()); ++i) {
        if (tri_inliers[i]) {
            if (!curframe_->is_index_set(valid_match[i].trainIdx)) {
                vo_ptr<MapPoint> target_pt;
                if (keyframe_->is_index_set(valid_match[i].queryIdx)) {
                    target_pt = keyframe_->get_map_pt(valid_match[i].queryIdx);
                    if (points_seen_.count(target_pt)) {
                        int seen_times = points_seen_[target_pt];
                        cv::Mat avg_tri =
                                (float(seen_times) * target_pt->get_coord() +
                                 tri_results[i]) /
                                (float(seen_times) + 1.0f);
                        target_pt->set_coord(avg_tri);
                        points_seen_[target_pt] += 1;
                    }
                } else {
                    target_pt = std::make_shared<MapPoint>(
                            MapPoint::create_map_point(
                                    tri_results[i],
                                    keyframe_->descriptor.row(
                                            valid_match[i].queryIdx)));
                    keyframe_->set_map_pt(valid_match[i].queryIdx, target_pt);
                    assert(!points_seen_.count(target_pt));
                    points_seen_[target_pt] = 1;
                    cnt_succ += 1;
                }
                curframe_->set_map_pt(valid_match[i].trainIdx, target_pt);
            }
        }
    }
    log_debug_line("Triangulated " << cnt_succ << " points.");
}

void LocalMap::insert_frame(const FrameMessage &message) {
    keyframe_ = map_->m_keyframes.back();
    curframe_ = message.frame;
    triangulate_with_keyframe(message.match_with_keyframe);
    map_->m_frames.push_back(message.frame);
    if (message.is_keyframe) {
        points_seen_.clear();
        map_->insert_key_frame(message.frame);
    }
}


void Map::insert_frame(const FrameMessage &message) {
    local_map_->insert_frame(message);
}

void Map::global_bundle_adjustment() {
    std::unique_lock<std::mutex> glb_lock(map_global_mutex);
    while (true) {
        if (mb_shutdown) { break; }
        m_global_ba_cv.wait(glb_lock, [&] { return mb_global_ba; });
        if (mb_shutdown) { break; }
        mb_global_ba = false;
        if (m_keyframes.size() < 5) { continue; }

        _global_bundle_adjustment(glb_lock);
    }
}

void Map::_global_bundle_adjustment(std::unique_lock<std::mutex> &lock) {}
}// namespace vo_nono