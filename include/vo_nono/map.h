#ifndef VO_NONO_MAP_H
#define VO_NONO_MAP_H

#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <thread>
#include <unordered_set>
#include <vector>

#include "vo_nono/frame.h"
#include "vo_nono/geometry.h"
#include "vo_nono/keypoint/triangulate.h"
#include "vo_nono/point.h"

namespace vo_nono {
struct FrameInfo {
    vo_id_t frame_id;
    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptors;
    cv::Mat Rcw;
    cv::Mat tcw;
    double time;

    std::vector<vo_id_t> map_pt_id;
    std::vector<cv::Mat> map_coord;
    cv::Mat image;

    void clear() {
        frame_id = vo_id_invalid;
        kpts.clear();
        map_pt_id.clear();
        map_coord.clear();
        descriptors = cv::Mat();
        image = cv::Mat();
        Rcw = cv::Mat();
        tcw = cv::Mat();
    }

    void init_map_info() {
        assert(int(kpts.size()) == descriptors.rows);
        map_pt_id = std::vector(kpts.size(), vo_id_invalid);
        map_coord.resize(kpts.size());
        Rcw = cv::Mat::eye(3, 3, CV_32F);
        tcw = cv::Mat::zeros(3, 1, CV_32F);
    }

    [[nodiscard]] bool is_index_set(int i) const {
        assert(map_pt_id.size() == map_coord.size());
        assert(i < int(map_pt_id.size()));
        return map_pt_id[i] != vo_id_invalid;
    }

    void set_point(int i, vo_id_t id, const cv::Mat &coord) {
        assert(map_pt_id.size() == map_coord.size());
        assert(i < int(map_pt_id.size()));
        map_pt_id[i] = id;
        map_coord[i] = coord;
    }

    void update(const vo_ptr<Frame> &pframe) {
        frame_id = pframe->get_id();
        kpts = pframe->get_kpts();
        descriptors = pframe->get_descs();
        Rcw = pframe->get_Rcw();
        tcw = pframe->get_Tcw();
        time = pframe->get_time();
        map_pt_id = std::vector<vo_id_t>(kpts.size(), vo_id_invalid);
        map_coord = std::vector<cv::Mat>(kpts.size());
        for (int i = 0; i < int(kpts.size()); ++i) {
            if (pframe->is_index_set(i)) {
                map_pt_id[i] = pframe->get_map_pt(i)->get_id();
                map_coord[i] = pframe->get_map_pt(i)->get_coord();
            }
        }
    }
};

class Map {
public:
    explicit Map(const Camera &camera)
        : mr_camera(camera),
          mb_shutdown(false),
          mb_global_ba(false) {
        mt_global_ba = std::thread(&Map::global_bundle_adjustment, this);
    }
    Map(const Map &) = delete;
    ~Map() { shutdown(); }

    using Trajectory = std::vector<std::pair<double, cv::Mat>>;
    void insert_key_frame(const vo_ptr<Frame> &frame) {
        m_keyframes.push_back(frame);
        m_cur_keyframe = frame;
        mb_global_ba = true;
        m_global_ba_cv.notify_all();
    }

    void global_bundle_adjustment();

    [[nodiscard]] Trajectory get_trajectory() const {
        Trajectory trajectory;
        trajectory.reserve(m_frames.size());
        for (const vo_ptr<Frame> &frame : m_frames) {
            trajectory.emplace_back(
                    std::make_pair(frame->get_time(), frame->get_pose()));
        }
        return trajectory;
    }

    std::vector<vo_ptr<MapPoint>> get_local_map_points() {
        std::unordered_set<vo_id_t> id_book;
        std::vector<vo_ptr<MapPoint>> result;
        int cnt = 0;
        for (auto iter = m_keyframes.rbegin(); iter != m_keyframes.rend();
             ++iter) {
            std::vector<vo_ptr<MapPoint>> frame_pts =
                    (*iter)->get_all_map_pts();
            for (auto &map_pt : frame_pts) {
                if (!id_book.count(map_pt->get_id())) {
                    id_book.insert(map_pt->get_id());
                    result.push_back(map_pt);
                }
            }
            cnt += 1;
            if (cnt >= 5) { break; }
        }
        return result;
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(map_global_mutex);
            mb_global_ba = true;
            mb_shutdown = true;
            m_global_ba_cv.notify_all();
        }
        mt_global_ba.join();
    }

    void initialize(FrameInfo &keyframe_info, FrameInfo &ref_frame_info,
                    const std::vector<cv::DMatch> &matches,
                    const std::vector<cv::Mat> &triangulate_res,
                    const std::vector<bool> &triangulate_inlier) {
        assert(matches.size() == triangulate_res.size());
        assert(matches.size() == triangulate_inlier.size());
        vo_ptr<Frame> p_keyframe = std::make_shared<Frame>(Frame::create_frame(
                keyframe_info.descriptors, keyframe_info.kpts,
                keyframe_info.time, keyframe_info.Rcw.clone(),
                keyframe_info.tcw.clone()));
        vo_ptr<Frame> p_refframe = std::make_shared<Frame>(Frame::create_frame(
                std::move(ref_frame_info.descriptors),
                std::move(ref_frame_info.kpts), ref_frame_info.time,
                ref_frame_info.Rcw.clone(), ref_frame_info.tcw.clone()));

        for (size_t i = 0; i < matches.size(); ++i) {
            if (triangulate_inlier[i]) {
                auto new_pt =
                        std::make_shared<MapPoint>(MapPoint::create_map_point(
                                triangulate_res[i],
                                p_keyframe->get_desc_by_index(
                                        matches[i].queryIdx)));
                p_keyframe->set_map_pt(matches[i].queryIdx, new_pt);
                p_refframe->set_map_pt(matches[i].trainIdx, new_pt);
                keyframe_info.set_point(matches[i].queryIdx, new_pt->get_id(),
                                        new_pt->get_coord());
            }
        }
        m_frames.push_back(p_keyframe);
        m_frames.push_back(p_refframe);
        insert_key_frame(p_keyframe);
    }

    void insert_frame(FrameInfo &frame_info,
                      const std::vector<cv::DMatch> &matches,
                      const std::vector<bool> &match_inlier) {
        vo_ptr<Frame> p_frame = std::make_shared<Frame>(Frame::create_frame(
                frame_info.descriptors, std::move(frame_info.kpts),
                frame_info.time, frame_info.Rcw.clone(),
                frame_info.tcw.clone()));

        std::vector<cv::Mat> tri_res;
        std::vector<bool> tri_inliers;
        Triangulator::triangulate_and_filter_frames(
                m_cur_keyframe.get(), p_frame.get(),
                mr_camera.get_intrinsic_mat(), matches, tri_res, tri_inliers,
                10000);
        for (int i = 0; i < int(tri_res.size()); ++i) {
            if (tri_inliers[i] &&
                !m_cur_keyframe->is_index_set(matches[i].queryIdx) &&
                !p_frame->is_index_set(matches[i].trainIdx)) {
                auto new_pt =
                        std::make_shared<MapPoint>(MapPoint::create_map_point(
                                tri_res[i], m_cur_keyframe->get_desc_by_index(
                                                    matches[i].queryIdx)));
                m_cur_keyframe->set_map_pt(matches[i].queryIdx, new_pt);
                p_frame->set_map_pt(matches[i].trainIdx, new_pt);
            }
        }
        m_frames.push_back(p_frame);
    }

    void update_keyframe_info(FrameInfo &key_frame_info) {
        key_frame_info.update(m_cur_keyframe);
    }

    std::mutex map_global_mutex;

private:
    std::vector<vo_ptr<Frame>> m_keyframes;
    std::vector<vo_ptr<Frame>> m_frames;
    vo_ptr<Frame> m_cur_keyframe;

    const Camera &mr_camera;

    std::condition_variable m_global_ba_cv;

    bool mb_shutdown;
    bool mb_global_ba;
    std::thread mt_global_ba;
};
}// namespace vo_nono

#endif//VO_NONO_MAP_H
