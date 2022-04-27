#ifndef VO_NONO_FRAME_H
#define VO_NONO_FRAME_H

#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vo_nono/camera.h"
#include "vo_nono/point.h"
#include "vo_nono/types.h"
#include "vo_nono/util/macro.h"

namespace vo_nono {
class FeaturePoint {
public:
    cv::KeyPoint keypoint;
    cv::Mat descriptor;
    vo_ptr<Frame> map_point;
    int index;

    FeaturePoint(cv::KeyPoint kpt, cv::Mat dscpts, int idx)
        : keypoint(kpt),
          descriptor(std::move(dscpts)),
          index(idx) {}
};

class Frame {
public:
    cv::Mat image;

    static Frame create_frame(cv::Mat descriptor,
                              std::vector<cv::KeyPoint> kpts, double time,
                              cv::Mat Rcw = cv::Mat::eye(3, 3, CV_32F),
                              cv::Mat Tcw = cv::Mat::zeros(3, 1, CV_32F));


    void set_Rcw(const cv::Mat &Rcw) {
        assert(Rcw.rows == 3);
        assert(Rcw.cols == 3);
        assert_float_eq(cv::determinant(Rcw), 1.0f);
        if (Rcw.type() == CV_32F) {
            Rcw_ = Rcw.clone();
        } else if (Rcw.type() == CV_64F) {
            Rcw.convertTo(Rcw_, CV_32F);
        } else {
            unimplemented();
        }
        assert(Rcw_.type() == CV_32F);
    }
    void set_Tcw(const cv::Mat &Tcw) {
        assert(Tcw.rows == 3);
        assert(Tcw.cols == 1);
        if (Tcw.type() == CV_32F) {
            Tcw_ = Tcw.clone();
        } else if (Tcw.type() == CV_64F) {
            Tcw.convertTo(Tcw_, CV_32F);
        } else {
            unimplemented();
        }
        assert(Tcw_.type() == CV_32F);
    }
    void set_pose(const cv::Mat &Rcw, const cv::Mat &Tcw) {
        set_Rcw(Rcw);
        set_Tcw(Tcw);
    }
    void set_pose(const cv::Mat &pose) {
        assert(pose.type() == CV_32F);
        assert(pose.rows == 3);
        assert(pose.cols == 4);
        set_Rcw(pose.colRange(0, 3));
        set_Tcw(pose.col(3));
    }
    [[nodiscard]] cv::Mat get_Rcw() const { return Rcw_.clone(); }
    [[nodiscard]] cv::Mat get_Tcw() const { return Tcw_.clone(); }
    [[nodiscard]] cv::Mat get_pose() const {
        cv::Mat pose = cv::Mat::zeros(3, 4, CV_32F);
        Rcw_.copyTo(pose.rowRange(0, 3).colRange(0, 3));
        Tcw_.copyTo(pose.rowRange(0, 3).col(3));
        return pose;
    }

    // keypoints that already has corresponding map point
    [[nodiscard]] size_t get_cnt_map_pt() const { return index_to_pt_.size(); }
    void set_map_pt(int i, const std::shared_ptr<MapPoint> &pt) {
        //assert(index_to_pt_.count(i) == 0);
        index_to_pt_.insert({i, pt});
        pt_to_index_.insert({pt->get_id(), i});
    }
    std::shared_ptr<MapPoint> get_map_pt(int i) const {
        assert(index_to_pt_.count(i) != 0);
        return index_to_pt_.at(i);
    }
    std::vector<vo_ptr<MapPoint>> get_all_map_pts() const {
        std::vector<vo_ptr<MapPoint>> res;
        res.reserve(index_to_pt_.size());
        for (auto &it : index_to_pt_) { res.push_back(it.second); }
        return res;
    }
    [[nodiscard]] std::vector<cv::KeyPoint> get_keypoints() const {
        std::vector<cv::KeyPoint> result;
        for (auto &point : feature_points) { result.push_back(point.keypoint); }
        return result;
    }
    [[nodiscard]] cv::Mat get_descriptors() const {
        cv::Mat result = cv::Mat(int(feature_points.size()), 32, CV_8U);
        for (int i = 0; i < int(feature_points.size()); ++i) {
            feature_points[i].descriptor.copyTo(result.row(i));
        }
        return result;
    }
    bool is_index_set(int i) const { return index_to_pt_.count(i) != 0; }

    [[nodiscard]] vo_id_t get_id() const { return id_; }
    [[nodiscard]] double get_time() const { return time; }

    std::vector<FeaturePoint> feature_points;

private:
    static vo_id_t frame_id_cnt;
    Frame(vo_id_t id, const cv::Mat &descriptor,
          const std::vector<cv::KeyPoint> &kpts, double time, cv::Mat Rcw,
          cv::Mat Tcw)
        : id_(id),
          time(time),
          Rcw_(std::move(Rcw)),
          Tcw_(std::move(Tcw)) {
        assert(descriptor.rows == int(kpts.size()));
        for (int i = 0; i < descriptor.rows; ++i) {
            feature_points.emplace_back(
                    FeaturePoint(kpts[i], descriptor.row(i), i));
        }
    }

    vo_id_t id_;
    double time;
    cv::Mat Rcw_;
    cv::Mat Tcw_;

    // from index of kpts to map points
    std::unordered_map<int, std::shared_ptr<MapPoint>> index_to_pt_;
    // from map point's id to index
    std::unordered_map<vo_id_t, int> pt_to_index_;
};
}// namespace vo_nono

#endif//VO_NONO_FRAME_H
