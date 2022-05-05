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
    Frame *frame;
    cv::KeyPoint keypoint;
    cv::Mat descriptor;
    vo_ptr<MapPoint> map_point;
    int index;

    FeaturePoint(Frame *pframe, cv::KeyPoint kpt, cv::Mat dscpts, int idx)
        : frame(pframe),
          keypoint(kpt),
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
    Frame(Frame &&other) noexcept { copy_from(other); }
    Frame &operator=(Frame &&other) noexcept {
        copy_from(other);
        return *this;
    }
    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;
    Frame() = delete;

    ~Frame() {
        for (auto p_feature : feature_points) {
            if (p_feature->map_point) {
                p_feature->map_point->unassociate_feature_point(p_feature);
            }
            delete p_feature;
        }
    }

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
    [[nodiscard]] int get_cnt_map_pt() const { return map_pt_cnt_; }
    void set_map_pt(int i, const std::shared_ptr<MapPoint> &pt) {
        assert(i < int(feature_points.size()));
        if (!feature_points[i]->map_point) { map_pt_cnt_ += 1; }
        feature_points[i]->map_point = pt;
    }
    [[nodiscard]] std::shared_ptr<MapPoint> get_map_pt(int i) const {
        assert(i < int(feature_points.size()));
        return feature_points[i]->map_point;
    }
    [[nodiscard]] std::vector<vo_ptr<MapPoint>> get_all_map_pts() {
        std::vector<vo_ptr<MapPoint>> res;
        for (auto &feat_point : feature_points) {
            if (feat_point->map_point) { res.push_back(feat_point->map_point); }
        }
        return res;
    }
    [[nodiscard]] std::vector<cv::KeyPoint> get_keypoints() const {
        std::vector<cv::KeyPoint> result;
        for (auto &point : feature_points) {
            result.push_back(point->keypoint);
        }
        return result;
    }
    [[nodiscard]] cv::Mat get_descriptors() const {
        cv::Mat result = cv::Mat(int(feature_points.size()), 32, CV_8U);
        for (int i = 0; i < int(feature_points.size()); ++i) {
            feature_points[i]->descriptor.copyTo(result.row(i));
        }
        return result;
    }
    [[nodiscard]] bool is_index_set(int i) const {
        return bool(feature_points[i]->map_point);
    }

    [[nodiscard]] vo_id_t get_id() const { return id_; }
    [[nodiscard]] double get_time() const { return time_; }
    [[nodiscard]] cv::Point2f get_pixel_pt(int index) const {
        return feature_points[index]->keypoint.pt;
    }

    std::vector<FeaturePoint *> feature_points;

private:
    static vo_id_t frame_id_cnt;
    Frame(vo_id_t id, const cv::Mat &descriptor,
          const std::vector<cv::KeyPoint> &kpts, double time, cv::Mat Rcw,
          cv::Mat Tcw)
        : id_(id),
          time_(time),
          Rcw_(std::move(Rcw)),
          Tcw_(std::move(Tcw)) {
        assert(descriptor.rows == int(kpts.size()));
        feature_points.resize(descriptor.rows);
        for (int i = 0; i < descriptor.rows; ++i) {
            feature_points[i] =
                    new FeaturePoint(this, kpts[i], descriptor.row(i), i);
        }
    }

    void copy_from(Frame &other) {
        id_ = other.id_;
        time_ = other.time_;
        Rcw_ = other.Rcw_;
        Tcw_ = other.Tcw_;
        map_pt_cnt_ = other.map_pt_cnt_;
        feature_points = std::move(other.feature_points);
        for (auto &feat : feature_points) { feat->frame = this; }
    }

    vo_id_t id_{};
    double time_{};
    cv::Mat Rcw_;
    cv::Mat Tcw_;

    int map_pt_cnt_ = 0;
};
}// namespace vo_nono

#endif//VO_NONO_FRAME_H
