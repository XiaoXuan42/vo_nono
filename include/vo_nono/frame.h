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
    [[nodiscard]] size_t get_cnt_map_pt() const {
        return index_to_pt_.size();
    }
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
    bool is_index_set(int i) const { return index_to_pt_.count(i) != 0; }

    static vo_id_t frame_id_cnt;

    vo_id_t id;
    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptor;
    double time;

private:
    Frame(vo_id_t id, cv::Mat descriptor, std::vector<cv::KeyPoint> kpts,
          double time, cv::Mat Rcw, cv::Mat Tcw)
        : id(id),
          kpts(std::move(kpts)),
          descriptor(std::move(descriptor)),
          time(time),
          Rcw_(std::move(Rcw)),
          Tcw_(std::move(Tcw)) {}

    cv::Mat Rcw_;
    cv::Mat Tcw_;

    // from index of kpts to map points
    std::unordered_map<int, std::shared_ptr<MapPoint>> index_to_pt_;
    // from map point's id to index
    std::unordered_map<vo_id_t, int> pt_to_index_;
};
}// namespace vo_nono

#endif//VO_NONO_FRAME_H
