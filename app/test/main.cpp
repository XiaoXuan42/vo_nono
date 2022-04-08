#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <random>

#include "vo_nono/camera.h"
#include "vo_nono/motion.h"
#include "vo_nono/optim.h"
#include "vo_nono/util.h"

cv::Mat get_proj(const cv::Mat &Rcw, const cv::Mat &tcw) {
    cv::Mat proj(3, 4, CV_32F);
    Rcw.copyTo(proj.rowRange(0, 3).colRange(0, 3));
    tcw.copyTo(proj.rowRange(0, 3).col(3));
    return proj;
}

void test_quaternion() {
    cv::Mat rot, rc_rot;
    cv::Mat rvec = cv::Mat({3.0f, 7.89f, -10.5f});
    cv::Rodrigues(rvec, rot);
    float q[4];
    vo_nono::rotation_mat_to_quaternion(rot, q);
    rc_rot = vo_nono::quaternion_to_rotation_mat(q);
    std::cout << rot << std::endl << rc_rot;
}

void test_init() {
    srand(time(NULL));

    cv::Mat iden33 = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat t1 = cv::Mat({0, 0, 0});
    cv::Mat t2 = cv::Mat({10, 0, 0});
    cv::Mat proj1 = cv::Mat::zeros(3, 4, CV_32F);
    iden33.copyTo(proj1.rowRange(0, 3).colRange(0, 3));
    t1.copyTo(proj1.rowRange(0, 3).col(3));
    cv::Mat proj2 = cv::Mat::zeros(3, 4, CV_32F);
    iden33.copyTo(proj2.rowRange(0, 3).colRange(0, 3));
    t2.copyTo(proj2.rowRange(0, 3).col(3));

    std::vector<cv::Mat> map_points;
    std::vector<cv::Point2f> img_points1, img_points2;

    for (int i = 0; i < 10; ++i) {
        cv::Mat point(4, 1, CV_32F);
        float x = (float) (rand() % 100) / 10;
        float y = (float) (rand() % 100) / 10;
        float z = 10 + (rand() % 100) / 10;
        point.at<float>(0, 0) = x;
        point.at<float>(1, 0) = y;
        point.at<float>(2, 0) = z;
        point.at<float>(3, 0) = 1.0f;
        map_points.push_back(point);
    }
    for (cv::Mat pt : map_points) {
        cv::Mat img_pt1 = proj1 * pt;
        img_pt1 /= img_pt1.at<float>(2);
        cv::Point2f pt1;
        pt1.x = img_pt1.at<float>(0, 0);
        pt1.y = img_pt1.at<float>(1, 0);
        img_points1.push_back(pt1);

        cv::Mat img_pt2 = proj2 * pt;
        img_pt2 /= img_pt2.at<float>(2);
        cv::Point2f pt2;
        pt2.x = img_pt2.at<float>(0, 0);
        pt2.y = img_pt2.at<float>(1, 0);
        img_points2.push_back(pt2);
    }

    cv::Mat Ess = cv::findEssentialMat(img_points1, img_points2, iden33,
                                       cv::RANSAC, 0.999, 0.1);
    cv::Mat float_ess;
    Ess.convertTo(float_ess, CV_32F);
    cv::Mat R, t;

    for (int i = 0; i < 10; ++i) {
        cv::Mat h_pt1(3, 1, CV_32F), h_pt2(3, 1, CV_32F);
        h_pt1.at<float>(0, 0) = img_points1[i].x;
        h_pt1.at<float>(0, 1) = img_points1[i].y;
        h_pt1.at<float>(0, 2) = 1.0f;
        h_pt2.at<float>(0, 0) = img_points2[i].x;
        h_pt2.at<float>(0, 1) = img_points2[i].y;
        h_pt2.at<float>(0, 2) = 1.0f;
        cv::Mat res = h_pt1.t() * float_ess * h_pt2;
        std::cout << "Ess loss: " << res.at<float>(0, 0) << std::endl;
    }
    cv::recoverPose(Ess, img_points1, img_points2, R, t);
    std::cout << "R:\n" << R << std::endl;
    std::cout << "m_t:\n" << t << std::endl;
    cv::Mat proj2_es = get_proj(R, t);

    cv::Mat tri_res;
    cv::triangulatePoints(proj1, proj2_es, img_points1, img_points2, tri_res);
    for (int i = 0; i < 10; ++i) {
        cv::Mat cur_pos = tri_res.col(i) / tri_res.at<float>(3, i);
        std::cout << "ground truth:\n"
                  << map_points[i].rowRange(0, 3) << std::endl;
        std::cout << "estimated:\n" << cur_pos << std::endl;
        for (int j = 0; j < 3; ++j) {
            std::cout << map_points[i].at<float>(j, 0) / cur_pos.at<float>(j, 0)
                      << " ";
        }
        std::cout << std::endl;
    }
}

void test_motion() {
    vo_nono::MotionPredictor predictor;
    cv::Mat v_t = cv::Mat({3.0f, 4.0f, 5.0f});
    cv::Mat rvec = cv::Mat({10.9f, 8.2f, 8.0f});
    for (int i = 0; i <= 30; ++i) {
        double t = (double) i / 30.0;
        cv::Mat cur_T = v_t * (float) t;
        cv::Mat cur_R;
        cv::Rodrigues(t * rvec, cur_R);
        if (predictor.is_available()) {
            cv::Mat pred_R, pred_t;
            predictor.predict_pose(t, pred_R, pred_t);
            assert(cv::norm(pred_R - cur_R) <= 0.0001);
            assert(cv::norm(pred_t - cur_T) <= 0.0001);
        }
        predictor.inform_pose(cur_R, cur_T, t);
    }
}

void test_histogram() {
    auto int_indexer = [](int a) { return a; };
    vo_nono::Histogram<int> histo(100, int_indexer);
    for (int i = 0; i < 100; ++i) { histo.insert_element(i); }
    histo.cal_topK(1);
    for (int i = 0; i < 100; ++i) { assert(histo.is_topK(i)); }
    for (int i = 0; i < 50; ++i) { histo.insert_element(i); }
    for (int i = 0; i < 75; ++i) { histo.insert_element(i); }
    histo.cal_topK(51);
    for (int i = 0; i < 100; ++i) {
        if (i < 75) {
            assert(histo.is_topK(i));
        } else {
            assert(!histo.is_topK(i));
        }
    }
}

void test_bundle_adjustment() {
    using namespace vo_nono;
    cv::Mat intrinsic_mat =
            (cv::Mat_<float>(3, 3) << 500, 0, 300, 0, 500, 250, 0, 0, 1);
    Camera camera(intrinsic_mat, 680, 420);
    cv::Mat pose1 =
            (cv::Mat_<float>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    cv::Mat pose2 =
            (cv::Mat_<float>(3, 4) << 0, -1, 0, 500, 1, 0, 0, 0, 0, 0, 1, 0);

    std::vector<cv::Mat> points, noise_points;
    std::vector<cv::Point2f> proj[2];
    constexpr int pt_cnt = 20;
    for (int i = 0; i < pt_cnt; ++i) {
        cv::Mat coord = (cv::Mat_<float>(3, 1) << i + 2, 4 * i + 3, 500);
        points.push_back(coord);
        cv::Mat proj1 = pose1.colRange(0, 3) * coord + pose1.col(3);
        cv::Mat proj2 = pose2.colRange(0, 3) * coord + pose2.col(3);
        proj1 = intrinsic_mat * proj1;
        proj2 = intrinsic_mat * proj2;
        proj1 /= proj1.at<float>(2);
        proj2 /= proj2.at<float>(2);
        proj[0].emplace_back(
                cv::Point2f(proj1.at<float>(0), proj1.at<float>(1)));
        proj[1].emplace_back(
                cv::Point2f(proj2.at<float>(0), proj2.at<float>(1)));
    }

    OptimizeGraph graph(camera, 2, pt_cnt);
    std::random_device rd_device;
    std::mt19937 gen(rd_device());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    cv::Mat noise_pose1 = pose1;
    cv::Mat noise_pose2 = pose2;
    for (int i = 0; i < points.size(); ++i) {
        cv::Mat noise_point = points[i].clone();
        noise_point.at<float>(0) += dis(gen);
        noise_point.at<float>(1) += dis(gen);
        noise_point.at<float>(2) += dis(gen);
        noise_points.push_back(noise_point);
    }
    graph.cam_poses[0] = noise_pose1;
    graph.cam_poses[1] = noise_pose2;
    for (int i = 0; i < points.size(); ++i) {
        graph.points[i] = noise_points[i];
        for (int j = 0; j < 2; ++j) { graph.add_edge(j, i, proj[j][i]); }
    }
    Optimizer::bundle_adjustment(graph, 40);
    for (int i = 0; i < 2; ++i) {
        if (i == 0) {
            std::cout << "Original: " << pose1 << std::endl;
            std::cout << "Noise: " << noise_pose1 << std::endl;
        } else {
            std::cout << "Original: " << pose2 << std::endl;
            std::cout << "Noise: " << noise_pose2 << std::endl;
        }
        std::cout << "Optimized: " << graph.optim_cam_poses[i] << std::endl;
    }
    for (int i = 0; i < points.size(); ++i) {
        std::cout << "Original: " << points[i] << std::endl;
        std::cout << "Noise: " << noise_points[i] << std::endl;
        std::cout << "Optimized: " << graph.optim_points[i] << std::endl;
    }
}

int main() {
    test_bundle_adjustment();
    return 0;
}