#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

cv::Mat get_proj(const cv::Mat &Rcw, const cv::Mat &tcw) {
    cv::Mat proj(3, 4, CV_32F);
    Rcw.copyTo(proj.rowRange(0, 3).colRange(0, 3));
    tcw.copyTo(proj.rowRange(0, 3).col(3));
    return proj;
}

int main() {
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
        float x = (float)(rand() % 100) / 10;
        float y = (float)(rand() % 100) / 10;
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

    cv::Mat Ess = cv::findEssentialMat(img_points1, img_points2, iden33, cv::RANSAC, 0.999, 0.1);
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
    std::cout << "t:\n" << t << std::endl;
    cv::Mat proj2_es = get_proj(R, t);

    cv::Mat tri_res;
    cv::triangulatePoints(proj1, proj2_es, img_points1, img_points2, tri_res);
    for (int i = 0; i < 10; ++i) {
        cv::Mat cur_pos = tri_res.col(i) / tri_res.at<float>(3, i);
        std::cout << "ground truth:\n" << map_points[i].rowRange(0, 3) << std::endl;
        std::cout << "estimated:\n" << cur_pos << std::endl;
        for (int j = 0; j < 3; ++j) {
            std::cout << map_points[i].at<float>(j, 0) / cur_pos.at<float>(j, 0) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}