#include "vo_nono/frontend.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, const char *argv[]) {
    if (argc == 3) {
        cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        cv::Mat dscpts1, dscpts2;
        std::vector<cv::KeyPoint> kpt1, kpt2;
        vo_nono::Frontend::detect_and_compute(img1, kpt1, dscpts1);
        vo_nono::Frontend::detect_and_compute(img2, kpt2, dscpts2);
        std::vector<cv::DMatch> matches = vo_nono::Frontend::match_descriptor(dscpts1, dscpts2);

        cv::Mat output_img;
        cv::drawMatches(img1, kpt1, img2, kpt2, matches, output_img);
        cv::imshow("match", output_img);
        cv::waitKey(0);
    }
    return 0;
}