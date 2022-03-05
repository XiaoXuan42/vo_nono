#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "vo_nono/frontend.h"

int main(int argc, const char *argv[]) {
    if (argc == 3) {
        cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        cv::Mat dscpts1, dscpts2;
        std::vector<cv::KeyPoint> kpt1, kpt2;
        vo_nono::Frontend::detect_and_compute(img1, kpt1, dscpts1);
        vo_nono::Frontend::detect_and_compute(img2, kpt2, dscpts2);
        std::vector<cv::DMatch> matches =
                vo_nono::Frontend::match_descriptor(dscpts1, dscpts2);

        cv::Mat output_img1, output_img2;
        cv::drawMatches(img1, kpt1, img2, kpt2, matches, output_img1);
        matches = vo_nono::Frontend::filter_matches(matches, kpt1, kpt2);
        cv::drawMatches(img1, kpt1, img2, kpt2, matches, output_img2);
        cv::imshow("match(no filter)", output_img1);
        cv::imshow("match(filter)", output_img2);
        cv::waitKey(0);
    }
    return 0;
}