//
// Created by 谢柏渊 on 2018/7/24.
//

#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "cv_image_reader.h"

int main() {

    std::cout << "hello world!" << std::endl;

//    cv::Mat srcImage = cv::imread(
//            "/Users/xiebaiyuan/PaddleProject/Paddle/paddle/fluid/inference/tests/images/hand.jpg");
//    cv::resize(srcImage, srcImage, cvSize(300, 300));
//    cv::imshow("hand", srcImage);
//    std::vector<cv::Mat> channels;
//
//
//    cv::split(srcImage,channels);
//
//    cv::Mat &channel0 = channels[0];
//    cv::Mat &channel1 = channels[1];
//    cv::Mat &channel2 = channels[2];
//
//    std::cout << "channel0.size=  " << channel0.size << std::endl;
//    std::cout << "channel1.size=  " << channel1.size << std::endl;
//    std::cout << "channel2.size=  " << channel2.size << std::endl;
//
//    cv::imshow("b", channel0);
//    cv::imshow("g", channel1);
//    cv::imshow("r", channel2);

    ReadImageShowChannels("/Users/xiebaiyuan/PaddleProject/Paddle/paddle/fluid/inference/tests/images/hand.jpg");
    // srcImage.resize(Size(300,300));
    cv::waitKey();
    return 0;
};

