//
// Created by 谢柏渊 on 2018/7/24.
//

#include <iostream>
#include <vector>
#include <paddle/fluid/framework/lod_tensor.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "cv_image_reader.h"


void ReadImageShowChannels(const std::string filename) {

    std::cout << "filename: " << (filename) << std::endl;

    cv::Mat srcImage = cv::imread(filename);
    cv::resize(srcImage, srcImage, cvSize(300, 300));
    cv::imshow("hand", srcImage);
    std::vector<cv::Mat> channels;


    cv::split(srcImage, channels);

    cv::Mat &channel0 = channels[0];
    cv::Mat &channel1 = channels[1];
    cv::Mat &channel2 = channels[2];

    std::cout << "channel0.size=  " << channel0.size << std::endl;
    std::cout << "channel1.size=  " << channel1.size << std::endl;
    std::cout << "channel2.size=  " << channel2.size << std::endl;

    cv::imshow("b", channel0);
    cv::imshow("g", channel1);
    cv::imshow("r", channel2);

}

/**
 * 读取图片到pTensor 以供模型使用
 *
 * @param filename  需要读取的文件名
 * @param pTensor   存放输入数据的pTensor
 * @param dim       矩阵尺度
 * @param pMeans    通道偏移,顺序需要与channel顺序一致
 * @param scale     缩放类型
 * @param channel   图片通道类型
 */
void ReadImage(const char *filename, paddle::framework::LoDTensor *input, paddle::framework::DDim &&dims,
               float *pMeans, float scale, CHANNEL_TYPE type) {

    std::cout << "pMeans: " << pMeans << std::endl;
    std::cout << "dims: " << dims << std::endl;

    std::cout << "dim2: " << dims[2] << std::endl;
    std::cout << "dim3: " << dims[3] << std::endl;

    std::cout << "dims size: " << dims.size() << std::endl;
    std::cout << "filename: " << filename << std::endl;


    cv::Mat srcImage = cv::imread(filename);
    cv::resize(srcImage, srcImage, cvSize(static_cast<int>(dims[2]), static_cast<int>(dims[3])));
    std::vector<float> bs;
    std::vector<float> gs;
    std::vector<float> rs;

    for (int row = 0; row < srcImage.rows; row++) {
        // data 是 uchar* 类型的, m.ptr(row) 返回第 row 行数据的首地址
        // 需要注意的是该行数据是按顺序存放的,也就是对于一个 3 通道的 Mat, 一个像素3个通道值, [B,G,R][B,G,R][B,G,R]...
        // 所以一行长度为:sizeof(uchar) * m.cols * m.channels() 个字节
        uchar *data = srcImage.ptr(row);
        for (int col = 0; col < srcImage.cols; col++) {
            bs.push_back(data[col * 3]); //第row行的第col个像素点的第一个通道值 Blue
            gs.push_back(data[col * 3 + 1]);// Green
            rs.push_back(data[col * 3 + 2]);// Red
        }
    }

    std::cout << "bs.size(): " << bs.size();
    std::cout << "gs.size(): " << gs.size();
    std::cout << "rs.size(): " << rs.size();
    auto *input_ptr = input->mutable_data<float>(dims, paddle::platform::CPUPlace());
    std::cout << "input->numel() size = " << input->numel() << std::endl;

    if (type == kRgb) {
        for (int i = 0; i < input->numel(); ++i) {
            if (i < rs.size()) {
                input_ptr[i] = (rs[i] - pMeans[0]) * scale;
            } else if (i < rs.size() + gs.size()) {
                input_ptr[i] = (gs[i - rs.size()] - pMeans[1]) * scale;
            } else {
                input_ptr[i] = (bs[i - rs.size() - gs.size()] - pMeans[2]) * scale;
            }
        }
    } else if (type == kBgr) {
        for (int i = 0; i < input->numel(); ++i) {
            if (i < bs.size()) {
                input_ptr[i] = (bs[i] - pMeans[0]) * scale;
            } else if (i < bs.size() + gs.size()) {
                input_ptr[i] = (gs[i - bs.size()] - pMeans[1]) * scale;
            } else {
                input_ptr[i] = (rs[i - bs.size() - gs.size()] - pMeans[2]) * scale;
            }
        }
    }

}