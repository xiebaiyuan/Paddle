//
// Created by 谢柏渊 on 2018/7/24.
//

#ifndef PADDLE_CV_IMAGE_READER_H
#define PADDLE_CV_IMAGE_READER_H

#include <paddle/fluid/framework/lod_tensor.h>


/**
* 读取图片显示不同channels
* @param filename
*/
void ReadImageShowChannels(std::string filename);

/**
 * 通道类型
 */
enum CHANNEL_TYPE {
    kRgb,
    kBgr
};

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
void ReadImage(const char *filename, paddle::framework::LoDTensor *pTensor, paddle::framework::DDim &&dim,
               float *pMeans, float scale, CHANNEL_TYPE channel);


#endif //PADDLE_CV_IMAGE_READER_H
