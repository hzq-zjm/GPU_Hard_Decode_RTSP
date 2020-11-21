/*
 * @Author: your name
 * @Date: 2020-11-09 19:09:02
 * @LastEditTime: 2020-11-12 14:18:52
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /cudaDecodeGL_copy/main.cpp
 */
#include "NvDecode.h"
#include "opencv2/opencv.hpp"
#include <iostream>

int main(int argc, char* argv[])
{	if(argc!=2){printf("输入参数错误!\n"); return -1;}
	const std::string rstpUrl(argv[1]);
	NvDecode decod;
	if(decod.start(rstpUrl)!=0) return -1;
	uchar *bgraPtr = nullptr;
	int width = 0, height = 0;
	unsigned long long timestamp = 0;
	while (!decod.m_pFrameQueue->isEndOfDecode())
	{
		if (decod.deQueueFrame(&bgraPtr, &width, &height, &timestamp)) {
			//std::cout<<timestamp<<std::endl;
			cv::Mat frame(height, width, CV_8UC4);
			frame.data = bgraPtr;
			cv::imshow("video", frame);
			cv::waitKey(30);
			frame.release();
		}
		else {
			cv::waitKey(30);
			continue;
		}
	}
	return 0;
}
