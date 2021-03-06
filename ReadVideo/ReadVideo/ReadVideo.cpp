// ReadVideo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <fstream>
#include <iostream>
#include <string>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

int main()
{
	// 打开视频文件
	VideoCapture capture("video.mov");
	// 检查视频是否成功打开
	if (!capture.isOpened())
	{
		return 1;
	}
	// 获取帧率
	double rate = capture.get(CV_CAP_PROP_FPS);
	bool stop(false);
	// 当前视频帧
	Mat frame;
	namedWindow("Extracted Frame");
	// 每一帧之间的延迟
	// 与视频的帧率相对应
	int delay = 1000 / rate;

	int i = 0; 

	// 遍历每一帧
	while (!stop)
	{
		// 尝试读取下一帧
		if (!capture.read(frame))
		{
			cout << "read faild" << endl;
			break;
		}
		
		i++;

		imshow("Extracted Frame", frame);
		// 引入延迟
		// 也可以通过按停止键停止
		waitKey(delay);

		cout << i << "\n" << endl;
	}
	// 关闭视频文件
	// 将由析构函数调用，因此非必须
	capture.release();
}

