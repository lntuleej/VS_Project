// VideoProcessor.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

class VideoProcessor
{
public:
	VideoProcessor() :callIt(true), delay(0), fnumber(0), stop(0), 
		frameToStop(-1){}
	// 设置回调函数
	void setFrameProcessor(
		void(*frameProcessingCallback)(Mat&, Mat&)) {
		process = frameProcessingCallback;
	};

	// 设置视频文件名称
	bool setInput(string filename);

	// 创建输入窗口
	void displayInput(string wn);

	// 创建输出窗口
	void displayOutput(string wn);

	// 不再显示处理后的帧
	void dontDisplay();

	// 获取并处理序列帧
	void run();

	// 是否开始了捕获设备？
	bool isOpened();

	// 是否已经停止？
	bool isStopped();

	// 停止运行
	void stopIt();

	// 得到下一帧
	bool readNextFrame(Mat& frame);

	// 返回下一帧的帧数
	long getFrameNumber();

	// 析构函数
	~VideoProcessor();

private:
	// OpenCV视频捕捉对象
	VideoCapture capture;
	// 每帧调用的回调函数
	void(*process)(Mat&, Mat&);
	// 确定是否调用回调函数的bool变量
	bool callIt;
	// 输入窗口名称
	string windowNameInput;
	// 输出窗口名称
	string windowNameOutput;
	// 延迟
	int delay;
	// 已处理的帧数
	long fnumber;
	// 在该帧数停止
	long frameToStop;
	// 是否停止处理
	bool stop;
};

VideoProcessor::VideoProcessor()
{
}

bool VideoProcessor::setInput(string filename)
{
	fnumber = 0;
	// 释放之前打开过的资源
	capture.release();
	// 打开视频文件
	return capture.open(filename);
}

void VideoProcessor::displayInput(string wn)
{
	windowNameInput = wn;
	namedWindow(windowNameInput);
}

void VideoProcessor::displayOutput(string wn)
{
	windowNameOutput = wn;
	namedWindow(windowNameOutput);
}

void VideoProcessor::dontDisplay()
{
	destroyWindow(windowNameInput);
	destroyWindow(windowNameOutput);
	windowNameInput.clear();
	windowNameOutput.clear();
}

void VideoProcessor::run()
{
	// 当前帧
	Mat frame;
	// 输出帧
	Mat output;
	// 打开失败时
	if (!isOpened())
	{
		return;
	}
	stop = false;
	while (!isStopped())
	{
		// 读取下一帧
		if (! readNextFrame(frame))
		{
			break;
		}

		// 显示输出帧
		if (windowNameInput.length() != 0)
		{
			imshow(windowNameInput, frame);
		}

		// 调用处理函数
		if (callIt)
		{
			// 处理当前帧
			process(frame, output);
			// 增加帧数
			fnumber++;
		}
		else
		{
			output = frame;
		}

		// 显示输出帧
		if (windowNameOutput.length() != 0)
		{
			imshow(windowNameOutput, output);
		}

		// 引入延迟
		if (delay >= 0 && waitKey(delay) >= 0)
		{
			stopIt();
		}

		// 检查是否需要停止运行
		if (frameToStop >= 0 && getFrameNumber() == frameToStop)
		{
			stopIt();
		}
	}
}

bool VideoProcessor::isOpened()
{
	return stop;
}

bool VideoProcessor::isStopped()
{
	return stop;
}

void VideoProcessor::stopIt()
{
	stop = true;
}

bool VideoProcessor::readNextFrame(Mat & frame)
{
	return capture.read(frame);
}

long VideoProcessor::getFrameNumber()
{
	// 得到不会设备信息
	long fnumber = static_cast<long>(capture.get(CV_CAP_PROP_POS_FRAMES));
	return fnumber;
}

VideoProcessor::~VideoProcessor()
{
}

void canny(Mat& img, Mat& out)
{
	// 灰度转换
	if (img.channels() == 3)
	{
		cvtColor(img, out, CV_BGR2GRAY);
	}
	// 计算边缘
	Canny(out, out, 100, 200);
	// 反转图像
	threshold(out, out, 128, 255, THRESH_BINARY_INV);
}

int main()
{
	;
}

