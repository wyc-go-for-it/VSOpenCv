#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
class OpenCVTest
{
public:
	OpenCVTest();
	~OpenCVTest();
	void readVideo(const string &path);
	void image2Gray(const string &path);
	void mouseCallback(const string &path);
private:
	Mat MoveDetect(Mat &background, Mat &frame);
	//НиЭМ
	static void mouseTrack(int event, int x, int y, int flags, void *userdata);
	const string windowName = "mouseTrack",captureWindow = "capture";
	Mat mMouseDrawImage;
	Point mDown;
	bool isDown;
};
