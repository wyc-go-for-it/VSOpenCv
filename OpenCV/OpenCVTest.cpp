#include "OpenCVTest.h"
OpenCVTest::OpenCVTest()
{
}
OpenCVTest::~OpenCVTest()
{
	destroyAllWindows();
}
void OpenCVTest::readVideo(const string &path)
{
	VideoCapture video(path);//定义VideoCapture类video  
	if (!video.isOpened())  //对video进行异常检测  
	{
		cout << "video open error!" << endl;
		return;
	}
	// 获取帧数
	double frameCount = video.get(CAP_PROP_FRAME_COUNT);
	// 获取FPS 
	double FPS = video.get(CAP_PROP_FPS);
	// 存储帧
	Mat frame;
	// 存储背景图像
	Mat background;
	// 存储结果图像
	Mat result;
	// 存储前一帧图像
	Mat temp;
	for (int i = 0; i < frameCount; i++)
	{
		// 读帧进frame
		video >> frame;
		// 对帧进行异常检测
		if (frame.empty())
		{
			cout << "frame is empty!" << endl;
			break;
		}
		imshow("frame", frame);

		// 获取帧位置(第几帧)
		double framePosition = video.get(CAP_PROP_POS_FRAMES);
		cout << "framePosition: " << framePosition << endl;
		// 如果为第一帧（temp还为空）
		if (i == 0)
		{
			// 调用MoveDetect()进行运动物体检测，返回值存入result
			result = MoveDetect(frame, frame);
		}
		//若不是第一帧（temp有值了）
		else
		{
			// 调用MoveDetect()进行运动物体检测，返回值存入result
			result = MoveDetect(temp, frame);
		}
		imshow("result", result);
		//按原FPS显示
		if (waitKey(500 / FPS) == 27)
		{
			cout << "ESC退出!" << endl;
			break;
		}
		temp = frame.clone();
	}

}
Mat OpenCVTest::MoveDetect(Mat &background, Mat &frame)
{
	Mat result = frame.clone();
	// 1.将background和frame转为灰度图  
	Mat gray1, gray2;
	cvtColor(background, gray1, COLOR_BGR2GRAY);
	cvtColor(frame, gray2, COLOR_BGR2GRAY);
	// 2.将background和frame做差  
	Mat diff;
	absdiff(gray1, gray2, diff);
	imshow("diff", diff);
	// 3.对差值图diff_thresh进行阈值化处理  
	Mat diff_thresh;
	threshold(diff, diff_thresh, 50, 255, THRESH_BINARY);
	imshow("diff_thresh", diff_thresh);
	// 4.腐蚀  
	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(diff_thresh, diff_thresh, kernel_erode);
	imshow("erode", diff_thresh);
	// 5.膨胀  
	dilate(diff_thresh, diff_thresh, kernel_dilate);
	imshow("dilate", diff_thresh);
	// 6.查找轮廓并绘制轮廓  
	vector<vector<Point>> contours;
	findContours(diff_thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	// 在result上绘制轮廓
	drawContours(result, contours, -1, Scalar(0, 0, 255), 2);
	// 7.查找正外接矩形  
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		// 在result上绘制正外接矩形
		rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);
	}
	// 返回result
	return result;
}
void OpenCVTest::image2Gray(const string & path)
{
	Mat im = imread(path);
	if (im.empty())
	{
		cout << "read image error...";
		waitKey(3000);
		return;
	}
	const string origin = "origin", gray = "gray";
	namedWindow(origin, WINDOW_KEEPRATIO);
	imshow(origin,im);

	Mat grayImage;
	cvtColor(im, grayImage,COLOR_BGR2GRAY);

	namedWindow(gray, WINDOW_KEEPRATIO);
	imshow(gray, grayImage);

}

void OpenCVTest::mouseCallback(const string & path)
{
	mMouseDrawImage = imread(path);
	if (mMouseDrawImage.empty())
	{
		cout << "read image error...";
		waitKey(3000);
		return;
	}
	namedWindow(windowName,WINDOW_FREERATIO);
	imshow(windowName, mMouseDrawImage);
	cv::setMouseCallback(windowName, mouseTrack,this);
}
void OpenCVTest::mouseTrack(int event, int x, int y, int flags, void * userdata)
{
	OpenCVTest *t =  ((OpenCVTest *)userdata);
 
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		if (!t->isDown) {
			t->isDown = true;
			t->mDown.x = x;
			t->mDown.y = y;
		}
		cout << "EVENT_LBUTTONDOWN " << "X:" << x << "Y:" << y << "\n";
		break;
	case EVENT_MOUSEMOVE:
		if (t->isDown)
		{
			Mat mask = t->mMouseDrawImage.clone();

			Point cur(x, y);

			int d_x = t->mDown.x;
			int d_y = t->mDown.y;

			Rect rect;

			if (x > d_x || y > d_y)
			{
				rect.x = d_x;
				rect.y = d_y;
				rect.width = abs(x -d_x);
				rect.height = abs(y - d_y);
			}
			else {
			
				rect.x = x;
				rect.y = y;
				rect.width = abs(d_x - x);
				rect.height = abs(d_y - y);

			}

			rectangle(mask, t->mDown, cur, Scalar(0, 0, 255), 2);

			stringstream ss;
			ss << "X:" << x << "Y:" << y;
			putText(mask,ss.str(),cur, FONT_HERSHEY_SCRIPT_SIMPLEX,2,Scalar::all(255),2, 8);

			ss.str("");
			ss << "width:" << rect.width << "height:" << rect.height;
			putText(mask, ss.str(), t->mDown, FONT_HERSHEY_SCRIPT_SIMPLEX,1, Scalar(0,0,255),2, 8);
 
			imshow(t->windowName, mask);

			if (rect.area() > 0 && x <= mask.rows  && y <= mask.cols && x >= 0  && y >=  0)
			{
				t->mMouseDrawImage.copyTo(mask);
				imshow(t->captureWindow,mask(rect));
			}
		}
		break;
	case EVENT_LBUTTONUP:
		if (t->isDown) {
			t->isDown = false;
			t->mDown.x = 0;
			t->mDown.y = 0;
		}
		cout << "EVENT_LBUTTONUP " << "X:" << x << "Y:" << y << "\n";
		break;
	default:
		break;
	}
}
