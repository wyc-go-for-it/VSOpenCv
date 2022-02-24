#include "OpenCVTest.h"
#include <opencv2/text.hpp>
#include <opencv2/dnn.hpp>
using namespace std;
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
		waitKey(100000);
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
	int i = 0;
	for (;;)
	{
		// 读帧进frame
		video >> frame;
		// 对帧进行异常检测
		if (frame.empty())
		{
			cout << "frame is empty!" << endl;
			break;
		}
		cv::imshow("frame", frame);

		// 获取帧位置(第几帧)
		double framePosition = video.get(CAP_PROP_POS_FRAMES);
		cout << "framePosition: " << framePosition << endl;
		// 如果为第一帧（temp还为空）
		if (i == 0)
		{
			// 调用MoveDetect()进行运动物体检测，返回值存入result
			result = MoveDetect(frame, frame);
			i++;
		}
		//若不是第一帧（temp有值了）
		else
		{
			// 调用MoveDetect()进行运动物体检测，返回值存入result
			result = MoveDetect(temp, frame);
		}
		cv::imshow("result", result);
		//按原FPS显示
 		if (waitKey(FPS) == 27)
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
	for (uint i = 0; i < contours.size(); i++)
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
	namedWindow(mouseTrackWindowName,WINDOW_FREERATIO);
	imshow(mouseTrackWindowName, mMouseDrawImage);
	cv::setMouseCallback(mouseTrackWindowName, mouseTrack,this);
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
 
			imshow(t->mouseTrackWindowName, mask);

			if (rect.area() > 0 && x <= mask.rows  && y <= mask.cols && x >= 0  && y >=  0)
			{
				t->mMouseDrawImage.copyTo(mask);
				t->mCaptureImage = mask(rect);
				imshow(t->captureWindow, t->mCaptureImage);
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
		if (!t->mCaptureImage.empty())
		{
			t->OMatchTemplate(t->mMouseDrawImage, t->mCaptureImage);
		}
		break;
	default:
		break;
	}
}
void OpenCVTest::textbox_draw(Mat src, std::vector<Rect>& groups, std::vector<float>& probs, std::vector<int>& indexes)
{
	for (size_t i = 0; i < indexes.size(); i++)
	{
		if (src.type() == CV_8UC3)
		{
			Rect currrentBox = groups[indexes[i]];
			rectangle(src, currrentBox, Scalar(0, 255, 255), 2, LINE_AA);
			String label = format("%.2f", probs[indexes[i]]);
			std::cout << "text box: " << currrentBox << " confidence: " << probs[indexes[i]] << "\n";

			int baseLine = 0;
			Size labelSize = getTextSize(label, FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
			int yLeftBottom = max(currrentBox.y, labelSize.height);
			rectangle(src, Point(currrentBox.x, yLeftBottom - labelSize.height),
				Point(currrentBox.x + labelSize.width, yLeftBottom + baseLine), Scalar(255, 255, 255), FILLED);

			putText(src, label, Point(currrentBox.x, yLeftBottom), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0), 1, LINE_AA);
		}
		else
			rectangle(src, groups[i], Scalar(255), 3, 8);
	}
}
void OpenCVTest::recognition(const string & path)
{
	const string model_dir = "./vc15/modelfiles/";
	const string modelArch = model_dir + "textbox.prototxt";
	const string moddelWeights = model_dir + "TextBoxes_icdar13.caffemodel";
 
	bool a = fileExists(modelArch);
	bool b = fileExists(moddelWeights);

	if (!a || !b)
	{
		cout << "Model files not found in the current directory. Aborting!" << endl;
		return;
	}

	Mat image = imread(path, IMREAD_COLOR);

	cout << "Starting Text Box Demo" << endl;
	Ptr<text::TextDetectorCNN> textSpotter =
		text::TextDetectorCNN::create(modelArch, moddelWeights);

	vector<Rect> bbox;
	vector<float> outProbabillities;
	textSpotter->detect(image, bbox, outProbabillities);
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(bbox, outProbabillities, 0.4f, 0.5f, indexes);

	Mat image_copy = image.clone();
	textbox_draw(image_copy, bbox, outProbabillities, indexes);
	imshow("Text detection", image_copy);
	image_copy = image.clone();

	Ptr<text::OCRHolisticWordRecognizer> wordSpotter = text::OCRHolisticWordRecognizer::create(model_dir + "dictnet_vgg_deploy.prototxt", model_dir + "dictnet_vgg.caffemodel", model_dir + "dictnet_vgg_labels.txt");

	for (size_t i = 0; i < indexes.size(); i++)
	{
		Mat wordImg;
		cvtColor(image(bbox[indexes[i]]), wordImg, COLOR_BGR2GRAY);
		string word;
		vector<float> confs;
		wordSpotter->run(wordImg, word, NULL, NULL, &confs);

		Rect currrentBox = bbox[indexes[i]];
		rectangle(image_copy, currrentBox, Scalar(0, 255, 255), 2, LINE_AA);

		int baseLine = 0;
		Size labelSize = getTextSize(word, FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
		int yLeftBottom = max(currrentBox.y, labelSize.height);
		rectangle(image_copy, Point(currrentBox.x, yLeftBottom - labelSize.height),
			Point(currrentBox.x + labelSize.width, yLeftBottom + baseLine), Scalar(255, 255, 255), FILLED);

		putText(image_copy, word, Point(currrentBox.x, yLeftBottom), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0), 1, LINE_AA);

	}
	imshow("Text recognition", image_copy);
	cout << "Recognition finished. Press any key to exit.\n";
}
void OpenCVTest::recognition_number(const string & path)
{
	Mat src = imread(path);

	Mat gary;
	cvtColor(src,gary,COLOR_BGR2GRAY);

	//二值
	Mat thresh_img, thresh_copy;
	threshold(gary, thresh_img,135,255, THRESH_BINARY);
	imshow("thresh_img", thresh_img);
	thresh_copy = thresh_img.clone();

	// 4.腐蚀  
	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(thresh_img, thresh_img, kernel_erode);
	imshow("erode", thresh_img);
  
	blur(thresh_img, thresh_img, Size(3, 3));
	//边缘检测
	Canny(thresh_img, thresh_img, 3, 9, 3);
	imshow("Canny", thresh_img);

	vector<vector<Point>> contours;
	findContours(thresh_img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
 
 
	vector<string> files_names = listFiles("./cut_img/*.*");
	string fname;

	Mat cut,comp,resize_img;
	Rect t_rect;
	for (uint i = 0; i < contours.size(); i++)
	{
		t_rect = boundingRect(contours[i]);
 
		cut = ~thresh_copy(t_rect);

		
		for (size_t i = 0,size = files_names.size(); i < size; i++)
		{
			fname ="./cut_img/"+ files_names.at(i);
			comp = imread(fname);
			threshold(comp, comp, 135, 255, THRESH_BINARY);
			imshow("comp", comp);
			if (!comp.empty()) {

				resize(cut, resize_img, comp.size(),0,0,INTER_AREA);
				imshow("resize", resize_img);

				waitKey();

				float percentage = compare(resize_img,comp);
				cout << "file:" << fname << " percentage:" << percentage << "\n";

				if (percentage > 0.5) {
					stringstream ss;
					ss << i;
					putText(src, ss.str(), Point(t_rect.x , t_rect.y + t_rect.height / 2), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar(0,0,255), 2, 8);
				}
			}
		}

		// 在result上绘制正外接矩形
		rectangle(src, t_rect, Scalar(0, 255, 0), 2);
	}
	imshow("src", src);
}
float OpenCVTest::compare(Mat src, Mat model)
{
 
	Mat re_model;
	resize(model, re_model, src.size());
	int rows, cols;
	uchar *src_data, *model_data;
	rows = re_model.rows;
	cols = re_model.cols*src.channels();
	float percentage, same = 0.0, different = 0.0;

	for (int i = 0; i < rows; i++)       //遍历图像像素
	{
		src_data = src.ptr<uchar>(i);
		model_data = re_model.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			if (src_data[j] == model_data[j])
			{
				same++;         //记录像素值相同的个数
			}
			else
			{
				different++;    //记录像素值不同的个数
			}
		}
	}
	percentage = same / (same + different);
	return percentage;                     //返回相似度
}
void OpenCVTest::Qpyr(const string & path)
{
	Mat img = imread(path);
	imshow("原始图", img);

	Mat dstUp, dstDown;
 
	pyrUp(img, dstUp,Size(img.cols * 2, img.rows * 2)); //放大一倍
	pyrDown(img, dstDown, cv::Size2i::Size_(img.cols << 1, img.rows << 1)); //缩小为原来的一半
	imshow("尺寸放大之后", dstUp);
	imshow("尺寸缩小之后", dstDown);

}
void OpenCVTest::OCanny(const string &path) {
	Mat img = imread(path);
	imshow("原始图", img);
	Mat edge, grayImage;

	//将原始图转化为灰度图
	cvtColor(img, grayImage, COLOR_BGR2GRAY);

	//先使用3*3内核来降噪
	blur(grayImage, edge, Size(3, 3));

	//运行canny算子
	Canny(edge, edge, 3, 9, 3);

	imshow("边缘提取效果", edge);


	//Sobel算法
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst;

	//求x方向梯度
	Sobel(grayImage, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	imshow("x方向soble", abs_grad_x);

	//求y方向梯度
	Sobel(grayImage, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("y向soble", abs_grad_y);

	//合并梯度
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	imshow("整体方向soble", dst);

	waitKey(0);
}
void OpenCVTest::OHoughLines(const string &path) {
	Mat srcImage = imread(path);
	imshow("Src Pic", srcImage);

	Mat midImage, dstImage;

	Canny(srcImage, midImage, 50, 200, 3);

	cvtColor(midImage, dstImage, COLOR_GRAY2BGR);
 
	vector<Vec2f> lines;
 
	HoughLines(midImage, lines, 1, CV_PI / 180, 350, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0]; 
		float theta = lines[i][1];

		cout << "theta:" << 180 / CV_PI * theta << "\n";

		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 - 280 * (b));
		pt1.y = cvRound(y0 + 280 * (a));
		pt2.x = cvRound(x0 + 280 * (b));
		pt2.y = cvRound(y0 - 280 * (a));

		circle(dstImage,Point(x0,y0), 5, Scalar(0, 0, 255));
	 

		line(dstImage, pt1, pt2, Scalar(55, 100, 195), 1, LINE_AA);
		imshow("边缘检测后的图", midImage);
		imshow("最终效果图", dstImage);
	}
}

void OpenCVTest::OHoughLinesP(const string & path)
{
	Mat srcImage = imread(path);
	imshow("Src Pic", srcImage);

	Mat midImage, dstImage;

	Canny(srcImage, midImage, 50, 200, 3);
	cvtColor(midImage, dstImage, COLOR_GRAY2BGR);




	vector<Vec4i> lines;
	//与HoughLines不同的是，HoughLinesP得到lines的是含有直线上点的坐标的，所以下面进行划线时就不再需要自己求出两个点来确定唯一的直线了
	HoughLinesP(midImage, lines, 1, CV_PI / 180, 80, 50, 10);//注意第五个参数，为阈值

	//依次画出每条线段
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];

		line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, LINE_AA); //Scalar函数用于调节线段颜色
	}
	imshow("边缘检测后的图", midImage);
	imshow("最终效果图", dstImage);
}

void OpenCVTest::ORemap(const string & path,Mat &dstImage)
{
	Mat srcImage = imread(path);

	if (!srcImage.data)
	{
		cout << "找不到这张图片！" << endl;
		return ;
	}

	imshow("Src Pic", srcImage);

	Mat map_x, map_y;
	dstImage.create(srcImage.size(), srcImage.type());//创建和原图一样的效果图
	map_x.create(srcImage.size(), CV_32FC1);
	map_y.create(srcImage.size(), CV_32FC1);

	//遍历每一个像素点，改变map_x & map_y的值,实现对角翻转
	for (int j = 0; j < srcImage.rows; j++)
	{
		for (int i = 0; i < srcImage.cols; i++)
		{
			map_x.at<float>(j, i) = static_cast<float>(srcImage.cols - i);
			map_y.at<float>(j, i) = static_cast<float>(srcImage.rows - j);
		}
	}

	//进行重映射操作
	remap(srcImage, dstImage, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	imshow("重映射效果图", dstImage);
}

void OpenCVTest::OMatchTemplate(const Mat &img, const Mat &templ)
{
	g_srcImage = img.clone();
	if (!g_srcImage.data)
	{
		cout << "原始图读取失败" << endl;
		return ;
	}
	g_tempalteImage = templ;
	if (!g_tempalteImage.data)
	{
		cout << "模板图读取失败" << endl;
		return ;
	}

	namedWindow("原始图", WINDOW_AUTOSIZE);
	createTrackbar("方法", "原始图", &g_nMatchMethod, g_nMaxTrackbarNum, on_matching,this);

	on_matching(0, this);

}
void OpenCVTest::on_matching(int, void * obj)
{
	OpenCVTest *t = ((OpenCVTest *)obj);

	Mat srcImage;
	t->g_srcImage.copyTo(srcImage);
	int resultImage_cols = t->g_srcImage.cols - t->g_tempalteImage.cols + 1;
	int resultImage_rows = t->g_srcImage.rows - t->g_tempalteImage.rows + 1;
	t->g_resultImage.create(resultImage_cols, resultImage_rows, CV_32FC3);

	matchTemplate(t->g_srcImage, t->g_tempalteImage, t->g_resultImage, t->g_nMatchMethod);
	normalize(t->g_resultImage, t->g_resultImage, 0, 2, NORM_MINMAX, -1, Mat());
	double minValue, maxValue;
	Point minLocation, maxLocation, matchLocation;
	minMaxLoc(t->g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation);

	if (t->g_nMatchMethod == TM_SQDIFF || t->g_nMatchMethod == TM_SQDIFF_NORMED)
	{
		matchLocation = minLocation;
	}
	else
	{
		matchLocation = maxLocation;
	}

	rectangle(srcImage, matchLocation, Point(matchLocation.x + t->g_tempalteImage.cols, matchLocation.y + t->g_tempalteImage.rows), Scalar(0, 0, 255), 2, 8, 0);
	rectangle(t->g_resultImage, matchLocation, Point(matchLocation.x + t->g_tempalteImage.cols, matchLocation.y + t->g_tempalteImage.rows), Scalar(0, 0, 255), 2, 8, 0);

	imshow("原始图", srcImage);
}

void OpenCVTest::ORepairImg(const string & path)
{
	Mat imageSource = imread(path);
	if (!imageSource.data)
	{
		cout << "load image error...";
		return;
	}
	imshow("原图", imageSource);
	Mat imageGray;
	//转换为灰度图
	cvtColor(imageSource, imageGray, COLOR_BGR2GRAY, 0);
	Mat imageMask = Mat(imageSource.size(), CV_8UC1, Scalar::all(0));

	//通过阈值处理生成Mask
	threshold(imageGray, imageMask, 120, 255,THRESH_BINARY);
	Mat Kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	//对Mask膨胀处理，增加Mask面积
	dilate(imageMask, imageMask, Kernel);

	//图像修复
	inpaint(imageSource, imageMask, imageSource, 5, INPAINT_NS);
	imshow("Mask", imageMask);
	imshow("修复后", imageSource);
}

void OpenCVTest::OContours(const string & path)
{
	ContourSrc = imread(path);

	/// 把原图像转化成灰度图像并进行平滑
	cvtColor(ContourSrc, ContourSrcGray, COLOR_BGR2GRAY);
	blur(ContourSrcGray, ContourSrcGray, Size(3, 3));

	/// 创建新窗口
	const char* source_window = "Source";
	namedWindow(source_window, WINDOW_AUTOSIZE);
	imshow(source_window, ContourSrc);

	createTrackbar("thresh:", "Source", &ContourThresh, ContourMaxThresh, thresh_callback,this);
	thresh_callback(0, this);
}

void OpenCVTest::thresh_callback(int, void* userdata)
{

	OpenCVTest *t = ((OpenCVTest *)userdata);

	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// 使用Threshold检测边缘
	threshold(t->ContourSrcGray, threshold_output, t->ContourThresh, 255, THRESH_BINARY);
	/// 找到轮廓
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// 多边形逼近轮廓 + 获取矩形和圆形边界框
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);
	}

	cv::RNG rng(12345);
	/// 画多边形轮廓 + 包围的矩形框 + 圆形框
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}

	/// 显示在一个窗口
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}

void OpenCVTest::OpenCamera() {
	VideoCapture cap(0, CAP_DSHOW);
	if (cap.isOpened())
	{
		Mat frame,dstFrame;
		Mat map_x, map_y;

		while (true)
		{
			cap >> frame;
			//cv::flip(frame,dstFrame,1);

			map_x.create(frame.size(), CV_32FC1);
			map_y.create(frame.size(), CV_32FC1);
			//遍历每一个像素点，改变map_x & map_y的值,实现对角翻转
			for (int j = 0; j < frame.rows; j++)
			{
				for (int i = 0; i < frame.cols; i++)
				{
					map_x.at<float>(j, i) = static_cast<float>(frame.cols - i);
					map_y.at<float>(j, i) = static_cast<float>(j);
				}
			}

			//进行重映射操作
			remap(frame, dstFrame, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

			cvtColor(frame, map_x, COLOR_BGR2GRAY);
			imshow("camera1", map_x);
			imshow("camera", dstFrame);
			if (waitKey(1) == 27)
			{
				break;
			}
		}
		destroyWindow("camera");
	}
}

void OpenCVTest::imgBlur() {
	int DELAY_BLUR = 100;
	int MAX_KERNEL_LENGTH = 31;
	Mat src; Mat dst;
 
	src = imread("../data/lena.jpg", IMREAD_COLOR);
	if (display_dst("Original Image",src) != 0) { return; }
	dst = src.clone();
	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
	{
		blur(src, dst, Size(i, i), Point(-1, -1));
		if (display_dst("Homogeneous Blur",dst) != 0) { return ; }
	}
 
	dst = src.clone();
	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
	{
		GaussianBlur(src, dst, Size(i, i), 0, 0);
		if (display_dst("Gaussian Blur",dst) != 0) { return ; }
	}

	dst = src.clone();
	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
	{
		medianBlur(src, dst, i);
		if (display_dst("Median Blur",dst) != 0) { return ; }
	}
 
	dst = src.clone();
	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
	{
		bilateralFilter(src, dst, i, i * 2, i / 2);
		if (display_dst("Bilateral Blur",dst) != 0) { return ; }
	}
	waitKey(0);
}

int OpenCVTest::display_dst(const char *caption, const Mat & dst)
{
	imshow(caption, dst);
	waitKey(150);
	return 0;
}

void OpenCVTest::morphologyOperations() {
	String imageName("../data/morphology.jpg"); // by default
 
	src = imread(imageName, IMREAD_COLOR); // Load an image
	if (src.empty())
	{
		return;
	}
	namedWindow(window_name, WINDOW_NORMAL); // Create window
	createTrackbar("Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat", "Morphology Transformations Demo", &morph_operator, max_operator, Morphology_Operations,this);
	createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
		&morph_elem, max_elem,
		Morphology_Operations, this);
	createTrackbar("Kernel size:\n 2n +1", window_name,
		&morph_size, max_kernel_size,
		Morphology_Operations, this);
	Morphology_Operations(0, this);
}
void OpenCVTest::Morphology_Operations(int, void * data) {
	const OpenCVTest *ins = static_cast<OpenCVTest *>(data);
	/*
	operation
	开：MORPH_OPEN：2 通过图像的侵蚀获得的，随后是扩张
	关：MORPH_CLOSE：3 图像的扩张，然后是侵蚀获得的
	形态梯度：MORPH_GRADIENT：4 图像的扩张和侵蚀的区别 ,找轮廓
	顶帽：MORPH_TOPHAT：5 图像与其《开》之间的区别
	黑帽子：MORPH_BLACKHAT：6	图像与其《必》之间的区别
	*/
  int operation = ins->morph_operator + 2;
  Mat element = getStructuringElement(ins->morph_elem, Size( 2* ins->morph_size + 1, 2* ins->morph_size+1 ), Point(ins->morph_size, ins->morph_size ) );
  Mat dst;
  morphologyEx(ins->src, dst, operation, element );
  imshow(ins->window_name, dst );
}

void OpenCVTest::hitAndMiss() {
	Mat input_image = (Mat_<uchar>(16, 16) <<
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 255, 255, 255, 0, 0, 0, 255, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 255, 255, 255, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 255, 255, 255, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 0, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 255, 255, 255, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 255, 255, 255, 0, 0, 0, 255, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 255, 255, 255, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 255, 255, 255, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 0, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 0, 0, 255,
		0, 255, 255, 255, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255);


	Mat kernel = (Mat_<int>(3, 3) <<
		0, -1, -1,
		1, 1, -1,
		0, 1, 0);
	Mat output_image;
	morphologyEx(input_image, output_image, MORPH_HITMISS, kernel);
	const int rate = 10;
	kernel = (kernel + 1) * 127;
	kernel.convertTo(kernel, CV_8U);
	cout << kernel << endl;
	resize(kernel, kernel, Size(), rate, rate, INTER_NEAREST);
	imshow("kernel", kernel);
	resize(input_image, input_image, Size(), rate, rate, INTER_NEAREST);
	imshow("Original", input_image);
	resize(output_image, output_image, Size(), rate, rate, INTER_NEAREST);
	imshow("Hit or Miss", output_image);
	waitKey(0);
}

void OpenCVTest::getVerticalAndHorizontalLine() {

	// Load the image
	Mat src = imread("../data/music.jpg");
	// Check if image is loaded fine
	if (!src.data)
		cerr << "Problem loading image!!!" << endl;
	// Show source image
	imshow("src", src);
	// Transform source image to gray if it is not
	Mat gray;
	if (src.channels() == 3)
	{
		cvtColor(src, gray, COLOR_BGR2GRAY);
	}
	else
	{
		gray = src;
	}
	// Show gray image
	imshow("gray", gray);
	// Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
	Mat bw;
	adaptiveThreshold(~gray, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, -2);
	// Show binary image
	imshow("binary", bw);
	// Create the images that will use to extract the horizontal and vertical lines
	Mat horizontal = bw.clone();
	Mat vertical = bw.clone();
	// Specify size on horizontal axis
	int horizontalsize = horizontal.cols / 30;
	// Create structure element for extracting horizontal lines through morphology operations
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
	// Apply morphology operations
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	// Show extracted horizontal lines
	horizontal = ~horizontal;
	imshow("horizontal", horizontal);
	// Specify size on vertical axis
	int verticalsize = vertical.rows / 30;
	// Create structure element for extracting vertical lines through morphology operations
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
	// Apply morphology operations
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));
	// Show extracted vertical lines
	imshow("vertical", vertical);
 
	// Inverse vertical image
	bitwise_not(vertical, vertical);

	imshow("vertical_bit", vertical);
	// Extract edges and smooth image according to the logic
	// 1. extract edges
	// 2. dilate(edges)
	// 3. src.copyTo(smooth)
	// 4. blur smooth img
	// 5. smooth.copyTo(src, edges)
	// Step 1
	Mat edges;
	adaptiveThreshold(vertical, edges, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
	imshow("edges", edges);
	// Step 2
	Mat kernel = Mat::ones(2, 2, CV_8UC1);
	dilate(edges, edges, kernel);
	imshow("dilate", edges);
	// Step 3
	Mat smooth;
	vertical.copyTo(smooth);
	// Step 4
	blur(smooth, smooth, Size(2, 2));
	// Step 5
	smooth.copyTo(vertical, edges);
	// Show final result
	imshow("smooth", vertical);
	waitKey(0);
}

void OpenCVTest::linearFilter() {
	Mat src, dst;
	Mat kernel;
	Point anchor;
	double delta;
	int ddepth;
	int kernel_size;
	string window_name = "filter2D Demo-";
	String imageName("../data/lena.jpg"); // by default
 
	src = imread(imageName, IMREAD_COLOR); // Load an image
	if (src.empty())
	{
		return;
	}
	anchor = Point(-1, -1);
	delta = 0;
	ddepth = -1;
	int ind = 0;

	stringstream s;
	int offset = window_name.find_first_of("-") + 1;

	for (;;)
	{
		char c = (char)waitKey(500);
		if (c == 27)
		{
			break;
		}
		kernel_size = 3 + 2 * (ind % 4);
		kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
		filter2D(src, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT);

		
		s.str("");
		s << kernel_size;

		int len = s.str().length();
		window_name.replace(offset, len,s.str());
		imshow(window_name, dst);
		ind++;
	}
}

void OpenCVTest::makeImgBorder() {

	Mat src, dst;
	int top, bottom, left, right;
	int borderType = BORDER_REPLICATE;
	const char* window_name = "copyMakeBorder Demo";
	const char* ori_window_name = "origin copyMakeBorder Demo";
	RNG rng(12345);

	String imageName("../data/lena.jpg"); // by default
 
	src = imread(imageName, IMREAD_COLOR); // Load an image
	if (src.empty())
	{
		printf(" No data entered, please enter the path to an image file \n");
		return;
	}
	printf("\n \t copyMakeBorder Demo: \n");
	printf("\t -------------------- \n");
	printf(" ** Press 'c' to set the border to a random constant value \n");
	printf(" ** Press 'r' to set the border to be replicated \n");
	printf(" ** Press 'ESC' to exit the program \n");

	namedWindow(ori_window_name, WINDOW_AUTOSIZE);
	imshow(ori_window_name, src);
	
	top = (int)(0.05*src.rows); bottom = (int)(0.05*src.rows);
	left = (int)(0.05*src.cols); right = (int)(0.05*src.cols);
	dst = src;


	namedWindow(window_name, WINDOW_AUTOSIZE);
	imshow(window_name, dst);
	for (;;)
	{
		char c = (char)waitKey();
		if (c == 27)
		{
			break;
		}
		else if (c == 'c')
		{
			borderType = BORDER_CONSTANT;
		}
		else if (c == 'r')
		{
			borderType = BORDER_REPLICATE;
		}
		else {
			const Mat reset = dst.adjustROI(-top >> 1, -bottom >> 1, -left >> 1, -right >> 1);
			imshow("cut_rest", reset);
		}

		Scalar value(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
		imshow(window_name, dst);
	}
}

void OpenCVTest::simpleEdgeDetector() {
	Mat src, src_gray;
	Mat grad;
	const char* window_name = "Sobel Demo - Simple Edge Detector";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	src = imread("../data/lena.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		return ;
	}
	imshow("源图像", src);

	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );  
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );  
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);

	imshow("X 方向", abs_grad_x);

	imshow("Y 方向", abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	imshow(window_name, grad);
}

void OpenCVTest::imgHistogram() {
	Mat src, dst;
	String imageName("../data/lena.jpg"); // by default

	src = imread(imageName, IMREAD_COLOR);
	if (src.empty())
	{
		return ;
	}

	imshow("original img", src);

	vector<Mat> bgr_planes;
	split(src, bgr_planes);
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	// Draw the histograms for B, G and R
	int hist_w = 1920; int hist_h = 768;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
 

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Scalar(255, 0, 0), 2, 16, 0);


		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	namedWindow("calcHist Demo", WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

}

void OpenCVTest::pointTest() {

	const int r = 100;
	Mat src = Mat::zeros(Size(4 * r, 4 * r), CV_8UC1);
	vector<Point2f> vert(6);
	vert[0] = Point(3 * r / 2, static_cast<int>(1.34*r));
	vert[1] = Point(1 * r, 2 * r);
	vert[2] = Point(3 * r / 2, static_cast<int>(2.866*r));
	vert[3] = Point(5 * r / 2, static_cast<int>(2.866*r));
	vert[4] = Point(3 * r, 2 * r);
	vert[5] = Point(5 * r / 2, static_cast<int>(1.34*r));
	for (int j = 0; j < 6; j++)
	{
		line(src, vert[j], vert[(j + 1) % 6], Scalar(255), 3, 8);
	}
	vector<vector<Point> > contours; vector<Vec4i> hierarchy;
	Mat src_copy = src.clone();
	findContours(src_copy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat raw_dist(src.size(), CV_32FC1);
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			raw_dist.at<float>(j, i) = (float)pointPolygonTest(contours[0], Point2f((float)i, (float)j), true);
		}
	}
	double minVal; double maxVal;
	minMaxLoc(raw_dist, &minVal, &maxVal, 0, 0, Mat());
	minVal = abs(minVal); maxVal = abs(maxVal);

	cout << "minVal:" << minVal << "maxVal:" << maxVal;

	Mat drawing = Mat::zeros(src.size(), CV_8UC3);
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			if (raw_dist.at<float>(j, i) < 0)
			{
				drawing.at<Vec3b>(j, i)[0] = (uchar)(255 - abs(raw_dist.at<float>(j, i)) * 255 / minVal);
			}
			else if (raw_dist.at<float>(j, i) > 0)
			{
				drawing.at<Vec3b>(j, i)[2] = (uchar)(255 - raw_dist.at<float>(j, i) * 255 / maxVal);
			}
			else
			{
				drawing.at<Vec3b>(j, i)[0] = 255; drawing.at<Vec3b>(j, i)[1] = 255; drawing.at<Vec3b>(j, i)[2] = 255;
			}
		}
	}
	const char* source_window = "Source";
	namedWindow(source_window, WINDOW_AUTOSIZE);
	imshow(source_window, src);
	namedWindow("Distance", WINDOW_AUTOSIZE);
	imshow("Distance", drawing);
	waitKey(0);

}

void OpenCVTest::writeVideo() {
	const string source = "F:/OpenCV/WYCOPENCV/OpenCV/data/Megamind.avi";           // the source file name
	const bool askOutputType = false;  // If false it will use the inputs codec type
	VideoCapture inputVideo(source);              // Open input
	if (!inputVideo.isOpened())
	{
		cout << "Could not open the input video: " << source << endl;
		return;
	}
	string::size_type pAt = source.find_last_of('.');                  // Find extension point
	const string NAME = source.substr(0, pAt) + "wyc" + ".avi";   // Form the new name with container
	int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
	// Transform from int to char via Bitwise operators
	char EXT[] = { (char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0 };
	Size S = Size((int)inputVideo.get(CAP_PROP_FRAME_WIDTH),(int)inputVideo.get(CAP_PROP_FRAME_HEIGHT));

	// 获取帧数
	double fps = inputVideo.get(CAP_PROP_FPS);

	VideoWriter outputVideo;                                        // Open the output
	if (askOutputType)
		outputVideo.open(NAME, ex = -1, inputVideo.get(CAP_PROP_FPS), S, true);
	else
		outputVideo.open(NAME, ex, inputVideo.get(CAP_PROP_FPS), S, true);

	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video for write: " << source << endl;
		return ;
	}
	cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
		<< " of nr#: " << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;
	cout << "Input codec type: " << EXT << endl;
	int channel = 2; // Select the channel to save
	switch ('R')
	{
	case 'R': channel = 2; break;
	case 'G': channel = 1; break;
	case 'B': channel = 0; break;
	}
	Mat src, res;
	vector<Mat> spl;
	for (;;) //Show the image captured in the window and repeat
	{
		inputVideo >> src;              // read
		if (src.empty()) break;         // check if at end

		split(src, spl);                // process - extract only the correct channel
		for (int i = 0; i < 3; ++i)
			if (i != channel)
				spl[i] = Mat::zeros(S, spl[0].type());
		merge(spl, res);
		//outputVideo.write(res); //save or
		outputVideo << res;
	}
	cout << "Finished writing" << endl;
}