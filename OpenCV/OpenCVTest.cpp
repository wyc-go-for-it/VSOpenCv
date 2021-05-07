#include "OpenCVTest.h"
#include <opencv2/text.hpp>
#include <opencv2/dnn.hpp>
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
		cv::imshow("frame", frame);

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
		cv::imshow("result", result);
		//按原FPS显示
		if (waitKey((int)(500 / FPS)) == 27)
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
