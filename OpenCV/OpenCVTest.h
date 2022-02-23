#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include  <fstream>
#include <cstring>
#include <io.h>
#include <Windows.h>

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
	//����ʶ��
	void recognition(const string &path);
	void recognition_number(const string &path);

	static inline bool fileExists(const string& filename)
	{
		ifstream f(filename.c_str());
		return f.good();
	}

	static std::vector<string> listFiles(const char * dir)
	{
		vector<string> files_name;
		HANDLE hFind;
		WIN32_FIND_DATA findData;
		LARGE_INTEGER size;
		hFind = FindFirstFile(dir, &findData);
		if (hFind == INVALID_HANDLE_VALUE)
		{
			cout << "Failed to find first file!\n";
			return files_name;
		}
		do
		{
			// ����"."��".."������� 
			if (strcmp(findData.cFileName, ".") == 0 || strcmp(findData.cFileName, "..") == 0)
				continue;
			if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)    // �Ƿ���Ŀ¼ 
			{
				cout << findData.cFileName << "\t<dir>\n";
			}
			else
			{
				size.LowPart = findData.nFileSizeLow;
				size.HighPart = findData.nFileSizeHigh;
				cout << findData.cFileName << "\t" << size.QuadPart << " bytes\n";
				files_name.push_back(findData.cFileName);
			}
		} while (FindNextFile(hFind, &findData));

		return files_name;
 	}

	void Qpyr(const string &path);//ͼ���ϡ��²���
	void OCanny(const string &path);//��Ե���
	void OHoughLines(const string &path);//����任����������
	void OHoughLinesP(const string &path);
	void ORemap(const string &path,Mat &dst);//ӳ��
	void OMatchTemplate(const Mat &src,const Mat &temp);//ģ��ƥ��
	void ORepairImg(const string &path);//ͼƬ�޸�
	void OContours(const string &path);//��������
	static void thresh_callback(int, void*);
	void OpenCamera();//��Ƶ
	void imgBlur();//ƽ��
	void morphologyOperations();//��̬����
	void hitAndMiss();
	void getVerticalAndHorizontalLine();
	void linearFilter();//�Զ����˲�
	void makeImgBorder();//��չͼ��߽�
	void simpleEdgeDetector();//��Ե���
	void imgHistogram();//�����ߵķ�ʽ��ʾͼƬֱ��ͼ
	void pointTest();//��������λ�ü��
private:
	Mat MoveDetect(Mat &background, Mat &frame);
	//��ͼ
	static void mouseTrack(int event, int x, int y, int flags, void *userdata);
	const string mouseTrackWindowName = "mouseTrack",captureWindow = "capture", TemplateWindow = "match";
	Mat mMouseDrawImage,mCaptureImage;
	Point mDown;
	bool isDown;
	//����ʶ��
	void textbox_draw(Mat src, std::vector<Rect>& groups, std::vector<float>& probs, std::vector<int>& indexes);
	float compare(Mat first,Mat second);
	//ģ��ƥ��
	static void on_matching(int, void*);
	Mat g_srcImage, g_tempalteImage, g_resultImage;
	int g_nMatchMethod;
	int g_nMaxTrackbarNum = 5;

	//��������
	Mat ContourSrc; 
	Mat ContourSrcGray;
	int ContourThresh = 100;
	int ContourMaxThresh = 255;
	//ƽ��
	int display_dst(const char *caption, const Mat & dst);
	//��̬����
	Mat src;
	int morph_elem = 0;
	int morph_size = 0;
	int morph_operator = 0;
	int const max_operator = 4;
	int const max_elem = 2;
	int const max_kernel_size = 21;
	const char* window_name = "Morphology Transformations Demo";
	static void Morphology_Operations(int, void*);
};
