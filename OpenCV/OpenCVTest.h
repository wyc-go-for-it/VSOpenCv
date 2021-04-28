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


private:
	Mat MoveDetect(Mat &background, Mat &frame);
	//��ͼ
	static void mouseTrack(int event, int x, int y, int flags, void *userdata);
	const string windowName = "mouseTrack",captureWindow = "capture";
	Mat mMouseDrawImage;
	Point mDown;
	bool isDown;
	//����ʶ��
	void textbox_draw(Mat src, std::vector<Rect>& groups, std::vector<float>& probs, std::vector<int>& indexes);
	float compare(Mat first,Mat second);
};
