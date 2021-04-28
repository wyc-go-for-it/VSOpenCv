#include <iostream>
#include "OpenCVTest.h"
 
using namespace std;

static string img_dir = "F:/OpenCV/WYCOPENCV/data/";

int main(int argc,char **argv) {
 
	OpenCVTest ot;
	ot.recognition_number(img_dir + "number.png");
	waitKey();
	destroyAllWindows();
	return 0;
}


