#include <iostream>
#include "OpenCVTest.h"
 
using namespace std;

static string img_dir = "F:/OpenCV/WYCOPENCV/data/";

int main(int argc,char **argv) {
 
	OpenCVTest ot;
	ot.mouseCallback(img_dir + "nba.jpg");
	waitKey();
	destroyAllWindows();
	return 0;
}


