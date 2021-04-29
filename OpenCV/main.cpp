#include <iostream>
#include "OpenCVTest.h"
 
using namespace std;

static string img_dir = "F:/OpenCV/WYCOPENCV/data/";

int main(int argc,char **argv) {
 
	OpenCVTest ot;
	ot.OHoughLines(img_dir + "leuvenA.jpg");
	waitKey();
	destroyAllWindows();
	return 0;
}


