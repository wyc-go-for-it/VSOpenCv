#include <iostream>
#include "OpenCVTest.h"
 
using namespace std;

static string img_dir = "../data/";

int main(int argc,char **argv) {
 
	OpenCVTest ot;
	ot.faceDetect();
	waitKey();
	destroyAllWindows();
	return 0;
}


