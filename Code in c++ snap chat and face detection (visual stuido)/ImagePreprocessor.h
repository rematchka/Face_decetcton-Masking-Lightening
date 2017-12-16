#pragma once
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <typeinfo.h>
#include<exception>
#include <iostream>
#include <stdio.h>
#include <vector>
#include<cmath>
#include<set>
#include<map>
#include"DSU.h"
#include <stack>
using namespace std;
using namespace cv;

#define imfirst 0
#define imsecond 1
#define imthird 2
#define imfirst_and_second 3
#define imfirst_and_third 4
#define imsecond_and_third 5
#define imfirst_and_second_and_third 6

class ImagePreprocessor
{

	Mat image;
public:
	Mat outImage;
public:
	ImagePreprocessor(Mat _image);
	~ImagePreprocessor();
	void EqualiseHistogram(int flag);
	void greyWord();
	void white_patch();
	void modified_white_patch(int varargin);
	Mat getOutput();
};

