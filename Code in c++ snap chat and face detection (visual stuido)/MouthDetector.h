#pragma once
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <typeinfo.h>

#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;

class MouthDetector
{
	Mat FaceROI;
	Rect faceRegion;
	Mat MouthROI;
	Rect mouthRegion;
	string mouth_cascade_name;
	CascadeClassifier mouthclassifier;

public:
	MouthDetector();
	MouthDetector(Mat _FaceROI, Rect _faceRegion);
	Mat getmouth();
	Rect getMouthROI();
	void detectMouth();
	~MouthDetector();
	void mouth_map(Mat I);
	vector<vector<pair<int, int>>> lip_rect;
	vector<vector<pair<int, int>>> lip_ellip;
	vector<vector<pair<int, int>>> lip_w_h;
	vector<vector<pair<int, int>>> lip_x_y;

};

