#pragma once
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <typeinfo.h>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

class EyesDetector
{
public:
	EyesDetector();
	~EyesDetector();
	EyesDetector(Mat _FaceROI);
	vector<Rect> getEyesROI();
	vector<vector<pair<int, int>>> eyes_w_h;
	vector<vector<pair<int, int>>> eyes_rect;
	vector<vector<pair<int, int>>> eyes_ellip;
	vector<vector<pair<int, int>>> eye_x_y;
private:
	Mat FaceROI;
	String eyes_cascade_name;
	CascadeClassifier eyes_cascade;
	vector<Rect>eyesROI;
	vector<Mat> eyes;
public:
	vector<Mat> getEyes();
	void detectEyes();
	void eye_map(Mat  face_image);
	Mat mat2gray(cv::Mat inMat);
	int otsuThreshold(Mat img);
};

