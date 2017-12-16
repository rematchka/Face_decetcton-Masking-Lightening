#pragma once
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>
#include<algorithm>
#include "opencv2/opencv.hpp"
#include <vector>
#include"FilterFactory.h"

using namespace cv;
using namespace std;

class FilterFactory
{
public:
	FilterFactory();
	~FilterFactory();
	void detectMouth(Mat& img, vector<Rect_<int> >& mouth, string cascade_path);
	void detectNose(Mat& img, vector<Rect_<int> >& nose);
	void detectEyes(Mat& img, vector<Rect_<int> >& eyes);
	void face_detect_dog_filter(Mat &I);
	void face_detect_hat_mustache(Mat &I);
	void face_detection_crown(Mat &I);
	void face_detection_harr_cascade_mustache(Mat &I);
	void flower_crown(Mat &I, vector<vector<pair<int, int>>> faces_max_x_y, vector<vector<pair<int, int>>> faces_x_y, vector<vector<pair<int, int>>> faces_w_h, vector<vector<pair<int, int>>> faces_ellip,int cnt);
	void dog_filter(Mat &I, vector<vector<pair<int, int>>> faces_max_x_y, vector<vector<pair<int, int>>> faces_x_y, vector<vector<pair<int, int>>> faces_w_h, vector<vector<pair<int, int>>> faces_ellip, int cnt);
	void hat_moustache(Mat &I, vector<vector<pair<int, int>>> faces_max_x_y, vector<vector<pair<int, int>>> faces_x_y, vector<vector<pair<int, int>>> faces_w_h, vector<vector<pair<int, int>>> faces_ellip, int cnt);




};

