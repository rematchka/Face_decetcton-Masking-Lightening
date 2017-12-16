#pragma once
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <typeinfo.h>

#include <iostream>
#include <stdio.h>
#include <vector>
#include<cmath>
using namespace std;
using namespace cv;

class FaceDetector
{
public:
	FaceDetector();
	~FaceDetector();
	Rect getFaceROI();
	FaceDetector(Mat _image);
	Mat image_forfeature;
private:
	Mat image;
	string face_cascade_name;
	CascadeClassifier face_cascade;
	Mat ellipsemask;
	
public:
	void detectFace();
	int minxf;
	int minyf;

	int minfex;
	int minfey;
	int maxx;
	int maxy;
	vector<vector<pair<int, int>>> faces_rect;
	vector<vector<pair<int, int>>> faces_ellip;
	vector<vector<pair<int, int>>> faces_w_h;
	vector<vector<pair<int, int>>> faces_x_y;
	vector<vector<pair<int, int>>> faces_max_x_y;

private:
	Mat FaceROI;
	Rect FaceRegion;
public:
	void shrinkFaceROI(Rect ROI);
private:
	vector<Point> _scale(Point _topLeft, Point _topRight, Point _bottomLeft, Point _bottomRight,float fx,float fy);
public:
	Mat getFace();
	Mat getEllipticalFaceMask();
	Mat getNegativeEllipticalFaceMask();
	Mat getEllipticalFace();
	int cnt_faces;
	void createEllipticalFaceMask();
	void contour(Mat I);
	void skinDetection(Mat img, Mat& binImg);
	void check_r(Mat img, Mat& binImg);
	vector<Rect> faceDetection(Mat img, Mat & binImg);
	vector<Rect> addFrame(int, void*);
	int rotate(cv::Mat &src);
	void savePoints(vector<Rect>faces);
	
	

};

