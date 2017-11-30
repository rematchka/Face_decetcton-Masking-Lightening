/**
* @file Morphology_1.cpp
* @brief Erosion and Dilation sample code
* @author OpenCV team
*/

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>
#include<algorithm>

using namespace cv;
using namespace std;


int main(int, char** argv)
{
	cv::Mat I = imread("FB_IMG_1463846848502.jpg", CV_LOAD_IMAGE_COLOR);
	if (I.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}

	namedWindow("Display window", CV_WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", I);
	//waitKey(0);
	 I.convertTo(I, CV_32FC3);
	 vector<Mat> channels1;
	 split(I , channels1);
	 Mat I0 = channels1[0];
	 Mat I1 = channels1[1];
	 Mat I2 = channels1[2];
	
	 cv::Mat fullImageHSV = cv::Mat::zeros(I.size(), CV_32FC3);
	 try
	 {
		 // ... Contents of your main
		// cvtColor(I, I, CV_RGB2BGR);

		 cvtColor(I, fullImageHSV, CV_BGR2HSV);
		 namedWindow("Display window1", CV_WINDOW_AUTOSIZE);
		// imshow("hsv", fullImageHSV);
		// waitKey(0);
		 vector<Mat> channels;
		 split(fullImageHSV, channels);
		 Mat hue = channels[0];
		 hue = hue / 360;
		// hue;
		
		// cout << hue.size() << " " << hue.channels() << endl;
		 Mat sat = channels[1];
		 sat.convertTo(sat, CV_32FC1);
		 Mat v = channels[2];
		 v.convertTo(v, CV_32FC1);
		 namedWindow("Display window2", CV_WINDOW_AUTOSIZE);
		 imshow("cropped_hsv", hue);
		 waitKey(0);
		 cout << hue.at<float>(0, 0) << endl;
		 cout << sat.at<float>(0, 0) << endl;
		 cout << v.at<float>(0, 0) << endl;
		 Mat cr=cv::Mat::zeros(hue.size(), CV_32FC1),cb= cv::Mat::zeros(hue.size(), CV_32FC1);
		 int rows = hue.rows;
		 int cols = hue.cols;
		 float val1, val2;
		 for(int i=0;i<rows;i++)
			 for (int j = 0; j < cols; j++)
			 {
			
				 cb.at<float>(i, j) = val1= 0.148* I2.at<float>(i, j) - 0.291*  I1.at<float>(i, j) + 0.439 * I0.at<float>(i, j) + 128;
				 cr.at<float>(i, j) = val2= 0.439 *I2.at<float>(i, j) - 0.368 * I1.at<float>(i, j) - 0.071 * I0.at<float>(i, j) + 128;
				

			 }

		 cout << cb.at<float>(0, 0) << endl;
		 cout << cr.at<float>(0, 0) << endl;
	//	 cb= 0.148* hue - 0.291*  sat + 0.439 * v + 128;
		// cr = 0.439 *hue - 0.368 * sat - 0.071 * v + 128;
		Mat segment = cv::Mat::zeros(hue.size(), CV_32FC1);
		int cnt = 0;
		for (int i = 0; i<rows; i++)
			for (int j = 0; j < cols; j++)
			{
				if (cr.at<float>(i, j) >= (float)140 && cr.at<float>(i, j) <= (float)165 && cb.at<float>(i, j) >= (float)140 && cb.at<float>(i, j) <= (float)195 && hue.at<float>(i, j) >= (float) 0.01&& hue.at<float>(i, j) <= (float) 0.1)
				{
					segment.at<float>(i, j) = 1; cnt++;
				}
				else
					segment.at<float>(i, j) = 0;

			}
		cout << cnt << endl;

		//vector<Mat> channelsimgbin;
	//	split(im, channelsimgbin);
		Mat im0;
		//channelsimgbin[0].convertTo(channelsimgbin[0], CV_32FC3);
		//segment.convertTo(segment, CV_32FC3);


		 //im0 = channelsimgbin[0];
	
		Mat  im1;
		Mat  im2;
		im0 =segment;
		im1 =segment;
		im2 =segment;

		cout << segment.size() << " " << segment.channels() << endl;
		vector<Mat> channelsmerge;

		
		channelsmerge.push_back(im0);
		channelsmerge.push_back(im1);
		channelsmerge.push_back(im2);
		cv::Mat in[] = { im0, im1, im2 };
		cout << channelsmerge.size() << endl;
		Mat eeeee ;
		merge(channelsmerge, eeeee);
		cout << eeeee.size()<<" "<< eeeee.channels()<<endl;
		imshow("img skin", eeeee);
		waitKey(0);


		Mat mat = Mat::ones(9, 9, CV_8U);
		
		mat.at<uchar>(0, 0) = 0;
		mat.at<uchar>(0, 1) = 0;
		mat.at<uchar>(0, 0) = 0;
		mat.at<uchar>(0, 7) = 0;
		mat.at<uchar>(0, 8) = 0;
		mat.at<uchar>(8, 0) = 0;
		mat.at<uchar>(8, 1) = 0;
		mat.at<uchar>(8, 7) = 0;
		mat.at<uchar>(8, 8) = 0;
		Mat dil= cv::Mat::zeros(hue.size(), CV_32FC1);
		
		cv::dilate(segment, dil, mat);
		Mat erodee= cv::Mat::zeros(hue.size(), CV_32FC1);
		cv::erode(dil, erodee, mat);


		vector<Mat> channelsmerge1;


		channelsmerge1.push_back(dil);
		channelsmerge1.push_back(dil);
		channelsmerge1.push_back(dil);
	
		cout << channelsmerge1.size() << endl;
		Mat outdil;
		merge(channelsmerge1, outdil);
		imshow("dilate", outdil);
		waitKey(0);
		imshow("segment", segment);
		vector<Mat> channelsmerge11;


		channelsmerge11.push_back(erodee);
		channelsmerge11.push_back(erodee);
		channelsmerge11.push_back(erodee);

		cout << channelsmerge11.size() << endl;
		Mat outerode;
		merge(channelsmerge11, outerode);
		imshow("eorde", outerode);
		waitKey(0);


		segment.convertTo(segment, CV_8S);
		Mat stats, centroids, labelImage;
		int nLabels = connectedComponentsWithStats(segment, labelImage, stats, centroids, 8, CV_32S);
		Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
		int maxi = 0;
		for (int label = 1; label < nLabels; ++label) { //label  0 is the background
			

			cout << "area del component: " << label << "-> " << stats.at<int>(label, CC_STAT_AREA) << endl;
			//colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
			maxi = max(maxi, stats.at<int>(label, CC_STAT_AREA));
		}
		Mat surfSup = stats.col(4)>=maxi;


		for (int i = 1; i < nLabels; i++)
		{
			if (surfSup.at<uchar>(i, 0))
			{
				mask = mask | (labelImage == i);
			}
		}
		Mat r(segment.size(), CV_8UC1, Scalar(0));
		outerode.copyTo(r, mask);
		imshow("Result", r);
		waitKey();

		
	 }
	 catch (cv::Exception & e)
	 {
		 cerr << e.msg << endl; // output exception message
	 }
	 return 0;
	// system("pause");
}

