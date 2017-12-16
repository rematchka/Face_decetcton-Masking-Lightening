#include "FilterFactory.h"
#include<iostream>
#include<string>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>
#include<algorithm>
#include "opencv2/opencv.hpp"
#include <vector>



using namespace cv;
using namespace std;




String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
string nose_cascde_name = "haarcascade_mcs_nose.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;

FilterFactory::FilterFactory()
{
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); };
	if (!nose_cascade.load(nose_cascde_name)) { printf("--(!)Error loading nose cascade\n");  };
}


FilterFactory::~FilterFactory()
{
}


 void FilterFactory::detectEyes(Mat& img, vector<Rect_<int> >& eyes)
{
	//CascadeClassifier eyes_cascade;
	//eyes_cascade.load(eyes_cascade_name);

	eyes_cascade.detectMultiScale(img, eyes, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}
void FilterFactory::detectNose(Mat& img, vector<Rect_<int> >& nose)
{
	//CascadeClassifier nose_cascade;
	//nose_cascade.load(nose_cascde_name);

	nose_cascade.detectMultiScale(img, nose, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

 void FilterFactory::detectMouth(Mat& img, vector<Rect_<int> >& mouth, string cascade_path)
{
	CascadeClassifier mouth_cascade;
	mouth_cascade.load(cascade_path);

	mouth_cascade.detectMultiScale(img, mouth, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}
void FilterFactory::face_detection_harr_cascade_mustache(Mat& I)
{
	try
	{

		Mat mustache = imread("mustache1.png", -1);
		if (mustache.empty())
		{
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
		}
		//imshow("moustache.png", mustache);
		//waitKey();
		vector<Mat> channels1;
		split(mustache, channels1);
		Mat I0 = channels1[0];
		Mat I1 = channels1[1];
		Mat I2 = channels1[2];
		Mat alpha = channels1[3];
		Mat orig_mask = alpha;
		cout << orig_mask.channels() << endl;
		vector<Mat>channels;
		channels.push_back(I0);
		channels.push_back(I1);
		channels.push_back(I2);
		merge(channels, mustache);
		//	mustache = imread("moustache.png");
		if (mustache.empty())
		{
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
		}

		//imshow("mustache", mustache);
		//waitKey();


		Mat origi_mask_inv;
		bitwise_not(orig_mask, origi_mask_inv);
		int origimostheight = mustache.size().height;
		int origimostwidth = mustache.size().width;


		cv::Mat frame = I;

		if (frame.empty())
		{
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
		}
		//	imshow("img", frame);
		//waitKey();
		std::vector<Rect> faces;
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		//-- Detect faces
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t i = 0; i < faces.size(); i++)
		{
			//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			Mat faceROI = frame_gray(faces[i]);
			
			Mat face_color = frame(faces[i]);
			
			/*std::vector<Rect> eyes;
			detectEyes(faceROI, eyes);
			//-- In each face, detect eyes
			//eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			for (size_t j = 0; j < eyes.size(); j++)
			{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
			}*/
			//nose detection 
			double nose_center_height = 0.0;
			if (!nose_cascade.empty())
			{
				vector<Rect_<int> > nose;
				detectNose(faceROI, nose);

				// Mark points corresponding to the centre (tip) of the nose
				for (unsigned int j = 0; j < nose.size(); ++j)
				{
					Rect n = nose[j];
					//circle(frame, Point(n.x + n.width / 2, n.y + n.height / 2), 3, Scalar(0, 255, 0), -1, 8);
					//nose_center_height = (n.y + n.height / 2);
					//imshow(" face detection", frame);
					//waitKey(0);


					int mostw = 3 * n.width;
					int mosth = mostw * origimostheight / origimostwidth;
					int x1 = n.x - (mostw / 4) + faces[i].x, x2 = n.x + n.width + (mostw / 4) + faces[i].x, y1 = n.y + n.height - (mosth / 2) + faces[i].y, y2 = n.y + n.height + (mosth / 2) + faces[i].y;
					/*	if (x1 < 0)
					x1 = 0;
					if (y1 < 0)
					y1 = 0;
					if (x2 > n.width)
					x2 = faces[i].width;
					if (y2 > n.height)
					y2 = faces[i].height;*/
					mostw = x2 - x1;
					mosth = y2 - y1;
					//cout << mustache.size() << endl;

					resize(mustache, mustache, cvSize(mostw, mosth));
					//cout << mustache.size() << endl;
					Mat	mask = orig_mask;
					//cout << orig_mask.size() << endl;
					resize(orig_mask, mask, cvSize(mostw, mosth));
					//cout << mask.size() << endl;
					Mat	mask_inv = origi_mask_inv;
					resize(origi_mask_inv, mask_inv, cvSize(mostw, mosth));
					Mat roi;
					frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(roi);
					//cout << roi.size() << endl;
					//cout << mask_inv.size() << endl;

					Mat	roi_bg;
					bitwise_and(roi, roi, roi_bg, mask = mask_inv);

					//cout << mustache.channels() << endl;
					Mat	roi_fg;
					bitwise_and(mustache, mustache, roi_fg, mask = mask);

					Mat	dst;
					add(roi_bg, roi_fg, dst);
					dst.copyTo(frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)));
					

				}
			}



		}
		//-- Show what you got
		I = frame;
		


	}

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}

}

void FilterFactory::face_detection_crown(Mat &I)
{

	try
	{

		Mat flower = imread("flower_crown.png");


		cv::Mat frame = I;
		cout << I.channels() << endl;
		if (frame.empty())
		{
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
		}
		//	imshow("img", frame);
		//waitKey();
		std::vector<Rect> faces;
		Mat frame_gray;
		if(frame.channels()!=1)
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		//-- Detect faces
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t i = 0; i < faces.size(); i++)
		{
			//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			Mat faceROI = frame_gray(faces[i]);
			
			Mat face_color = frame(faces[i]);
			
			resize(flower, flower, cvSize(faces[i].width, faces[i].height));
			//flower.copyTo(frame(cv::Rect(faces[i].x, -faces[i].y+faces[i].height, faces[i].width, faces[i].height)));
			int rows = flower.rows;
			int cols = flower.cols;
			for (int k = 0; k<rows; k++)
				for (int l = 0; l<cols; l++)
					for (int m = 0; m < 3; m++)
					{

						if (flower.at<Vec3b>(k, l)[m]>0)
						{
							frame.at<Vec3b>(abs(faces[i].y + k - int(0.5*faces[i].height)), l + faces[i].x)[m] = flower.at<Vec3b>(k, l)[m];



						}
					}



		}

		I = frame;
	}

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
}

void FilterFactory::face_detect_hat_mustache(Mat &I)
{

	try
	{

		Mat hat = imread("cowboy_hat.png");


		cv::Mat frame = I;
		if (frame.empty())
		{
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
		}
		//	imshow("img", frame);
		//waitKey();
		std::vector<Rect> faces;
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		//-- Detect faces
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t i = 0; i < faces.size(); i++)
		{
			//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			Mat faceROI = frame_gray(faces[i]);
			
			Mat face_color = frame(faces[i]);
			
			resize(hat, hat, cvSize((faces[i].width + 1), (faces[i].height*0.5) + 1));
			//flower.copyTo(frame(cv::Rect(faces[i].x, -faces[i].y+faces[i].height, faces[i].width, faces[i].height)));
			int rows = hat.rows;
			int cols = hat.cols;
			for (int k = 0; k<rows; k++)
				for (int l = 0; l<cols; l++)
					for (int m = 0; m < 3; m++)
					{

						if (hat.at<Vec3b>(k, l)[m]<235)
						{
							frame.at<Vec3b>(abs(faces[i].y + k - int(0.25*faces[i].height)), l + faces[i].x)[m] = hat.at<Vec3b>(k, l)[m];



						}
					}



			Mat mst = imread("moustache.png");
			int mst_width = int(faces[i].width*0.4166666) + 1;
			int	mst_height = int(faces[i].height*0.14285) + 1;



			resize(mst, mst, cvSize(mst_width, mst_height));

			for (int ii = (int(0.62857142857*faces[i].height)); ii<int(0.62857142857*faces[i].height) + mst_height; ii++)
				for (int j = int(0.29166666666*faces[i].width); j< int(0.29166666666*faces[i].width) + mst_width; j++)
					for (int k = 0; k < 3; k++)
					{
						if (mst.at<Vec3b>(ii - int(0.62857142857*faces[i].height), j - int(0.29166666666*faces[i].width))[k] < 235)
							frame.at < Vec3b>(faces[i].y + ii, faces[i].x + j)[k] = mst.at<Vec3b>(ii - int(0.62857142857*faces[i].height), j - int(0.29166666666*faces[i].width))[k];

					}


			


		}
		I = frame;

	}

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}

}
void FilterFactory::face_detect_dog_filter(Mat &I)
{
	try
	{

		Mat dog_nose = imread("dog_nose.png");
		Mat dog_left_ear = imread("dog_left_ear.png");
		Mat dog_right_ear = imread("dog_right_ear.png");


		cv::Mat frame = I;
		if (frame.empty())
		{
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
		}
		//	imshow("img", frame);
		//waitKey();
		std::vector<Rect> faces;
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		//-- Detect faces
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t i = 0; i < faces.size(); i++)
		{
			//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			Mat faceROI = frame_gray(faces[i]);

			
			Mat face_color = frame(faces[i]);
			
			resize(dog_left_ear, dog_left_ear, cvSize((faces[i].width *0.25), faces[i].height*0.5));
			resize(dog_right_ear, dog_right_ear, cvSize((faces[i].width *0.25), faces[i].height*0.5));
			int rowss = dog_left_ear.rows;
			int colss = dog_left_ear.cols;
			for (int k = 0; k<rowss; k++)
				for (int l = 0; l<colss; l++)
					for (int m = 0; m < 3; m++)
					{

						if (dog_left_ear.at<Vec3b>(k, l)[m]>0)
						{
							frame.at<Vec3b>(faces[i].y + k - int(0.25*faces[i].height), l + faces[i].x)[m] = dog_left_ear.at<Vec3b>(k, l)[m];



						}
					}


			for (int k = 0; k<rowss; k++)
				for (int l = 0; l<colss; l++)
					for (int m = 0; m < 3; m++)
					{

						if (dog_right_ear.at<Vec3b>(k, l)[m]>0)
						{
							frame.at<Vec3b>(faces[i].y + k - int(0.25*faces[i].height), faces[i].x + l + faces[i].width - colss)[m] = dog_right_ear.at<Vec3b>(k, l)[m];



						}
					}







			vector<Rect_<int> > nose;
			detectNose(faceROI, nose);

			// Mark points corresponding to the centre (tip) of the nose
			for (unsigned int j = 0; j < nose.size(); ++j)
			{
				Rect n = nose[j];
				//circle(frame, Point(n.x + n.width / 2, n.y + n.height / 2), 3, Scalar(0, 255, 0), -1, 8);
				//nose_center_height = (n.y + n.height / 2);
				//imshow(" face detection", frame);
				//waitKey(0);

				resize(dog_nose, dog_nose, cvSize((n.width), n.height));
				int rows = dog_nose.rows;
				int cols = dog_nose.cols;
				for (int k = 0; k<rows; k++)
					for (int l = 0; l<cols; l++)
						for (int m = 0; m < 3; m++)
						{

							if (dog_nose.at<Vec3b>(k, l)[m]>0)
							{
								frame.at<Vec3b>(n.y + k + faces[i].y, l + n.x + faces[i].x)[m] = dog_nose.at<Vec3b>(k, l)[m];



							}
						}





			}
		}





		I = frame;



	}




	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
}


void FilterFactory::flower_crown(Mat &I, vector<vector<pair<int, int>>> faces_max_x_y, vector<vector<pair<int, int>>> faces_x_y, vector<vector<pair<int, int>>> faces_w_h, vector<vector<pair<int, int>>> faces_ellip,int cnt)
{

	try
	{
		

		Mat flower = imread("flower_crown.png");


		cv::Mat frame =I;
		if (frame.empty())
		{
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
		}

		for (size_t i = 0; i <cnt; i++)
		{

			resize(flower, flower, cvSize(abs(faces_w_h[i][0].first), 0.5*abs(faces_w_h[i][0].second)));

			int rows = flower.rows;
			int cols = flower.cols;
			for (int k = 0; k<rows; k++)
				for (int l = 0; l<cols; l++)
					for (int m = 0; m < 3; m++)
					{

						if (flower.at<Vec3b>(k, l)[m]>0)
						{
							frame.at<Vec3b>(abs(faces_x_y[i][0].second + k - (0.15*faces_w_h[i][0].second)), abs(l + faces_x_y[i][0].first))[m] = flower.at<Vec3b>(k, l)[m];



						}
					}

		


		}
		I = frame;

	}

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
}


void FilterFactory::dog_filter(Mat &I, vector<vector<pair<int, int>>> faces_max_x_y, vector<vector<pair<int, int>>> faces_x_y, vector<vector<pair<int, int>>> faces_w_h, vector<vector<pair<int, int>>> faces_ellip, int cnt)
{
	try
	{
		
		Mat dog_nose = imread("dog_nose.png");
		Mat dog_left_ear = imread("dog_left_ear.png");
		Mat dog_right_ear = imread("dog_right_ear.png");


		cv::Mat frame =I;
		if (frame.empty())
		{
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
		}
		//	imshow("img", frame);
		//waitKey();

		for (size_t i = 0; i < cnt; i++)
		{

			resize(dog_left_ear, dog_left_ear, cvSize((faces_w_h[i][0].first *0.25), faces_w_h[i][0].second*0.5));
			resize(dog_right_ear, dog_right_ear, cvSize((faces_w_h[i][0].first *0.25), faces_w_h[i][0].second*0.5));
			int rowss = dog_left_ear.rows;
			int colss = dog_left_ear.cols;
			for (int k = 0; k<rowss; k++)
				for (int l = 0; l<colss; l++)
					for (int m = 0; m < 3; m++)
					{

						if (dog_left_ear.at<Vec3b>(k, l)[m]>0)
						{
							frame.at<Vec3b>(abs(faces_x_y[i][0].second + k - int(0.25*faces_w_h[i][0].second)), l + faces_x_y[i][0].first)[m] = dog_left_ear.at<Vec3b>(k, l)[m];



						}
					}


			for (int k = 0; k<rowss; k++)
				for (int l = 0; l<colss; l++)
					for (int m = 0; m < 3; m++)
					{

						if (dog_right_ear.at<Vec3b>(k, l)[m]>0)
						{
							frame.at<Vec3b>(abs(faces_x_y[i][0].second + k - int(0.25*faces_w_h[i][0].second)), faces_x_y[i][0].first + l + faces_w_h[i][0].first - colss)[m] = dog_right_ear.at<Vec3b>(k, l)[m];



						}
					}






		}





		I = frame;



	}




	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
}


void FilterFactory::hat_moustache(Mat &I, vector<vector<pair<int, int>>> faces_max_x_y, vector<vector<pair<int, int>>> faces_x_y, vector<vector<pair<int, int>>> faces_w_h, vector<vector<pair<int, int>>> faces_ellip, int cnt)
{

	try
	{

		Mat hat = imread("cowboy_hat.png");

		
		cv::Mat frame =I;
		if (frame.empty())
		{
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
		}
		//	imshow("img", frame);
		//waitKey();

		for (size_t i = 0; i < cnt; i++)
		{
			//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

			resize(hat, hat, cvSize((faces_w_h[i][0].first + 1), (faces_w_h[i][0].second*0.5) + 1));
			//flower.copyTo(frame(cv::Rect(faces[i].x, -faces[i].y+faces[i].height, faces[i].width, faces[i].height)));
			int rows = hat.rows;
			int cols = hat.cols;
			for (int k = 0; k<rows; k++)
				for (int l = 0; l<cols; l++)
					for (int m = 0; m < 3; m++)
					{

						if (hat.at<Vec3b>(k, l)[m]<235)
						{
							frame.at<Vec3b>(abs(faces_x_y[i][0].second + k - int(0.25*faces_w_h[i][0].second)), l + faces_x_y[i][0].first)[m] = hat.at<Vec3b>(k, l)[m];



						}
					}



			Mat mst = imread("moustache.png");
			int mst_width = int(faces_w_h[i][0].first*0.4166666) + 1;
			int	mst_height = int(faces_w_h[i][0].second*0.14285) + 1;



			resize(mst, mst, cvSize(mst_width, mst_height));

			for (int ii = (int(0.62857142857*faces_w_h[i][0].second)); ii<int(0.62857142857*faces_w_h[i][0].second) + mst_height; ii++)
				for (int j = int(0.29166666666*faces_w_h[i][0].first); j< int(0.29166666666*faces_w_h[i][0].first) + mst_width; j++)
					for (int k = 0; k < 3; k++)
					{
						if (mst.at<Vec3b>(ii - int(0.62857142857*faces_w_h[i][0].second), j - int(0.29166666666*faces_w_h[i][0].first))[k] < 235)
							frame.at < Vec3b>(faces_x_y[i][0].second + ii - 0.055555*faces_w_h[i][0].second, faces_x_y[i][0].first+ j)[k] = mst.at<Vec3b>(ii - int(0.62857142857*faces_w_h[i][i].second), j - int(0.29166666666*faces_w_h[i][i].first))[k];

					}


			


		}
		I = frame;

	}

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}

}
