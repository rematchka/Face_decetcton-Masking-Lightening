
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>
#include<algorithm>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;




String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
string nose_cascde_name = "haarcascade_mcs_nose.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;


void face_detection_manually()
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
	split(I, channels1);
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
		Mat cr = cv::Mat::zeros(hue.size(), CV_32FC1), cb = cv::Mat::zeros(hue.size(), CV_32FC1);
		int rows = hue.rows;
		int cols = hue.cols;
		float val1, val2;
		for (int i = 0; i<rows; i++)
			for (int j = 0; j < cols; j++)
			{

				cb.at<float>(i, j) = val1 = 0.148* I2.at<float>(i, j) - 0.291*  I1.at<float>(i, j) + 0.439 * I0.at<float>(i, j) + 128;
				cr.at<float>(i, j) = val2 = 0.439 *I2.at<float>(i, j) - 0.368 * I1.at<float>(i, j) - 0.071 * I0.at<float>(i, j) + 128;


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
		im0 = segment;
		im1 = segment;
		im2 = segment;

		cout << segment.size() << " " << segment.channels() << endl;
		vector<Mat> channelsmerge;


		channelsmerge.push_back(im0);
		channelsmerge.push_back(im1);
		channelsmerge.push_back(im2);
		cv::Mat in[] = { im0, im1, im2 };
		cout << channelsmerge.size() << endl;
		Mat eeeee;
		merge(channelsmerge, eeeee);
		cout << eeeee.size() << " " << eeeee.channels() << endl;
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
		Mat dil = cv::Mat::zeros(hue.size(), CV_32FC1);

		cv::dilate(segment, dil, mat);
		Mat erodee = cv::Mat::zeros(hue.size(), CV_32FC1);
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
		Mat surfSup = stats.col(4) >= maxi;


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
}
static void detectEyes(Mat& img, vector<Rect_<int> >& eyes)
{
	//CascadeClassifier eyes_cascade;
	//eyes_cascade.load(eyes_cascade_name);

	eyes_cascade.detectMultiScale(img, eyes, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}
static void detectNose(Mat& img, vector<Rect_<int> >& nose)
{
	//CascadeClassifier nose_cascade;
	//nose_cascade.load(nose_cascde_name);

	nose_cascade.detectMultiScale(img, nose, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

static void detectMouth(Mat& img, vector<Rect_<int> >& mouth, string cascade_path)
{
	CascadeClassifier mouth_cascade;
	mouth_cascade.load(cascade_path);

	mouth_cascade.detectMultiScale(img, mouth, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}
void face_detection_harr_cascade_mustache()
{
	try
	{
	
		Mat mustache = imread("mustache1.png",-1);
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


		cv::Mat frame = imread("FB_IMG_1463846848502.jpg", CV_LOAD_IMAGE_COLOR);
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
			imshow("mmm", faceROI);
			waitKey();
			Mat face_color = frame(faces[i]);
			namedWindow("Display window1", CV_WINDOW_AUTOSIZE);
			imshow(" face detection", frame);
			waitKey(0);
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
				detectNose(frame, nose);

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
					int x1 = n.x - (mostw / 4), x2 = n.x + n.width + (mostw / 4), y1 = n.y + n.height - (mosth / 2), y2 = n.y + n.height + (mosth / 2);
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
					cout << mustache.size() << endl;
			
					resize(mustache, mustache, cvSize(mostw, mosth));
					cout << mustache.size() << endl;
					Mat	mask = orig_mask;
					cout << orig_mask.size() << endl;
					resize(orig_mask, mask, cvSize(mostw, mosth));
					cout << mask.size() << endl;
					Mat	mask_inv = origi_mask_inv;
					resize(origi_mask_inv, mask_inv, cvSize(mostw, mosth));
					Mat roi ;
					frame(cv::Rect(x1, y1,  x2-x1,  y2-y1)).copyTo(roi);
					cout << roi.size() << endl;
					cout << mask_inv.size() << endl;

					Mat	roi_bg;
					bitwise_and(roi, roi, roi_bg, mask = mask_inv);

					cout << mustache.channels() << endl;
					Mat	roi_fg;
					bitwise_and(mustache, mustache, roi_fg, mask = mask);

					Mat	dst;
					add(roi_bg, roi_fg, dst);
					dst.copyTo(frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)));
					namedWindow("Display window1", CV_WINDOW_AUTOSIZE);
					imshow(" face detection", dst);
					waitKey(0);

				}
			}



		}
		//-- Show what you got
		namedWindow("Display window1", CV_WINDOW_AUTOSIZE);
		imshow(" face detection", frame);
		waitKey(0);


	}

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}

}

void face_detection_crown()
{

	try
	{

		Mat flower = imread("flower_crown.png");
		

		cv::Mat frame = imread("FB_IMG_1463846848502.jpg", CV_LOAD_IMAGE_COLOR);
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
			imshow("mmm", faceROI);
			waitKey();
			Mat face_color = frame(faces[i]);
			namedWindow("Display window1", CV_WINDOW_AUTOSIZE);
			imshow(" face detection", frame);
			waitKey(0);
			resize(flower, flower, cvSize(faces[i].width, faces[i].height));
			//flower.copyTo(frame(cv::Rect(faces[i].x, -faces[i].y+faces[i].height, faces[i].width, faces[i].height)));
			int rows = flower.rows;
			int cols = flower.cols;
			for(int k= 0;k<rows;k++)
				for (int l =0 ;l<cols;l++)
					for (int m = 0; m < 3; m++)
					{
						
						if (flower.at<Vec3b>(k, l)[m]>0)
						{
							frame.at<Vec3b>(faces[i].y + k - int(0.5*faces[i].height), l + faces[i].x)[m] = flower.at<Vec3b>(k, l)[m];
							
							

						}
					}
			
			imshow(" face detection", frame);
			waitKey(0);
			

		}


	}

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
}

void face_detect_hat_mustache()
{

	try
	{

		Mat hat = imread("cowboy_hat.png");


		cv::Mat frame = imread("FB_IMG_1463846848502.jpg", CV_LOAD_IMAGE_COLOR);
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
			imshow("mmm", faceROI);
			waitKey();
			Mat face_color = frame(faces[i]);
			namedWindow("Display window1", CV_WINDOW_AUTOSIZE);
			imshow(" face detection", frame);
			waitKey(0);
			resize(hat, hat, cvSize((faces[i].width+1), (faces[i].height*0.35)+1));
			//flower.copyTo(frame(cv::Rect(faces[i].x, -faces[i].y+faces[i].height, faces[i].width, faces[i].height)));
			int rows = hat.rows;
			int cols = hat.cols;
			for (int k = 0; k<rows; k++)
				for (int l = 0; l<cols; l++)
					for (int m = 0; m < 3; m++)
					{

						if (hat.at<Vec3b>(k, l)[m]<235)
						{
							frame.at<Vec3b>(faces[i].y + k - int(0.25*faces[i].height), l + faces[i].x)[m] = hat.at<Vec3b>(k, l)[m];



						}
					}



			Mat mst= imread("moustache.png");
			int mst_width = int(faces[i].width*0.4166666) + 1;
			int	mst_height = int(faces[i].height*0.142857) + 1;



			resize(mst, mst, cvSize(mst_width, mst_height));

			for (int ii = (int(0.62857142857*faces[i].height)); ii<int(0.62857142857*faces[i].height) + mst_height; ii++)
				for (int j =int(0.29166666666*faces[i].width);j< int(0.29166666666*faces[i].width) + mst_width;j++)
					for (int k = 0; k < 3; k++)
					{
						if (mst.at<Vec3b>(ii - int(0.62857142857*faces[i].height), j - int(0.29166666666*faces[i].width))[k] < 235)
							frame.at < Vec3b>(faces[i].y + i, faces[i].x + j)[k] =mst.at<Vec3b>(ii - int(0.62857142857*faces[i].height), j - int(0.29166666666*faces[i].width))[k];
						
					}


			imshow(" face detection", frame);
			waitKey(0);


		}


	}

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}

}
void face_detect_dog_filter()
{
	try
	{

		Mat dog_nose = imread("dog_nose.png");
		Mat dog_left_ear = imread("dog_left_ear.png");
		Mat dog_right_ear = imread("dog_right_ear.png");


		cv::Mat frame = imread("FB_IMG_1463846848502.jpg", CV_LOAD_IMAGE_COLOR);
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
			imshow("mmm", faceROI);
			waitKey();
			Mat face_color = frame(faces[i]);
			namedWindow("Display window1", CV_WINDOW_AUTOSIZE);
			imshow(" face detection", frame);
			waitKey(0);
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
							frame.at<Vec3b>(faces[i].y + k - int(0.25*faces[i].height), faces[i].x+l +faces[i].width-colss)[m] = dog_right_ear.at<Vec3b>(k, l)[m];



						}
					}






			
			vector<Rect_<int> > nose;
			detectNose(frame, nose);

			// Mark points corresponding to the centre (tip) of the nose
			for (unsigned int j = 0; j < nose.size(); ++j)
			{
				Rect n = nose[j];
				//circle(frame, Point(n.x + n.width / 2, n.y + n.height / 2), 3, Scalar(0, 255, 0), -1, 8);
				//nose_center_height = (n.y + n.height / 2);
				//imshow(" face detection", frame);
				//waitKey(0);

				resize(dog_nose, dog_nose, cvSize((n.width ),n.height ));
				int rows = dog_nose.rows;
				int cols = dog_nose.cols;
				for (int k = 0; k<rows; k++)
					for (int l = 0; l<cols; l++)
						for (int m = 0; m < 3; m++)
						{

							if (dog_nose.at<Vec3b>(k, l)[m]>0)
							{
								frame.at<Vec3b>(n.y + k , l + n.x)[m] = dog_nose.at<Vec3b>(k, l)[m];



							}
						}

				

				

			}
		}

		
			


			imshow(" face detection", frame);
			waitKey(0);



		}


	

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
}
void grey_world()
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

	split(I, channels1);
	Mat I0 = channels1[0];
	Mat I1 = channels1[1];
	Mat I2 = channels1[2];
	int m = I.rows;
	int n = I.cols;
	cv::Scalar  Bmean = sum(sum(I0)) / (m*n);
	cv::Scalar Gmean = sum(sum(I1)) / (m*n);
	cv::Scalar Rmean = sum(sum(I2)) / (m*n);
	double bmean = Bmean[0];
	double rmean = Rmean[0];
	double gmean = Gmean[0];
	vector<double > means;
	means.push_back(bmean);
	means.push_back(rmean);
	means.push_back(gmean);
	cv::Scalar Avg = mean(means);
	cv::Scalar Kr = Avg / Rmean;
	cv::Scalar	Kg = Avg / Gmean;
	cv::Scalar	Kb = Avg / Bmean;
	cout << Kr[0] << endl;
	Mat OUT1 = Kr[0]*I2;
	cout << OUT1.at<float>(0, 0)<<endl;
	Mat OUT2 = Kg[0]*I1;
	Mat OUT3 = Kb[0]*I0;
	vector<Mat>mergechannel;
	mergechannel.push_back(OUT3);
	mergechannel.push_back(OUT2);
	mergechannel.push_back(OUT1);
	Mat fff;
	merge(mergechannel, fff);
	double Min, Max;
	cv::minMaxLoc(fff, &Min, &Max);
	if (Min != Max) {
		fff -= Min;
		fff.convertTo(fff, CV_8U, 255.0 / (Max - Min));
	}
//	cout << eeeee.size() << " " << eeeee.channels() << endl;
	imshow("image after gey world", fff);
	//cout << fff.at<float>(0, 0) << endl;
	waitKey(0);

	
	
}

void white_patch()
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

	split(I, channels1);
	Mat I0 = channels1[0];
	Mat I1 = channels1[1];
	Mat I2 = channels1[2];
	double Min1, Max1, Min2, Max2, Min3, Max3;
	
	cv::minMaxLoc(I0, &Min1, &Max1);
	cv::minMaxLoc(I1, &Min2, &Max2);
	cv::minMaxLoc(I2, &Min3, &Max3);
	
	
 double	Kr = 255 / Max3;
 double Kg = 255 / Max2;
 double Kb = 255 / Max1;
 Mat OUT1 = Kr * I2;
 cout << OUT1.at<float>(0, 0) << endl;
 Mat OUT2 = Kg * I1;
 Mat OUT3 = Kb * I0;
 vector<Mat>mergechannel;
 mergechannel.push_back(OUT3);
 mergechannel.push_back(OUT2);
 mergechannel.push_back(OUT1);
 Mat fff;
 merge(mergechannel, fff);
 double Min, Max;
 cv::minMaxLoc(fff, &Min, &Max);
 if (Min != Max) {
	 fff -= Min;
	 fff.convertTo(fff, CV_8U, 255.0 / (Max - Min));
 }
 //	cout << eeeee.size() << " " << eeeee.channels() << endl;
 imshow("image after gey world", fff);
 //cout << fff.at<float>(0, 0) << endl;
 waitKey(0);
	/*OUT(:, : , 1) = Kr*double(I(:, : , 1));
	OUT(:, : , 2) = Kg*double(I(:, : , 2));
	OUT(:, : , 3) = Kb*double(I(:, : , 3));
	OUT = uint8(OUT);*/
}
void modified_white_patch(int varargin)
{
	int th = varargin;
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

	split(I, channels1);
	Mat I0 = channels1[0];
	Mat I1 = channels1[1];
	Mat I2 = channels1[2];
	Mat R = I2;
	R = I2 > th;
	cv::Scalar	Kr =mean(R);
	Kr[0] = 255 / Kr[0];
	Mat G = I1;
	G = I1 > th;
	cv::Scalar	Kg = mean(G);
	Kg[0] = 255 / Kg[0];
	Mat B = I0;
	B = I0 > th;
	cv::Scalar	Kb = mean(B);
	Kb[0] = 255 / Kb[0];
	cout << Kr[0] << endl;
	Mat OUT1 = Kr[0] * I2;
	cout << OUT1.at<float>(0, 0) << endl;
	Mat OUT2 = Kg[0] * I1;
	Mat OUT3 = Kb[0] * I0;
	vector<Mat>mergechannel;
	mergechannel.push_back(OUT3);
	mergechannel.push_back(OUT2);
	mergechannel.push_back(OUT1);
	Mat fff;
	merge(mergechannel, fff);
	double Min, Max;
	cv::minMaxLoc(fff, &Min, &Max);
	if (Min != Max) {
		fff -= Min;
		fff.convertTo(fff, CV_8U, 255.0 / (Max - Min));
	}
	//	cout << eeeee.size() << " " << eeeee.channels() << endl;
	imshow("image after gey world", fff);
	//cout << fff.at<float>(0, 0) << endl;
	waitKey(0);

}

void progressive(int num1, int num2)
{
}

void face_detection_haar_cascade_mariam()
{
	try
	{

		

		//imshow("mustache", mustache);
		//waitKey();


		cv::Mat frame = imread("FB_IMG_1463846848502.jpg", CV_LOAD_IMAGE_COLOR);
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
			Mat face_without_eliipse;
			frame(faces[i]).copyTo(face_without_eliipse);
			//= frame(faces[i]);
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			//grey image for face
			Mat faceROI = frame_gray(faces[i]);
			imshow("mmm", faceROI);
			waitKey();
			Mat face_color = frame(faces[i]);
			//face with ellipse
			namedWindow("Display window1", CV_WINDOW_AUTOSIZE);
			imshow(" face detection", face_color);
			waitKey(0);
			//face witout ellipse
			namedWindow("Display window2", CV_WINDOW_AUTOSIZE);
			imshow(" face detection2", face_without_eliipse);
			waitKey(0);
			
			}
	

	}

	//nose detection
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
}
int main(int, char** argv)
{
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };
	if(!nose_cascade.load(nose_cascde_name)) { printf("--(!)Error loading nose cascade\n"); return -1; };
	//face_detection_manually();
	//face_detection_harr_cascade_mustache();
	//face_detection_crown();
	//face_detect_hat_mustache();
	//face_detect_dog_filter();
	//grey_world();
	//white_patch();
	//modified_white_patch(200);
	//progressive(200, 100);
	face_detection_haar_cascade_mariam();

	
	 return 0;
	
}

