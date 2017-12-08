
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

int isodata(Mat I)
{
	//I.convertTo(I, CV_8UC1);
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	// Calculate histogram
	MatND hist;
	calcHist(&I, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	// Show the calculated histogram in command window
	

	return 0;

}
int otsuThreshold(Mat img)
{
	int rows = img.rows;
	int cols = img.cols;
	vector<int> ihist(256);
	vector<float> hist_val(256);
	float prbn;
	float meanitr;
	float meanglb;
	int OPT_THRESH_VAL;
	float param1;
	float param2;
	double param3;
	int pos;
	prbn = 0.0;
	meanitr = 0.0;
	meanglb = 0.0;
	OPT_THRESH_VAL = 0;

	param3 = 0.0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			pos = img.at<uchar>(i, j);  // Check the pixel value
			ihist[pos] += 1; // Use the pixel value as the position/"Weight"
		}
	}


	//Normalise histogram values and calculate global mean level
	for (int i = 0; i < 256; ++i)
	{
		hist_val[i] = ihist[i] / (float)(rows * cols);
		meanglb += ((float)i * hist_val[i]);
	}
	

	// Implementation of OTSU algorithm
	for (int i = 0; i < 255; i++)
	{
		prbn += (float)hist_val[i];
		meanitr += ((float)i * hist_val[i]);

		param1 = (float)((meanglb * prbn) - meanitr);
		param2 = (float)(param1 * param1) / (float)(prbn * (1.0f - prbn));

		if (param2 > param3)
		{
			param3 = param2;
			OPT_THRESH_VAL = i;     // Update the "Weight/Value" as Optimum Threshold value
		}
	}
	return OPT_THRESH_VAL;


}
/*int isodata(Mat I)
{
	//I.convertTo(I, CV_8UC1);
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	// Calculate histogram
	MatND hist;
	calcHist(&I, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	// Show the calculated histogram in command window


	return 0;

}*/

void eye_map(Mat& face_image)
{
	try {
		Mat ttt = face_image;
		ttt.convertTo(ttt, CV_32FC3);
		vector<Mat> channells1;
		split(ttt, channells1);
		Mat I0 = channells1[0];
		Mat I1 = channells1[1];
		Mat I2 = channells1[2];
		cout << I0.channels() << endl;
		Mat iycbcr;
		cvtColor(face_image, iycbcr, CV_BGR2YCrCb);
		vector<Mat> channels1;
		split(iycbcr, channels1);
		
		Mat y = channels1[0];
		Mat cr = channels1[1];
		Mat cb = channels1[2];
		cout << " y value 0 " << y.at<uchar>(0, 0) << endl;
		y = 0.299  *I2 + 0.587 *I1 + 0.114 *I0+16;
		//cout << " y value 0 " << y.at<float>(0, 0) << endl;
		//y.convertTo(y, CV_32FC1);
		cr.convertTo(cr, CV_32FC1);
		cb.convertTo(cb, CV_32FC1);
		//cout<<" y value "<< y.at<float>(0, 0) << endl;
		
		Mat Q;
		pow(cb, 2, Q);
		Mat R = 255 - cr;
		pow(R, 2, R);
		Mat G = cv::Mat(cb / cr);
		Mat CrCb = cv::Mat(cr / cb);
		Mat	EyeC = (Q + R + G) / 3;
		Mat CRS; pow(cr, 2, CRS);
		Scalar	ssCRS = sum(sum(CRS));
		Scalar ssCrCb = sum(sum(CrCb));
		double eta = 0.95 * ssCRS[0] / ssCrCb[0];
		Mat x = CRS - eta * CrCb;
		Mat xx = CRS.mul(x);
		Mat MM = xx.mul(x);
		Mat mat = Mat::ones(9, 9, CV_8U);

		mat.at<uchar>(0, 0) = 0;
		mat.at<uchar>(0, 1) = 0;
		
		mat.at<uchar>(0, 7) = 0;
		mat.at<uchar>(0, 8) = 0;
		mat.at<uchar>(1, 0) = 0;
		mat.at<uchar>(1, 8) = 0;
		mat.at<uchar>(8, 0) = 0;
		mat.at<uchar>(8, 1) = 0;
		mat.at<uchar>(7, 0) = 0;
		mat.at<uchar>(7, 8) = 0;
		mat.at<uchar>(8, 7) = 0;
		mat.at<uchar>(8, 8) = 0;
		cout << mat << endl;
		//cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * 4 - 1, 2 * 4 - 1));
		Mat dil1 = cv::Mat::zeros(y.size(), CV_32FC1);
		cv::dilate(y, dil1, mat);
		Mat erodee1 = cv::Mat::zeros(y.size(), CV_32FC1);
		cv::erode(y, erodee1, mat);
		Mat EyeY = cv::Mat(dil1 / (erodee1+1));
		Mat EyeMap = EyeY.mul(EyeC);
		//Initialize m
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;

		minMaxLoc(EyeC, &minVal, &maxVal, &minLoc, &maxLoc);
		EyeC = EyeC / maxVal;
		minMaxLoc(EyeMap, &minVal, &maxVal, &minLoc, &maxLoc);
		EyeMap = EyeMap / maxVal;
		minMaxLoc(EyeY, &minVal, &maxVal, &minLoc, &maxLoc);
		EyeY = EyeY / maxVal;

		minMaxLoc(dil1, &minVal, &maxVal, &minLoc, &maxLoc);
		dil1 = dil1 / maxVal;

		minMaxLoc(erodee1, &minVal, &maxVal, &minLoc, &maxLoc);
		erodee1 = erodee1 / maxVal;
		imshow("Eye map ", EyeMap);

		waitKey(0);

		imshow(" Eye c", EyeC);
		waitKey(0);

		imshow(" Eye y", EyeY);
		waitKey(0);
		imshow(" erode", erodee1);
		waitKey(0);
		imshow(" dil", dil1);
		waitKey(0);

		Mat normalizedImage;
		normalize(EyeMap, normalizedImage, 1.0, 0.0, NORM_MINMAX,  CV_64F);
		//EyeMap.convertTo(normalizedImage, CV_64FC3, 1.0 / 255.0);
		Mat img_bw;
		imshow(" threshold image", normalizedImage);
		waitKey(0);
		EyeMap.convertTo(EyeMap,CV_8U );
		cout << normalizedImage.channels() << endl;
		cv::threshold(EyeMap, img_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		imshow(" threshold image", img_bw);
		waitKey(0);

	}
	catch (cv::Exception & e) {
		cerr << e.msg << endl; // output exception message
	}

}
cv::Mat mat2gray(cv::Mat inMat)
{
	//idea: to be scaled between 0 and 1, compute for each value: val := (val - minVal)/(maxVal-minVal)

	if (inMat.channels() != 1) std::cout << "mat2gray only works for single channel floating point matrices" << std::endl;

	// here we assume floating point matrices (single/double precision both ok)
	double minVal, maxVal;

	cv::minMaxLoc(inMat, &minVal, &maxVal);

	cv::Mat scaledMat = inMat.clone();
	scaledMat = scaledMat - cv::Mat(inMat.size(), inMat.type(), cv::Scalar(minVal));
	scaledMat = ((1.0) / (maxVal - minVal))*scaledMat;

	return scaledMat;
}
void another_eye_map(Mat I)
{
	try {
		Mat ttt = I;

		ttt.convertTo(ttt, CV_32FC3);
		Mat iycbcr;
		cvtColor(ttt, iycbcr, CV_BGR2YCrCb);
		vector<Mat> channels1;
		split(iycbcr, channels1);
		Mat y = channels1[0];
		Mat cr = channels1[1];
		Mat cb = channels1[2];
	
		y = 255 * mat2gray(y);
		cr = 255 * mat2gray(cr);
		cb = 255 * mat2gray(cb);
		y.convertTo(y, CV_8U);
		cr.convertTo(cr, CV_8U);
		cb.convertTo(cb, CV_8U);
		cv::Mat fullImageHSV ;


		cvtColor(I, fullImageHSV, CV_BGR2HSV);

		vector<Mat> channels;
		split(fullImageHSV, channels);
		Mat hue = channels[0];
		Mat s = channels[1];

		//	normalize(cb, cb, 1.0, 0.0, NORM_MINMAX, CV_32F);
		//normalize(cr, cr, 1.0, 0.0, NORM_MINMAX, CV_32F);
		//cr.convertTo(cr, CV_32F, 255.0);
		//cb.convertTo(cb, CV_32F, 255.0);
		//cb.convertTo(cb, CV_32F, 1.0 / 255.0);
		//cr.convertTo(cr, CV_32F, 1.0 / 255.0);
	
		//Mat R;
		//pow((cr), 2.0, R);
		//s.convertTo(s, CV_32FC1);
		Mat powcr;
		pow(cr, 2, powcr);

		//cout << R.size() << endl;
		//cout << powcr << endl;
		int rows = powcr.rows;
		int cols = powcr.cols;
		Mat result = cb;
		int maxi = max(rows, cols);
		int mini = min(rows, cols);
	
		//invert(cb, inverted, cv::DECOMP_SVD);
		//resize(cr, cr, cvSize(maxi, maxi));
		//resize(cb, cb, cvSize(maxi, maxi));
		//cb = abs(cb);
		//cout << inverted << endl;
		//	
		//Mat xx = cb.reshape(1, maxi);
		//xx.convertTo(xx, CV_32F);
		//Mat invcb = xx.inv(DECOMP_LU);
		//cv::solve(cr, cb, result, DECOMP_LU);



		Scalar  ss = sum(sum(powcr));
		double ee = ss[0];

		Mat CrCb = cv::Mat(cr / cb);
		Scalar dCrCb = sum(sum(CrCb));
		cout << dCrCb[0] << endl;

		double eitha = 0.95*(ss[0] / dCrCb[0]);

		cout << eitha << endl;



		Mat powcrcb;

		Mat xxx = powcr - (eitha*CrCb);
		pow(xxx, 2, xxx);
		Mat	Mouthmap = powcr.mul(xxx);
		Mat	EnLip = s.mul(Mouthmap);
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;

/*		minMaxLoc(Mouthmap, &minVal, &maxVal, &minLoc, &maxLoc);
		Mouthmap = Mouthmap / maxVal;


		minMaxLoc(EnLip, &minVal, &maxVal, &minLoc, &maxLoc);
		EnLip = EnLip / maxVal;*/

		imshow("mouth map", Mouthmap);
		waitKey();
		imshow("EnLip", EnLip);
		waitKey();
		Mat dst;
		int morph_operator = 0;
		int operation = morph_operator + 5;
		cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * 25 - 1, 2 * 25 - 1));
		Mat mat = Mat::ones(9, 9, CV_8U);

		mat.at<uchar>(0, 0) = 0;
		mat.at<uchar>(0, 1) = 0;

		mat.at<uchar>(0, 7) = 0;
		mat.at<uchar>(0, 8) = 0;
		mat.at<uchar>(1, 0) = 0;
		mat.at<uchar>(1, 8) = 0;
		mat.at<uchar>(8, 0) = 0;
		mat.at<uchar>(8, 1) = 0;
		mat.at<uchar>(7, 0) = 0;
		mat.at<uchar>(7, 8) = 0;
		mat.at<uchar>(8, 7) = 0;
		mat.at<uchar>(8, 8) = 0;
		/// Apply the specified morphology operation
		morphologyEx(Mouthmap, dst, operation, mat);
		Mat img_bw;
		imshow(" morphing image", dst);
		waitKey(0);
		
		Mat x = Mouthmap;
		/*
		imshow("  image", dst);
		waitKey(0);*/
		/////////////////////////////////////////////////////////////////////////////////////////////////////

		dst.convertTo(dst, CV_8U);
		x.convertTo(x, CV_8U);
		int thresh = otsuThreshold(dst);
		cv::threshold(dst, img_bw, thresh, 255, CV_THRESH_BINARY);

		imshow(" otsu image", img_bw);
		waitKey(0);
		cv::threshold(x, img_bw, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);

		imshow(" threshold image", img_bw);
		waitKey(0);

		cv::threshold(dst, img_bw, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);

		imshow(" ther image", img_bw);
		waitKey(0);
	}
	catch (cv::Exception & e) {
		cerr << e.msg << endl; // output exception message
	}

}

void mouth_map(Mat I)
{
	try {
		//Mat img = imread("ssss.png");
		Mat ttt = I;

		ttt.convertTo(ttt, CV_32FC3);
		vector<Mat> channells1;
		cv::Mat iycbcr = cv::Mat::zeros(I.size(), CV_32FC3);
		/*split(ttt, channells1);
		Mat I0 = channells1[0];
		Mat I1 = channells1[1];
		Mat I2 = channells1[2];
		I0.convertTo(I0, CV_32FC1);
		I1.convertTo(I1, CV_32FC1);
		I2.convertTo(I2, CV_32FC1);
		I0 = I0 / 255.0;
		I1 = I1 / 255.0;
		I2 = I2 / 255.0;*/
		//Mat iycbcr;
		cvtColor(ttt, iycbcr, CV_BGR2YCrCb);
		vector<Mat> channels1;
		split(iycbcr, channels1);
		Mat y = channels1[0];
		Mat cr = channels1[1];
		Mat cb = channels1[2];
		//cout << " y value 0 " << y.at<uchar>(0, 0) << endl;

		//cout << " y value 0 " << y.at<float>(0, 0) << endl;
		//y.convertTo(y, CV_32FC1);
		//cr.convertTo(cr, CV_32FC1);
		//cb.convertTo(cb, CV_32FC1);
		//y = 0.299  *I2 + 0.587 *I1 + 0.114 *I0 + 16;
		//cb = 0.148* I2 - 0.291*  I1 + 0.439 * I0 +128;
		//cr = 0.439 *I2 - 0.368 * I1 - 0.071 * I0 +128;
		//cout << " y value 0 " << y.at<uchar>(0, 0) << endl;

		//cout << " y value 0 " << y.at<float>(0, 0) << endl;
		cv::Mat fullImageHSV = cv::Mat::zeros(I.size(), CV_32FC3);


		cvtColor(I, fullImageHSV, CV_BGR2HSV);

		vector<Mat> channels;
		split(fullImageHSV, channels);
		Mat hue = channels[0];
		Mat s = channels[1];

		//	normalize(cb, cb, 1.0, 0.0, NORM_MINMAX, CV_32F);
			//normalize(cr, cr, 1.0, 0.0, NORM_MINMAX, CV_32F);
			//cr.convertTo(cr, CV_32F, 255.0);
			//cb.convertTo(cb, CV_32F, 255.0);
			//cb.convertTo(cb, CV_32F, 1.0 / 255.0);
			//cr.convertTo(cr, CV_32F, 1.0 / 255.0);
		cout << cb.at<float>(0, 0) << endl;
		cout << cr.at<float>(0, 0) << endl;
		//Mat R;
		//pow((cr), 2.0, R);
		s.convertTo(s, CV_32FC1);
		Mat powcr;
		pow(cr, 2, powcr);
		
		//cout << R.size() << endl;
		//cout << powcr << endl;
		int rows = powcr.rows;
		int cols = powcr.cols;
		Mat result=cb;
		int maxi = max(rows, cols);
		int mini= min(rows, cols);
		//cv::Mat inverted(cols, rows, CV_32FC1);
		//invert(cb, inverted, cv::DECOMP_SVD);
		//resize(cr, cr, cvSize(maxi, maxi));
		//resize(cb, cb, cvSize(maxi, maxi));
		//cb = abs(cb);
		//cout << inverted << endl;
	//	
		//Mat xx = cb.reshape(1, maxi);
		//xx.convertTo(xx, CV_32F);
		//Mat invcb = xx.inv(DECOMP_LU);
		//cv::solve(cr, cb, result, DECOMP_LU);
		
		

		Scalar  ss = sum(sum(powcr));
		double ee = ss[0];
		
		Mat CrCb = cv::Mat(cr / cb);
		Scalar dCrCb = sum(sum(CrCb));
		cout << dCrCb[0] << endl;
		
		double eitha = 0.95*(ss[0] / dCrCb[0]);

		cout << eitha << endl;



		Mat powcrcb;
		
		Mat xxx = powcr - (eitha*CrCb);
		pow(xxx, 2, xxx);
		Mat	Mouthmap = powcr.mul(xxx);
		Mat	EnLip = s.mul(Mouthmap);
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;

		minMaxLoc(Mouthmap, &minVal, &maxVal, &minLoc, &maxLoc);
		Mouthmap = Mouthmap / maxVal;


		minMaxLoc(EnLip, &minVal, &maxVal, &minLoc, &maxLoc);
		EnLip = EnLip / maxVal;

		imshow("mouth map", Mouthmap);
		waitKey();
		imshow("EnLip", EnLip);
		waitKey();
		Mat dst;
		int morph_operator = 0;
		int operation = morph_operator + 5;
		//cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * 25- 1, 2 * 25 - 1));
		Mat mat = Mat::ones(9, 9, CV_8U);

		mat.at<uchar>(0, 0) = 0;
		mat.at<uchar>(0, 1) = 0;

		mat.at<uchar>(0, 7) = 0;
		mat.at<uchar>(0, 8) = 0;
		mat.at<uchar>(1, 0) = 0;
		mat.at<uchar>(1, 8) = 0;
		mat.at<uchar>(8, 0) = 0;
		mat.at<uchar>(8, 1) = 0;
		mat.at<uchar>(7, 0) = 0;
		mat.at<uchar>(7, 8) = 0;
		mat.at<uchar>(8, 7) = 0;
		mat.at<uchar>(8, 8) = 0;
		/// Apply the specified morphology operation
		morphologyEx(EnLip, dst, operation, mat);
		Mat img_bw;
		imshow(" morphing image", dst);
		waitKey(0);
		cout << dst.at<float>(0, 0);
		Mat x = Mouthmap;
	/*
	imshow("  image", dst);
	waitKey(0);*/
	/////////////////////////////////////////////////////////////////////////////////////////////////////
		
		dst.convertTo(dst, CV_8U);
		x.convertTo(x, CV_8U);
		int thresh = otsuThreshold(dst);
		cv::threshold(dst, img_bw, thresh, 255, CV_THRESH_BINARY );

		imshow(" otsu image", img_bw);
		waitKey(0);
		cv::threshold(x, img_bw, 0, 255, CV_THRESH_BINARY);

		imshow(" threshold image", img_bw);
		waitKey(0);

		cv::threshold(dst, img_bw, thresh, 255, CV_THRESH_BINARY );

		imshow(" ther image", img_bw);
		waitKey(0);
	}
	catch (cv::Exception & e) {
		cerr << e.msg << endl; // output exception message
	}

}
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
	//imshow("Display window", I);
	//waitKey(0);
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

		cv::Mat sel = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(15, 15));
		Mat dil = cv::Mat::zeros(hue.size(), CV_32FC1);

		cv::dilate(segment, dil, sel);
		Mat erodee = cv::Mat::zeros(hue.size(), CV_32FC1);
		cv::erode(dil, erodee, sel);


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


		//segment.convertTo(segment, CV_8S);
		
		erodee.convertTo(erodee, CV_8U);
		cout << erodee.channels() << endl;
		
		//CV_Assert(r.type() == CV_8UC1);
		Mat dst;
		vector<vector<cv::Point>> contours;
		vector<Vec4i> hierarchy;

		findContours(erodee, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		CvScalar color = cvScalar(255);
		dst = Mat::zeros(erodee.size(), CV_8UC1);

		for (int i = 0; i<contours.size(); i++)
		{
			drawContours(dst, contours, i, color, -1, 8, hierarchy, 0, cv::Point());
		}
		namedWindow("Contours", CV_WINDOW_AUTOSIZE);
		medianBlur(dst, dst, (5, 5));
		
		Mat dil1 = cv::Mat::zeros(hue.size(), CV_32FC1);

		cv::dilate(dst, dil1, sel);
		Mat erodee1 = cv::Mat::zeros(hue.size(), CV_32FC1);
		cv::erode(dil1, erodee1, sel);

		imshow("Contours", dst);
		
		waitKey();

		imshow("erode 22", erodee1);

		waitKey();

		//erodee1.convertTo(erodee1, CV_8S);
		Mat stats, centroids, labelImage;
		int nLabels = connectedComponentsWithStats(erodee1, labelImage, stats, centroids, 8, CV_32S);
		Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
		int maxi = 0;
		for (int label = 1; label < nLabels; ++label) { //label  0 is the background


			cout << "area del component: " << label << "-> " << stats.at<int>(label, CC_STAT_AREA) << endl;
			//colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
			maxi = max(maxi, stats.at<int>(label, CC_STAT_AREA));
		}
		Mat surfSup = stats.col(4) >= 30000;


		for (int i = 1; i < nLabels; i++)
		{
			if (surfSup.at<uchar>(i, 0))
			{
				mask = mask | (labelImage == i);
			}
		}
		Mat r(erodee1.size(), CV_8UC1, Scalar(0));
		erodee1.copyTo(r, mask);
	
	
		imshow("Result", r);
		waitKey();
		vector<Mat> channels11;
		split(r, channels11);
		Mat ro = channels11[0];
		
		//to do and face with contour
		/*ro.convertTo(ro, CV_32F);
		I0 = I0*ro;
		I1 = I1*ro;
		I2 = I2*ro;*/
		I0.convertTo(I0, CV_8U);
		I1.convertTo(I1, CV_8U);
		I2.convertTo(I2, CV_8U);
		bitwise_and(I0, ro,I0);
		bitwise_and(I1, ro, I1);
		bitwise_and(I2, ro, I2);
		vector<Mat> channelorigi;
		channelorigi.push_back(I0);
		channelorigi.push_back(I1);
		channelorigi.push_back(I2);
		Mat out;
		merge(channelorigi, out);
		imshow("extract face", out);




		waitKey(0);

		//eye_map(out);
		mouth_map(out);
		//another_eye_map(out);


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


		cv::Mat frame = imread("hhhhhhhh#.png", CV_LOAD_IMAGE_COLOR);
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
					int x1 = n.x - (mostw / 4) + faces[i].x, x2 = n.x + n.width + (mostw / 4)+faces[i].x, y1 = n.y + n.height - (mosth / 2)+faces[i].y, y2 = n.y + n.height + (mosth / 2) + faces[i].y;
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
		

		cv::Mat frame = imread("hhhhhhhh#.png", CV_LOAD_IMAGE_COLOR);
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
							frame.at<Vec3b>(abs(faces[i].y + k - int(0.5*faces[i].height)), l + faces[i].x)[m] = flower.at<Vec3b>(k, l)[m];
							
							

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


		cv::Mat frame = imread("hhhhhhhh#.png", CV_LOAD_IMAGE_COLOR);
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
			resize(hat, hat, cvSize((faces[i].width+1), (faces[i].height*0.5)+1));
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



			Mat mst= imread("moustache.png");
			int mst_width = int(faces[i].width*0.4166666) + 1;
			int	mst_height = int(faces[i].height*0.14285) + 1;



			resize(mst, mst, cvSize(mst_width, mst_height));

			for (int ii = (int(0.62857142857*faces[i].height)); ii<int(0.62857142857*faces[i].height) + mst_height; ii++)
				for (int j =int(0.29166666666*faces[i].width);j< int(0.29166666666*faces[i].width) + mst_width;j++)
					for (int k = 0; k < 3; k++)
					{
						if (mst.at<Vec3b>(ii - int(0.62857142857*faces[i].height), j - int(0.29166666666*faces[i].width))[k] < 235)
							frame.at < Vec3b>(faces[i].y + ii, faces[i].x + j)[k] =mst.at<Vec3b>(ii - int(0.62857142857*faces[i].height), j - int(0.29166666666*faces[i].width))[k];
						
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


		cv::Mat frame = imread("hhhhhhhh#.png", CV_LOAD_IMAGE_COLOR);
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
			detectNose(faceROI, nose);

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
								frame.at<Vec3b>(n.y + k+faces[i].y, l + n.x+ faces[i].x)[m] = dog_nose.at<Vec3b>(k, l)[m];



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
			Point center(faces[i].x + faces[i].width /2, faces[i].y + faces[i].height/2 );
			ellipse(frame, center, Size(faces[i].width/2 , faces[i].height/2 ), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
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
	face_detection_manually();
	//face_detection_harr_cascade_mustache();
	//face_detection_crown();
	//face_detect_hat_mustache();
	//face_detect_dog_filter();
	//grey_world();
	//white_patch();
	//modified_white_patch(200);
	//progressive(200, 100);
	//face_detection_haar_cascade_mariam();

	
	 return 0;
	
}
