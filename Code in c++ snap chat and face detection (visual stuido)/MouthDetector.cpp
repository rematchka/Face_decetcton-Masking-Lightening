#include "MouthDetector.h"



MouthDetector::MouthDetector()
{
	mouth_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	if (!mouthclassifier.load(mouth_cascade_name)) { printf("--(!)Error loading\n"); };
}


MouthDetector::~MouthDetector()
{
	
}


MouthDetector::MouthDetector(Mat _FaceROI, Rect _faceRegion)
{
	mouth_cascade_name = "haarcascade_smile.xml";
	if (!mouthclassifier.load(mouth_cascade_name)) { printf("--(!)Error loading\n"); };

	int topleftx =0;
	int toplefty=_faceRegion.height*0.7;
	faceRegion = Rect(topleftx, toplefty, _faceRegion.width,_faceRegion.height*0.3);
	FaceROI = _FaceROI(faceRegion);


}
Mat MouthDetector::getmouth()
{
	return MouthROI;
}
Rect MouthDetector::getMouthROI()
{
	return mouthRegion;
}

void MouthDetector::detectMouth()
{
	vector<Rect>mouthRegions;
	mouthclassifier.detectMultiScale(FaceROI, mouthRegions, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (int i = 0; i < mouthRegions.size(); i++)
	{
		mouthRegion = mouthRegions[i];
	}
	MouthROI = FaceROI(mouthRegion);
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
void MouthDetector::mouth_map(Mat I)
{
	try {
		//Mat img = imread("ssss.png");
		Mat ttt = I;


		vector<Mat> channells1;
		cv::Mat iycbcr;


		cvtColor(ttt, iycbcr, CV_BGR2YCrCb);
		iycbcr.convertTo(iycbcr, CV_32FC3, 1.0 / 255.0);
		vector<Mat> channels1;
		split(iycbcr, channels1);
		Mat y = channels1[0];
		Mat cr = channels1[1];
		Mat cb = channels1[2];

		cv::Mat fullImageHSV = cv::Mat::zeros(I.size(), CV_32FC3);


		cvtColor(I, fullImageHSV, CV_BGR2HSV);

		vector<Mat> channels;
		split(fullImageHSV, channels);
		Mat hue = channels[0];
		Mat s = channels[1];


		cout << cb.at<float>(0, 0) << endl;
		cout << cr.at<float>(0, 0) << endl;

		s.convertTo(s, CV_32FC1);
		Mat powcr;
		pow(cr, 2, powcr);


		int rows = powcr.rows;
		int cols = powcr.cols;
		Mat result = cb;
		int maxi = max(rows, cols);
		int mini = min(rows, cols);


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


		Mat dst;
		int morph_operator = 0;
		int operation = morph_operator + 5;

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

		cout << dst.at<float>(0, 0);
		Mat x = Mouthmap;
		/*
		imshow("  image", dst);
		waitKey(0);*/
		/////////////////////////////////////////////////////////////////////////////////////////////////////

		dst.convertTo(dst, CV_8U);
		x.convertTo(x, CV_8U);
		int thresh = otsuThreshold(dst);
		cv::threshold(dst, img_bw, thresh, 255, CV_THRESH_BINARY);


		//best valueeeeee/////////////////////////////////////////////////
		cv::threshold(x, img_bw, 0, 255, CV_THRESH_BINARY);


		cv::threshold(x, img_bw, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);










		int minxl;
		int minyl;

		Mat stats, centroids, labelImage;
		int nLabels = connectedComponentsWithStats(img_bw, labelImage, stats, centroids, 8, CV_32S);

		for (int label = 1; label < nLabels; ++label) { //label  0 is the background


			cout << "area del component: " << label << "-> " << stats.at<int>(label, CC_STAT_AREA) << endl;

		}
		Mat surfSup = stats.col(4) <500 & stats.col(4) >100;

		Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
		for (int i = 1; i < nLabels; i++)
		{
			if (surfSup.at<uchar>(i, 0))
			{
				mask = mask | (labelImage == i);
			}
		}
		Mat r(img_bw.size(), CV_8UC1, Scalar(0));
		img_bw.copyTo(r, mask);




		vector<vector<cv::Point>> contours;


		vector<Vec4i> hierarchy;

		Mat drawing = Mat::zeros(r.size(), CV_8UC3);
		findContours(r, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		lip_w_h.resize(contours.size());
		vector<RotatedRect> minRect(contours.size());
		vector<RotatedRect> minEllipse(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			minRect[i] = minAreaRect(Mat(contours[i]));
			RotatedRect box = minAreaRect(Mat(contours[i]));
			if (box.size.width > box.size.height)
			{
				swap(box.size.width, box.size.height);
				//box.angle += 90.f;
			}
			lip_w_h[i].push_back({ box.size.width, box.size.height });
			if (contours[i].size() > 5)
			{
				minEllipse[i] = fitEllipse(Mat(contours[i]));
			}
		}
		RNG rng(12345);
		/// Draw contours + rotated rects + ellipses
		//	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

		lip_rect.resize(contours.size());
		lip_ellip.resize(contours.size());
		lip_x_y.resize(contours.size());
		for (int i = 0; i< contours.size(); i++)
		{


			minxl = 100000000;
			minyl = 100000000;
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			// contour
			drawContours(I, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			// ellipse
			ellipse(I, minEllipse[i], color, 2, 8);
			//faces_ellip[i].push_back({ minEllipse[i]. ,minEllipse[i].y });
			// rotated rectangle
			Point2f rect_points[4]; minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
			{
				line(I, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
				lip_rect[i].push_back({ rect_points[j].x, rect_points[j].y });
				minxl = min(minxl, (int)rect_points[j].x);
				minyl = min(minyl, (int)rect_points[j].y);
			}
			lip_x_y[i].push_back({ minxl, minxl });
			for (size_t cP = 0; cP < contours[i].size(); cP++)
			{
				Point currentContourPixel = contours[i][cP];
				lip_ellip[i].push_back({ currentContourPixel.x, currentContourPixel.y });
				// do whatever you want
			}
		}



		imshow("Contours lips", I);
		waitKey(0);

	}
	catch (cv::Exception & e) {
		cerr << e.msg << endl; // output exception message
	}

}
