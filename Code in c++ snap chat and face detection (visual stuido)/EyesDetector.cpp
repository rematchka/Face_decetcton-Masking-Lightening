#include "EyesDetector.h"



EyesDetector::EyesDetector()
{
	eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); };
}


EyesDetector::~EyesDetector()
{
}


EyesDetector::EyesDetector(Mat _FaceROI)
{
	eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); };
	FaceROI = _FaceROI;

}


vector<Rect> EyesDetector::getEyesROI()
{
	return eyesROI;
}


vector<Mat> EyesDetector::getEyes()
{
	return eyes;
}


void EyesDetector::detectEyes()
{
	eyes_cascade.detectMultiScale(FaceROI, eyesROI, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (int i = 0; i < eyesROI.size(); i++)
	{
		eyes.push_back(FaceROI(eyesROI[i]));
	}
}
int EyesDetector::otsuThreshold(Mat img)
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

Mat EyesDetector:: mat2gray(cv::Mat inMat)
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
void EyesDetector::eye_map(Mat  face_image)
{
	//face_image = imread("download (4).jpg");
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

		iycbcr.convertTo(iycbcr, CV_32FC3, 1.0 / 255.0);
		vector<Mat> channels1;
		split(iycbcr, channels1);

		Mat y = channels1[0];
		Mat cr = channels1[1];
		Mat cb = channels1[2];




		Mat Q;
		pow(cb, 2, Q);
		Mat R = 1 - cr;
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

		Mat dil1 = cv::Mat::zeros(y.size(), CV_32FC1);
		cv::dilate(y, dil1, mat);
		Mat erodee1 = cv::Mat::zeros(y.size(), CV_32FC1);
		cv::erode(y, erodee1, mat);

		Mat EyeY = cv::Mat(dil1 / (erodee1 + 1));

		Mat EyeMap = EyeY.mul(EyeC);

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

		Mat norm = EyeMap;
		Mat normalizedImage;
		norm = mat2gray(EyeMap);

		normalize(EyeMap, normalizedImage, 1.0, 0.0, NORM_MINMAX, CV_64F);

		Mat img_bw;
		//imshow(" threshold image", normalizedImage);
		//waitKey(0);
		norm.convertTo(norm, CV_8U);
		normalizedImage.convertTo(normalizedImage, CV_8U);
		cout << normalizedImage.channels() << endl;
		/*cv::threshold(normalizedImage, img_bw, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);

		imshow(" threshold image", img_bw);
		waitKey(0);
		*/
		cv::threshold(norm, img_bw, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);

		//imshow("norm  threshold image", img_bw);
		//waitKey(0);


		Mat stats, centroids, labelImage;
		int nLabels = connectedComponentsWithStats(img_bw, labelImage, stats, centroids, 8, CV_32S);

		int maxi = 0;
		for (int label = 1; label < nLabels; ++label) { //label  0 is the background


			cout << "area del component: " << label << "-> " << stats.at<int>(label, CC_STAT_AREA) << endl;

			maxi = max(maxi, stats.at<int>(label, CC_STAT_AREA));
		}
		Mat surfSup = stats.col(4) <200 & stats.col(4) >100;

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
		int minxe;
		int minye;
		Mat drawing = Mat::zeros(r.size(), CV_8UC3);
		findContours(r, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		eyes_w_h.resize(contours.size());
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
			eyes_w_h[i].push_back({ box.size.width, box.size.height });

			if (contours[i].size() > 5)
			{
				minEllipse[i] = fitEllipse(Mat(contours[i]));
			}
		}
		RNG rng(12345);
		/// Draw contours + rotated rects + ellipses
		//	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

		eyes_rect.resize(contours.size());
		eyes_ellip.resize(contours.size());
		eye_x_y.resize(contours.size());

		for (int i = 0; i< contours.size(); i++)
		{

			minxe = 100000000;
			minye = 100000000;
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			// contour
			drawContours(face_image, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			// ellipse
			ellipse(face_image, minEllipse[i], color, 2, 8);
			//faces_ellip[i].push_back({ minEllipse[i]. ,minEllipse[i].y });
			// rotated rectangle
			Point2f rect_points[4]; minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
			{
				line(face_image, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
				eyes_rect[i].push_back({ rect_points[j].x, rect_points[j].y });
				minxe = min(minxe, (int)rect_points[j].x);
				minye = min(minye, (int)rect_points[j].y);
			}
			eye_x_y[i].push_back({ minxe, minye });
			for (size_t cP = 0; cP < contours[i].size(); cP++)
			{
				Point currentContourPixel = contours[i][cP];
				eyes_ellip[i].push_back({ currentContourPixel.x, currentContourPixel.y });
				// do whatever you want
			}
		}



		imshow("Contours eye", face_image);
		//waitKey(0);
	}
	catch (cv::Exception & e) {
		cerr << e.msg << endl; // output exception message
	}

}