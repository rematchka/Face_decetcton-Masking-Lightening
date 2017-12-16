#include "FaceDetector.h"


Mat src, test;
int thresh = 100;
int max_thresh = 255;
RNG rng(1);

FaceDetector::FaceDetector()
{
	face_cascade_name = "haarcascade_frontalface_alt.xml";
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); };
	faces_rect.resize(5);
	faces_ellip.resize(5);
	faces_x_y.resize(5);
	faces_max_x_y.resize(5);
	faces_w_h.resize(5);
	
	cnt_faces = 0;
}


FaceDetector::~FaceDetector()
{
}


Rect FaceDetector::getFaceROI()
{
	return FaceRegion;
}


FaceDetector::FaceDetector(Mat _image)
{
	face_cascade_name = "haarcascade_frontalface_alt.xml";
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); };
	image = _image;
}


void FaceDetector::detectFace()
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(image, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		 shrinkFaceROI(faces[i]);
	}

	FaceROI = image(FaceRegion);
}


void FaceDetector::shrinkFaceROI(Rect ROI)
{
	 Point center(ROI.x + ROI.width*0.5,ROI.y + ROI.height*0.5);

	 Point topLeft = Point(ROI.x, ROI.y) - center;
	 Point topRight= Point(ROI.x+ROI.width, ROI.y) - center;
	 Point bottomLeft = Point(ROI.x, ROI.y + ROI.height) - center;
	 Point bottomRight = Point(ROI.x + ROI.width, ROI.y + ROI.height)-center;

	 vector<Point> scaledRect = _scale(topLeft, topRight, bottomLeft, bottomRight,0.9,0.9);
	 for (int i = 0; i < scaledRect.size(); i++)
	 {
		 scaledRect[i] += center;
	 }
	 int ROIwidth = (int)abs(scaledRect[1].x - scaledRect[0].x);
	 int ROIheight =(int) abs(scaledRect[0].y - scaledRect[2].y);

	 FaceRegion = Rect(scaledRect[0], Size(ROIwidth, ROIheight));



}

vector<Point> FaceDetector::_scale(Point _topLeft, Point _topRight, Point _bottomLeft, Point _bottomRight,float fx,float fy)
{
	vector<Point> scaledpoints;

	scaledpoints.push_back(Point(int(_topLeft.x*fx), int(_topLeft.y*fy)));
	scaledpoints.push_back(Point(int(_topRight.x*fx),int( _topRight.y*fy)));
	scaledpoints.push_back(Point(int(_bottomLeft.x*fx),int( _bottomLeft.y*fy)));
	scaledpoints.push_back(Point(int(_bottomRight.x*fx), int(_bottomRight.y*fy)));

	return scaledpoints;
}


Mat FaceDetector::getFace()
{
	return FaceROI;
}

void FaceDetector::createEllipticalFaceMask()
{
	int x = (int)FaceRegion.width / 2.0;
	int y = (int)FaceRegion.height / 2.0;
	ellipsemask = Mat(FaceROI.rows, FaceROI.cols, CV_8UC3, Scalar(0, 0, 0));
	ellipse(ellipsemask, Point(x, y), Size(FaceRegion.width*0.5, FaceRegion.height*0.6), 0, 0, 360, Scalar(255, 255, 255), -1, 8);
	imshow("ellipsemask", ellipsemask);
}
Mat FaceDetector::getEllipticalFaceMask()
{
	
	return ellipsemask;
}
Mat FaceDetector::getNegativeEllipticalFaceMask()
{
	Mat negativemask;
	bitwise_not(ellipsemask, negativemask);
	return negativemask;
}
Mat FaceDetector::getEllipticalFace()
{
	//createEllipticalFaceMask();

	Mat faceEllipse;
	bitwise_and(ellipsemask, FaceROI, faceEllipse);
	imshow("ellipseface", faceEllipse);
	return faceEllipse;


}

void FaceDetector::contour(Mat I)
{

	faces_ellip.clear();
	faces_max_x_y.clear();
	faces_rect.clear();
	faces_w_h.clear();
	faces_x_y.clear();
	Mat threshold_output=I;
	cv::cvtColor(I, threshold_output, CV_RGB2GRAY);



	Mat stats, centroids, labelImage;
	int nLabels = connectedComponentsWithStats(threshold_output, labelImage, stats, centroids, 8, CV_32S);

	
	Mat surfSup = stats.col(4)  >10000;

	Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
	for (int i = 1; i < nLabels; i++)
	{
		if (surfSup.at<uchar>(i, 0))
		{
			mask = mask | (labelImage == i);
		}
	}
	Mat r(threshold_output.size(), CV_8UC1, Scalar(0));
	threshold_output.copyTo(r, mask);





	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat drawing = Mat::zeros(r.size(), CV_8UC3);
	findContours(r, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	int cnt = 0;
	for(int i=0;i<contours.size();i++)
		
	faces_w_h.resize(contours.size());
	faces_x_y.reserve(contours.size());
	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		//if (contours[i].area()>10000)
		minRect[i] = minAreaRect(Mat(contours[i]));
		RotatedRect box = minAreaRect(Mat(contours[i]));
		if (box.size.width > box.size.height)
		{
			swap(box.size.width, box.size.height);
			//box.angle += 90.f;
		}
		
		faces_w_h[i].push_back({ (box.size.width) , (box.size.height) });
		//lip_rect[i].push_back({ rect_points[j].x, rect_points[j].y });
		if (contours[i].size() > 5)
		{
			minEllipse[i] = fitEllipse(Mat(contours[i]));
		}
	}
	RNG rng(12345);
	/// Draw contours + rotated rects + ellipses

	faces_rect.resize(contours.size());
	faces_ellip.resize(contours.size());
	faces_x_y.resize(contours.size());
	faces_max_x_y.resize(contours.size());

	int w, h;
	minxf = 100000000;
	minyf = 100000000;

	//RotatedRect box = minAreaRect(pts);

	// Be sure that largest side is the height

	for (int i = 0; i< contours.size(); i++)
	{

		minxf = 100000000;
		minyf = 100000000;
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		// contour
		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		// ellipse
		ellipse(drawing, minEllipse[i], color, 2, 8);

		// rotated rectangle
		Point2f rect_points[4]; minRect[i].points(rect_points);

		minfex = 100000000;
		minfey = 100000000;

		for (int j = 0; j < 4; j++)
		{
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);

			faces_rect[i].push_back({ rect_points[j].x, rect_points[j].y });
			minxf = min(minxf, (int)rect_points[j].x);
			minyf = min(minyf, (int)rect_points[j].y);
		}
		faces_x_y[i].push_back({ minxf, minyf });
		for (size_t cP = 0; cP < contours[i].size(); cP++)
		{
			Point currentContourPixel = contours[i][cP];

			faces_ellip[i].push_back({ currentContourPixel.x, currentContourPixel.y });
			minfex = min(currentContourPixel.x, minfex);
			minfey = min(currentContourPixel.y, minfey);
			maxx = max(currentContourPixel.x, maxx);
			maxy = max(currentContourPixel.y, maxy);
			// do whatever you want
		}
		faces_max_x_y[i].push_back({ maxx ,maxy });
	}


}


void YCgCr(double b, double g, double r, double &y, double &cg, double& cr)
{
	y = 16 + 0.256*r + 0.5041*g + 0.0979*b;
	cg = 128 - 0.3180*r - 0.4392*g - 0.1212*b;
	cr = 128 + 0.4392*r - 0.3677*g - 0.0714*b;// R: 79 G : 62 B : 42

}
void Dilation(Mat & src, int dilationElem, int dilationSize)
{
	Mat temp;
	int dilationType;
	if (dilationElem == 0) { dilationType = MORPH_RECT; }
	else if (dilationElem == 1) { dilationType = MORPH_CROSS; }
	else if (dilationElem == 2) { dilationType = MORPH_ELLIPSE; }
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


	Mat element = getStructuringElement(dilationType,
		Size(2 * dilationSize + 1, 2 * dilationSize + 1),
		Point(dilationSize, dilationSize));
	/// Apply the dilation operation
	dilate(src, src, mat);

}
void Erosion(Mat & src, int erosionElm, int erosionSize)
{
	Mat temp;
	int erosionType;
	if (erosionElm == 0) { erosionType = MORPH_RECT; }
	else if (erosionElm == 1) { erosionType = MORPH_CROSS; }
	else if (erosionElm == 2) { erosionType = MORPH_ELLIPSE; }
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


	Mat element = getStructuringElement(erosionType,
		Size(2 * erosionSize + 1, 2 * erosionSize + 1),
		Point(erosionSize, erosionSize));
	/// Apply the dilation operation
	erode(src, src, mat);

}
Mat lightening(Mat img)
{
	Mat src, dst, ycrcb;
	vector<Mat> channels;
	img.copyTo(src);

	if (img.channels() == 3) {
		//cout << "converting to gray from 3\n";
		//convert to 1 channel because of equalizehist 
		cvtColor(src, ycrcb, CV_BGR2YCrCb);
		split(ycrcb, channels);

		//apply histogram equalization to light the pic 
		//there is another algorithms like reference white ( it is slow )
		//so i decided to use histogram equalization instead 
		equalizeHist(channels[0], channels[0]);
		//merge 3 channels together and convert to BGR again
		merge(channels, ycrcb);
		cvtColor(ycrcb, dst, CV_YCrCb2BGR);
		//imshow("frame", dst);
	}
	else if (img.channels() == 4) {
		cout << "Error it is BGRA image \n";
		//	cvtColor(src, src, CV_BGRA2HSV);
	}

	return dst;
}
void FaceDetector::skinDetection(Mat img, Mat& binImg)
{
	double y, cg, cr;
	vector<Mat>channels;
	split(img, channels);


	//here i will segment the picture to skin regions and non skin regions
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			//1-transform to Cg-Cr plane
			YCgCr(img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2], y, cg, cr);

			//2-check if it is a skkin rigion or not
			if ((y > 80) && (cr < 180) && (cr>135))
			{
				//skin region
				binImg.at<Vec3b>(i, j)[0] = 255;
				binImg.at<Vec3b>(i, j)[1] = 255;
				binImg.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				//non skin region
				binImg.at<Vec3b>(i, j)[0] = 0;
				binImg.at<Vec3b>(i, j)[1] = 0;
				binImg.at<Vec3b>(i, j)[2] = 0;
			}

		}

	}



}


void FaceDetector::check_r(Mat img, Mat& binImg)
{
	Mat I = img;
	I.convertTo(I, CV_32FC3);
	vector<Mat> channels1;
	split(I, channels1);
	Mat I0 = channels1[0];
	Mat I1 = channels1[1];
	Mat I2 = channels1[2];

	cv::Mat fullImageHSV = cv::Mat::zeros(I.size(), CV_32FC3);
	cvtColor(I, fullImageHSV, CV_BGR2HSV);
	vector<Mat> channels;
	split(fullImageHSV, channels);
	Mat hue = channels[0];
	hue = hue / 360;
	Mat cr = cv::Mat::zeros(hue.size(), CV_32FC1), cb = cv::Mat::zeros(hue.size(), CV_32FC1);
	int rows = hue.rows;
	int cols = hue.cols;
	float val1, val2;

	cb = 0.148* I2 - 0.291*  I1 + 0.439 * I0 + 128;
	cr = 0.439 *I2 - 0.368 * I1 - 0.071 * I0 + 128;



	int cnt = 0;
	for (int i = 0; i<rows; i++)
		for (int j = 0; j < cols; j++)
		{
			if (cr.at<float>(i, j) >= (float)140 && cr.at<float>(i, j) <= (float)165 && cb.at<float>(i, j) >= (float)140 && cb.at<float>(i, j) <= (float)195 && hue.at<float>(i, j) >= (float) 0.01&& hue.at<float>(i, j) <= (float) 0.1)
			{
				binImg.at<Vec3b>(i, j)[0] = 255;
				binImg.at<Vec3b>(i, j)[1] = 255;
				binImg.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				binImg.at<Vec3b>(i, j)[0] = 0;
				binImg.at<Vec3b>(i, j)[1] = 0;
				binImg.at<Vec3b>(i, j)[2] = 0;
			}

		}

}
vector<Rect> FaceDetector::faceDetection(Mat img, Mat & binImg)
{
	Mat temp, temp2, inv, face, x,origi;
	image = img;
	origi = img;
	//1- lightening
	binImg = lightening(img);
	img.copyTo(temp);
	//get the lightening img to the rignal one to deal with it 
	img = binImg;

	//img.copyTo(binImg);
	//2-skin detection
	///skinDetection(img, binImg);
	check_r(img, binImg);
	//3-morphological operation
	Dilation(binImg, 0, 2);
	/*	Erosion(binImg, 0, 2);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	binImg.copyTo(temp2);
	cv::cvtColor(temp2, temp2, CV_RGB2GRAY);
	findContours(temp2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	//imshow("frame", temp2);
	//floodfill
	binImg.copyTo(temp2);
	cv::cvtColor(temp2, temp2, CV_RGB2GRAY);
	floodFill(temp2, Point(0, 0), Scalar(255));*/

	//inv binimg
	//bitwise_not(binImg, inv);
	//x=bin&img or(inv) // this will give me the exact face
	//bitwise_and(binImg, temp, x);
	//bitwise_or(x, inv, face);

	//imshow("frame", face);
	//4-add rectangles
	
	binImg.copyTo(src);
	
	//Mat r = src;
	
	//contour(binImg);
	vector<Rect> faces = addFrame(0, 0);
	
	savePoints(faces);
	
	
	//vector<Rect> faces = addFrame(0, 0);
//	Mat vvvv(origi.rows, origi.cols, CV_8UC3, Scalar(255, 255, 255));

	//image_forfeature = img;
	return faces;
}
void FaceDetector::savePoints(vector<Rect>faces)
{
	Mat vvvv(image.rows, image.cols, CV_8UC3, Scalar(255, 255, 255));
	int cnt = 0;
	faces_rect.resize(5);
	faces_ellip.resize(5);
	faces_x_y.resize(5);
	faces_max_x_y.resize(5);
	faces_w_h.resize(5);

	cnt_faces = 0;
	//Mat vvvv;
	for (int i = 0; i < faces.size(); i++)
	{
		minxf = 100000000;
		minyf = 100000000;
		if (faces[i].area()>10000)
		{
			faces_w_h[cnt].push_back({ (faces[i].width) , (faces[i].height) });
			faces_x_y[cnt].push_back({ faces[i].x ,faces[i].y });
			image(Rect(faces[i].x, faces[i].y, (faces[i].width), (faces[i].height))).copyTo(vvvv(Rect(faces[i].x, faces[i].y, (faces[i].width), (faces[i].height))));
			cnt++;
		}
	}
	image_forfeature = vvvv;
	cnt_faces = cnt;
}
vector<Rect>  FaceDetector::addFrame(int, void*)
{
	cv::cvtColor(src, src, CV_RGB2GRAY);
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	threshold(src, threshold_output, thresh, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));

	}
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	///this section put green rectangle
	for (size_t i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(0, 0, 255);
		if (boundRect[i].area()>10000)
			rectangle(test, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);

	}


	return boundRect;
}
int FaceDetector::rotate(cv::Mat &src)
{
	int x1, x2, y1, y2;
	float angle, slope;
	x1 = faces_x_y[0][0].first;
	x2 = abs(faces_x_y[0][0].first+faces_w_h[0][0].first);
	y1 = faces_x_y[0][0].second;
	y2 = abs(faces_w_h[0][0].second+faces_x_y[0][0].second);

	slope = float(y1 - y2) / float(x1 - x2);
	angle = atan(slope) * 180 / 3.14;

	if (slope <0)
	{
		// get rotation matrix for rotating the image around its center
		cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
		cv::Mat rot = cv::getRotationMatrix2D(center, 360 - angle, 1.0);
		// determine bounding rectangle
		cv::Rect bbox = cv::RotatedRect(center, src.size(), 360 - angle).boundingRect();
		// adjust transformation matrix
		rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
		rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

		cv::Mat dst;
		cv::warpAffine(src, dst, rot, bbox.size());
		dst.copyTo(src);
		return 1;

	}
	else if (slope>0)
	{
		// get rotation matrix for rotating the image around its center
		cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
		cv::Mat rot = cv::getRotationMatrix2D(center, -1 * angle, 1.0);
		// determine bounding rectangle
		cv::Rect bbox = cv::RotatedRect(center, src.size(), -1 * angle).boundingRect();
		// adjust transformation matrix
		rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
		rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

		cv::Mat dst;
		cv::warpAffine(src, dst, rot, bbox.size());
		dst.copyTo(src);
		return 2;
	}
	return 0;
}
