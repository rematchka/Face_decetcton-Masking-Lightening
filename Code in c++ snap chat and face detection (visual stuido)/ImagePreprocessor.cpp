#include "ImagePreprocessor.h"



ImagePreprocessor::ImagePreprocessor(Mat _image)
{
	image = _image;
}


ImagePreprocessor::~ImagePreprocessor()
{
}

void ImagePreprocessor::EqualiseHistogram(int flag)
{
	vector<Mat>channels;
	split(image, channels);

	switch (flag)
	{
	case imfirst:
	{
		equalizeHist(channels[0], image);
		break;
	}
	case imsecond:
	{
		equalizeHist(channels[1], image);
		break;
	}
	case imthird:
	{
		equalizeHist(channels[2], image);
		break;
	}
	case imfirst_and_second:
	{
		equalizeHist(channels[0], image);
		equalizeHist(channels[1], image);
		break;
	}
	case imsecond_and_third:
	{
		equalizeHist(channels[1], image);
		equalizeHist(channels[2], image);
		break;
	}
	case imfirst_and_third:
	{
		equalizeHist(channels[0], image);
		
		equalizeHist(channels[2], image);
		break;
	}
	case imfirst_and_second_and_third:
	{
		equalizeHist(channels[0], image);
		equalizeHist(channels[1], image);
		equalizeHist(channels[2], image);
		break;
	}
	default:
		break;
	}

	merge(channels, image);
}

Mat ImagePreprocessor::getOutput()
{
	return image;
}
void ImagePreprocessor::greyWord()
{
	Mat I = image;

	
	if (I.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}


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
	
	outImage = fff;



}

void ImagePreprocessor::white_patch()
{
  Mat	I = image;
	if (I.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}

	
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
	outImage = fff;
	
}
void ImagePreprocessor::modified_white_patch(int varargin)
{
	int th = varargin;
	cv::Mat I = image;
	if (I.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}



	I.convertTo(I, CV_32FC3);
	vector<Mat> channels1;

	split(I, channels1);
	Mat I0 = channels1[0];
	Mat I1 = channels1[1];
	Mat I2 = channels1[2];
	Mat R = I2;
	R = I2 > th;
	cv::Scalar	Kr = mean(R);
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
	outImage = fff;

}
