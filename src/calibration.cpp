#include "calibration.h"

using namespace calib;

// 
// Getting the coordinates of points in the world coordinate system
void Calibration::createKnownPosition(std::vector<cv::Point3f>& corners)
{
	corners.clear();

	switch (m_pattern)
	{
	case Pattern::CHESSBOARD:
	case Pattern::CIRCLES_GRID:
		for (int i = 0; i < m_patternSize.height; ++i)
			for (int j = 0; j < m_patternSize.width; ++j)
				corners.push_back(cv::Point3f(std::float_t(j*m_squareSize), std::float_t(i*m_squareSize), 0));
		break;
	case Pattern::ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < m_patternSize.height; i++)
			for (int j = 0; j < m_patternSize.width; j++)
				corners.push_back(cv::Point3f(std::float_t((2 * j + i % 2)*m_squareSize), std::float_t(i*m_squareSize), 0));
		break;
	default:
		CV_Error(cv::Error::StsError, "Error template");
		break;
	}
}

// 
// Getting the coordinates of corners on the projection plane
bool Calibration::getCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners)
{
	cv::Mat gray;
	bool isFound = false;

	switch (m_pattern)
	{
	case Pattern::CHESSBOARD:
		// Translate the image into grayscale (necessary for sub-pixel refinement)
		if (image.channels() == 3)			
			cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		else if (image.channels() == 1)	
			image.copyTo(gray);

		isFound = findChessboardCorners(gray, m_patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

		if (isFound)
		{
			// Subpixel corner refinement
			cornerSubPix(gray, corners, m_patternSize, cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
				30,		// max number of iterations 
				0.1));     // min accuracy);
		}
		break;
	case Pattern::CIRCLES_GRID:
		isFound = findCirclesGrid(image, m_patternSize, corners);
		break;
	case Pattern::ASYMMETRIC_CIRCLES_GRID:
		isFound = findCirclesGrid(image, m_patternSize, corners, cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING);
		break;
	default:
		CV_Error(cv::Error::StsError, "Error template");
		break;
	}

	if (!isFound)	
		return false;

	return true;
}



std::double_t calib::calculateDistance(std::double_t baselineMetres, std::double_t focalLenght, std::double_t disparity)
{
	if (disparity == 0.000f)
	{
		std::cout << "Disparity = 0" << std::endl;
		return 0;
	}

	// Print parameters
	/*std::cout << "F:" << focalLenght << std::endl
		<< "Baseline: " << baselineMetres << std::endl
		<< "Disparity: " << disparity << std::endl;*/

		//std::cout << "Method 1: Z = (f * d) / (x1 - x2) = "	<< 
		//	(focalLenght * baselineMetres) / (disparity) << std::endl;

	return (focalLenght * baselineMetres) / (disparity);
}