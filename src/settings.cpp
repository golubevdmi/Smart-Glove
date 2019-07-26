#include "calibration.h"



bool Settings::calibByVideo(std::string videoPath)
{
	CV_Assert(!videoPath.empty());

	grabVideo(videoPath);

	return true;
}
bool Settings::calibBySavedImages(const std::vector<cv::Mat> &images)
{
	if (images.size() < 1)
	{
		std::cout << "Need more images. savedImage.size(): " << images.size() << std::endl;
		return false;
	}

	grabImages(images);

	return true;
}

// 
// Получение координат точек в мировой системе координат
void Settings::createKnownPosition(std::vector<cv::Point3f>& corners)
{
	corners.clear();

	switch (m_pattern)
	{
	case Pattern::CHESSBOARD:
	case Pattern::CIRCLES_GRID:
		for (int i = 0; i < m_patternSize.height; ++i)
			for (int j = 0; j < m_patternSize.width; ++j)
				corners.push_back(cv::Point3f(float(j*m_squareSize), float(i*m_squareSize), 0));
		break;
	case Pattern::ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < m_patternSize.height; i++)
			for (int j = 0; j < m_patternSize.width; j++)
				corners.push_back(cv::Point3f(float((2 * j + i % 2)*m_squareSize), float(i*m_squareSize), 0));
		break;
	default:
		CV_Error(cv::Error::StsError, "Error template");
		break;
	}
}

// 
// Получение координат углов на проекционной плоскости
bool Settings::getCorners(const cv::Mat& imageArray, std::vector<cv::Point2f>& corners)
{
	cv::Mat gray;
	bool isFound = false;

	switch (m_pattern)
	{
	case Pattern::CHESSBOARD:
		// Переводим изображение в оттенки серого(необходимо для субпиксельного уточнения)
		if (imageArray.channels() == 3)			cvtColor(imageArray, gray, cv::COLOR_BGR2GRAY);
		else if (imageArray.channels() == 1)	imageArray.copyTo(gray);

		isFound = findChessboardCorners(gray, m_patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

		if (isFound)
		{
			// Субпиксельное уточнение углов
			cornerSubPix(gray, corners, m_patternSize, cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
				30,		// max number of iterations 
				0.1));     // min accuracy);
		}
		break;
	case Pattern::CIRCLES_GRID:
		isFound = findCirclesGrid(imageArray, m_patternSize, corners);
		break;
	case Pattern::ASYMMETRIC_CIRCLES_GRID:
		isFound = findCirclesGrid(imageArray, m_patternSize, corners, cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING);
		break;
	default:
		CV_Error(cv::Error::StsError, "Error template");
		break;
	}

	if (!isFound)	
		return false;

	return true;
}