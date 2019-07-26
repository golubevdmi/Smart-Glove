#include "SettingsReader.h"

SettingsReader::SettingsReader(std::string filename)
{
	settingsReceived = false;

	settingsProgram = "";

	imageSize = cv::Size(0, 0);

	m_cameraMatrix1 = cv::Mat::zeros(3, 3, CV_64FC1);
	m_cameraMatrix2 = cv::Mat::zeros(3, 3, CV_64FC1);

	m_distCoeffs1 = cv::Mat::zeros(1, 5, CV_64FC1);
	m_distCoeffs2 = cv::Mat::zeros(1, 5, CV_64FC1);

	Rotate = cv::Mat::zeros(3, 3, CV_64FC1);
	T.zeros();

	settingsReceived = getStereoCalibParams(filename);
}
SettingsReader::~SettingsReader()
{
	settingsProgram = "";
	imageSize = cv::Size(0, 0);
	m_cameraMatrix1.release();
	m_cameraMatrix2.release();
	m_distCoeffs1.release();
	m_distCoeffs2.release();
	Rotate.release();
	T.zeros();
}

bool SettingsReader::getDistanceParams(double& baselineMetres, double& focalLenght)
{
	std::cout << "--> Get distance parameters" << std::endl;
	if (!settingsReceived)
	{
		std::cout << "Settings not loaded" << std::endl;
		return false;
	}

	cv::Mat R1, R2, P1, P2, Q;
	stereoRectify(m_cameraMatrix1, m_distCoeffs1, m_cameraMatrix2, m_distCoeffs2, imageSize, Rotate, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1);

	baselineMetres = T[0];

	if (baselineMetres < 0)				baselineMetres *= (-1);
	if (settingsProgram == "MATLAB")	baselineMetres /= 1000;

	focalLenght = Q.at<double>(2, 3);

	R1.release();
	R2.release();
	P1.release();
	P2.release();
	Q.release();

	return true;
}
bool SettingsReader::getUndistMap(cv::Mat& map1X, cv::Mat& map1Y, cv::Mat& map2X, cv::Mat& map2Y)
{
	std::cout << "--> Get undistorted maps" << std::endl;
	if (!settingsReceived)
	{
		std::cout << "Settings not loaded" << std::endl;
		return false;
	}
	
	cv::Mat R1, R2;
	cv::Mat P1, P2;
	cv::Mat Q;

	stereoRectify(m_cameraMatrix1, m_distCoeffs1, m_cameraMatrix2, m_distCoeffs2, imageSize, Rotate, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1);

	initUndistortRectifyMap(m_cameraMatrix1, m_distCoeffs1, R1, P1, imageSize, CV_32FC1, map1X, map1Y);
	initUndistortRectifyMap(m_cameraMatrix2, m_distCoeffs2, R2, P2, imageSize, CV_32FC1, map2X, map2Y);

	return true;
}
bool SettingsReader::getSettingsToFuncUndistortPoints(cv::Mat& M1, cv::Mat& M2, cv::Mat& D1, cv::Mat& D2, cv::Mat& P1, cv::Mat& P2, cv::Mat& R1, cv::Mat& R2)
{
	std::cout << "--> Get New cameraMatrix and Rotate Camera1 and Camera2" << std::endl;
	if (!settingsReceived)
	{
		std::cout << "Settings not loaded" << std::endl;
		return false;
	}

	M1 = m_cameraMatrix1.clone();
	M2 = m_cameraMatrix2.clone();
	D1 = m_distCoeffs1.clone();
	D2 = m_distCoeffs2.clone();

	cv::Mat Q;
	cv::stereoRectify(M1, D1, M2, D2, imageSize, Rotate, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1);

	if (M1.empty() || M2.empty() || D1.empty() || D2.empty()
		|| R1.empty() || R2.empty() || P1.empty() || P2.empty())
	{
		std::cout << "Err params" << std::endl;
		return false;
	}

	return true;
}


bool SettingsReader::getStereoCalibParams(std::string filename)
{
	std::cout << "--> Download stereo calibrate parameters from " << filename << std::endl;

	try
	{

		cv::FileStorage node(filename, cv::FileStorage::READ);
		if (!node.isOpened())
		{
			std::cout << "Calibration file not open" << std::endl;
			return false;
		}

		node["SettingsProgram"] >> settingsProgram;

		bool received = false;
		if (settingsProgram == "MATLAB")		received = readSettingsMatlab(node);
		else if (settingsProgram == "OPENCV")	received = readSettingsOpencv(node);
		else									received = readSettings(node);

		node.release();

		if (received)		return true;
	}
	catch (...)
	{
		std::cout << "Error: Settings not loaded" << std::endl;
		return false;
	}

	std::cout << "Settings not loaded" << std::endl;
	return false;
}
bool SettingsReader::readSettingsMatlab(cv::FileStorage node)
{
	cv::Vec2d focalLength(0, 0), princPoint(0, 0), radialDistortion(0, 0);

	node["ImageSize"] >> imageSize;

	if (imageSize.width == 0 || imageSize.height == 0)	return false;
	if (imageSize.width < imageSize.height)
	{
		int tmp = imageSize.height;
		imageSize.height = imageSize.width;
		imageSize.width = tmp;
	}

	// Intrinsic params
	// Camera 1
	node["FocalLength1"] >> focalLength;
	node["PrincipalPoint1"] >> princPoint;
	node["RadialDistortion1"] >> radialDistortion;

	m_cameraMatrix1.at<double>(0, 0) = focalLength[0];
	m_cameraMatrix1.at<double>(1, 1) = focalLength[1];
	m_cameraMatrix1.at<double>(0, 2) = princPoint[0];
	m_cameraMatrix1.at<double>(1, 2) = princPoint[1];

	m_distCoeffs1.at<double>(0) = radialDistortion[0];
	m_distCoeffs1.at<double>(1) = radialDistortion[1];

	// Camera 2
	node["FocalLength2"] >> focalLength;
	node["PrincipalPoint2"] >> princPoint;
	node["RadialDistortion2"] >> radialDistortion;

	m_cameraMatrix2.at<double>(0, 0) = focalLength[0];
	m_cameraMatrix2.at<double>(1, 1) = focalLength[1];
	m_cameraMatrix2.at<double>(0, 2) = princPoint[0];
	m_cameraMatrix2.at<double>(1, 2) = princPoint[1];

	m_distCoeffs2.at<double>(0) = radialDistortion[0];
	m_distCoeffs2.at<double>(1) = radialDistortion[1];

	if (m_cameraMatrix1.at<double>(0, 0) == 0 || m_cameraMatrix1.at<double>(1, 1) == 0 ||
		m_cameraMatrix2.at<double>(0, 2) == 0 || m_cameraMatrix2.at<double>(1, 2) == 0)
		return false;

	// Extrinsic params
	node["RotationOfCamera2"] >> Rotate;
	node["TranslationOfCamera2"] >> T;

	if (Rotate.empty())	return false;
	Rotate = Rotate.t();

	if (T[0] == 0)		return false;

	return true;
}
bool SettingsReader::readSettingsOpencv(cv::FileStorage node)
{
	node["imageSize"] >> imageSize;
	if (imageSize.width == 0 || imageSize.height == 0)	return false;

	// Intrinsic params
	node["M1"] >> m_cameraMatrix1;
	node["D1"] >> m_distCoeffs1;

	node["M2"] >> m_cameraMatrix2;
	node["D2"] >> m_distCoeffs2;

	if (m_cameraMatrix1.at<double>(0, 0) == 0 || m_cameraMatrix1.at<double>(1, 1) == 0 ||
		m_cameraMatrix2.at<double>(0, 2) == 0 || m_cameraMatrix2.at<double>(1, 2) == 0)
		return false;

	// Extrinsic params
	node["Rotate"] >> Rotate;
	node["T"] >> T;

	node["E"] >> E;
	node["F"] >> F;

	if (Rotate.empty())	return false;
	if (T[0] == 0)		return false;

	return true;
}
bool SettingsReader::readSettings(cv::FileStorage node)
{
	std::cout << "WARNING: SettingsReader - General" << std::endl;
	cv::Vec2d focalLength(0, 0), princPoint(0, 0), radialDistortion(0, 0);

	node["ImageSize"] >> imageSize;

	if (imageSize.width == 0 || imageSize.height == 0)	return false;
	if (imageSize.width < imageSize.height)
	{
		int tmp = imageSize.height;
		imageSize.height = imageSize.width;
		imageSize.width = tmp;
	}

	// Intrinsic params
	// Camera 1
	node["FocalLength1"] >> focalLength;
	node["PrincipalPoint1"] >> princPoint;
	node["RadialDistortion1"] >> radialDistortion;

	m_cameraMatrix1.at<double>(0, 0) = focalLength[0];
	m_cameraMatrix1.at<double>(1, 1) = focalLength[1];
	m_cameraMatrix1.at<double>(0, 2) = princPoint[0];
	m_cameraMatrix1.at<double>(1, 2) = princPoint[1];

	m_distCoeffs1.at<double>(0) = radialDistortion[0];
	m_distCoeffs1.at<double>(1) = radialDistortion[1];

	// Camera 2
	node["FocalLength2"] >> focalLength;
	node["PrincipalPoint2"] >> princPoint;
	node["RadialDistortion2"] >> radialDistortion;

	m_cameraMatrix2.at<double>(0, 0) = focalLength[0];
	m_cameraMatrix2.at<double>(1, 1) = focalLength[1];
	m_cameraMatrix2.at<double>(0, 2) = princPoint[0];
	m_cameraMatrix2.at<double>(1, 2) = princPoint[1];

	m_distCoeffs2.at<double>(0) = radialDistortion[0];
	m_distCoeffs2.at<double>(1) = radialDistortion[1];

	if (m_cameraMatrix1.at<double>(0, 0) == 0 || m_cameraMatrix1.at<double>(1, 1) == 0 ||
		m_cameraMatrix2.at<double>(0, 2) == 0 || m_cameraMatrix2.at<double>(1, 2) == 0)
		return false;

	// Extrinsic params
	node["RotationOfCamera2"] >> Rotate;
	node["TranslationOfCamera2"] >> T;

	if (Rotate.empty())	return false;
	//Rotate = Rotate.t();

	if (T[0] == 0)		return false;

	return true;
}

void SettingsReader::showParams()
{
	std::cout << "m_cameraMatrix1: " << std::endl << m_cameraMatrix1 << std::endl << std::endl
		<< "m_cameraMatrix2: " << std::endl << m_cameraMatrix2 << std::endl << std::endl
		<< "m_distCoeffs1: " << m_distCoeffs1 << std::endl
		<< "m_distCoeffs2: " << m_distCoeffs2 << std::endl
		<< "Rotate: " << std::endl << Rotate << std::endl << std::endl
		<< "T: " << T << std::endl << std::endl;
}



double calculateDistance(double baselineMetres, double focalLenght, double disparity)
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