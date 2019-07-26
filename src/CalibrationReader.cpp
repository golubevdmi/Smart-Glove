#include "calibration.h"

using namespace calib;

//
// Show intrinsic and extrinsic params
void StereoCalibrationReader::show()
{
	std::cout << "ImageSize: "                  << imageSize <<
		       "\nCamera matrix 1:\n"           << camera_matrix1 <<
		       "\nCamera matrix 2:\n"           << camera_matrix2 <<
		       "\nDistortion coefficients 1:\n" << distortion_coeffs1 <<
		       "\nDistortion coefficients 2:\n" << distortion_coeffs2 <<
		       "\nRotate:\n"                    << Rotate <<
		       "\nT: "                          << T <<
		       "\nEssential:\n"                 << E <<
		       "\nFundamental:\n"               << F << std::endl;

	std::cout << std::endl << std::endl;

	std::cout << "R1:\n" << R1 <<
		       "\nR2:\n" << R2 <<
		       "\nP1:\n" << P1 <<
			   "\nP2:\n" << P2 <<
			   "\nQ:\n" << Q << std::endl;

	std::cout << std::endl << std::endl;

	std::cout << "Baseline: " << baseline <<
		       "\nFocalLenght: " << focallenght << std::endl;
}

//
// Reading intrinsic and extrinsic params
bool StereoCalibrationReader::read()
{
	std::cout << ">> Download stereo calibrate parameters" << std::endl;

	if (!m_fsParams.isOpened())
	{
		std::cout << "Stereo Params not received" << std::endl;
		return false;
	}

	m_fsParams["UsedFramework"] >> UsedFramework;

	if (UsedFramework == "MATLAB")
		m_isReceived = readAsMatlab();
	else if (UsedFramework == "OPENCV")
		m_isReceived = readAsOpenCV();
	else
		std::cout << "UsedFramework: Framework not defined" << std::endl;

	if (!m_isReceived)
	{
		std::cout << "Stereo params not received" << std::endl;
		return false;
	}


	return true;
}
bool StereoCalibrationReader::readAsMatlab()
{
	// ImageSize
	m_fsParams["ImageSize"] >> imageSize;

	if (imageSize.width < imageSize.height)
	{
		std::uint32_t tmp = imageSize.height;
		imageSize.height = imageSize.width;
		imageSize.width = tmp;
	}

	// Intrinsic params
	// Camera 1
	cv::Vec2d focalLength1(0, 0), princPoint1(0, 0), radialDistortion1(0, 0);
	m_fsParams["FocalLength1"] >> focalLength1;
	m_fsParams["PrincipalPoint1"] >> princPoint1;
	m_fsParams["RadialDistortion1"] >> radialDistortion1;

	camera_matrix1.at<std::double_t>(0, 0) = focalLength1[0];
	camera_matrix1.at<std::double_t>(1, 1) = focalLength1[1];
	camera_matrix1.at<std::double_t>(0, 2) = princPoint1[0];
	camera_matrix1.at<std::double_t>(1, 2) = princPoint1[1];
	camera_matrix1.at<std::double_t>(2, 2) = 1.0;

	distortion_coeffs1.at<std::double_t>(0) = radialDistortion1[0];
	distortion_coeffs1.at<std::double_t>(1) = radialDistortion1[1];

	// Camera 2
	cv::Vec2d focalLength2(0, 0), princPoint2(0, 0), radialDistortion2(0, 0);
	m_fsParams["FocalLength2"] >> focalLength2;
	m_fsParams["PrincipalPoint2"] >> princPoint2;
	m_fsParams["RadialDistortion2"] >> radialDistortion2;

	camera_matrix2.at<std::double_t>(0, 0) = focalLength2[0];
	camera_matrix2.at<std::double_t>(1, 1) = focalLength2[1];
	camera_matrix2.at<std::double_t>(0, 2) = princPoint2[0];
	camera_matrix2.at<std::double_t>(1, 2) = princPoint2[1];
	camera_matrix2.at<std::double_t>(2, 2) = 1.0;

	distortion_coeffs2.at<std::double_t>(0) = radialDistortion2[0];
	distortion_coeffs2.at<std::double_t>(1) = radialDistortion2[1];


	// Extrinsic params
	m_fsParams["RotationOfCamera2"] >> Rotate;
	Rotate = Rotate.t();

	m_fsParams["TranslationOfCamera2"] >> T;


	if (camera_matrix1.at<std::double_t>(0, 0) == 0.0 || camera_matrix1.at<std::double_t>(1, 1) == 0.0 ||
		camera_matrix2.at<std::double_t>(0, 2) == 0.0 || camera_matrix2.at<std::double_t>(1, 2) == 0.0)
		return false;

	if (Rotate.empty())	
		return false;

	if (T[0] == 0.0)		
		return false;

	if (imageSize.width == 0 || imageSize.height == 0)	
		return false;

	return true;
}
bool StereoCalibrationReader::readAsOpenCV()
{
	m_fsParams["imageSize"] >> imageSize;

	// Intrinsic params
	m_fsParams["M1"] >> camera_matrix1;
	m_fsParams["D1"] >> distortion_coeffs1;

	m_fsParams["M2"] >> camera_matrix2;
	m_fsParams["D2"] >> distortion_coeffs2;

	// Extrinsic params
	m_fsParams["Rotate"] >> Rotate;
	m_fsParams["T"] >> T;

	m_fsParams["E"] >> E;
	m_fsParams["F"] >> F;


	if (imageSize.width == 0 || imageSize.height == 0)	
		return false;

	if (camera_matrix1.at<std::double_t>(0, 0) == 0 || 
		camera_matrix1.at<std::double_t>(1, 1) == 0 ||
		camera_matrix2.at<std::double_t>(0, 2) == 0 ||
		camera_matrix2.at<std::double_t>(1, 2) == 0)
		return false;

	if (Rotate.empty())	return false;
	if (T[0] == 0.0)		return false;

	return true;
}


bool StereoCalibrationReader::computeParams()
{
	std::cout << ">> Compute All params" << std::endl;

	return computeRectifyParams() &&
		   computeUndistortMap()  &&
		   computeBaseline()      &&
		   computeFocalLenght();
}
//
// Compute R1, R2, P1, P2, Q
bool StereoCalibrationReader::computeRectifyParams()
{
	if (!m_isReceived && !read())
	{
		std::cout << "Rectify: Params not loaded" << std::endl;
		return false;
	}

	std::cout << ">> Compute Rectify" << std::endl;


	cv::stereoRectify(camera_matrix1, distortion_coeffs1, camera_matrix2, distortion_coeffs2,
		imageSize, Rotate, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1);

	return true;
}

//
// Compute maps (for remapping)
bool StereoCalibrationReader::computeUndistortMap()
{
	if (!m_isReceived && !read())
	{
		std::cout << "Undistort map: Params not loaded" << std::endl;
		return false;
	}

	if (R1.empty() || R2.empty() ||
		P1.empty() || P2.empty())
	{
		std::cout << "Undistort map: R1, R2, P1, P2 empty" << std::endl;
		std::cout << ">> Compute rectify" << std::endl;
		if (!computeRectifyParams())
		{
			std::cout << ">> Undistort map: Params not loaded" << std::endl;
			return false;
		}
	}

	std::cout << ">> Compute Undistort map" << std::endl;

	initUndistortRectifyMap(camera_matrix1, distortion_coeffs1, R1, P1, imageSize, CV_32FC1, map1[0], map1[1]);
	initUndistortRectifyMap(camera_matrix2, distortion_coeffs2, R2, P2, imageSize, CV_32FC1, map2[0], map2[1]);

	return true;
}

//
// Main distance params
bool StereoCalibrationReader::computeBaseline()
{
	if (!m_isReceived && !read())
	{
		std::cout << "Baseline: Params not loaded" << std::endl;
		return false;
	}

	std::cout << ">> Compute baseline" << std::endl;

	baseline = T[0];

	if (baseline < 0)
		baseline *= (-1);

	if (UsedFramework == "MATLAB")
		baseline /= 1000;

	return true;
}
bool StereoCalibrationReader::computeFocalLenght()
{
	if (!m_isReceived && !read())
	{
		std::cout << "FocalLenght: Params not loaded" << std::endl;
		return false;
	}

	if (Q.empty())
	{
		std::cout << "FocalLenght: Q is empty" << std::endl;
		std::cout << ">> Compute Rectify" << std::endl;
		if (!computeRectifyParams())
		{
			std::cout << "FocalLenght: Params not received" << std::endl;
			return false;
		}
	}

	focallenght = Q.at<std::double_t>(2, 3);

	return true;
}