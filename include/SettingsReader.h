#pragma once

#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


class SettingsReader
{
public:
	SettingsReader(std::string filename);
	~SettingsReader();

	bool getDistanceParams(double& baselineMetres, double& focalLenght);
	bool getUndistMap(cv::Mat& map1X, cv::Mat& map1Y, cv::Mat& map2X, cv::Mat& map2Y);
	bool getSettingsToFuncUndistortPoints(cv::Mat& M1, cv::Mat& M2, cv::Mat& D1, cv::Mat& D2, cv::Mat& P1, cv::Mat& P2, cv::Mat& R1, cv::Mat& R2);

	// Checks if the settings have been received.
	bool isOpened() { return settingsReceived; }

	cv::Size getImageSize() { return imageSize; }
	cv::Mat getCameraMatrix1() { return m_cameraMatrix1.clone(); }
	cv::Mat getCameraMatrix2() { return m_cameraMatrix2.clone(); }
	cv::Mat getDistCoeffs1() { return m_distCoeffs1.clone(); }
	cv::Mat getDistCoeffs2() { return m_distCoeffs2.clone(); }
	cv::Mat getRotationMatrix() { return Rotate.clone(); }
	cv::Vec3d getTranslVec() { return T; }

	cv::Mat getEssentialMat()   { return E.clone(); }
	cv::Mat getFundamentalMat() { return F.clone(); }

	void showParams();

private:
	bool settingsReceived;

	std::string settingsProgram;

	cv::Size imageSize;

	// Intrinsic params
	cv::Mat m_cameraMatrix1, m_cameraMatrix2;
	cv::Mat m_distCoeffs1, m_distCoeffs2;

	// Extrinsic params
	cv::Mat Rotate;
	cv::Vec3d T;

	cv::Mat E, F;

	// Load main settings
	bool getStereoCalibParams(std::string filename);
	bool readSettingsMatlab(cv::FileStorage node);
	bool readSettingsOpencv(cv::FileStorage node);
	bool readSettings(cv::FileStorage node);

};


double calculateDistance(double baselineMetres, double focalLenght, double disparity);