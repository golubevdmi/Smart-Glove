#include "calibration.h"


using namespace calib;


//
// Search corners in video frames
void SingleCalibration::CalibrationByVideo(std::string videoPath)
{
	std::vector<std::vector<cv::Point2f>> imagePoints;

	cv::Mat frame;
	cv::VideoCapture cap(videoPath);
	CV_Assert(!cap.isOpened());

	cap >> frame;
	imageSize = frame.size();


	std::string single_winname  = "Single Calibration";
	std::string corners_winname = "Corners";

	std::cout << "Frame size: " << imageSize << std::endl;
	std::cout << "Press SPACE to find corners\n" << std::endl;

	std::int8_t key = NULL;
	std::int8_t pause = 1;
	std::uint32_t counter = 0;
	while (cap.read(frame))
	{

		// Search for corner points by pressing the key
		if (key == ' ')
		{
			std::vector<cv::Point2f> corners;

			// Saving corner points if found
			bool isFound = getCorners(frame, corners);
			if (isFound)
			{
				imagePoints.push_back(corners);

				cv::Mat drawCorn = frame.clone();
				cv::drawChessboardCorners(drawCorn, m_patternSize, corners, isFound);

				cv::namedWindow(corners_winname, cv::WINDOW_FREERATIO);
				cv::imshow(corners_winname, drawCorn);

				std::cout << ">> Corners Save: " << counter << std::endl;
				counter++;
			}
			else
			{
				std::cout << "Corners is not Found" << std::endl;
			}
		}

		key = cv::waitKey(pause);

		cv::namedWindow(single_winname, cv::WINDOW_FREERATIO);
		cv::imshow(single_winname, frame);
	}

	cv::destroyWindow(single_winname);
	cv::destroyWindow(corners_winname);

	// Calibration
	std::cout << "Number of images: " << counter << std::endl;

	if (counter > 1)
		calibrate(imagePoints);

	imagePoints.clear();
}

//
// Search corners in an array of images
void SingleCalibration::CalibrationByImagesVector(const std::vector<cv::Mat>& images)
{
	std::vector<std::vector<cv::Point2f>> imagePoints;

	cv::Mat image = images[0];
	imageSize = image.size();

	std::string single_winname = "Single Calibration";
	std::string corners_winname = "Corners";
	\
	std::cout << "Frame size: " << imageSize << std::endl;

	std::uint32_t counter = 0;
	for (int i = 0; i < images.size(); i++)
	{
		image = images[i];

		// Saving corner points if found
		std::vector<cv::Point2f> corners;
		bool isFound = getCorners(image, corners);
		
		if (isFound)
		{
			imagePoints.push_back(corners);

			cv::Mat drawCorn = image.clone();
			cv::drawChessboardCorners(drawCorn, m_patternSize, corners, isFound);

			cv::namedWindow(corners_winname, cv::WINDOW_FREERATIO);
			cv::imshow(corners_winname, drawCorn);

			std::cout << ">>> Corners Save: " << counter << std::endl;
			counter++;
		}
		else
		{
			std::cout << "Corners not isFound" << std::endl;
		}

		// Text on image
		std::string putCount = "Corners isFound: " + std::to_string(counter);
		cv::putText(image, putCount, cv::Size(image.cols * 0.01, image.rows * 0.8), 2, 2, cv::Scalar(255, 255, 255), 7, 8, false);

		cv::namedWindow(single_winname, cv::WINDOW_FREERATIO);
		cv::imshow(single_winname, image);

		cv::waitKey(100);
	}

	cv::destroyWindow(single_winname);
	cv::destroyWindow(corners_winname);

	// Calibration
	std::cout << "Number of images: " << counter << std::endl;
	if (counter > 0)
		calibrate(imagePoints);

	imagePoints.clear();
}

//
// Calculation of parameters for the data obtained
void SingleCalibration::calibrate(std::vector<std::vector<cv::Point2f>>& imagePoints)
{
	//	Creating a vector containing material
    // coordinates of points from all images
	std::vector<std::vector<cv::Point3f>> objectPoints;
	objectPoints = std::vector<std::vector<cv::Point3f>>(1);
	createKnownPosition(objectPoints[0]);
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	// The function returns the average re-projection error.
	// This number gives a good estimate of the accuracy of the parameters found.
	// It should be as close to zero as possible.
	std::cout << std::endl << "Running calibration... ";
	std::double_t rms = calibrateCamera(objectPoints, imagePoints, imageSize, camera_matrix,
		distortion_coeffs, rvecs, tvecs, cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST);

	std::cout << "Complete" << std::endl;
	std::cout << "Done with RMS error = " << rms << std::endl;

	return;
}

//
// --TODO
void SingleCalibration::writeParams()
{

	return;
}