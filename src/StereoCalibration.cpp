#include "calibration.h"


using namespace calib;


//
// Search corners in video frames
void StereoCalibration::CalibrationByVideo(std::string videoPath)
{
	std::vector<std::vector<cv::Point2f>> imagePoints1, imagePoints2;

	cv::Mat frame;
	cv::VideoCapture cap(videoPath);

	cap >> frame;
	imageSize = cv::Size(frame.cols / 2, frame.rows);

	std::string stereo_winname  = "Stereo Calibration";
	std::string corners_winname = "Corners";

	std::cout << "Frame size: " << imageSize << std::endl;
	std::cout << "Press SPACE to find corners\n" << std::endl;

	std::int8_t key = NULL;
	std::int8_t pause = 1;
	std::uint32_t counter = 0;
	while (true)
	{
		cap >> frame;
		if (frame.empty())	break;

		cv::Mat left = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows)).clone();
		cv::Mat right = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows)).clone();

		// Search for corner points by pressing a key
		if (key == ' ')
		{
			std::vector<cv::Point2f> corners1, corners2;

			bool found1 = getCorners(left, corners1);
			bool found2 = getCorners(right, corners2);

			// Saving corner points if found
			if (found1 && found2)
			{
				imagePoints1.push_back(corners1);
				imagePoints2.push_back(corners2);

				cv::Mat drawStereoCorn = frame.clone();
				cv::Mat drawCorn1 = drawStereoCorn(cv::Rect(0, 0, drawStereoCorn.cols / 2, drawStereoCorn.rows));
				cv::Mat drawCorn2 = drawStereoCorn(cv::Rect(drawStereoCorn.cols / 2, 0, drawStereoCorn.cols / 2, drawStereoCorn.rows));

				cv::drawChessboardCorners(drawCorn1, m_patternSize, corners1, found1);
				cv::drawChessboardCorners(drawCorn2, m_patternSize, corners2, found2);

				cv::namedWindow(corners_winname, cv::WINDOW_FREERATIO);
				cv::imshow(corners_winname, drawStereoCorn);

				std::cout << ">> Corners Save: " << counter << std::endl;
				counter++;
			}
			else
			{
				std::cout << "Corners is not Found" << std::endl;
			}
		}

		cv::namedWindow(stereo_winname, cv::WINDOW_FREERATIO);
		cv::imshow(stereo_winname, frame);
		key = cv::waitKey(1);
		if (key == 27)	break;
	}

	frame.release();
	cap.release();

	cv::destroyWindow(stereo_winname);
	cv::destroyWindow(corners_winname);

	// Calibration
	std::cout << "Number of images: " << counter << std::endl;

	if (counter > 1)
		calibrate(imagePoints1, imagePoints2);

	imagePoints1.clear();
	imagePoints2.clear();
}

//
// Search corners in an array of images
void StereoCalibration::CalibrationByImagesVector(const std::vector<cv::Mat> &stereopairs)
{
	std::vector<std::vector<cv::Point2f>> imagePoints1, imagePoints2;

	cv::Mat image = stereopairs[0];

	imageSize = cv::Size(image.cols / 2, image.rows);

	std::cout << "Frame size: " << imageSize << std::endl;

	std::int8_t key = NULL;
	std::uint32_t counter = 0;

	for (auto stereoImg : stereopairs)
	{
		cv::Mat left = cv::Mat(image, cv::Rect(0, 0, image.cols / 2, image.rows));
		cv::Mat right = cv::Mat(image, cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows));

		std::vector<cv::Point2f> corners1, corners2;

		bool found1 = getCorners(left, corners1);
		bool found2 = getCorners(right, corners2);

		// Saving corner points if found
		if (found1 && found2)
		{
			imagePoints1.push_back(corners1);
			imagePoints2.push_back(corners2);

			std::cout << "--> Corners Save: " << counter << std::endl;

			// Drawing
			cv::Mat drawStereoCorn = stereoImg.clone();

			// TODO -- using merge or convertTo
			cvtColor(drawStereoCorn, drawStereoCorn, cv::COLOR_GRAY2BGR);

			cv::Mat tmp = cv::Mat::zeros(cv::Size(image.rows, image.cols), CV_8UC1);
			cv::Mat tmp1 = cv::Mat::zeros(cv::Size(image.rows, image.cols), CV_8UC1);

			cv::Mat drawCorn1 = cv::Mat(drawStereoCorn, cv::Rect(0, 0, drawStereoCorn.cols / 2, drawStereoCorn.rows));
			cv::Mat drawCorn2 = cv::Mat(drawStereoCorn, cv::Rect(drawStereoCorn.cols / 2, 0, drawStereoCorn.cols / 2, drawStereoCorn.rows));

			cv::drawChessboardCorners(drawCorn1, m_patternSize, corners1, found1);
			cv::drawChessboardCorners(drawCorn2, m_patternSize, corners2, found2);

			// Text on image
			std::string putCount = "Corners isFound: " + std::to_string(counter);
			cv::putText(drawStereoCorn, putCount, cv::Size(drawStereoCorn.cols * 0.01, drawStereoCorn.rows * 0.8), 2, 2, cv::Scalar(255, 255, 255), 7, 8, false);

			cv::namedWindow("CornersImage", cv::WINDOW_FREERATIO);
			cv::imshow("CornersImage", drawStereoCorn);
		}
		else
		{
			std::cout << "Corners not isFound" << std::endl;
		}

		char key = cv::waitKey(1);
	}

	cv::destroyWindow("CornersImage");


	// Calibration
	std::cout << "Number of images: " << counter << std::endl;
	if (counter > 0)
		calibrate(imagePoints1, imagePoints2);

	imagePoints1.clear();
	imagePoints2.clear();
}

//
// Calculation of parameters for the data obtained
bool StereoCalibration::calibrate(const std::vector<std::vector<cv::Point2f>>& imagePoints1, const std::vector<std::vector<cv::Point2f>>& imagePoints2)
{
	// Creating a vector containing material
	// coordinates of points from all images
	std::vector<std::vector<cv::Point3f>> objectPoints;
	objectPoints = std::vector<std::vector<cv::Point3f>>(1);
	createKnownPosition(objectPoints[0]);
	objectPoints.resize(imagePoints1.size(), objectPoints[0]);

	// The function returns the average re-projection error.
	// This number gives a good estimate of the accuracy of the parameters found.
	// It should be as close to zero as possible.
	std::cout << std::endl << "Running stereo calibration... " << std::endl;

	// Calibration camera 1
	cv::Mat rvecs1, tvecs1;
	rms_camera1 = calibrateCamera(objectPoints, imagePoints1, imageSize, camera_matrix1, distortion_coeffs1, rvecs1, tvecs1, cv::CALIB_FIX_K3 + cv::CALIB_FIX_TANGENT_DIST);

	// Calibration camera 2
	cv::Mat rvecs2, tvecs2;
	rms_camera2 = calibrateCamera(objectPoints, imagePoints2, imageSize, camera_matrix2, distortion_coeffs2, rvecs2, tvecs2, cv::CALIB_FIX_K3 + cv::CALIB_FIX_TANGENT_DIST);

	// Output rms
	std::cout << "rms1: " << rms_camera1 << std::endl
		      << "rms2: " << rms_camera2 << std::endl;


	// Stereo calibration
	m_rms = cv::stereoCalibrate(objectPoints, imagePoints1, imagePoints2,
		camera_matrix1, distortion_coeffs1, camera_matrix2, distortion_coeffs2, imageSize, Rotate, T,
		E, F, cv::CALIB_FIX_INTRINSIC);

	std::cout << "Complete" << std::endl
		<< "Stereo RMS error = " << m_rms << std::endl;

	writeParams();

	return true;
}

//
// Save calibrate params
void StereoCalibration::writeParams()
{
	if (m_output.empty())	
		m_output = "stereo_params.yml";

	cv::FileStorage fs(m_output, cv::FileStorage::WRITE);

	fs << "UsedFramework" << "OPENCV";

	fs << "imageSize" << imageSize;

	// Intrinsic params
	fs << "M1" << camera_matrix1;
	fs << "D1" << distortion_coeffs1;

	fs << "M2" << camera_matrix2;
	fs << "D2" << distortion_coeffs2;

	// Extrinsic params
	fs << "Rotate" << Rotate;
	fs << "T" << T;

	fs << "E" << E;
	fs << "F" << F;

	fs.release();
}