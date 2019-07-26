#pragma once
#include <iostream>
#include <string>
#include <algorithm>

#include "calib3d.hpp"
#include "highgui.hpp"
#include "imgcodecs.hpp"
#include "imgproc.hpp"


namespace calib
{
	//! Calibration pattern
	enum class Pattern
	{
		CHESSBOARD,
		CIRCLES_GRID,
		ASYMMETRIC_CIRCLES_GRID
	};

	//
	// Parameters for single camera
	struct  SingleCamera
	{
	public:
		SingleCamera() :
			imageSize(0, 0)
		{
			camera_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
			distortion_coeffs = cv::Mat::zeros(1, 5, CV_64FC1);
		}

		cv::Mat camera_matrix;			// Intrinsic camera params
		cv::Mat distortion_coeffs;		// Distortion coefficient

		std::vector<cv::Mat> rvecs;			// Rotate vectors
		std::vector<cv::Mat> tvecs;			// Offset vectors

		cv::Size imageSize;
	};

	//
	// Parameters for stereo camera
	struct  StereoCamera
	{
		StereoCamera() :
			baseline(0.0),
			focallenght(0.0),
			imageSize(0, 0)
		{
			camera_matrix1 = cv::Mat::zeros(3, 3, CV_64FC1);
			camera_matrix2 = cv::Mat::zeros(3, 3, CV_64FC1);
			
			distortion_coeffs1 = cv::Mat::zeros(1, 5, CV_64FC1);
			distortion_coeffs2 = cv::Mat::zeros(1, 5, CV_64FC1);
		}

		cv::Size imageSize;

		std::double_t baseline;
		std::double_t focallenght;
		
		// Intrinsic camera params 3x3
		cv::Mat camera_matrix1, camera_matrix2;
		// Distortion coefficent 1x5
		cv::Mat distortion_coeffs1, distortion_coeffs2;

		// 3x3
		cv::Mat Rotate;
		// 1x3
		cv::Vec3d T;
		// Essential 3x3, Fundamental 3x3 matrix
		cv::Mat E, F;

		// Rotate matrix camera1, camera2
		cv::Mat R1, R2;
		// New camera matrix
		cv::Mat P1, P2;
		// Reprojection matrix
		cv::Mat Q;

		cv::Mat map1[2], map2[2];
	};



	//
	// A set of common characteristics for calibration
	//
	//
	class  Calibration
	{
	public:
		Calibration(Pattern pattern, cv::Size patternSize, std::float_t squareSize, std::string outputName = std::string()) :
			m_pattern(pattern),
			m_patternSize(patternSize),
			m_squareSize(squareSize),
			m_output(outputName),
			m_rms(0.0)
		{}
		~Calibration() {}


		virtual void CalibrationByVideo(std::string videoPath) = 0;
		virtual void CalibrationByImagesVector(const std::vector<cv::Mat>& images) = 0;
		virtual void writeParams() = 0;

	protected:
		std::float_t m_squareSize;		// The size of the side of the square / diameter circle
		Pattern      m_pattern;			// Calibration pattern
		cv::Size     m_patternSize;		// Number of corners (width / height)

		std::string m_output;

		// Roor mean square
		std::double_t m_rms;

		virtual void createKnownPosition(std::vector<cv::Point3f>& corners);
		virtual bool getCorners(const cv::Mat& imageArray, std::vector<cv::Point2f>& FoundCorners);
	};
	class  SingleCalibration : public Calibration, public SingleCamera
	{
	public:
		SingleCalibration(Pattern pattern, cv::Size patternSize, std::float_t squareSize, std::string outputName = std::string()) :
			SingleCamera(),
			Calibration(pattern, patternSize, squareSize, outputName)
		{}
		~SingleCalibration() {}

		virtual void CalibrationByVideo(std::string videoPath) final;
		virtual void CalibrationByImagesVector(const std::vector<cv::Mat>& images) final;
		virtual void writeParams() final;

	private:
		void calibrate(std::vector<std::vector<cv::Point2f>>& imagePoints);
	};
	class  StereoCalibration : public Calibration, public StereoCamera
	{
	public:
		StereoCalibration(Pattern pattern, cv::Size patternSize, std::float_t squareSize, std::string outputName = std::string()) :
			StereoCamera(),
			Calibration(pattern, patternSize, squareSize, outputName),
			rms_camera1(0.0),
			rms_camera2(0.0)
		{}
		~StereoCalibration() {}

		virtual void CalibrationByVideo(std::string videoPath) final;
		virtual void CalibrationByImagesVector(const std::vector<cv::Mat> &images) final;
		virtual void writeParams() final;

	private:
		/// Root mean square cameras
		std::double_t rms_camera1, rms_camera2;

		bool calibrate(const std::vector<std::vector<cv::Point2f>>& imagePoints1, const std::vector<std::vector<cv::Point2f>>& imagePoints2);
	};

	

	//
	// Calibration Reading Class Set 
	//
	//
	class  CalibrationReader
	{
	public:
		CalibrationReader(std::string filename) :
			m_fsParams(filename, cv::FileStorage::READ),
			m_isReceived(false)
		{
			CV_Assert(m_fsParams.isOpened());
		}
		~CalibrationReader() {}

		virtual bool read() = 0;
		virtual void show() = 0;

		/// Checks if the settings have been received.
		virtual bool isOpened() { return m_isReceived; }

	protected:
		cv::FileStorage m_fsParams;

		/// MATLAB, OpenCV
		std::string UsedFramework;

		bool m_isReceived;

		virtual bool readAsOpenCV() = 0;
		virtual bool readAsMatlab() = 0;
	};
	class  SingleCalibrationReader : public CalibrationReader, private SingleCamera
	{
	public:

	private:

	};
	class  StereoCalibrationReader : public CalibrationReader, private StereoCamera
	{
	public:
		StereoCalibrationReader(std::string filename) :
			StereoCamera(),
			CalibrationReader(filename)
		{}
		~StereoCalibrationReader() {}

		/// Reading intrinsic and extrinsic params
		virtual bool read() final;
		/// Compute all params
		bool computeParams();
		/// Compute R1, R2, P1, P2, Q
		bool computeRectifyParams();
		/// Compute maps (for remapping)
		bool computeUndistortMap();
		/// Main distance params
		bool computeBaseline();
		bool computeFocalLenght();

		/// Show intrinsic and extrinsic params
		virtual void show() final;

		std::double_t getBaseline()    { return baseline; }
		std::double_t getFocalLenght() { return focallenght; }

		cv::Mat getM1() { return camera_matrix1.clone(); }
		cv::Mat getM2() { return camera_matrix2.clone(); }

		cv::Mat getD1() { return distortion_coeffs1.clone(); }
		cv::Mat getD2() { return distortion_coeffs2.clone(); }

		cv::Mat getR1() { return R1.clone(); }
		cv::Mat getR2() { return R2.clone(); }

		cv::Mat getP1() { return P1.clone(); }
		cv::Mat getP2() { return P2.clone(); }

		cv::Mat getEssential()   { return E.clone(); }
		cv::Mat getFundamental() { return F.clone(); }

		cv::Mat getMap1x() { return map1[0].clone(); }
		cv::Mat getMap1y() { return map1[1].clone(); }
		cv::Mat getMap2x() { return map2[0].clone(); }
		cv::Mat getMap2y() { return map2[1].clone(); }

	protected:
		virtual bool readAsOpenCV() final;
		virtual bool readAsMatlab() final;
	};



	 std::double_t calculateDistance(std::double_t baselineMetres, std::double_t focalLenght, std::double_t disparity);
}