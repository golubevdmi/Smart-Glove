#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <thread>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


enum class DetectorModel
{
	MOBILENET_SSD_V1,
	MOBILENET_SSD_V2_COCO
};



struct DetectedObject
{
	std::int32_t class_id;
	std::string classname;

	std::double_t confidence;

	cv::Rect box;

	DetectedObject(std::int32_t class_id, std::string classname, std::double_t confidence, cv::Rect box) :
		class_id(class_id),
		classname(classname),
		confidence(confidence),
		box(box)
	{}
};



class DnnDetector
{
public:
	DnnDetector(std::string pathToModel, std::string pathToConfig = std::string()) :
		m_size(0, 0),
		m_scale(1.0),
		m_mean(0, 0, 0, 0),
		m_swapRB(false),
		m_path_model(pathToModel),
		m_path_config(pathToConfig)
	{}
	~DnnDetector() {}

	bool Detect(const cv::Mat &src, std::vector<DetectedObject> &detectedObjects);

	std::thread DetectTh(const cv::Mat &src, std::vector<DetectedObject> &detectedObjects)
	{
		return std::thread(&DnnDetector::Detect, this, src, std::ref(detectedObjects));
	}

	void setSize (cv::Size size)		     { m_size = size; }
	void setScale(std::double_t scale)	     { m_scale = 1 / scale; }
	void setMean (cv::Scalar mean)		     { m_mean = mean; }
	void setSwap (bool swap)			     { m_swapRB = swap; }
	void setModel(DetectorModel model)	     { m_model = model; }
	void setLabel(std::string pathToLabel)   { m_path_label = pathToLabel; }
	void setConfig(std::string pathToConfig) { m_path_config = pathToConfig; }

	std::vector<std::string> getClassesNames() { return m_classes_names; }

private:
	cv::Size		m_size;
	std::double_t	m_scale;
	cv::Scalar		m_mean;
	bool			m_swapRB;
	cv::dnn::Net	m_net;

	DetectorModel m_model;

	std::string m_path_model, m_path_config, m_path_label;
	std::vector<std::string> m_classes_names;


	bool netInitialization();
	void addClassesToVector();

	std::vector<DetectedObject> convertToDetectedObjectVec(const cv::Mat &prob, cv::Size srcSize) const;
};