#include "DnnDetector.h"



bool DnnDetector::netInitialization()
{
	if (!m_net.empty())	        return true;

	CV_Assert(!m_path_model.empty());

	switch (m_model)
	{
	case DetectorModel::MOBILENET_SSD_V1:
		CV_Assert(!m_path_config.empty());
		m_net = cv::dnn::readNet(m_path_model, m_path_config);
		break;
	case DetectorModel::MOBILENET_SSD_V2_COCO:
		CV_Assert(!m_path_config.empty());
		m_net = cv::dnn::readNet(m_path_model, m_path_config);
		break;
	default:
		break;
	}

	if (m_net.empty())
		CV_Error(cv::Error::StsError, "Cannot read net");

	m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);


	return true;
}



bool DnnDetector::Detect(const cv::Mat &src, std::vector<DetectedObject> &detectedObjects)
{
	if (!netInitialization())	return false;
	if (m_classes_names.empty() && !m_path_label.empty()) addClassesToVector();


	cv::Mat inputTensor;

	switch (m_model)
	{
	case DetectorModel::MOBILENET_SSD_V1:
		CV_Assert(m_size.width > 0);
		CV_Assert(m_size.height > 0);
		CV_Assert(m_scale >= 0 && m_scale <= 1.0);
		cv::dnn::blobFromImage(src, inputTensor, m_scale, m_size, m_swapRB);
		break;
	case DetectorModel::MOBILENET_SSD_V2_COCO:
		cv::dnn::blobFromImage(src, inputTensor, 1.0, cv::Size(), m_swapRB, false, 5);
		break;
	default:
		std::cout << "Model Error" << std::endl;
		return false;
		break;
	}
	
	try
	{
		m_net.setInput(inputTensor);
		cv::Mat prob = m_net.forward();

		cv::Mat detectionAsMat(prob.size[2], prob.size[3], CV_32F, prob.ptr<std::float_t>());

		if (detectionAsMat.empty())	return false;

		detectedObjects = convertToDetectedObjectVec(detectionAsMat, src.size());
	}
	catch (cv::Exception e)
	{
		std::cout << e.what() << std::endl;
		return false;
	}


	return true;
}

//
// Add classes to vector
void DnnDetector::addClassesToVector()
{
	std::ifstream in(m_path_label, std::ios::in);
	if (in.is_open())
	{
		std::string line;
		while (std::getline(in, line))
			m_classes_names.push_back(line);
	
		in.close();
	}
}

//
// Convert from mobilenet_ssd2 v2 to DetectedObject
std::vector<DetectedObject> DnnDetector::convertToDetectedObjectVec(const cv::Mat &prob, cv::Size srcSize) const
{
	std::float_t scale = 0.5;
	std::vector<DetectedObject> detObjects;
	
	for (std::uint32_t i = 0; i < prob.rows; i++)
	{
		std::double_t confidence = static_cast<std::double_t>(prob.at<std::float_t>(i, 2));
		if (confidence < scale) continue;

		std::int32_t classId	 = static_cast<std::int32_t>(prob.at<std::float_t>(i, 1));

		std::int32_t xLeft	 = static_cast<std::int32_t>(prob.at<std::float_t>(i, 3) * srcSize.width);
		std::int32_t yBottom = static_cast<std::int32_t>(prob.at<std::float_t>(i, 4) * srcSize.height);
		std::int32_t xRight  = static_cast<std::int32_t>(prob.at<std::float_t>(i, 5) * srcSize.width);
		std::int32_t yTop	 = static_cast<std::int32_t>(prob.at<std::float_t>(i, 6) * srcSize.height);

		std::string className = "None";
		if (!m_classes_names.empty())
			if (classId < m_classes_names.size())
				className = m_classes_names[classId];

		detObjects.push_back(DetectedObject(classId, className, confidence, cv::Rect(cv::Point(xLeft, yBottom), cv::Point(xRight, yTop))));
	}

	return detObjects;
}