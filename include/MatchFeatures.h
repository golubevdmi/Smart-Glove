#pragma once


#include "highgui.hpp"
#include "imgproc.hpp"
#include "calib3d.hpp"
#include "xfeatures2d.hpp"

#include <iostream>
#include <fstream>


#define MIN_MATCH_COUNT 10

enum class FeatureDetectorType
{
	DETECTOR_FAST,
	DETECTOR_STAR,
	DETECTOR_SIFT,		// SIFT (Scale-Invariant Feature Transform) - nonfree
	DETECTOR_SURF,		// SURF (Speeded-Up Robust Features) - nonfree
	DETECTOR_ORB,		// ORB (Oriented FAST and Rotated BRIEF)
	DETECTOR_BRISK,
	DETECTOR_MSER,
	DETECTOR_GFTT,
	DETECTOR_HARRIS,
	DETECTOR_SIMPLEBLOB
};
enum class DescriptorExtractorType
{
	EXTRACTOR_SIFT,		// nonfree
	EXTRACTOR_SURF,		// nonfree
	EXTRACTOR_ORB,
	EXTRACTOR_BRIEF,
	EXTRACTOR_FREAK,
	EXTRACTOR_BRISK,
	EXTRACTOR_AKAZE
};
enum class MatcherType
{
	MATCHER_FLANNBASED,
	MATCHER_BRUTEFORCE
};


class MatchFeatures
{
public:
	MatchFeatures(FeatureDetectorType detectorType = FeatureDetectorType::DETECTOR_ORB,
		DescriptorExtractorType extractorType = DescriptorExtractorType::EXTRACTOR_ORB, MatcherType matcherType = MatcherType::MATCHER_BRUTEFORCE);
	MatchFeatures(cv::Ptr<cv::Feature2D> detector, cv::Ptr<cv::Feature2D> extractor, cv::Ptr<cv::DescriptorMatcher> matcher) :
		m_detector(detector),
		m_extractor(extractor),
		m_matcher(matcher)
	{}
	~MatchFeatures();

	bool ComputeFeatures(const cv::Mat& query_image, const cv::Mat& train_image, cv::Mat& destination, cv::Mat mask = cv::Mat());
	bool ComputeFeaturesForStereo(const cv::Mat& stereopair, cv::Mat& destination, cv::Mat mask = cv::Mat());

	bool writeGoodPoints(std::string filename);
	bool readGoodPoints(std::string filename, std::vector<cv::Point2f>& pointsQuery, std::vector<cv::Point2f>& pointsTrain);

	std::vector<cv::KeyPoint>	getKeypoints1() { return m_keypoints_query; }
	std::vector<cv::KeyPoint>	getKeypoints2() { return m_keypoints_train; }
	std::vector<cv::DMatch>		getm_good_matches() { return m_good_matches; }
	std::double_t				getMeanDisparity();

	void getImageWithKeypoints(const cv::Mat &source, cv::Mat& destination);
	void getImageWithKeypoints(const cv::Mat &query_image, const cv::Mat& train_image, cv::Mat& destination);
	bool getImageWithGoodPoints(const cv::Mat& query_image, const cv::Mat& train_image, cv::Mat& dst,
		std::vector<cv::Point2f> pointsQuery = std::vector<cv::Point2f>(), std::vector<cv::Point2f> pointsTrain = std::vector<cv::Point2f>());

	// --TODO
	void getMatchedPoints(std::vector<cv::Point2f> &queryPts, std::vector<cv::Point2f> &trainPts)
	{
		for (auto idx : m_good_matches)
		{
			queryPts.push_back(cv::Point2f(m_keypoints_query[idx.queryIdx].pt.x, m_keypoints_query[idx.queryIdx].pt.y));
			trainPts.push_back(cv::Point2f(m_keypoints_train[idx.trainIdx].pt.x, m_keypoints_train[idx.trainIdx].pt.y));
		}
	}

	void printGoodMatches();

private:
	std::vector<cv::KeyPoint> m_keypoints_query, m_keypoints_train;
	std::vector<cv::DMatch> m_good_matches;

	FeatureDetectorType m_detector_type;
	DescriptorExtractorType m_extractor_type;

	cv::Ptr<cv::Feature2D> m_detector;
	cv::Ptr<cv::Feature2D> m_extractor;
	cv::Ptr<cv::DescriptorMatcher> m_matcher;

	void detectKeypoints(const cv::Mat &query_image, const cv::Mat& train_image, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, cv::Mat mask = cv::Mat());
	void computeDescriptors(const cv::Mat &query_image, const cv::Mat& train_image,
		std::vector<cv::KeyPoint>& keypoints1, cv::Mat& descriptors1, std::vector<cv::KeyPoint>& keypoints2, cv::Mat& descriptors2);

	std::vector<cv::DMatch> matchDescriptors(cv::Mat& descriptors1, cv::Mat& descriptors2, std::float_t distanceCoeffMin = 3.5f, cv::Mat mask = cv::Mat());
	std::vector<cv::DMatch> matchDescriptors(cv::Mat& descriptors1, cv::Mat& descriptors2, int k, cv::Mat mask = cv::Mat());
	std::vector<cv::DMatch> ratioTest(const std::vector<std::vector<cv::DMatch>>& knn_matches, std::float_t ratio_thresh);
	std::vector<cv::DMatch> distanceFilter(const cv::Mat& descriptors, const std::vector<cv::DMatch>& matches, std::float_t distanceCoeffMin = 3.5f);

	void localizeTheObject(const std::vector<cv::KeyPoint> keypoints1, const std::vector<cv::KeyPoint> keypoints2,
		const std::vector<cv::DMatch>& m_good_matches, const cv::Mat& query_image, cv::Mat& destination, cv::Mat mask = cv::Mat());
	cv::Mat setMaskOnTrainImage(const cv::Size& trainImageSize, cv::Mat mask);

};