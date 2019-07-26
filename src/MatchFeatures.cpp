#include "MatchFeatures.h"

#define ORB_NFEATURES 1000


MatchFeatures::MatchFeatures(FeatureDetectorType detectorType, DescriptorExtractorType extractorType, MatcherType matcherType)
{
	std::double_t hessianThreshold = 400.0;
	int nOctaves = 4;
	int	nOctaveLayers = 3;
	bool extended = false;
	bool upright = false;


	m_detector_type = detectorType;
	m_extractor_type = extractorType;

	// Detector
	switch (m_detector_type)
	{
	case FeatureDetectorType::DETECTOR_FAST:
		m_detector = cv::FastFeatureDetector::create();
		break;
	case FeatureDetectorType::DETECTOR_STAR:
		m_detector  = cv::xfeatures2d::StarDetector::create();
		break;
	case FeatureDetectorType::DETECTOR_SIFT:
		m_detector  = cv::xfeatures2d::SIFT::create();
		break;
	case FeatureDetectorType::DETECTOR_SURF:
		m_detector  = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
		break;
	case FeatureDetectorType::DETECTOR_ORB:
		m_detector  = cv::ORB::create(ORB_NFEATURES);
		break;
	case FeatureDetectorType::DETECTOR_BRISK:
		m_detector = cv::BRISK::create();
		break;
	case FeatureDetectorType::DETECTOR_MSER:
		m_detector = cv::MSER::create();
		break;
	case FeatureDetectorType::DETECTOR_GFTT:
		m_detector = cv::GFTTDetector::create();
		break;
	case FeatureDetectorType::DETECTOR_HARRIS:
		m_detector = cv::xfeatures2d::HarrisLaplaceFeatureDetector::create();
		break;
	case FeatureDetectorType::DETECTOR_SIMPLEBLOB:
		m_detector = cv::SimpleBlobDetector::create();
		break;
	default:
		m_detector = cv::ORB::create();
		break;
	}
	std::cout << ">> Detector: " << m_detector->getDefaultName() << std::endl;

	// Extractor
	switch (m_extractor_type)
	{
	case DescriptorExtractorType::EXTRACTOR_SIFT:
		m_extractor = cv::xfeatures2d::SIFT::create();
		break;
	case DescriptorExtractorType::EXTRACTOR_SURF:
		m_extractor = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
		break;
	case DescriptorExtractorType::EXTRACTOR_ORB:
		m_extractor = cv::ORB::create(ORB_NFEATURES);
		break;
	case DescriptorExtractorType::EXTRACTOR_BRIEF:
		m_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
		break;
	case DescriptorExtractorType::EXTRACTOR_FREAK:
		m_extractor = cv::xfeatures2d::FREAK::create();
		break;
	case DescriptorExtractorType::EXTRACTOR_BRISK:
		m_extractor = cv::BRISK::create();
		break;
	case DescriptorExtractorType::EXTRACTOR_AKAZE:
		m_extractor = cv::AKAZE::create();
		break;
	default:
		m_extractor = cv::ORB::create();
		break;
	}
	std::cout << ">> Extractor: " << m_extractor->getDefaultName() << std::endl;

	// Matcher
	switch (matcherType)
	{
	case MatcherType::MATCHER_FLANNBASED:
		switch (m_extractor_type)
		{
		case DescriptorExtractorType::EXTRACTOR_ORB:
			m_matcher = (new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)));
			break;
		default:
			m_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
			break;
		}
		std::cout << ">> Matcher: FLANN" << std::endl;
		break;
	case MatcherType::MATCHER_BRUTEFORCE:
		switch (extractorType)
		{
		case DescriptorExtractorType::EXTRACTOR_SIFT:
			m_matcher = cv::BFMatcher::create(cv::NORM_L1, false);
			break;
		case DescriptorExtractorType::EXTRACTOR_SURF:
			m_matcher = cv::BFMatcher::create(cv::NORM_L2, false);
			break;
		case DescriptorExtractorType::EXTRACTOR_ORB:
			m_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
			break;
		case DescriptorExtractorType::EXTRACTOR_BRIEF:
			m_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
			break;
		case DescriptorExtractorType::EXTRACTOR_BRISK:
			m_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
			break;
		case DescriptorExtractorType::EXTRACTOR_AKAZE:
			m_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
			break;
		default:
			m_matcher = cv::BFMatcher::create(cv::BFMatcher::BRUTEFORCE, false);
			break;
		}
		std::cout << ">> Matcher: BRUTEFORCE" << std::endl;
		break;
	}
}

MatchFeatures::~MatchFeatures()
{
	m_detector.release();
	m_extractor.release();
	m_matcher.release();

	m_keypoints_query.clear();
	m_keypoints_train.clear();

	m_good_matches.clear();
}



bool MatchFeatures::ComputeFeatures(const cv::Mat &query_image, const cv::Mat& train_image, cv::Mat& destination, cv::Mat mask)
{
	detectKeypoints(query_image, train_image, m_keypoints_query, m_keypoints_train, mask);

	cv::Mat descriptors1, descriptors2;
	computeDescriptors(query_image, train_image, m_keypoints_query, descriptors1, m_keypoints_train, descriptors2);

	// Match descriptors
	std::float_t distanceCoeffMin = 5.5;
	int k = 2;

	if (m_extractor_type == DescriptorExtractorType::EXTRACTOR_SURF)
		distanceCoeffMin = 2.0f;

	m_good_matches = matchDescriptors(descriptors1, descriptors2, k, cv::Mat());

	// Draw top matches
	//std::cout << "--> Draw top matches" << std::endl;
	drawMatches(query_image, m_keypoints_query, train_image, m_keypoints_train, m_good_matches, destination, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	// Находим объект на втором изображении
	// Проверка на минимальное количество хороших точек
	if ((!mask.empty() || query_image.size != train_image.size)
		&& m_good_matches.size() > MIN_MATCH_COUNT)
	{
		localizeTheObject(m_keypoints_query, m_keypoints_train, m_good_matches, query_image, destination, mask);
	}

	return true;
}
bool MatchFeatures::ComputeFeaturesForStereo(const cv::Mat& stereopair, cv::Mat& destination, cv::Mat mask)
{
	// Разделяем оригинальный кадр на два
	cv::Mat query_image  = cv::Mat(stereopair, cv::Rect(0, 0, stereopair.cols / 2, stereopair.rows));
	cv::Mat train_image = cv::Mat(stereopair, cv::Rect(stereopair.cols / 2, 0, stereopair.cols / 2, stereopair.rows));

	if (!ComputeFeatures(query_image, train_image, destination, mask)) return false;

	return true;
}

// 
// Detect the keypoints using Detector
void MatchFeatures::detectKeypoints(const cv::Mat &query_image, const cv::Mat &train_image, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, cv::Mat mask)
{
	if (!keypoints1.empty() || !keypoints2.empty())
	{
		keypoints1.clear();
		keypoints2.clear();
	}

	cv::Mat maskTrain = setMaskOnTrainImage(train_image.size(), mask);

	//std::cout << "--> Detect keypoints" << std::endl;
	m_detector->detect(query_image, keypoints1, mask);
	m_detector->detect(train_image, keypoints2, maskTrain);
}

//
// Compute the descriptors using Extractor
void MatchFeatures::computeDescriptors(const cv::Mat &query_image, const cv::Mat& train_image,
	std::vector<cv::KeyPoint>& keypoints1, cv::Mat& descriptors1, std::vector<cv::KeyPoint>& keypoints2, cv::Mat& descriptors2)
{
	if (keypoints1.empty() || keypoints2.empty())		return;
	if (!descriptors1.empty() || !descriptors2.empty())
	{
		descriptors1.release();
		descriptors2.release();
	}

	//std::cout << "--> Compute the descriptors" << std::endl;
	m_extractor->compute(query_image, keypoints1, descriptors1);
	m_extractor->compute(train_image, keypoints2, descriptors2);
}

//
// Find the best match for each descriptor from a query set
std::vector<cv::DMatch> MatchFeatures::matchDescriptors(cv::Mat& descriptors1, cv::Mat& descriptors2, std::float_t distanceCoeffMin, cv::Mat mask)
{
	std::vector<cv::DMatch> matches;
	m_matcher->match(descriptors1, descriptors2, matches, mask);

	// Filter matches using min-max distance
	std::vector<cv::DMatch> good_matches = distanceFilter(descriptors1, matches, distanceCoeffMin);

	return good_matches;
}

//
// Find the k best matches for each descriptor from a query set
std::vector<cv::DMatch> MatchFeatures::matchDescriptors(cv::Mat& descriptors1, cv::Mat& descriptors2, int k, cv::Mat mask)
{
	std::float_t ratioThresh = 0.4f;
	std::vector<std::vector<cv::DMatch>> knn_matches;
	m_matcher->knnMatch(descriptors1, descriptors2, knn_matches, k, mask, false);

	// Filter matches using the Lowe's ratio test
	std::vector<cv::DMatch> good_matches = ratioTest(knn_matches, ratioThresh);

	return good_matches;
}

// 
// Filter knn_matches using the Lowe's ratio test
std::vector<cv::DMatch> MatchFeatures::ratioTest(const std::vector<std::vector<cv::DMatch>>& knn_matches, const std::float_t ratioThresh)
{
	//std::cout << "--> Filter matches using the Lowe's ratio test" << std::endl;

	std::vector<cv::DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
		if (knn_matches[i][0].distance < ratioThresh * knn_matches[i][1].distance)
			good_matches.push_back(knn_matches[i][0]);

	return good_matches;
}

//
// Filter matches using min-max distance 
std::vector<cv::DMatch> MatchFeatures::distanceFilter(const cv::Mat& descriptors, const std::vector<cv::DMatch>& matches, std::float_t distanceCoeffMin)
{
	//std::cout << "--> Quick calculation of max and min distances between keypoints" << std::endl;

	std::vector<cv::DMatch> good_matches;

	std::double_t max_dist = 80;
	std::double_t min_dist = 5;

	// Найдем максимальное и минимальное расстояние между ключевыми точками
	for (int i = 0; i < descriptors.rows; i++) 
	{
		std::double_t dist = matches[i].distance;

		if (dist < min_dist) 	min_dist = dist;
		if (dist > max_dist) 	max_dist = dist;
	}


	// Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	// or a small arbitary value ( 0.02 ) in the event that min_dist is very small)
	// PS.- radiusMatch can also be used here.
	for (int i = 0; i < descriptors.rows; i++)
		if (matches[i].distance <= cv::max(distanceCoeffMin * min_dist, 0.02))
			good_matches.push_back(matches[i]);

	//std::cout << "Min: " << min_dist << " -- Max: " << max_dist << std::endl;

	return good_matches;
}

// --TODO Works not stable
// Localize the object
void MatchFeatures::localizeTheObject(const std::vector<cv::KeyPoint> keypoints1, const std::vector<cv::KeyPoint> keypoints2,
	const std::vector<cv::DMatch>& good_matches, const cv::Mat& query_image, cv::Mat& destination, cv::Mat mask)
{
	//std::cout << "--> Localize the object" << std::endl;
	try
	{
		// Draw roi rectangle on query_image
		if (!mask.empty())
		{
			cv::Rect rec = boundingRect(mask);
			rectangle(destination, rec, cv::Scalar(255), 5, 8, 0);
		}

		std::vector<cv::Point2f> queryPoints;
		std::vector<cv::Point2f> trainPoints;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			queryPoints.push_back(keypoints1[good_matches[i].queryIdx].pt);
			trainPoints.push_back(keypoints2[good_matches[i].trainIdx].pt);
		}

		cv::Mat H = findHomography(queryPoints, trainPoints, cv::RANSAC);

		// Get the corners from the objImage ( the object to be "detected" )
		std::vector<cv::Point2f> queryCorners(4);
		if (!mask.empty())
		{
			cv::Rect rec = boundingRect(mask);
			queryCorners[0] = cv::Point(rec.x, rec.y);
			queryCorners[1] = cv::Point(rec.x + rec.width, rec.y);
			queryCorners[2] = cv::Point(rec.x + rec.width, rec.y + rec.height);
			queryCorners[3] = cv::Point(rec.x, rec.y + rec.height);
		}
		else
		{
			queryCorners[0] = cv::Point(0, 0);
			queryCorners[1] = cv::Point(query_image.cols, 0);
			queryCorners[2] = cv::Point(query_image.cols, query_image.rows);
			queryCorners[3] = cv::Point(0, query_image.rows);
		}

		std::vector<cv::Point2f> trainCorners(4);
		perspectiveTransform(queryCorners, trainCorners, H);

		// Draw lines between the corners (the mapped object in the scene - image_2 )
		line(destination, trainCorners[0] + cv::Point2f(query_image.cols, 0), trainCorners[1] + cv::Point2f(query_image.cols, 0), cv::Scalar(255, 255, 255), 5);
		line(destination, trainCorners[1] + cv::Point2f(query_image.cols, 0), trainCorners[2] + cv::Point2f(query_image.cols, 0), cv::Scalar(255, 255, 255), 5);
		line(destination, trainCorners[2] + cv::Point2f(query_image.cols, 0), trainCorners[3] + cv::Point2f(query_image.cols, 0), cv::Scalar(255, 255, 255), 5);
		line(destination, trainCorners[3] + cv::Point2f(query_image.cols, 0), trainCorners[0] + cv::Point2f(query_image.cols, 0), cv::Scalar(255, 255, 255), 5);
	}
	catch (cv::Exception)
	{
	}
}

//
// --TEMP May be recycled or removed.
bool MatchFeatures::writeGoodPoints(std::string filename)
{
	std::cout << ">> Saving points to " << filename << std::endl;

	std::vector<cv::Point2f> pointsQuery, pointsTrain;

	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	if (!fs.isOpened())
	{
		std::cout << ">> Failed" << std::endl;
		return false;
	}

	fs << "Points" << std::string("OPENCV");

	for (int i = 0; i < (int)m_good_matches.size(); i++)
	{
		pointsQuery.push_back(m_keypoints_query[m_good_matches[i].queryIdx].pt);
		pointsTrain.push_back(m_keypoints_train[m_good_matches[i].trainIdx].pt);
	}

	fs << "pointsQuery" << pointsQuery;
	fs << "pointsTrain" << pointsTrain;

	std::cout << ">> Complete" << std::endl;

	return true;
}

// 
// --TEMP May be recycled or removed.
bool MatchFeatures::readGoodPoints(std::string filename, std::vector<cv::Point2f>& pointsQuery, std::vector<cv::Point2f>& pointsTrain)
{
	std::cout << ">> Reading points from " << filename << std::endl;

	cv::FileStorage node(filename, cv::FileStorage::READ);
	if (!node.isOpened())
	{
		std::cout << ">> Failed" << std::endl;
		return false;
	}

	node["pointsQuery"] >> pointsQuery;
	node["pointsTrain"] >> pointsTrain;

	std::cout << ">> Complete" << std::endl;

	return true;
}


std::double_t MatchFeatures::getMeanDisparity()
{
	std::double_t meanDisparity = 0;

	if (m_keypoints_query.size() == 0 || m_keypoints_train.size() == 0 || m_good_matches.size() == 0)
		return 0.0;

	for (int i = 0; i < (int)m_good_matches.size(); i++)
	{
		std::double_t disparity = m_keypoints_query[m_good_matches[i].queryIdx].pt.x - m_keypoints_train[m_good_matches[i].trainIdx].pt.x;
		
		// --TODO
		//if (disparity > 0)
			meanDisparity += disparity;
	}

	return meanDisparity /= (int)m_good_matches.size();
}

//
// --TEMP May be recycled or removed.
void MatchFeatures::getImageWithKeypoints(const cv::Mat& source, cv::Mat& destination)
{
	cv::Mat query_image = cv::Mat(source, cv::Rect(0, 0, source.cols / 2, source.rows));
	cv::Mat train_image = cv::Mat(source, cv::Rect(source.cols / 2, 0, source.cols / 2, source.rows));

	getImageWithKeypoints(query_image, train_image, destination);
}
//
// --TEMP May be recycled or removed.
void MatchFeatures::getImageWithKeypoints(const cv::Mat &query_image, const cv::Mat& train_image, cv::Mat& destination)
{
	if (m_keypoints_query.empty() || m_keypoints_train.empty())
	{
		m_detector->detect(query_image, m_keypoints_query, cv::Mat());
		m_detector->detect(train_image, m_keypoints_train, cv::Mat());
	}

	cv::Mat queryDst, trainDst;
	drawKeypoints(query_image, m_keypoints_query, queryDst);
	drawKeypoints(train_image, m_keypoints_train, trainDst);

	hconcat(queryDst, trainDst, destination);
}
//
// --TEMP May be recycled or removed.
bool MatchFeatures::getImageWithGoodPoints(const cv::Mat& query_image, const cv::Mat& train_image, cv::Mat& dst, std::vector<cv::Point2f> pointsQuery, std::vector<cv::Point2f> pointsTrain)
{
	if (pointsQuery.empty() || pointsTrain.empty())
	{
		if (m_good_matches.empty())	return false;

		for (int i = 0; i < (int)m_good_matches.size(); i++)
		{
			pointsQuery.push_back(m_keypoints_query[m_good_matches[i].queryIdx].pt);
			pointsTrain.push_back(m_keypoints_train[m_good_matches[i].trainIdx].pt);
		}
	}

	cv::Mat queryIMG_clone = query_image.clone();
	cv::Mat trainIMG_clone = train_image.clone();

	int counter = 1;
	for (std::vector<cv::Point2f>::iterator it1 = pointsQuery.begin(), it2 = pointsTrain.begin(); it1 != pointsQuery.end() || it2 != pointsTrain.end(); it1++, it2++)
	{
		cv::putText(queryIMG_clone, std::to_string(counter), cv::Point2f(it1->x + 10, it1->y - 10), 6, 2, cv::Scalar(0, 255, 0), 7, 8, false);
		cv::circle(queryIMG_clone, *it1, 10, cv::Scalar(255, 0, 0), 3, 8, 0);

		cv::putText(trainIMG_clone, std::to_string(counter), cv::Point2f(it2->x + 10, it2->y - 10), 6, 2, cv::Scalar(0, 255, 0), 7, 8, false);
		cv::circle(trainIMG_clone, *it2, 10, cv::Scalar(255, 0, 0), 3, 8, 0);



		std::cout << counter << ": " << *it1 << " -- " << *it2 << std::endl;
		counter++;
	}
	cv::hconcat(queryIMG_clone, trainIMG_clone, dst);

	return true;
}
void MatchFeatures::printGoodMatches()
{
	// Show good matches
	std::cout << std::endl << "Good Match" << std::endl;
	for (int i = 0; i < (int)m_good_matches.size(); i++)
	{
		std::cout << "[ " << i << " ]\t"
			<< "K1[ " << m_good_matches[i].queryIdx << " ]: " << m_keypoints_query[m_good_matches[i].queryIdx].pt.x
			<< " -- K2[ " << m_good_matches[i].trainIdx << " ]: " << m_keypoints_train[m_good_matches[i].trainIdx].pt.x << std::endl;
	}
}

// --TODO Works
//
cv::Mat MatchFeatures::setMaskOnTrainImage(const cv::Size& trainImageSize, cv::Mat mask)
{
	if (mask.empty())	
		return cv::Mat();

	cv::Mat maskTrain = cv::Mat::zeros(trainImageSize, CV_8U);

	try
	{
		cv::Rect rec = boundingRect(mask);

		rec.width += rec.x;

		// Установка точки в начале координат по Ох
		rec.x = 0;

		// Погрешность ректификации
		rec.y -= 5;
		rec.height += 10;

		cv::Mat roi = cv::Mat(maskTrain, rec);
		roi = cv::Scalar(255);

		//namedWindow("mask1", WINDOW_FREERATIO);
		//imshow("mask1", mask);

		//namedWindow("mask2", WINDOW_FREERATIO);
		//imshow("mask2", maskTrain);
	}
	catch (cv::Exception)
	{

	}

	return maskTrain;
}