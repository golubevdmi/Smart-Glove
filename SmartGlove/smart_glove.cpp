#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>


#include "DnnDetector.h"
#include "TrackingByMatching.h"
#include "calibration.h"
#include "ControlDisplayedObjects.h"
#include "MatchFeatures.h"


using namespace calib;


const char* cmdOptions =
"{ i  video                             |                       ../data/video/video1.avi                      | image to process                  }"
"{ w  width                             |                                300                                  | image width for classification    }"
"{ h  heigth                            |                                300                                  | image heigth fro classification   }"
"{ model_path                           |      ../data/dnn/mobilenet-ssd/v2/frozen_inference_graph.pb         | path to model                     }"
"{ config_path                          | ../data/dnn/mobilenet-ssd/v2/ssd_mobilenet_v2_coco_2018_03_29.pbtxt | path to model configuration       }"
"{ label_path                           |   ../data/dnn/mobilenet-ssd/v2/object_detection_classes_coco.txt    | path to class labels              }"
"{ calib_path                           |                    ../data/calib/params.yml                         | path to calibrate params          }"
"{ scale                                |                                                                     | scale					          }"
"{ mean                                 |                        127.5 127.5 127.5 0                          | vector of mean model values       }"
"{ swap                                 |                                  0                                  | swap R and B channels. TRUE|FALSE }"
"{ writer_path                          |                              output.avi                             | path to output video			  }"
"{ q ? help usage                       |                                                                     | print help message                }";



bool getFrame(cv::Mat &frame, cv::VideoCapture &cap1, cv::VideoCapture &cap2, std::string videoPath = std::string());

void runDetect(DnnDetector **m_detector, std::string modelPath, std::string configPath,
	std::string labelPath, cv::Size size, std::double_t scale, cv::Scalar mean, bool swapRB);
void runTrack(TrackingByMatching **tracker);

void CalcDistance(MatchFeatures &mf, cv::Mat &frame, std::vector<TrackedObject> &tObjects, std::double_t base,
	std::double_t focalLenght, const cv::Mat *M, const cv::Mat *D, const cv::Mat *R, const cv::Mat *P);

void ControlObjects(ControlDisplayedObjects **controller, cv::Size imgSize, std::string classesPath);

void drawObjects(cv::Mat &image, const std::vector<TrackedObject> &tObjects, const std::vector<std::int32_t> &desIds, std::int32_t idNav);
void drawObject(cv::Mat &image, const TrackedObject &detObject);
void drawStat(cv::Mat &image, std::int32_t timeDetect, std::int32_t timeTracker, std::int32_t numberOfObjects,
	std::int32_t idNavigation = -1, std::vector<std::int32_t> desIds = std::vector<std::int32_t>());

// Color Vector (for coloring areas)
std::vector<cv::Scalar> colors;

//
// Hot keys
void info()
{
	std::cout << "Press '0' to choose work with tracked objects\n" <<
				 "Press '+' for sound prompt(NOT WORKS)\n"
				 "Press ENTER to start m_detector and tracker\n" <<
		         "Press SPACE to pause\n" <<
		         "Press Esc to exit\n" << std::endl;
}

int main(int argc, const char* argv[])
{
	// CommandLine
	std::string videoPath, modelPath, configPath, labelPath, calibPath;
	cv::Size size(0, 0);
	std::double_t scale = 1.0;
	cv::Scalar mean(0, 0, 0, 0);
	bool swapRB = false;

	// Process input arguments
	cv::CommandLineParser parser(argc, argv, cmdOptions);

	if (parser.has("help"))
	{
		parser.printMessage();
		return -1;
	}
	if (!parser.check())
	{
		parser.printErrors();
		return -1;
	}

	// Load video and init parameters
	videoPath = parser.get<std::string>("video");

	size.width = parser.get<std::int32_t>("width");
	size.height = parser.get<std::int32_t>("heigth");

	modelPath = parser.get <std::string>("model_path");
	configPath = parser.get <std::string>("config_path");
	labelPath = parser.get <std::string>("label_path");

	calibPath = parser.get <std::string>("calib_path");

	scale = parser.get<std::double_t>("scale");
	mean = parser.get<cv::Scalar>("mean");
	swapRB = parser.get<bool>("swap");


	// Get random colors
	cv::RNG rng;
	for (int i = 0; i < 100; i++)
	{
		int r = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int b = rng.uniform(0, 256);
		colors.push_back(cv::Scalar(r, g, b));
	}

	// Cameras
	cv::Mat frame;
	cv::VideoCapture cap1, cap2;

	// Detector, tracker
	DnnDetector *m_detector = nullptr, *detector2 = nullptr;
	TrackingByMatching *tracker = nullptr, *tracker2 = nullptr;

	// Cameras params
	StereoCalibrationReader params(calibPath);
	params.computeParams();
	cv::Mat M[2], D[2], R[2], P[2];
	M[0] = params.getM1();
	M[1] = params.getM2();
	D[0] = params.getD1();
	D[1] = params.getD2();
	R[0] = params.getR1();
	R[1] = params.getR2();
	P[0] = params.getP1();
	P[1] = params.getP2();

	MatchFeatures mf;

	// ControlObjects
	ControlDisplayedObjects *controller = nullptr;

	std::int8_t key = NULL;
	std::int8_t pause = 1;

	info();

	//videoPath.clear();

	while (true)
	{
		std::vector<DetectedObject> detected_objects;
		std::vector<TrackedObject> tracked_objects;

		if (!getFrame(frame, cap1, cap2, videoPath))	
			break;


		cv::Rect left(0, 0, frame.size().width / 2, frame.size().height);
		cv::Rect right(frame.size().width / 2, 0, frame.size().width / 2, frame.size().height);
		
		// Detector and tracker (initialization)
		if (key == '\r')
		{
			runDetect(&m_detector, modelPath, configPath, labelPath, size, scale, mean, swapRB);
			runTrack(&tracker);
		}

		// Detector
		std::uint32_t timeD = clock();
		if (m_detector)
			m_detector->Detect(frame(left), detected_objects);
		timeD = clock() - timeD;

		// Tracker
		std::uint32_t timeT = clock();
		if (tracker && !detected_objects.empty())
			tracked_objects = tracker->track(detected_objects);
		timeT = clock() - timeT;

		// --TODO Not works
		// Playing voice prompt
		if (key == '+')
		{
			if (controller)
				controller->navigate();
		}

		// Controller
		std::vector<std::int32_t> desIds;
		std::int32_t idNav = -1;
		if (controller)
		{
			// Получение классов для контроля
			bool isDesEnable = controller->getDesClasses(desIds);
			bool isNavEnable = controller->getNavigationId(idNav);

			// Получение области для навигатора
			for (auto tObj : tracked_objects)
			{
				if (tObj.id_ext != -1 && tObj.class_id == idNav && tObj.missed < TRACKER_MIN_MISSED)
				{
					controller->setNavigationBox(tObj.box);
					break;
				}
				else
				{
					controller->setNavigationBox(cv::Rect());
				}
			}
		}

		// Distance
		CalcDistance(mf, frame, tracked_objects, params.getBaseline(), params.getFocalLenght(), M, D, R, P);

		// Draw
		cv::Mat frame_left = frame(left);
		drawObjects(frame_left, tracked_objects, desIds, idNav);
		drawStat(frame, timeD, timeT, tracked_objects.size(), idNav, desIds);

		// Show
		std::string win_name = "Video";
		cv::namedWindow(win_name, cv::WINDOW_FREERATIO);
		cv::imshow(win_name, frame);

		key = cv::waitKey(pause);
		if (key == 27)	break;
		if (key == ' ')	pause *= -1;
		if (key == '0')
		{
			cv::Size imgSize;

			if (cap1.isOpened() && cap2.isOpened())
				imgSize = frame(left).size();
			else
				imgSize = frame.size();

			ControlObjects(&controller, imgSize, labelPath);
		}
		
	}

	if (m_detector)
	{
		delete m_detector;
		m_detector = nullptr;
	}
	if (tracker)
	{
		delete tracker;
		tracker = nullptr;
	}

	if (detector2)
	{
		delete detector2;
		detector2 = nullptr;
	}
	if (tracker2)
	{
		delete tracker2;
		tracker2 = nullptr;
	}

	if (controller)
	{
		delete controller;
		controller = nullptr;
	}

	cap1.release();
	cap2.release();
	frame.release();

	cv::destroyAllWindows();

	return 0;
}

// 
// Grab frame from video or frame from camera or cameras
bool getFrame(cv::Mat &frame, cv::VideoCapture &cap1, cv::VideoCapture &cap2, std::string videoPath)
{
	if (!cap1.isOpened() && !videoPath.empty())
		cap1.open(videoPath);

	if (!cap1.isOpened() && !cap2.isOpened())
	{
		cap1.open(0, cv::CAP_DSHOW);
		cap2.open(1, cv::CAP_DSHOW);
	}

	if (cap1.isOpened() && cap2.isOpened())
	{
		cv::Mat left, right;
		cap1 >> left;
		cap2 >> right;

		CV_Assert(left.size() == right.size());
		CV_Assert(left.channels() == right.channels());

		cv::hconcat(left, right, frame);
	}
	else if (cap1.isOpened())
		cap1 >> frame;
	else if (cap2.isOpened())
		cap2 >> frame;

	if (frame.empty())	return false;

	return true;
}

//
// Initialization m_detector and tracker
void runDetect(DnnDetector **m_detector, std::string modelPath, std::string configPath,
	std::string labelPath, cv::Size size, std::double_t scale, cv::Scalar mean, bool swapRB)
{
	CV_Assert(!modelPath.empty());

	if (!(*m_detector))
	{
		(*m_detector) = new DnnDetector(modelPath);

		(*m_detector)->setModel(DetectorModel::MOBILENET_SSD_V2_COCO);

		if (!configPath.empty())
			(*m_detector)->setConfig(configPath);
		if (!labelPath.empty())
			(*m_detector)->setLabel(labelPath);


		if (size.width > 0 && size.height > 0)
			(*m_detector)->setSize(size);

		(*m_detector)->setScale(scale);
		(*m_detector)->setMean(mean);
		(*m_detector)->setSwap(swapRB);
	}
	else
	{
		delete (*m_detector);
		(*m_detector) = nullptr;
	}
}
void runTrack(TrackingByMatching **tracker)
{
	if (!(*tracker))
	{
		(*tracker) = new TrackingByMatching();
	}
	else
	{
		delete (*tracker);
		(*tracker) = nullptr;
	}
}

//
// Match left and right frames. Calculate distance
void CalcDistance(MatchFeatures &mf, cv::Mat &frame, std::vector<TrackedObject> &tObjects, std::double_t base,
	std::double_t focalLenght, const cv::Mat *M, const cv::Mat *D, const cv::Mat *R, const cv::Mat *P)
{
	cv::Rect left(0, 0, frame.size().width / 2, frame.size().height);
	cv::Rect right(frame.size().width / 2, 0, frame.size().width / 2, frame.size().height);

	for (auto &tObj : tObjects)
	{
		if (tObj.id_ext == -1 || tObj.missed > TRACKER_MIN_MISSED)	continue;


		cv::Mat mask = cv::Mat::zeros(frame(left).size(), CV_8UC1);
		mask(tObj.box).setTo(255);
		cv::Mat dst;
		mf.ComputeFeatures(frame(left), frame(right), dst, mask);

		std::vector<cv::Point2f> pt1, pt2;
		mf.getMatchedPoints(pt1, pt2);

		// Caclulate right rectangle obj
		cv::Point2f ptCentral(0, 0);
		for (auto p : pt2)
			ptCentral = cv::Point2f(ptCentral.x + p.x, ptCentral.y + p.y);

		ptCentral = cv::Point2f(ptCentral.x / pt2.size(), ptCentral.y / pt2.size());
		cv::Rect recRight(ptCentral.x - tObj.box.width / 2, ptCentral.y - tObj.box.height / 2, tObj.box.width, tObj.box.height);
		cv::rectangle(frame(right), recRight, colors[tObj.id_ext]);

		// Get undistort pts
		if (pt1.empty() || pt2.empty())	continue;
		CV_Assert(pt1.size() == pt2.size());
		cv::undistortPoints(pt1, pt1, M[0], D[0], R[0], P[0]);
		cv::undistortPoints(pt2, pt2, M[1], D[1], R[1], P[1]);

		// Calculate mean dx
		std::double_t meanDx = 0;
		std::vector<cv::Point2f>::iterator it1 = pt1.begin(), it2 = pt2.begin();
		for (; it1 != pt1.end(), it2 != pt2.end(); ++it1, ++it2)
		{
			if ((*it1).x > (*it2).x)
			{
				std::double_t dx = (*it1).x - (*it2).x;
				meanDx += dx;
			}
		}
		meanDx /= pt1.size();

		// Set distance
		if (meanDx > 18)
		{
			tObj.distance = calculateDistance(base, focalLenght, meanDx);

			if (tObj.distAvg != -1)
				tObj.distAvg = 0.9 * tObj.distAvg + 0.1 * tObj.distance;
			else
				tObj.distAvg = tObj.distance;
		}
		else
		{
			tObj.distance = -1;
		}
	}
}

//
// Draw
void drawObjects(cv::Mat &image, const std::vector<TrackedObject> &tObjects, const std::vector<std::int32_t> &desIds, std::int32_t idNav)
{
	for (auto &tObj : tObjects)
	{
		// Присвоен внешний ид и проверка на количество пропусков
		if (tObj.id_ext != -1 && tObj.missed < TRACKER_MIN_MISSED)
		{
			// Если класс желаемых ид пуст
			if (desIds.empty())
			{
				// Вывод первого объекта с совпадающим класс ид
				if (idNav != -1)
				{
					if (tObj.class_id == idNav)
					{
						{
							drawObject(image, tObj);
							break;
						}
					}
				}
				else
				{
					// Вывод всех объектов с внешним ид
					drawObject(image, tObj);
				}
			}
			else
			{
				// Вывод только нужных идс
				for (auto visibleId : desIds)
					if (visibleId == tObj.class_id)
						drawObject(image, tObj);
			}
		}
	}
}
void drawObject(cv::Mat &image, const TrackedObject &tracked_object)
{
	cv::rectangle(image, tracked_object.box, colors[tracked_object.id_ext]);
	cv::circle(image, tracked_object.cm, 4, colors[tracked_object.id_ext], 1, 8, 0);

	cv::Point2d pt;
	std::string text;
	std::int32_t fontFace = 1;
	std::double_t fontScale = 0.7;
	cv::Scalar color(colors[tracked_object.id_ext]);
	std::int32_t thickness = 1;
	std::int32_t linetype = 1;

	// id
	pt = cv::Point2d(tracked_object.box.x + tracked_object.box.width * 1.1, tracked_object.box.y + tracked_object.box.height * 0.0);
	text = "IdE: " + std::to_string(tracked_object.id_ext);
	cv::putText(image, text, pt, fontFace, fontScale, color, thickness, linetype, false);

	// className
	pt = cv::Point2d(tracked_object.box.x + tracked_object.box.width * 1.1, tracked_object.box.y + tracked_object.box.height * 0.2);
	text = "Name: " + tracked_object.classname;
	cv::putText(image, text, pt, fontFace, fontScale, color, thickness, linetype, false);

	// confidence
	//pt = cv::Point2d(tracked_object.box.x + tracked_object.box.width * 0.1, tracked_object.box.y + tracked_object.box.height * 0.15);
	//text = "c: " + std::to_string(tracked_object.confidence);
	//cv::putText(image, text, pt, fontFace, fontScale, color, thickness, linetype, false);

	if (tracked_object.distance != -1)
	{
		text = "D: " + std::to_string(tracked_object.distance);

		pt = cv::Point2d(tracked_object.box.x + tracked_object.box.width * 1.1, tracked_object.box.y + tracked_object.box.height * 0.4);
		cv::putText(image, text, pt, fontFace, fontScale, color, thickness, linetype, false);
	}
}
void drawStat(cv::Mat &image, std::int32_t timeDetect, std::int32_t timeTracker, std::int32_t numberOfObjects, std::int32_t idNavigation, std::vector<std::int32_t> idDes)
{

	cv::putText(image, "dTime: " + std::to_string(timeDetect) + "ms", cv::Point2d(image.size().width * 0.05, image.size().height * 0.05),
		1, 2, cv::Scalar(255, 255, 255), 4, 8, false);

	cv::putText(image, "tTime: " + std::to_string(timeTracker) + "ms", cv::Point2d(image.size().width * 0.05, image.size().height * 0.1),
		1, 2, cv::Scalar(255, 255, 255), 4, 8, false);

	cv::putText(image, "nObj: " + std::to_string(numberOfObjects), cv::Point2d(image.size().width * 0.05, image.size().height * 0.15),
		1, 2, cv::Scalar(255, 255, 255), 4, 8, false);

	if (idNavigation != -1)
	{
		cv::putText(image, "idN: " + std::to_string(idNavigation), cv::Point2d(image.size().width * 0.05, image.size().height * 0.20),
			1, 2, cv::Scalar(255, 255, 255), 4, 8, false);
	}

	if (!idDes.empty())
	{
		std::string text;
		for (auto id : idDes)
			text += std::to_string(id) + " ";

		cv::putText(image, "idDes: " + text, cv::Point2d(image.size().width * 0.05, image.size().height * 0.25),
			1, 2, cv::Scalar(255, 255, 255), 4, 8, false);
	}
}

//
// Controller class navigation
void ControlObjects(ControlDisplayedObjects **controller, cv::Size imgSize, std::string classesPath)
{
	cv::destroyAllWindows();
	system("cls");

	std::int32_t choice = 0;
	std::cout << "--------- Desired classes --------\n"
				 "-- 1. Set class desired by voice(NOT WORKS)\n"
			     "-- 2. Add desired object\n"
				 "-- 3. Delete desired object\n"
				 "-- 4. Enable track by Desired\n"
				 "------------ Navigation ----------\n"
				 "-- 5. Set class navigation by voice(NOT WORKS)\n"
				 "-- 6. Set class navigation\n"
				 "-- 7. Enable Navigation\n"
				 "-----------------------------------\n"
				 "-- 9. Clear current class(-es)\n"
				 "-- 0. Cancel\n"
				 "Select: ";
	std::cout << std::flush;
	std::cin >> choice;
	
	if (!(*controller))
		(*controller) = new ControlDisplayedObjects(imgSize, classesPath);

	switch (choice)
	{
	case 1:
		//(*controller)->setDesClass( (*controller)->recognizeSpeech() );
		//break;
	case 2:
		(*controller)->setDesClass();
		break;
	case 3:
		(*controller)->deleteDesiredClass();
		break;
	case 4:
		(*controller)->enableDesiredClasses();
		break;
	case 5:
		//(*controller)->setNavigationClass((*controller)->recognizeSpeech());
		//break;
	case 6:
		(*controller)->setNavigationClass();
		break;
	case 7:
		(*controller)->enableNavigation();
		break;
	case 9:
		(*controller)->clear();
		break;
	default:
		std::cout << std::endl << ">> Cancel" << std::endl;
		break;
	}


	return;
}