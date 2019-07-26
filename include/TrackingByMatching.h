#pragma once
#include <cmath>

#include <core.hpp>
#include <core/ocl.hpp>

#include <video.hpp>

#include "DnnDetector.h"



// --TODO
// Coefficients (need to conduct tests to good worked)

#define TRACKER_MIN_AREAS_PERCENT	   0.7
#define TRACKER_MIN_COVERAGE		   0.6
#define TRACKER_MAX_EUC_DISTANCE	   100
#define TRACKER_MIN_CONFIDENCE_PERCENT 0.95

#define TRAKER_CHECK_WEIGHT        0.7

#define TRACKER_WEIGHT_AREA        0.25
#define TRACKER_WEIGHT_COVERAGE    0.2
#define TRACKER_WEIGHT_EUC         0.1
#define TRACKER_WEIGHT_CONFIDENCE  0.05
#define TRACKER_WEIGHT_CLASS_ID    0.4

#define TRACKER_MIN_MISSED  7
#define TRACKER_MAX_MISSED  100
#define TRACKER_MIN_TRACKED 20



//
// Tracked object
struct TrackedObject
{
	// External / internal id
	std::int32_t id_ext;
	std::int32_t id_int;

	// Name and id of object class (from detector)
	std::int32_t class_id;
	std::string classname;

	// From detector
	std::double_t confidence;

	// Object area on frame (from detector)
	cv::Rect box;
	cv::Point2d cm, cmPrev;

	// Distance + Smoothed Distance
	// (calculation using stereopair)
	// Calculation implemented in another module
	std::double_t distance;
	std::double_t distAvg;

	// Counters
	std::int32_t missed;
	std::int32_t tracked;

	// --TODO Works
	// Path from center points
	std::vector<cv::Point2d> objPath;

	TrackedObject() :
		id_ext(-1),
		id_int(-1),
		class_id(-1),
		classname("None"),
		confidence(-1.0),
		box(0, 0, 0, 0),
		cm(-1, -1),
		distance(-1),
		distAvg(-1),
		missed(0),
		tracked(0)
	{}
};



// The class implements object tracking based on matching.
// Data for analysis comes from the detector
//
// Трекер реализует несколько проверок, каждая из которых имеет свой вес
// Сумма весов равна 1. Для хорошей работы необходимо подбирать коэффициенты
class TrackingByMatching
{
public:
	TrackingByMatching() {}
	~TrackingByMatching() {}

	std::vector<TrackedObject> track(const std::vector<DetectedObject> &objects);

	std::vector<TrackedObject> getTrackedObjects() const { return m_tracked_objects; }

private:
	std::vector<TrackedObject> m_tracked_objects;

	void initializationObjects(const std::vector<DetectedObject> &detected_objects);
	void addTrObject(const DetectedObject &dObj);
	void updateTrObject(const DetectedObject &dObj, TrackedObject &tObj);
	void updateTrObject(const TrackedObject &tObj1, TrackedObject &tObj2);

	void checkTracked();
	void checkMissed();
	void checkRepeatObjects();
	void eraseObject(std::int32_t id_int);

	std::int32_t createUniqueExtId() const;
	std::int32_t createUniqueIntId() const;
};

