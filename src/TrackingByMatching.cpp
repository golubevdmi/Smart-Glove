#include "TrackingByMatching.h"


// Check for intersection of object areas
bool checkCoverage(cv::Rect box1, cv::Rect box2);
// Check distance between center points
bool checkEucDistance(cv::Point cm1, cv::Point cm2);
// Area check
bool checkAreas(std::uint32_t area1, std::uint32_t area2) { return (std::double_t(area1) / std::double_t(area2)) > TRACKER_MIN_AREAS_PERCENT; }

// Returns the center point
cv::Point2d calcCm(cv::Rect box) { return cv::Point2d(box.x + box.width / 2, box.y + box.height / 2); }
// Check class ids objects
bool checkIds(std::int32_t class_id1, std::int32_t class_id2)					{ return class_id1 == class_id2; }
// Check objects classes names
bool checkClassnames(std::string classname1, std::string classname2)		{ return classname1 == classname2; }
// Check on the confidence of the detector for the object
bool checkConfidence(std::double_t confidence1, std::double_t confidence2)  { return (confidence1 / confidence2) > TRACKER_MIN_CONFIDENCE_PERCENT; }



// Matching objects. Matching to the previous frame.
// Выполняются несколько проверок, каждая из которых имеет свой вес
// Сумма весов равна 1. Если объект проходит установленный весовой порог - объекты сопоставлены
std::vector<TrackedObject> TrackingByMatching::track(const std::vector<DetectedObject> &detected_objects)
{
	// Инициализирует первоначальными значениями,
	// если пустой
	if (m_tracked_objects.empty())	initializationObjects(detected_objects);

	// Добавляет пропуск ко всем объектам
	// В дальнейшем эта переменная зануляется,
	// если объект был сопоставлен
	for (auto &tObj : m_tracked_objects)	
		tObj.missed++;

	for (auto dObj : detected_objects)
	{
		bool isMatch = false;
		for (auto &tObj : m_tracked_objects)
		{
			bool isAreas	  = checkAreas(tObj.box.area(), dObj.box.area());
			bool isCoverage	  = checkCoverage(tObj.box, dObj.box);
			bool isEuc		  = checkEucDistance(tObj.cm, calcCm(dObj.box));
			bool isId		  = checkIds(tObj.class_id, dObj.class_id);
			bool isConfidence = checkConfidence(tObj.confidence, dObj.confidence);

			
			if ( isAreas *		TRACKER_WEIGHT_AREA + 
				 isCoverage *	TRACKER_WEIGHT_COVERAGE +
				 isEuc *		TRACKER_WEIGHT_EUC +
				 isId *			TRACKER_WEIGHT_CLASS_ID +
				 isConfidence * TRACKER_WEIGHT_CONFIDENCE
	             > TRAKER_CHECK_WEIGHT)
			{
				updateTrObject(dObj, tObj);

				isMatch = true;
				break;
			}
		}

		if (!isMatch)
			addTrObject(dObj);
	}

	checkRepeatObjects();

	checkMissed();
	checkTracked();

	return m_tracked_objects;
}

// 
// Generate the initial object vector.
void TrackingByMatching::initializationObjects(const std::vector<DetectedObject> &detected_objects)
{
	for (auto &dObj : detected_objects)
		addTrObject(dObj);
}

// --TODO
// Repeat check
// Same check as match, but different threshold for passing
void TrackingByMatching::checkRepeatObjects()
{
	std::vector<std::int32_t> ids;

	for (auto tObjSrc : m_tracked_objects)
	{
		if (tObjSrc.id_ext == -1)	continue;

		bool isFound = false;
		for (auto id : ids)
		{
			if (id == tObjSrc.id_int)
				isFound = true;
		}
		if (isFound)
			continue;

		for (auto tObjVer : m_tracked_objects)
		{
			if (tObjSrc.id_int == tObjVer.id_int)	
				continue;

			bool isAreas	  = checkAreas(tObjSrc.box.area(), tObjVer.box.area());
			bool isCoverage   = checkCoverage(tObjSrc.box, tObjVer.box);
			bool isEuc		  = checkEucDistance(tObjSrc.cm, tObjVer.cm);
			bool isId		  = checkIds(tObjSrc.class_id, tObjVer.class_id);
			bool isConfidence = checkConfidence(tObjSrc.confidence, tObjVer.confidence);

			if (isAreas *		TRACKER_WEIGHT_AREA +
				isCoverage *	TRACKER_WEIGHT_COVERAGE +
				isEuc *			TRACKER_WEIGHT_EUC +
				isId *			TRACKER_WEIGHT_CLASS_ID +
				isConfidence *	TRACKER_WEIGHT_CONFIDENCE
				 > TRAKER_CHECK_WEIGHT)
			{
				//std::cout << "ver: " << tObjVer.id_int << " ," << tObjVer.id_ext << "," << tObjVer.classname << std::endl
				//	<< "src: " << tObjSrc.id_int << ", " << tObjSrc.id_ext << ", " << tObjSrc.classname << "\n" << std::endl;

				if (tObjVer.id_ext != -1 && tObjSrc.id_ext > tObjVer.id_ext)
				{
					if (tObjSrc.missed < tObjVer.missed)
						updateTrObject(tObjSrc, tObjVer);

					ids.push_back(tObjSrc.id_int);
				}
				else
					ids.push_back(tObjVer.id_int);
			}
		}
	}

	for (auto id : ids)
		eraseObject(id);
}

//
// Add new tracked object
void TrackingByMatching::addTrObject(const DetectedObject &dObj)
{
	TrackedObject tObj;

	updateTrObject(dObj, tObj);

	m_tracked_objects.push_back(tObj);
}

//
// Update tracked object fields by detected object
void TrackingByMatching::updateTrObject(const DetectedObject &dObj, TrackedObject &tObj)
{
	tObj.class_id = dObj.class_id;
	tObj.classname = dObj.classname;
	tObj.confidence = dObj.confidence;
	tObj.box = dObj.box;

	if (tObj.id_int == -1)
		tObj.id_int = createUniqueIntId();

	tObj.tracked++;
	tObj.missed = 0;

	tObj.cm = calcCm(tObj.box);
	tObj.objPath.push_back(tObj.cm);
}
// Update tracked object fields by tracked object
void TrackingByMatching::updateTrObject(const TrackedObject &tObj1, TrackedObject &tObj2)
{
	tObj2.confidence = tObj1.confidence;
	tObj2.box = tObj1.box;

	tObj2.tracked++;
	tObj2.missed = 0;

	tObj2.cm = calcCm(tObj2.box);
	tObj2.objPath.push_back(tObj2.cm);
}

// Checks how many objects were tracked
// If enough, assign an external id
void TrackingByMatching::checkTracked()
{
	for (auto &tObj : m_tracked_objects)
		if (tObj.id_ext == -1 && tObj.tracked > TRACKER_MIN_TRACKED)
			tObj.id_ext = createUniqueExtId();

	// Сортирует массив по внешним ид
	std::sort(m_tracked_objects.begin(), m_tracked_objects.end(), [](const TrackedObject &tObj1, const TrackedObject &tObj2) -> bool
	{
		if (tObj1.id_ext != -1 && tObj2.id_ext == -1)	    return true;
		else if (tObj1.id_ext == -1 && tObj2.id_ext != -1)	return false;
		else if (tObj1.id_ext == -1 && tObj2.id_ext == -1)	return false;
		else														return tObj1.id_ext < tObj2.id_ext;
	});

	//for (auto tObj : m_tracked_objects)
	//	std::cout << tObj.id_ext << " ";
	//std::cout << std::endl;
}

// Checks how many frames the object was not detected.
// If more than a certain number, delete
void TrackingByMatching::checkMissed()
{
	std::vector<std::int32_t> idsInt;

	for (auto tObj : m_tracked_objects)
		if (tObj.missed > TRACKER_MAX_MISSED)
			idsInt.push_back(tObj.id_int);

	for (auto idInt : idsInt)
		eraseObject(idInt);
}

//
// Deleting object by internal id
void TrackingByMatching::eraseObject(std::int32_t id_int)
{
	std::uint32_t counter = 0;
	for (auto tObj : m_tracked_objects)
	{
		if (tObj.id_int == id_int)
			break;

		counter++;
	}

	if (counter >= m_tracked_objects.size())
		return;

	m_tracked_objects.erase(m_tracked_objects.begin() + counter);
}

//
// Returns a unique external id
std::int32_t TrackingByMatching::createUniqueExtId() const
{
	static std::int32_t id = 0;
	return id++;
}

//
// Returns a unique internal id
std::int32_t TrackingByMatching::createUniqueIntId() const
{
	static std::int32_t id = 0;
	return id++;
}


//
// Check for hit of one area in another
bool isHit(cv::Rect box1, cv::Rect box2)
{
	return	!(box2.x > box1.x + box1.width ||
		box2.y > box1.y + box1.height ||
		box1.x > box2.x + box2.width ||
		box1.y > box2.y + box2.height);
}

//
// Returns areas coverage percentage
std::double_t getAreasCoverage(cv::Rect box1, cv::Rect box2)
{
	// Top left
	std::double_t x1 = box1.x;
	std::double_t y1 = box1.y;
	std::double_t x3 = box2.x;
	std::double_t y3 = box2.y;

	// Bottom right
	std::double_t x2 = box1.x + box1.width;
	std::double_t y2 = box1.y + box1.height;
	std::double_t x4 = box2.x + box2.width;
	std::double_t y4 = box2.y + box2.height;

	std::double_t left = 0.0, right = 0.0;
	left = std::max(x1, x3);
	right = std::min(x2, x4);

	std::double_t top = 0.0, bottom = 0.0;
	top = std::max(y1, y3);
	bottom = std::min(y2, y4);

	std::double_t square = (right - left) * (bottom - top);

	return square / (box1.area() + box2.area() - square);
}

// Hit test and crossing percentage
// If above a certain threshold, the test passed
bool checkCoverage(cv::Rect box1, cv::Rect box2) { return isHit(box1, box2) && (getAreasCoverage(box1, box2) > TRACKER_MIN_COVERAGE); }


std::double_t getEuclideanDistance(cv::Point pt1, cv::Point pt2) { return sqrt(pow(pt1.x - pt2.x, 2) - pow((pt1.y - pt2.y), 2)); }

// Check for maximum Euclidean distance
// If more than the threshold, then false
bool checkEucDistance(cv::Point cm1, cv::Point cm2) { return (getEuclideanDistance(cm1, cm2) < TRACKER_MAX_EUC_DISTANCE); }