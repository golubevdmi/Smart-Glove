#include "ControlDisplayedObjects.h"

// --TODO 
// Sound prompt
//sf::Music sound;
void playSound(std::string playPath)
{
	if (playPath.empty())	return;

	//sound.openFromFile(playPath);
	//sound.play();
}

bool isNumber(const std::string& s)
{
	return !s.empty() && std::find_if(s.begin(),
		s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}

//
// Check at the intersection of areas
bool ControlIsHit(cv::Rect box1, cv::Rect box2)
{
	return	!(box2.x > box1.x + box1.width ||
		box2.y > box1.y + box1.height ||
		box1.x > box2.x + box2.width ||
		box1.y > box2.y + box2.height);
}

//
// Returns area coverage percentage
std::double_t ControlGetAreasCoverage(cv::Rect box1, cv::Rect box2)
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

	if (box2.area() > box1.area()) return square / box1.area();

	return square / box2.area();
}


ControlDisplayedObjects::ControlDisplayedObjects(cv::Size imgSize, std::string classesPath) :
	m_img_size(imgSize),
	m_box_nav(cv::Rect()),
	m_isNavigationSet(false),
	m_isDesiredSet(false)
{
	// Get classes names
	if (!classesPath.empty())
		addClassesToVector(classesPath);

	CV_Assert( (m_img_size.width) > 0 && (m_img_size.height > 0) );
	
	m_box_center = cv::Rect(cv::Point(imgSize.width / 4, imgSize.height / 4),
		cv::Point(imgSize.width - imgSize.width / 4, imgSize.height - imgSize.height / 4));
}

// Defining a new tracked object
// Get a string with id class or class name
void ControlDisplayedObjects::setDesClass(std::string strClass)
{
	std::int32_t classId = getClassIdByString(strClass);
	if (classId >= 0 && classId < m_classes_names.size())
	{
		std::cout << ">> Class < " << m_classes_names[classId] << " > added" << std::endl;
		m_desired_classes.push_back(classId);
	}
	else
	{
		std::cout << "Class not found" << std::endl;
	}

	if (!m_desired_classes.empty())
	{
		m_isDesiredSet = true;
		m_isNavigationSet = false;
	}
}

// 
// Defining a new navigation class
void ControlDisplayedObjects::setNavigationClass(std::string strClass)
{
	m_id_nav = getClassIdByString(strClass);

	if (m_id_nav >= 0 && m_id_nav < m_classes_names.size())
	{
		std::cout << "Class < " << m_classes_names[m_id_nav] << " > added" << std::endl;
	}
	else
	{
		std::cout << "Class not found" << std::endl;
		return;
	}

	m_isNavigationSet = true;
	m_isDesiredSet = false;
}

//
// Voice navigation
void ControlDisplayedObjects::navigate()
{
	if (m_id_nav == -1)
	{
		std::cout << "Speech: No class selected" << std::endl;
		return;
	}
	if (!m_isNavigationSet)
	{
		std::cout << "Speech: Navigation disabled" << std::endl;
		return;
	}
	if (m_box_nav.width <= 0 || m_box_nav.height <= 0)
	{
		std::cout << "Speech: Box empty" << std::endl;
		return;
	}


	// --TODO
	if (ControlIsHit(m_box_nav, m_box_center) && ControlGetAreasCoverage(m_box_nav, m_box_center) > 0.7)
	{
		playSound(SOUND_PLAY_CENTER);
	}
	else
	{
		//if (m_box_nav.x < m_box_center.width / 2) playSound(SOUND_PLAY_LEFT);
		//else playSound(SOUND_PLAY_RIGHT);

		bool Right = false;
		bool Left = false;
		bool Top = false;
		bool Bottom = false;

		if (m_box_nav.x + m_box_nav.width/2 < m_box_center.x)
		{
			playSound(SOUND_PLAY_LEFT);
			std::cout << ">> The object is on the left" << std::endl;
		}
		else if (m_box_nav.x + m_box_nav.width / 2 > m_box_center.width)
		{
			playSound(SOUND_PLAY_RIGHT);
			std::cout << ">> The object is on the right" << std::endl;
		}


		if (!Right && !Left)
		{

			if (m_box_nav.y + m_box_nav.height/2 < m_box_center.y)
			{
				Bottom = true;
				playSound(SOUND_PLAY_TOP);
				std::cout << ">> The object is on the top" << std::endl;
			}
			else if (m_box_nav.y + m_box_nav.height/2 > m_box_center.height)
			{
				Top = true;
				playSound(SOUND_PLAY_BOTTOM);
				std::cout << ">> The object is on the bottom" << std::endl;
			}
		}
	}


	m_box_nav.x = 0;
	m_box_nav.y = 0;
	m_box_nav.width = 0;
	m_box_nav.height = 0;
}


// --TODO
// Voice recognition
std::string ControlDisplayedObjects::recognizeSpeech()
{
	auto config = SpeechConfig::FromSubscription("dbd4dc8a08674841b68501658af95203", "westus");
	auto recognizer = SpeechRecognizer::FromConfig(config);

	std::cout << ">> Speak!" << std::endl;

	auto result = recognizer->RecognizeOnceAsync().get();

	if (result->Reason == ResultReason::RecognizedSpeech)
	{
		// Speech recognized

		std::cout << "We recognized: " << result->Text << std::endl;
		std::string speech = result->Text;
		speech.pop_back();
		std::transform(speech.begin(), speech.end(), speech.begin(), tolower);
		return speech;
	}
	else if (result->Reason == ResultReason::NoMatch)
	{
		// Speech Unrecognized

		std::cout << "NOMATCH: Speech could not be recognized." << std::endl;
	}
	else if (result->Reason == ResultReason::Canceled)
	{
		// Cancel

		auto cancellation = CancellationDetails::FromResult(result);
		std::cout << "CANCELED: Reason=" << (int)cancellation->Reason << std::endl;

		if (cancellation->Reason == CancellationReason::Error)
		{
			std::cout << "CANCELED: ErrorCode= " << (int)cancellation->ErrorCode << std::endl;
			std::cout << "CANCELED: ErrorDetails=" << cancellation->ErrorDetails << std::endl;
			std::cout << "CANCELED: Did you update the subscription info?" << std::endl;
		}
	}

	return std::string();
}

//
// Enables navigation through the selected class.
void ControlDisplayedObjects::enableNavigation()
{
	if (m_id_nav != -1)
	{
		m_isNavigationSet = true;
		m_isDesiredSet = false;
	}
	else
	{
		std::cout << "Speech: No class selected" << std::endl;
	}
}

//
// Enables the display of selected classes.
void ControlDisplayedObjects::enableDesiredClasses()
{
	if (!m_desired_classes.empty())
	{
		m_isNavigationSet = false;
		m_isDesiredSet = true;
	}
	else
	{
		std::cout << "No classes selected" << std::endl;
	}
}

//
// Deleting desired
void ControlDisplayedObjects::deleteDesiredClass()
{
	if (m_desired_classes.empty())
	{
		std::cout << "Classes not found" << std::endl;
		return;
	}
	if (m_desired_classes.size() == 1)
	{
		clear();
		return;
	}
	std::int32_t counter = 0;
	for (auto classId : m_desired_classes)
	{
		std::cout << "- [ " << counter << "]: " << m_classes_names[classId] << std::endl;
		counter++;
	}

	std::int32_t choice = 0;
	std::cout << ">> Press -1 to clear vector of objects\n"
		         "Set class: ";
	std::cout << std::flush;
	std::cin >> choice;

	if (choice > counter)
		std::cout << "Class not found" << std::endl;
	else if (choice == -1)
		clear();
	else if (choice >= 0)
		m_desired_classes.erase(m_desired_classes.begin() + choice);
}

//
// Clears current tracking
void ControlDisplayedObjects::clear()
{
	if (m_isDesiredSet)
	{
		m_desired_classes.clear();
		m_isDesiredSet = false;
	}
	if (m_isNavigationSet)
	{
		m_id_nav = -1;
		m_isNavigationSet = false;
	}

	std::cout << ">> Objects clear" << std::endl;
}

//
// Add classes to vector
void ControlDisplayedObjects::addClassesToVector(std::string classesPath)
{
	std::ifstream in(classesPath, std::ios::in);
	if (in.is_open())
	{
		std::string line;
		while (std::getline(in, line))
			m_classes_names.push_back(line);

		in.close();
	}
}

//
// String parsing
std::int32_t ControlDisplayedObjects::getClassIdByString(std::string strClass)
{
	// Если классов нет
	if (m_classes_names.empty())
	{
		std::cout << "Classes not found" << std::endl;
		return -1;
	}

	if (strClass.empty())
	{
		std::cout << ">> Enter classId or name: ";
		std::cin >> strClass;
	}

	std::int32_t classId = -1;

	// Поиск по ид
	if (isNumber(strClass))
	{
		classId = std::stoi(strClass);
	}
	else
	{
		// Поиск по названию класса
		std::uint32_t counter = 0;
		for (auto className : m_classes_names)
		{
			if (strClass == className)
				break;

			counter++;
		}

		classId = counter;
	}

	// Проверка на существование
	for (auto id : m_desired_classes)
	{
		if (id == classId)
		{
			classId = -1;
			break;
		}
	}

	return classId;
}

//
// If navigation is enabled, returns id
bool ControlDisplayedObjects::getNavigationId(std::int32_t &id)
{
	if (!m_isNavigationSet || m_id_nav == -1)
	{
		id = -1;
		return false;
	}
	id = m_id_nav;

	return true;
}

//
// If mapping is enabled, returns the id vector
bool ControlDisplayedObjects::getDesClasses(std::vector<std::int32_t> &desiredClasses)
{
	if (!m_isDesiredSet || m_desired_classes.empty())	
		return false;

	desiredClasses = m_desired_classes;

	return true;
}