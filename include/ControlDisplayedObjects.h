#pragma once

#include <iostream>
#include <fstream>
#include <string>


#include <speechapi_cxx.h>

#include <SFML/Audio.hpp>

#include "highgui/highgui.hpp"


#define SOUND_PLAY_TOP   std::string("../data/sounds/Top.wav")
#define SOUND_PLAY_BOTTOM   std::string("../data/sounds/Bottom.wav")
#define SOUND_PLAY_LEFT   std::string("../data/sounds/Left.wav")
#define SOUND_PLAY_RIGHT  std::string("../data/sounds/Right.wav")
#define SOUND_PLAY_LEFT_TOP   std::string("../data/sounds/Top_left.wav")
#define SOUND_PLAY_LEFT_BOTTOM   std::string("../data/sounds/Bottom_left.wav")
#define SOUND_PLAY_RIGHT_TOP   std::string("../data/sounds/Top_right.wav")
#define SOUND_PLAY_RIGHT_BOTTOM  std::string("../data/sounds/Bottom_right.wav")
#define SOUND_PLAY_CENTER std::string("../data/sounds/Center.wav")


using namespace Microsoft::CognitiveServices::Speech;


// Allows the user to see only certain classes.
// either select one specific one and track it with the help of a voice assistant
class ControlDisplayedObjects
{
public:
	ControlDisplayedObjects(cv::Size imgSize, std::string classesPath = std::string());
	~ControlDisplayedObjects() {}


	void navigate();
	std::string recognizeSpeech();

	void deleteDesiredClass();
	void clear();

	void enableNavigation();
	void enableDesiredClasses();

	void setNavigationBox(cv::Rect box) { m_box_nav = box; }
	void setNavigationClass(std::string strClass = std::string());
	void setDesClass(std::string strClass = std::string());
	void setClassesNames(const std::vector<std::string> &classesNames) { m_classes_names = classesNames; }

	
	bool getNavigationId(std::int32_t &id);
	bool getDesClasses(std::vector<std::int32_t> &desiredClasses);

private:
	cv::Size m_img_size;

	// Parameters for voice navigation
	bool m_isNavigationSet;
	std::int32_t m_id_nav;
	cv::Rect m_box_center;
	cv::Rect m_box_nav;
	
	// Options to display the desired classes
	bool m_isDesiredSet;
	std::vector<std::int32_t> m_desired_classes;

	std::vector<std::string> m_classes_names;


	void addClassesToVector(std::string classesPath);
	std::int32_t getClassIdByString(std::string strClass);
};