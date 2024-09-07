#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>

namespace ORB_SLAM3
{
class YoloDetect
{
public:
	//struct for the objects.
	typedef struct{
		cv::Rect area;
		int label; 
		std::vector<cv::KeyPoint> keyPoints;
		std::vector<MapPoint*> mapPoints;
	} Object;	
	//Constructor 
	YoloDetect();
	//yolo detection function
	void Detect();
	// yolo detection variables
	torch::jit::script::Module mModule;
	std::vector<std::string> mClassnames;
	//thread function 
	void Run();
	void AddNewObject(int area_x, int area_y, int area_width, int area_height);
	void SetMapPoints(int objectIndex, const std::vector<MapPoint*>& newMapPoints);
	void SetKeyPoints(int objectIndex, const std::vector<cv::KeyPoint>& newKeyPoints);
	std::vector<Object> GetObjects();
	bool newObjct;
private:
	std::mutex mMutex;
	std::vector<Object> mObjects;
};
}// namespace ORB_SLAM3
#endif // DETECTOR_H