#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H

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
	//thread function 
	void Run();
	void AddNewObject(int area_x, int area_y, int area_width, int area_height);
	std::vector<Object> mObjects;
};
}// namespace ORB_SLAM3
#endif // DETECTOR_H