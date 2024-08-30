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
		std::vector<MapPoint> mapPoints;
	} Object;	
	//Constructor 
	YoloDetect();
	//yolo detection function
	void Detect();
	//thread function 
	void Run();
private:
	Object mObject;
};
}// namespace ORB_SLAM3
#endif // DETECTOR_H