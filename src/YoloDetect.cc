#include "MapPoint.h"
#include <opencv2/opencv.hpp>
#include "YoloDetect.h"


namespace ORB_SLAM3
{
	static const int INPUT_W = 640;
	static const int INPUT_H = 640;
	YoloDetect::YoloDetect()
	{
		mObject.area = cv::Rect(10, 20, 10, 10);
		mObject.label = 0;
	}
	//Just for test
	void YoloDetect::Detect()
	{
		mObject.area = cv::Rect(10, 20, 10, 10);
	}
	void YoloDetect::Run()
	{
		while(true)
		{
			Detect();
			cout <<"YoloDetect is here!"<<endl;
			usleep(1000);
		}
	}
} //namespace ORB_SLAM3
