#include "MapPoint.h"
#include <opencv2/opencv.hpp>
#include "YoloDetect.h"


namespace ORB_SLAM3
{
	static const int INPUT_W = 640;
	static const int INPUT_H = 640;
	YoloDetect::YoloDetect()
	{


	}
	//Just for test
	void YoloDetect::AddNewObject(int area_x, int area_y, int area_width, int area_height)
	{
		Object newObject;
		newObject.area = cv::Rect(area_x, area_y, area_width, area_height);
		newObject.label = 0;
		newObject.mapPoints = vector<MapPoint*>(1,static_cast<MapPoint*>(NULL));
		mObjects.push_back(newObject);			
	}
	void YoloDetect::Detect()
	{
		if(mObjects.size()<1)
			AddNewObject(10,10,100,100);
	}
	void YoloDetect::Run()
	{
		while(true)
		{
			Detect();
			cout <<"YoloDetect is here!"<<endl;
			usleep(100000);
		}
	}
} //namespace ORB_SLAM3
