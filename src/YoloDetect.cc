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
		std::lock_guard<std::mutex> lock(mMutex);
		Object newObject;
		newObject.area = cv::Rect(area_x, area_y, area_width, area_height);
		newObject.label = 0;
		newObject.mapPoints = vector<MapPoint*>(1,static_cast<MapPoint*>(NULL));
		mObjects.push_back(newObject);			
	}
	void YoloDetect::Detect()
	{
		//std::lock_guard<std::mutex> lock(mMutex);
		if(mObjects.size()<=1){
			AddNewObject(150,150,200,200);
			AddNewObject(200,200,100,200);
		}
	}
	std::vector<YoloDetect::Object> YoloDetect::GetObjects()
    {
        std::lock_guard<std::mutex> lock(mMutex); 
        return mObjects;
    }
    void YoloDetect::SetMapPoints(int objectIndex, const std::vector<MapPoint*>& newMapPoints)
    {
    	std::lock_guard<std::mutex> lock(mMutex);
    	if (objectIndex >= 0 && objectIndex < mObjects.size())
	    {
	        mObjects[objectIndex].mapPoints = newMapPoints; // Set new mapPoints for the specified object
	    }
    }
    void YoloDetect::SetKeyPoints(int objectIndex, const std::vector<cv::KeyPoint>& newKeyPoints)
    {
        std::lock_guard<std::mutex> lock(mMutex);
	    if (objectIndex >= 0 && objectIndex < mObjects.size())
	    {
	        mObjects[objectIndex].keyPoints = newKeyPoints; // Set new keyPoints for the specified object.
	    }
    }
	void YoloDetect::Run()
	{
		while(true)
		{
			Detect();
			usleep(1000);
		}
	}
} //namespace ORB_SLAM3
